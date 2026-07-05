#include <stdint.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef int ncclResult_t;
typedef void *cudaStream_t;

typedef struct {
    char internal[128];
} ncclUniqueId;

#define HETGPU_NCCL_MAX_RANKS 256

typedef struct HetGpuNcclComm {
    int nranks;
    int rank;
    int device;
    int aborted;
    int recovery_enabled;
    int failed_rank;
    int replacement_rank;
    int active_nranks;
    int active_ranks[HETGPU_NCCL_MAX_RANKS];
    ncclResult_t async_error;
    uint32_t consecutive_errors;
    uint64_t sequence;
} *ncclComm_t;

enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInvalidArgument = 3,
    ncclInvalidUsage = 5,
};

static int hetgpu_nccl_logs_enabled(void) {
    const char *env = getenv("HETGPU_NCCL_LOGS");
    return env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0 || strcmp(env, "on") == 0);
}

#define NCCL_LOG(...) do { if (hetgpu_nccl_logs_enabled()) fprintf(stderr, __VA_ARGS__); } while (0)

extern int hetgpu_sifive_nccl_all_reduce_f32(
    const float *sendbuff,
    float *recvbuff,
    size_t count,
    int op,
    int rank,
    int nranks
) __attribute__((weak));

extern int hetgpu_sifive_nccl_reduce_sum_f32(
    const float *rank_inputs,
    float *recvbuff,
    size_t count,
    int nranks
) __attribute__((weak));

extern int hetgpu_concordia_checkpoint_boundary(const char *boundary) __attribute__((weak));
extern int hetgpu_concordia_nccl_plan_replacement(
    int nranks,
    int failed_rank,
    int replacement_rank,
    int *out_ranks,
    int out_len
) __attribute__((weak));
extern int hetgpu_concordia_nccl_classify_failure(
    int result,
    int async_error,
    uint32_t consecutive_errors,
    int health_degraded
) __attribute__((weak));

static int ensure_dir(const char *path);
static void concordia_record_recovery_event(
    ncclComm_t comm,
    const char *event,
    const char *status,
    const char *reason
);

static int env_enabled(const char *name) {
    const char *env = getenv(name);
    return env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0 || strcmp(env, "on") == 0);
}

static void concordia_checkpoint_boundary(const char *boundary) {
    if (!hetgpu_concordia_checkpoint_boundary) return;
    if (!env_enabled("HETGPU_CONCORDIA_NCCL_BOUNDARY") &&
        !env_enabled("HETGPU_CONCORDIA_BOUNDARY") &&
        !env_enabled("CONCORDIA_CHECKPOINT_ON_BOUNDARY")) {
        return;
    }
    (void)hetgpu_concordia_checkpoint_boundary(boundary);
}

static int read_env_i32(const char *name, int fallback) {
    const char *env = getenv(name);
    if (!env || !*env) return fallback;
    char *end = NULL;
    long v = strtol(env, &end, 10);
    if (!end || *end != '\0') return fallback;
    return (int)v;
}

static int read_env_i32_any(const char **names, size_t count, int fallback) {
    for (size_t i = 0; i < count; ++i) {
        const char *env = getenv(names[i]);
        if (!env || !*env) continue;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (end && *end == '\0') return (int)v;
    }
    return fallback;
}

static const char *read_env_any(const char **names, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const char *env = getenv(names[i]);
        if (env && *env) return env;
    }
    return NULL;
}

static void sanitize_path_token(const char *input, char *output, size_t output_len) {
    if (!output || output_len == 0) return;
    if (!input || !*input) input = "default";
    size_t j = 0;
    for (size_t i = 0; input[i] && j + 1 < output_len; ++i) {
        char c = input[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.') {
            output[j++] = c;
        } else {
            output[j++] = '_';
        }
    }
    output[j] = '\0';
}

static int concordia_recovery_enabled(void) {
    return env_enabled("HETGPU_CONCORDIA_NCCL_RECOVERY") ||
           env_enabled("CONCORDIA_NCCL_RECOVERY");
}

static void concordia_job_token(char *output, size_t output_len) {
    static const char *job_keys[] = {
        "HETGPU_NCCL_COMM_ID",
        "MASTER_PORT",
        "OMPI_COMM_WORLD_JOBID",
        "PMIX_NAMESPACE",
        "PMI_JOBID",
        "SLURM_JOB_ID"
    };
    const char *job = getenv("HETGPU_NCCL_COMM_ID");
    if (!job || !*job) job = read_env_any(job_keys, sizeof(job_keys) / sizeof(job_keys[0]));
    if (!job || !*job) job = "default";
    sanitize_path_token(job, output, output_len);
}

static void format_active_ring(const int *ranks, int count, char *output, size_t output_len) {
    if (!output || output_len == 0) return;
    output[0] = '\0';
    size_t used = 0;
    for (int i = 0; i < count; ++i) {
        int written = snprintf(output + used, output_len - used, "%s%d", i == 0 ? "" : ",", ranks[i]);
        if (written < 0) break;
        if ((size_t)written >= output_len - used) {
            output[output_len - 1] = '\0';
            break;
        }
        used += (size_t)written;
    }
}

static int fallback_plan_replacement(
    int nranks,
    int failed_rank,
    int replacement_rank,
    int *out_ranks,
    int out_len
) {
    if (!out_ranks || nranks <= 0 || out_len < nranks) return -1;
    if (failed_rank < 0 || failed_rank >= nranks) return -2;
    if (replacement_rank >= 0 && replacement_rank < nranks && replacement_rank != failed_rank) return -3;
    for (int rank = 0; rank < nranks; ++rank) {
        out_ranks[rank] = (rank == failed_rank) ? replacement_rank : rank;
    }
    return nranks;
}

static int concordia_plan_replacement(
    int nranks,
    int failed_rank,
    int replacement_rank,
    int *out_ranks,
    int out_len
) {
    if (hetgpu_concordia_nccl_plan_replacement) {
        return hetgpu_concordia_nccl_plan_replacement(
            nranks,
            failed_rank,
            replacement_rank,
            out_ranks,
            out_len
        );
    }
    return fallback_plan_replacement(nranks, failed_rank, replacement_rank, out_ranks, out_len);
}

static int concordia_rank_allowed_for_recovery(int nranks, int rank) {
    if (rank >= 0 && rank < nranks) return 1;
    if (!concordia_recovery_enabled()) return 0;
    int replacement_rank = read_env_i32("HETGPU_CONCORDIA_NCCL_REPLACEMENT_RANK", -1);
    return rank == replacement_rank;
}

static int comm_rank_is_active(ncclComm_t comm, int rank) {
    if (!comm) return 0;
    int active = comm->active_nranks > 0 ? comm->active_nranks : comm->nranks;
    for (int i = 0; i < active; ++i) {
        int active_rank = comm->active_nranks > 0 ? comm->active_ranks[i] : i;
        if (active_rank == rank) return 1;
    }
    return 0;
}

static int comm_coordinator_rank(ncclComm_t comm) {
    if (!comm) return 0;
    if (comm->active_nranks > 0) return comm->active_ranks[0];
    return 0;
}

static int comm_active_count(ncclComm_t comm) {
    if (!comm) return 0;
    return comm->active_nranks > 0 ? comm->active_nranks : comm->nranks;
}

static int comm_active_rank_at(ncclComm_t comm, int index) {
    if (!comm) return index;
    if (comm->active_nranks > 0) return comm->active_ranks[index];
    return index;
}

static void concordia_configure_recovery_plan(ncclComm_t comm) {
    if (!comm) return;
    comm->recovery_enabled = concordia_recovery_enabled();
    comm->failed_rank = read_env_i32("HETGPU_CONCORDIA_NCCL_FAILED_RANK", -1);
    comm->replacement_rank = read_env_i32("HETGPU_CONCORDIA_NCCL_REPLACEMENT_RANK", -1);
    comm->active_nranks = comm->nranks < HETGPU_NCCL_MAX_RANKS ? comm->nranks : HETGPU_NCCL_MAX_RANKS;
    for (int rank = 0; rank < comm->active_nranks; ++rank) {
        comm->active_ranks[rank] = rank;
    }

    if (!comm->recovery_enabled || comm->failed_rank < 0 || comm->replacement_rank < 0) {
        return;
    }

    int planned[HETGPU_NCCL_MAX_RANKS];
    int count = concordia_plan_replacement(
        comm->nranks,
        comm->failed_rank,
        comm->replacement_rank,
        planned,
        HETGPU_NCCL_MAX_RANKS
    );
    if (count <= 0 || count > HETGPU_NCCL_MAX_RANKS) {
        NCCL_LOG("[hetGPU nccl_shim] Concordia recovery plan failed rc=%d nranks=%d failed=%d replacement=%d\n",
                 count, comm->nranks, comm->failed_rank, comm->replacement_rank);
        concordia_record_recovery_event(comm, "recovery_plan", "error", "plan_failed");
        return;
    }

    comm->active_nranks = count;
    memcpy(comm->active_ranks, planned, sizeof(int) * (size_t)count);

    char ring[1024];
    format_active_ring(comm->active_ranks, comm->active_nranks, ring, sizeof(ring));
    NCCL_LOG("[hetGPU nccl_shim] Concordia recovery plan failed=%d replacement=%d active_ring=%s\n",
             comm->failed_rank, comm->replacement_rank, ring);
    concordia_record_recovery_event(comm, "recovery_plan", "configured", "env");
}

static void concordia_record_recovery_event(
    ncclComm_t comm,
    const char *event,
    const char *status,
    const char *reason
) {
    if (!concordia_recovery_enabled() && !(comm && comm->recovery_enabled)) return;
    const char *dir = getenv("HETGPU_CONCORDIA_NCCL_RECOVERY_DIR");
    if (!dir || !*dir) dir = "/tmp/hetgpu_concordia_nccl_recovery";
    if (ensure_dir(dir) != 0) return;

    char job_token[128];
    concordia_job_token(job_token, sizeof(job_token));

    char path[512];
    int rank = comm ? comm->rank : read_env_i32("RANK", -1);
    snprintf(path, sizeof(path), "%s/events.%s.rank%d.jsonl", dir, job_token, rank);

    FILE *f = fopen(path, "a");
    if (!f) return;

    int active_nranks = comm ? comm_active_count(comm) : 0;
    int nranks = comm ? comm->nranks : 0;
    int failed_rank = comm ? comm->failed_rank : read_env_i32("HETGPU_CONCORDIA_NCCL_FAILED_RANK", -1);
    int replacement_rank = comm ? comm->replacement_rank : read_env_i32("HETGPU_CONCORDIA_NCCL_REPLACEMENT_RANK", -1);
    uint64_t sequence = comm ? comm->sequence : 0;
    char ring[1024];
    if (comm && comm->active_nranks > 0) {
        format_active_ring(comm->active_ranks, comm->active_nranks, ring, sizeof(ring));
    } else {
        ring[0] = '\0';
    }

    fprintf(
        f,
        "{\"event\":\"%s\",\"status\":\"%s\",\"rank\":%d,\"nranks\":%d,\"active_nranks\":%d,"
        "\"failed_rank\":%d,\"replacement_rank\":%d,\"active_ring\":\"%s\","
        "\"sequence\":%llu,\"reason\":\"%s\"}\n",
        event ? event : "unknown",
        status ? status : "unknown",
        rank,
        nranks,
        active_nranks,
        failed_rank,
        replacement_rank,
        ring,
        (unsigned long long)sequence,
        reason ? reason : ""
    );
    fclose(f);
}

static const char *concordia_action_status(int action) {
    switch (action) {
        case 0: return "retry";
        case 1: return "migrate";
        case 2: return "replace";
        default: return "unknown";
    }
}

static int concordia_classify_failure_action(ncclComm_t comm, ncclResult_t result) {
    if (!comm || result == ncclSuccess) return -1;
    if (hetgpu_concordia_nccl_classify_failure) {
        return hetgpu_concordia_nccl_classify_failure(
            (int)result,
            (int)comm->async_error,
            comm->consecutive_errors,
            0
        );
    }
    return comm->consecutive_errors >= 3 ? 2 : 0;
}

static void concordia_note_collective_result(ncclComm_t comm, ncclResult_t result) {
    if (!comm) return;
    if (result == ncclSuccess) {
        comm->consecutive_errors = 0;
        comm->async_error = ncclSuccess;
        return;
    }
    comm->consecutive_errors++;
    comm->async_error = result;
    int action = concordia_classify_failure_action(comm, result);
    concordia_record_recovery_event(
        comm,
        "failure_classified",
        concordia_action_status(action),
        "nccl_collective_error"
    );
}

static size_t dtype_size(int datatype) {
    switch (datatype) {
        case 0: return 1;  /* int8/char */
        case 1: return 1;  /* uint8 */
        case 2: return 4;  /* int32 */
        case 3: return 4;  /* uint32 */
        case 4: return 8;  /* int64 */
        case 5: return 8;  /* uint64 */
        case 6: return 2;  /* float16 */
        case 7: return 4;  /* float32 */
        case 8: return 8;  /* float64 */
        case 9: return 2;  /* bfloat16 */
        default: return 0;
    }
}

const char *ncclGetErrorString(ncclResult_t result) {
    switch (result) {
        case ncclSuccess: return "ncclSuccess";
        case ncclUnhandledCudaError: return "ncclUnhandledCudaError";
        case ncclSystemError: return "ncclSystemError";
        case ncclInvalidArgument: return "ncclInvalidArgument";
        case ncclInvalidUsage: return "ncclInvalidUsage";
        default: return "ncclUnknownError";
    }
}

const char *ncclGetLastError(ncclComm_t comm) {
    (void)comm;
    return "hetGPU NCCL shim: no async error";
}

ncclResult_t ncclGetVersion(int *version) {
    if (!version) return ncclInvalidArgument;
    *version = 21200;
    return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId) {
    if (!uniqueId) return ncclInvalidArgument;
    memset(uniqueId, 0x50, sizeof(*uniqueId));
    memcpy(uniqueId->internal, "HETGPU-SIFIVE-NCCL-SHIM", 21);
    return ncclSuccess;
}

static ncclResult_t make_comm(ncclComm_t *comm, int nranks, int rank, int device) {
    if (!comm || nranks <= 0 || rank < 0 || !concordia_rank_allowed_for_recovery(nranks, rank)) {
        return ncclInvalidArgument;
    }
    struct HetGpuNcclComm *created = (struct HetGpuNcclComm *)calloc(1, sizeof(*created));
    if (!created) return ncclSystemError;
    created->nranks = nranks;
    created->rank = rank;
    created->device = device;
    created->sequence = 0;
    created->async_error = ncclSuccess;
    concordia_configure_recovery_plan(created);
    concordia_record_recovery_event(created, "comm_init", "ok", "created");
    *comm = created;
    return ncclSuccess;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
    (void)commId;
    static const char *world_keys[] = {
        "WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMIX_SIZE",
        "PMI_SIZE",
        "MV2_COMM_WORLD_SIZE",
        "SLURM_NTASKS"
    };
    static const char *rank_keys[] = {
        "RANK",
        "OMPI_COMM_WORLD_RANK",
        "PMIX_RANK",
        "PMI_RANK",
        "MV2_COMM_WORLD_RANK",
        "SLURM_PROCID"
    };
    if (nranks <= 0) nranks = read_env_i32_any(world_keys, sizeof(world_keys) / sizeof(world_keys[0]), 1);
    if (rank < 0) rank = read_env_i32_any(rank_keys, sizeof(rank_keys) / sizeof(rank_keys[0]), 0);
    NCCL_LOG("[hetGPU nccl_shim] ncclCommInitRank nranks=%d rank=%d\n", nranks, rank);
    return make_comm(comm, nranks, rank, rank);
}

ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist) {
    if (!comms || ndev <= 0) return ncclInvalidArgument;
    NCCL_LOG("[hetGPU nccl_shim] ncclCommInitAll ndev=%d\n", ndev);
    for (int i = 0; i < ndev; ++i) {
        int device = devlist ? devlist[i] : i;
        ncclResult_t r = make_comm(&comms[i], ndev, i, device);
        if (r != ncclSuccess) return r;
    }
    return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    free(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    if (comm) {
        comm->aborted = 1;
        comm->async_error = ncclSystemError;
        concordia_record_recovery_event(comm, "comm_abort", "aborted", "api");
    }
    free(comm);
    return ncclSuccess;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
    (void)comm;
    return ncclSuccess;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
    if (!asyncError) return ncclInvalidArgument;
    *asyncError = comm ? comm->async_error : ncclSuccess;
    return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
    if (!comm || !count) return ncclInvalidArgument;
    *count = comm->nranks;
    return ncclSuccess;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
    if (!comm || !device) return ncclInvalidArgument;
    *device = comm->device;
    return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
    if (!comm || !rank) return ncclInvalidArgument;
    *rank = comm->rank;
    return ncclSuccess;
}

ncclResult_t ncclGroupStart(void) {
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd(void) {
    return ncclSuccess;
}

static int ensure_dir(const char *path) {
    if (mkdir(path, 0777) == 0 || errno == EEXIST) return 0;
    return -1;
}

static int wait_for_file(const char *path, int timeout_ms) {
    struct timespec sleep_time;
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 10 * 1000 * 1000;
    int waited = 0;
    while (access(path, R_OK) != 0) {
        if (waited >= timeout_ms) return -1;
        nanosleep(&sleep_time, NULL);
        waited += 10;
    }
    return 0;
}

static int write_file_exact(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t written = fwrite(data, 1, bytes, f);
    int close_rc = fclose(f);
    return (written == bytes && close_rc == 0) ? 0 : -1;
}

static int read_file_exact(const char *path, void *data, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t got = fread(data, 1, bytes, f);
    int close_rc = fclose(f);
    return (got == bytes && close_rc == 0) ? 0 : -1;
}

static void publish_abort(const char *abort_path, const char *reason) {
    if (!abort_path || !*abort_path) return;
    if (!reason || !*reason) reason = "aborted\n";
    (void)write_file_exact(abort_path, reason, strlen(reason));
}

static int wait_for_result_or_abort(const char *result_path, const char *abort_path, int timeout_ms) {
    struct timespec sleep_time;
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = 10 * 1000 * 1000;
    int waited = 0;
    while (access(result_path, R_OK) != 0) {
        if (abort_path && access(abort_path, R_OK) == 0) return 1;
        if (waited >= timeout_ms) return -1;
        nanosleep(&sleep_time, NULL);
        waited += 10;
    }
    return 0;
}

static ncclResult_t rendezvous_allreduce_f32(
    const float *sendbuff,
    float *recvbuff,
    size_t count,
    int op,
    ncclComm_t comm
) {
    if (!comm || op != 0) return ncclInvalidArgument;
    if (!comm_rank_is_active(comm, comm->rank)) {
        comm->async_error = ncclSystemError;
        concordia_record_recovery_event(comm, "collective_skip", "error", "rank_not_active");
        return ncclSystemError;
    }

    int timeout_ms = read_env_i32("HETGPU_NCCL_TIMEOUT_MS", 60000);
    char job_token[128];
    concordia_job_token(job_token, sizeof(job_token));

    char root[256];
    char call_dir[320];
    char rank_path[384];
    char result_path[384];
    char abort_path[384];
    snprintf(root, sizeof(root), "/dev/shm/hetgpu_nccl_%s", job_token);
    snprintf(call_dir, sizeof(call_dir), "%s/%llu", root, (unsigned long long)comm->sequence++);
    snprintf(rank_path, sizeof(rank_path), "%s/rank_%d.bin", call_dir, comm->rank);
    snprintf(result_path, sizeof(result_path), "%s/result.bin", call_dir);
    snprintf(abort_path, sizeof(abort_path), "%s/abort", call_dir);

    if (ensure_dir(root) != 0 || ensure_dir(call_dir) != 0) {
        perror("[hetGPU nccl_shim] mkdir");
        comm->async_error = ncclSystemError;
        concordia_record_recovery_event(comm, "collective_begin", "error", "mkdir_failed");
        return ncclSystemError;
    }
    concordia_record_recovery_event(comm, "collective_begin", "ok", "allreduce");
    if (write_file_exact(rank_path, sendbuff, count * sizeof(float)) != 0) {
        perror("[hetGPU nccl_shim] write rank file");
        comm->async_error = ncclSystemError;
        concordia_record_recovery_event(comm, "collective_abort", "error", "rank_write_failed");
        return ncclSystemError;
    }

    int active_count = comm_active_count(comm);
    int coordinator_rank = comm_coordinator_rank(comm);

    if (comm->rank == coordinator_rank) {
        int use_sifive = env_enabled("HETGPU_NCCL_USE_SIFIVE");
        float *accum = (float *)calloc(count, sizeof(float));
        float *rank_inputs = NULL;
        float *tmp = NULL;
        if (use_sifive) {
            rank_inputs = (float *)malloc(count * sizeof(float) * (size_t)active_count);
        } else {
            tmp = (float *)malloc(count * sizeof(float));
        }
        if (!accum || (use_sifive && !rank_inputs) || (!use_sifive && !tmp)) {
            publish_abort(abort_path, "coordinator allocation failed\n");
            comm->async_error = ncclSystemError;
            concordia_record_recovery_event(comm, "collective_abort", "error", "allocation_failed");
            free(accum);
            free(rank_inputs);
            free(tmp);
            return ncclSystemError;
        }

        for (int idx = 0; idx < active_count; ++idx) {
            int r = comm_active_rank_at(comm, idx);
            char path[384];
            float *dst = rank_inputs ? rank_inputs + ((size_t)idx * count) : tmp;
            snprintf(path, sizeof(path), "%s/rank_%d.bin", call_dir, r);
            if (wait_for_file(path, timeout_ms) != 0 || read_file_exact(path, dst, count * sizeof(float)) != 0) {
                fprintf(stderr, "[hetGPU nccl_shim] timed out reading rank %d for allreduce\n", r);
                publish_abort(abort_path, "rank input timeout\n");
                comm->async_error = ncclSystemError;
                concordia_record_recovery_event(comm, "collective_abort", "error", "rank_input_timeout");
                free(accum);
                free(rank_inputs);
                free(tmp);
                return ncclSystemError;
            }
            if (!rank_inputs) {
                for (size_t i = 0; i < count; ++i) accum[i] += dst[i];
            }
        }

        if (use_sifive) {
            if (hetgpu_sifive_nccl_reduce_sum_f32) {
                int rc = hetgpu_sifive_nccl_reduce_sum_f32(
                    rank_inputs,
                    accum,
                    count,
                    active_count
                );
                if (rc != 0) {
                    if (env_enabled("HETGPU_NCCL_CPU_FALLBACK_AFTER_SIFIVE")) {
                        fprintf(stderr, "[hetGPU nccl_shim] SIFIVE reduce-sum hook failed rc=%d; falling back to CPU sum\n", rc);
                        for (int r = 0; r < active_count; ++r) {
                            float *src = rank_inputs + ((size_t)r * count);
                            for (size_t i = 0; i < count; ++i) accum[i] += src[i];
                        }
                    } else {
                        fprintf(stderr, "[hetGPU nccl_shim] SIFIVE reduce-sum hook failed rc=%d\n", rc);
                        publish_abort(abort_path, "SIFIVE reduce-sum hook failed\n");
                        comm->async_error = ncclSystemError;
                        concordia_record_recovery_event(comm, "collective_abort", "error", "sifive_reduce_failed");
                        free(accum);
                        free(rank_inputs);
                        free(tmp);
                        return ncclSystemError;
                    }
                }
            } else if (hetgpu_sifive_nccl_all_reduce_f32) {
                for (int r = 0; r < active_count; ++r) {
                    float *src = rank_inputs + ((size_t)r * count);
                    for (size_t i = 0; i < count; ++i) accum[i] += src[i];
                }
                int rc = hetgpu_sifive_nccl_all_reduce_f32(accum, accum, count, op, comm->rank, active_count);
                if (rc != 0) {
                    if (env_enabled("HETGPU_NCCL_CPU_FALLBACK_AFTER_SIFIVE")) {
                        fprintf(stderr, "[hetGPU nccl_shim] SIFIVE allreduce hook failed rc=%d; keeping CPU sum result\n", rc);
                    } else {
                        fprintf(stderr, "[hetGPU nccl_shim] SIFIVE allreduce hook failed rc=%d\n", rc);
                        publish_abort(abort_path, "SIFIVE allreduce hook failed\n");
                        comm->async_error = ncclSystemError;
                        concordia_record_recovery_event(comm, "collective_abort", "error", "sifive_allreduce_failed");
                        free(accum);
                        free(rank_inputs);
                        free(tmp);
                        return ncclSystemError;
                    }
                }
            } else {
                fprintf(stderr, "[hetGPU nccl_shim] SIFIVE hook symbol unavailable\n");
                publish_abort(abort_path, "SIFIVE hook symbol unavailable\n");
                comm->async_error = ncclSystemError;
                concordia_record_recovery_event(comm, "collective_abort", "error", "sifive_symbol_unavailable");
                free(accum);
                free(rank_inputs);
                free(tmp);
                return ncclSystemError;
            }
        }

        if (write_file_exact(result_path, accum, count * sizeof(float)) != 0) {
            perror("[hetGPU nccl_shim] write result");
            publish_abort(abort_path, "result write failed\n");
            comm->async_error = ncclSystemError;
            concordia_record_recovery_event(comm, "collective_abort", "error", "result_write_failed");
            free(accum);
            free(rank_inputs);
            free(tmp);
            return ncclSystemError;
        }
        free(accum);
        free(rank_inputs);
        free(tmp);
    }

    int wait_rc = wait_for_result_or_abort(result_path, abort_path, timeout_ms);
    if (wait_rc == 1) {
        fprintf(stderr, "[hetGPU nccl_shim] allreduce aborted by rank 0\n");
        comm->async_error = ncclSystemError;
        concordia_record_recovery_event(comm, "collective_abort", "error", "abort_file");
        return ncclSystemError;
    }
    if (wait_rc != 0 || read_file_exact(result_path, recvbuff, count * sizeof(float)) != 0) {
        fprintf(stderr, "[hetGPU nccl_shim] timed out reading allreduce result\n");
        comm->async_error = ncclSystemError;
        concordia_record_recovery_event(comm, "collective_abort", "error", "result_timeout");
        return ncclSystemError;
    }

    comm->async_error = ncclSuccess;
    concordia_record_recovery_event(comm, "collective_complete", "ok", "allreduce");
    return ncclSuccess;
}

ncclResult_t ncclAllReduce(
    const void *sendbuff,
    void *recvbuff,
    size_t count,
    int datatype,
    int op,
    ncclComm_t comm,
    cudaStream_t stream
) {
    (void)op;
    (void)comm;
    (void)stream;
    size_t item_size = dtype_size(datatype);
    if (!sendbuff || !recvbuff || item_size == 0) return ncclInvalidArgument;
    NCCL_LOG("[hetGPU nccl_shim] ncclAllReduce count=%zu datatype=%d bytes=%zu\n",
             count, datatype, count * item_size);
    concordia_checkpoint_boundary("ncclAllReduce:begin");

    if (datatype == 7 && op == 0 && comm && comm->nranks > 1) {
        ncclResult_t result = rendezvous_allreduce_f32((const float *)sendbuff, (float *)recvbuff, count, op, comm);
        concordia_note_collective_result(comm, result);
        concordia_checkpoint_boundary("ncclAllReduce:end");
        return result;
    }

    if (datatype == 7 && op == 0 && env_enabled("HETGPU_NCCL_USE_SIFIVE") && hetgpu_sifive_nccl_all_reduce_f32) {
        int rank = comm ? comm->rank : read_env_i32("RANK", 0);
        int nranks = comm ? comm->nranks : read_env_i32("WORLD_SIZE", 1);
        int rc = hetgpu_sifive_nccl_all_reduce_f32(
            (const float *)sendbuff,
            (float *)recvbuff,
            count,
            op,
            rank,
            nranks
        );
        ncclResult_t result = rc == 0 ? ncclSuccess : ncclSystemError;
        concordia_note_collective_result(comm, result);
        concordia_checkpoint_boundary("ncclAllReduce:end");
        return result;
    }

    if (sendbuff != recvbuff) memcpy(recvbuff, sendbuff, count * item_size);
    concordia_note_collective_result(comm, ncclSuccess);
    concordia_checkpoint_boundary("ncclAllReduce:end");
    return ncclSuccess;
}

ncclResult_t ncclBroadcast(
    const void *sendbuff,
    void *recvbuff,
    size_t count,
    int datatype,
    int root,
    ncclComm_t comm,
    cudaStream_t stream
) {
    (void)root;
    return ncclAllReduce(sendbuff, recvbuff, count, datatype, 0, comm, stream);
}

ncclResult_t ncclReduce(
    const void *sendbuff,
    void *recvbuff,
    size_t count,
    int datatype,
    int op,
    int root,
    ncclComm_t comm,
    cudaStream_t stream
) {
    (void)root;
    return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t ncclAllGather(
    const void *sendbuff,
    void *recvbuff,
    size_t sendcount,
    int datatype,
    ncclComm_t comm,
    cudaStream_t stream
) {
    return ncclAllReduce(sendbuff, recvbuff, sendcount, datatype, 0, comm, stream);
}

ncclResult_t ncclReduceScatter(
    const void *sendbuff,
    void *recvbuff,
    size_t recvcount,
    int datatype,
    int op,
    ncclComm_t comm,
    cudaStream_t stream
) {
    return ncclAllReduce(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
}

ncclResult_t ncclSend(const void *sendbuff, size_t count, int datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    (void)sendbuff;
    (void)count;
    (void)datatype;
    (void)peer;
    (void)comm;
    (void)stream;
    return ncclSuccess;
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, int datatype, int peer, ncclComm_t comm, cudaStream_t stream) {
    (void)recvbuff;
    (void)count;
    (void)datatype;
    (void)peer;
    (void)comm;
    (void)stream;
    return ncclSuccess;
}
