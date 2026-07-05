#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="${HETGPU_NCCL_RECOVERY_ARTIFACT_DIR:-$ROOT/bench/concordia_mpi_nccl_recovery/artifacts/latest}"
WORKDIR="$ARTIFACT_DIR/build"
rm -rf "$ARTIFACT_DIR"
mkdir -p "$WORKDIR" "$ARTIFACT_DIR/evidence"
on_error() {
    rc=$?
    printf 'mpi_nccl_recovery_smoke=fail rc=%d artifact_dir=%s\n' "$rc" "$ARTIFACT_DIR" >&2
    for f in "$ARTIFACT_DIR"/*.out "$ARTIFACT_DIR"/*.err; do
        [ -f "$f" ] || continue
        printf '--- %s ---\n' "$f" >&2
        sed -n '1,160p' "$f" >&2
    done
    exit "$rc"
}
trap on_error ERR
trap 'rm -rf "$WORKDIR"' EXIT

cat >"$WORKDIR/harness.c" <<'C'
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int ncclResult_t;
typedef void *ncclComm_t;
typedef void *cudaStream_t;
typedef struct {
    char internal[128];
} ncclUniqueId;

extern ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId);
extern ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank);
extern ncclResult_t ncclCommDestroy(ncclComm_t comm);
extern ncclResult_t ncclAllReduce(
    const void *sendbuff,
    void *recvbuff,
    size_t count,
    int datatype,
    int op,
    ncclComm_t comm,
    cudaStream_t stream
);

static int read_env_i32_any(const char **names, int count, int fallback) {
    for (int i = 0; i < count; ++i) {
        const char *env = getenv(names[i]);
        if (!env || !*env) continue;
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (end && *end == '\0') return (int)v;
    }
    return fallback;
}

int main(int argc, char **argv) {
    int rank = -1;
    if (argc == 2) {
        rank = atoi(argv[1]);
    } else {
        const char *rank_keys[] = {
            "RANK",
            "OMPI_COMM_WORLD_RANK",
            "PMIX_RANK",
            "PMI_RANK",
            "MV2_COMM_WORLD_RANK",
            "SLURM_PROCID"
        };
        int world_rank = read_env_i32_any(rank_keys, 6, 0);
        int failed = atoi(getenv("HETGPU_CONCORDIA_NCCL_FAILED_RANK"));
        int replacement = atoi(getenv("HETGPU_CONCORDIA_NCCL_REPLACEMENT_RANK"));
        rank = (world_rank == failed) ? replacement : world_rank;
    }
    ncclUniqueId id;
    memset(&id, 0, sizeof(id));
    ncclComm_t comm = NULL;

    if (ncclGetUniqueId(&id) != 0) return 65;
    ncclResult_t init = ncclCommInitRank(&comm, 2, id, rank);
    if (init != 0) {
        fprintf(stderr, "rank %d init failed rc=%d\n", rank, init);
        return 66;
    }

    float input = (float)(rank + 1);
    float output = 0.0f;
    ncclResult_t ar = ncclAllReduce(&input, &output, 1, 7, 0, comm, NULL);
    if (ar != 0) {
        fprintf(stderr, "rank %d allreduce failed rc=%d output=%f\n", rank, ar, output);
        return 67;
    }
    if (fabsf(output - 5.0f) > 0.001f) {
        fprintf(stderr, "rank %d expected 5.0 got %f\n", rank, output);
        return 68;
    }
    ncclCommDestroy(comm);
    return 0;
}
C

cc -D_GNU_SOURCE -Wall -Wextra -Werror -o "$WORKDIR/harness" \
    "$WORKDIR/harness.c" "$ROOT/zluda/src/nccl_shim.c"

rm -rf /dev/shm/hetgpu_nccl_recovery_smoke

export HETGPU_CONCORDIA_NCCL_RECOVERY=1
export HETGPU_CONCORDIA_NCCL_FAILED_RANK=1
export HETGPU_CONCORDIA_NCCL_REPLACEMENT_RANK=3
export HETGPU_CONCORDIA_NCCL_RECOVERY_DIR="$ARTIFACT_DIR/evidence"
export HETGPU_NCCL_COMM_ID=recovery_smoke
export HETGPU_NCCL_TIMEOUT_MS=5000

if command -v mpirun >/dev/null 2>&1 && [ "${HETGPU_NCCL_RECOVERY_NO_MPI:-0}" != "1" ]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1
    export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    mpirun --allow-run-as-root -np 2 "$WORKDIR/harness" \
        >"$ARTIFACT_DIR/mpirun.out" 2>"$ARTIFACT_DIR/mpirun.err"
else
    "$WORKDIR/harness" 0 >"$ARTIFACT_DIR/rank0.out" 2>"$ARTIFACT_DIR/rank0.err" &
    p0=$!
    "$WORKDIR/harness" 3 >"$ARTIFACT_DIR/rank3.out" 2>"$ARTIFACT_DIR/rank3.err" &
    p1=$!

    wait "$p0"
    wait "$p1"
fi

grep -R '"event":"collective_complete"' "$ARTIFACT_DIR/evidence" >/dev/null
grep -R '"active_ring":"0,3"' "$ARTIFACT_DIR/evidence" >/dev/null

printf 'mpi_nccl_recovery_smoke=pass artifact_dir=%s evidence_dir=%s\n' \
    "$ARTIFACT_DIR" "$ARTIFACT_DIR/evidence"
