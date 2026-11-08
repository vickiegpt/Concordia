#define main jobd_program_main
#include "hetgpu_pacc_jobd.c"
#undef main
#include <assert.h>

int main(void) {
    _Static_assert(HETGPU_PACC_RMS_DEBUG_SLOT >= HETGPU_PACC_DIAG_RING_SLOT + HETGPU_PACC_COUNT,
                   "RMS debug overlaps per-device event rings");
    _Static_assert(HETGPU_PACC_DIAG_RING_OFF + HETGPU_PACC_DIAG_RING_RECORDS * sizeof(struct JobdDiagEvent)
                   <= HETGPU_PACC_CONTROL_BYTES, "event ring spills into next device");
    _Static_assert(HETGPU_PACC_COUNT * HETGPU_PACC_RMS_DEBUG_RECORD_BYTES <= HETGPU_PACC_CONTROL_BYTES,
                   "RMS records exceed reserved slot");
    unsetenv("HETGPU_PACC_ID");
    unsetenv("PACC_JOBD_PACC_ID");
    assert(!read_pacc_id_from_mbox(-1) && g_pacc_id == UINT64_MAX);
    setenv("HETGPU_PACC_ID", "invalid", 1);
    setenv("PACC_JOBD_PACC_ID", "0", 1);
    assert(!read_pacc_id_from_mbox(-1) && g_pacc_id == UINT64_MAX);
    setenv("HETGPU_PACC_ID", "4", 1);
    assert(!read_pacc_id_from_mbox(-1) && g_pacc_id == UINT64_MAX);

    FILE *file = tmpfile();
    assert(file);
    int fd = fileno(file);
    const size_t size = HETGPU_PACC_COUNT * HETGPU_PACC_CONTROL_BYTES;
    uint8_t *before = malloc(size), *after = malloc(size);
    assert(before && after && ftruncate(fd, size) == 0);
    for (unsigned id = 0; id < HETGPU_PACC_COUNT; ++id) {
        char value[8];
        snprintf(value, sizeof(value), "%u", id);
        setenv("HETGPU_PACC_ID", value, 1);
        assert(read_pacc_id_from_mbox(-1) && g_pacc_id == id);
        memset(before, 0xa5, size);
        assert(pwrite(fd, before, size, 0) == (ssize_t)size);
        assert(mirror_boot_marker_owned_slot_mbox_mmap(fd, 0x6a010000));
        assert(pread(fd, after, size, 0) == (ssize_t)size);
        size_t start = id * HETGPU_PACC_CONTROL_BYTES + HETGPU_PACC_COMPLETION_OFF;
        for (size_t byte = 0; byte < size; ++byte) {
            if (byte < start || byte >= start + sizeof(struct HostStatus))
                assert(after[byte] == before[byte]);
        }
        struct HostStatus status;
        memcpy(&status, after + start, sizeof(status));
        assert(status.magic == HETGPU_PACC_JOB_MAGIC && status.status == (0x6a010000 | id));
        assert(status.seq == 0);
    }
    g_pacc_id = UINT64_MAX;
    assert(!mirror_boot_marker_owned_slot_mbox_mmap(fd, 0));
    free(before);
    free(after);
    fclose(file);
    puts("PASS four-device marker ownership, explicit identity, diagnostic region non-overlap");
    return 0;
}
