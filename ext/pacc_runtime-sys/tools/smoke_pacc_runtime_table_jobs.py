#!/usr/bin/env python3
import argparse
import math
import os
import struct
import time

MAGIC = 0x4847505550414343
VERSION = 1
TABLE_MAGIC = 0x4847505554424C31
TABLE_VERSION = 1
STATUS_BUSY = 1

JOB_GEMM = 1
JOB_SOFTMAX = 2
JOB_RMSNORM = 3
DTypeF32 = 4

TABLE_OFF = 0x1400
CONTROL_BYTES = 0x2000
CONTROL_RESERVED = 4 * CONTROL_BYTES
COMPLETION_OFF = 0x1F20
SHARED_DDR_USER_OFF = 0x100000

GEMM_A_OFF = 0x1800
GEMM_B_OFF = 0x1840
GEMM_C_OFF = 0x1000
SOFTMAX_SRC_OFF = 0x1900
SOFTMAX_DST_OFF = 0x1100
RMS_X_OFF = 0x1A00
RMS_W_OFF = 0x1A40
RMS_Y_OFF = 0x1200

AP2PACC_PHYS = 0x20000000
PACC2AP_PHYS = 0x20002000

GEMM_FMT = "<IIIIIIQQQQQQQQqqqqqqQ"
SOFTMAX_FMT = "<QQQQQII"
RMS_FMT = "<QQQQQfI"


def pack_f32(values):
    return struct.pack("<" + "f" * len(values), *values)


def unpack_f32(data):
    return list(struct.unpack("<" + "f" * (len(data) // 4), data))


def parse_u64_text(text):
    text = text.strip()
    if text.lower().startswith("0x"):
        return int(text, 16)
    return int(text, 10)


def shared_ddr_base():
    env = os.environ.get("HETGPU_PACC_SHARED_DDR_BASE")
    if env:
        return parse_u64_text(env)
    for path in (
        "/sys/kernel/debug/hetgpu_pacc_mbox/shared_ddr_base",
    ):
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = parse_u64_text(f.read())
                if value:
                    return value
        except OSError:
            pass
    raise RuntimeError("shared DDR base is not available")


def write_dev(dev, off, data):
    with open(dev, "r+b", buffering=0) as f:
        f.seek(off)
        view = memoryview(data)
        written = 0
        while written < len(view):
            n = f.write(view[written:])
            if n is None:
                n = len(view) - written
            if n <= 0:
                raise OSError(f"short write to {dev} at 0x{off + written:x}")
            written += n


def read_dev(dev, off, n):
    with open(dev, "rb", buffering=0) as f:
        f.seek(off)
        return f.read(n)


def control_off(pacc_id, off):
    return pacc_id * CONTROL_BYTES + off


def shared_user_off(off):
    return SHARED_DDR_USER_OFF + off


def write_control(dev, backend, pacc_id, off, data):
    if backend == "shared-ddr":
        return write_dev(dev, shared_user_off(control_off(pacc_id, off)), data)
    return write_dev(dev, off, data)


def write_data(dev, backend, off, data):
    if backend == "shared-ddr":
        return write_dev(dev, shared_user_off(off), data)
    return write_dev(dev, off, data)


def read_data(dev, backend, off, n):
    if backend == "shared-ddr":
        return read_dev(dev, shared_user_off(off), n)
    return read_dev(dev, off, n)


def phys_addr(backend, shared_base, off, legacy_phys):
    if backend == "shared-ddr":
        return shared_base + off
    return legacy_phys + off


def read_status(dev, backend, pacc_id):
    if backend == "shared-ddr":
        data = read_dev(dev, shared_user_off(control_off(pacc_id, COMPLETION_OFF)), 32)
    else:
        data = read_dev(dev, 0, 32)
    magic, ver, job, status, seq = struct.unpack_from("<QIII4xQ", data, 0)
    return magic, ver, job, status, seq, data


def make_table(seq, gemm=None, softmax=None, rmsnorm=None):
    if gemm is None:
        gemm = bytes(struct.calcsize(GEMM_FMT))
        have_gemm = 0
    else:
        have_gemm = 1
    if softmax is None:
        softmax = bytes(struct.calcsize(SOFTMAX_FMT))
        have_softmax = 0
    else:
        have_softmax = 1
    if rmsnorm is None:
        rmsnorm = bytes(struct.calcsize(RMS_FMT))
        have_rmsnorm = 0
    else:
        have_rmsnorm = 1
    header = struct.pack(
        "<QIIQIIII",
        TABLE_MAGIC,
        TABLE_VERSION,
        0,
        seq,
        have_gemm,
        have_softmax,
        have_rmsnorm,
        0,
    )
    return header + gemm + softmax + rmsnorm


def submit(dev, backend, pacc_id, job_id, seq):
    doorbell = struct.pack("<QIIIIQ", MAGIC, VERSION, job_id, 0, 0, seq)
    write_control(dev, backend, pacc_id, 0, doorbell)
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        last = read_status(dev, backend, pacc_id)
        magic, ver, job, status, got_seq, _ = last
        if magic == MAGIC and ver == VERSION and job == job_id and got_seq == seq and status != STATUS_BUSY:
            return status
        time.sleep(0.02)
    raise TimeoutError(f"{dev} job_id={job_id} seq={seq} timed out; last={last}")


def assert_close(name, got, want, tol=1e-3):
    if len(got) != len(want):
        raise AssertionError(f"{name}: len got {len(got)} want {len(want)}")
    for i, (g, w) in enumerate(zip(got, want)):
        if abs(g - w) > tol:
            raise AssertionError(f"{name}[{i}] got {g:.6f} want {w:.6f}")


def smoke_one(dev, pacc_id, base_seq, backend, shared_base):
    data_base = CONTROL_RESERVED + pacc_id * 0x4000 if backend == "shared-ddr" else 0
    seq = base_seq + 1
    write_data(dev, backend, data_base + GEMM_A_OFF, pack_f32([1.0, 2.0, 3.0, 4.0]))
    write_data(dev, backend, data_base + GEMM_B_OFF, pack_f32([5.0, 6.0, 7.0, 8.0]))
    gemm_job = struct.pack(
        GEMM_FMT,
        0, 0, DTypeF32, DTypeF32, DTypeF32, DTypeF32,
        2, 2, 2,
        phys_addr(backend, shared_base, data_base + GEMM_A_OFF, AP2PACC_PHYS),
        phys_addr(backend, shared_base, data_base + GEMM_B_OFF, AP2PACC_PHYS),
        phys_addr(backend, shared_base, data_base + GEMM_C_OFF, PACC2AP_PHYS),
        0, 0,
        2, 2, 2,
        0, 0, 0,
        1,
    )
    write_control(dev, backend, pacc_id, TABLE_OFF, make_table(seq, gemm=gemm_job))
    status = submit(dev, backend, pacc_id, JOB_GEMM, seq)
    if status != 0:
        raise RuntimeError(f"{dev} GEMM status=0x{status:x}")
    gemm = unpack_f32(read_data(dev, backend, data_base + GEMM_C_OFF, 16))
    assert_close(f"{dev} GEMM", gemm, [19.0, 22.0, 43.0, 50.0])

    seq = base_seq + 2
    write_data(dev, backend, data_base + SOFTMAX_SRC_OFF, pack_f32([1.0, 2.0, 3.0, 4.0]))
    softmax_job = struct.pack(
        SOFTMAX_FMT,
        phys_addr(backend, shared_base, data_base + SOFTMAX_SRC_OFF, AP2PACC_PHYS),
        phys_addr(backend, shared_base, data_base + SOFTMAX_DST_OFF, PACC2AP_PHYS),
        1, 4, 4,
        DTypeF32, 0,
    )
    write_control(dev, backend, pacc_id, TABLE_OFF, make_table(seq, softmax=softmax_job))
    status = submit(dev, backend, pacc_id, JOB_SOFTMAX, seq)
    if status != 0:
        raise RuntimeError(f"{dev} softmax status=0x{status:x}")
    softmax = unpack_f32(read_data(dev, backend, data_base + SOFTMAX_DST_OFF, 16))
    if abs(sum(softmax) - 1.0) > 1e-3 or any(softmax[i] >= softmax[i + 1] for i in range(3)):
        raise AssertionError(f"{dev} softmax bad result {softmax}")

    seq = base_seq + 3
    write_data(dev, backend, data_base + RMS_X_OFF, pack_f32([1.0, 2.0, 3.0, 4.0]))
    write_data(dev, backend, data_base + RMS_W_OFF, pack_f32([1.0, 1.0, 1.0, 1.0]))
    rms_job = struct.pack(
        RMS_FMT,
        phys_addr(backend, shared_base, data_base + RMS_X_OFF, AP2PACC_PHYS),
        phys_addr(backend, shared_base, data_base + RMS_W_OFF, AP2PACC_PHYS),
        phys_addr(backend, shared_base, data_base + RMS_Y_OFF, PACC2AP_PHYS),
        1, 4,
        0.00001,
        DTypeF32,
    )
    write_control(dev, backend, pacc_id, TABLE_OFF, make_table(seq, rmsnorm=rms_job))
    status = submit(dev, backend, pacc_id, JOB_RMSNORM, seq)
    if status != 0:
        raise RuntimeError(f"{dev} rmsnorm status=0x{status:x}")
    rms = unpack_f32(read_data(dev, backend, data_base + RMS_Y_OFF, 16))
    scale = 1.0 / math.sqrt(7.5 + 0.00001)
    assert_close(f"{dev} RMSNorm", rms, [scale, 2 * scale, 3 * scale, 4 * scale], tol=2e-3)

    return gemm, softmax, rms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--backend", choices=("legacy", "shared-ddr"), default="legacy")
    args = parser.parse_args()

    shared_base = shared_ddr_base() if args.backend == "shared-ddr" else 0
    base = int(time.time() * 1000)
    for i in range(args.devices):
        dev = f"/dev/hetgpu_pacc_mbox{i}"
        if not os.path.exists(dev):
            raise FileNotFoundError(dev)
        gemm, softmax, rms = smoke_one(dev, i, base + i * 100, args.backend, shared_base)
        print(f"pacc{i} {args.backend} runtime-table OK gemm={gemm} softmax={softmax} rmsnorm={rms}")


if __name__ == "__main__":
    main()
