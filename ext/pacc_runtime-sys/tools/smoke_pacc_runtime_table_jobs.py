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


def write_ap(dev, off, data):
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


def read_pacc(dev, off, n):
    with open(dev, "rb", buffering=0) as f:
        f.seek(off)
        return f.read(n)


def read_status(dev):
    data = read_pacc(dev, 0, 32)
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


def submit(dev, job_id, seq):
    doorbell = struct.pack("<QIIIIQ", MAGIC, VERSION, job_id, 0, 0, seq)
    write_ap(dev, 0, doorbell)
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        last = read_status(dev)
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


def smoke_one(dev, base_seq):
    seq = base_seq + 1
    write_ap(dev, GEMM_A_OFF, pack_f32([1.0, 2.0, 3.0, 4.0]))
    write_ap(dev, GEMM_B_OFF, pack_f32([5.0, 6.0, 7.0, 8.0]))
    gemm_job = struct.pack(
        GEMM_FMT,
        0, 0, DTypeF32, DTypeF32, DTypeF32, DTypeF32,
        2, 2, 2,
        AP2PACC_PHYS + GEMM_A_OFF,
        AP2PACC_PHYS + GEMM_B_OFF,
        PACC2AP_PHYS + GEMM_C_OFF,
        0, 0,
        2, 2, 2,
        0, 0, 0,
        1,
    )
    write_ap(dev, TABLE_OFF, make_table(seq, gemm=gemm_job))
    status = submit(dev, JOB_GEMM, seq)
    if status != 0:
        raise RuntimeError(f"{dev} GEMM status=0x{status:x}")
    gemm = unpack_f32(read_pacc(dev, GEMM_C_OFF, 16))
    assert_close(f"{dev} GEMM", gemm, [19.0, 22.0, 43.0, 50.0])

    seq = base_seq + 2
    write_ap(dev, SOFTMAX_SRC_OFF, pack_f32([1.0, 2.0, 3.0, 4.0]))
    softmax_job = struct.pack(
        SOFTMAX_FMT,
        AP2PACC_PHYS + SOFTMAX_SRC_OFF,
        PACC2AP_PHYS + SOFTMAX_DST_OFF,
        1, 4, 4,
        DTypeF32, 0,
    )
    write_ap(dev, TABLE_OFF, make_table(seq, softmax=softmax_job))
    status = submit(dev, JOB_SOFTMAX, seq)
    if status != 0:
        raise RuntimeError(f"{dev} softmax status=0x{status:x}")
    softmax = unpack_f32(read_pacc(dev, SOFTMAX_DST_OFF, 16))
    if abs(sum(softmax) - 1.0) > 1e-3 or any(softmax[i] >= softmax[i + 1] for i in range(3)):
        raise AssertionError(f"{dev} softmax bad result {softmax}")

    seq = base_seq + 3
    write_ap(dev, RMS_X_OFF, pack_f32([1.0, 2.0, 3.0, 4.0]))
    write_ap(dev, RMS_W_OFF, pack_f32([1.0, 1.0, 1.0, 1.0]))
    rms_job = struct.pack(
        RMS_FMT,
        AP2PACC_PHYS + RMS_X_OFF,
        AP2PACC_PHYS + RMS_W_OFF,
        PACC2AP_PHYS + RMS_Y_OFF,
        1, 4,
        0.00001,
        DTypeF32,
    )
    write_ap(dev, TABLE_OFF, make_table(seq, rmsnorm=rms_job))
    status = submit(dev, JOB_RMSNORM, seq)
    if status != 0:
        raise RuntimeError(f"{dev} rmsnorm status=0x{status:x}")
    rms = unpack_f32(read_pacc(dev, RMS_Y_OFF, 16))
    scale = 1.0 / math.sqrt(7.5 + 0.00001)
    assert_close(f"{dev} RMSNorm", rms, [scale, 2 * scale, 3 * scale, 4 * scale], tol=2e-3)

    return gemm, softmax, rms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=int, default=4)
    args = parser.parse_args()

    base = int(time.time() * 1000)
    for i in range(args.devices):
        dev = f"/dev/hetgpu_pacc_mbox{i}"
        if not os.path.exists(dev):
            raise FileNotFoundError(dev)
        gemm, softmax, rms = smoke_one(dev, base + i * 100)
        print(f"pacc{i} runtime-table OK gemm={gemm} softmax={softmax} rmsnorm={rms}")


if __name__ == "__main__":
    main()
