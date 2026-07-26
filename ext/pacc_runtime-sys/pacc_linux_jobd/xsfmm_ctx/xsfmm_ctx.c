// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#include <linux/types.h>

#include <asm/csr.h>

#define SR_MS_SHIFT 29
#define SR_MS_MASK (3UL << SR_MS_SHIFT)
#define XSFMM_REQUEST_MAGIC 0x5853464d4d524551ULL
#define XSFMM_MAX_M 32ULL
#define XSFMM_MAX_REPEATS 4096ULL
#define XSFMM_MAX_BATCHES 64ULL

#ifndef HETGPU_XSFMM_MS_STATE
#define HETGPU_XSFMM_MS_STATE 1
#endif

#if HETGPU_XSFMM_MS_STATE < 1 || HETGPU_XSFMM_MS_STATE > 3
#error "HETGPU_XSFMM_MS_STATE must be Initial(1), Clean(2), or Dirty(3)"
#endif

#define SR_MS_VALUE ((unsigned long)HETGPU_XSFMM_MS_STATE << SR_MS_SHIFT)

struct xsfmm_request {
	u64 magic;
	u64 a;
	u64 b;
	u64 c;
	u64 m;
	u64 n;
	u64 k;
	u64 repeats;
	u64 cycles;
	u64 completed_repeats;
	s32 status;
	u32 reserved;
	u64 batch_count;
	u64 a_batch_stride;
	u64 b_batch_stride;
	u64 c_batch_stride;
};

__asm__(
	".text\n"
	".align 2\n"
	".global xsfmm_kernel_bf16\n"
	".type xsfmm_kernel_bf16,@function\n"
	"xsfmm_kernel_bf16:\n"
	"  csrwi vstart, 0\n"
	"  .word 0x508772d7\n"
	"  .word 0x8416f2d7\n"
	"  li t0, 2\n"
	"  .word 0x8422f357\n"
	"  .word 0x43e06057\n"
	"  mv t3, a5\n"
	"1:\n"
	"  beqz t3, 2f\n"
	"  mv t0, a3\n"
	"  .word 0x8402f357\n"
	"  .word 0x02055407\n"
	"  slli t2, a3, 1\n"
	"  add t4, a0, t2\n"
	"  .word 0x020ed607\n"
	"  slli t2, a3, 2\n"
	"  add a0, a0, t2\n"
	"  mv t0, a4\n"
	"  .word 0x8402f357\n"
	"  .word 0x0205d807\n"
	"  slli t2, a4, 1\n"
	"  add t4, a1, t2\n"
	"  .word 0x020eda07\n"
	"  slli t2, a4, 2\n"
	"  add a1, a1, t2\n"
	"  .word 0xf2881077\n"
	"  addi t3, t3, -2\n"
	"  j 1b\n"
	"2:\n"
	"  li t0, 0\n"
	"  mv t2, a3\n"
	"  slli t4, a4, 2\n"
	"3:\n"
	"  beqz t2, 4f\n"
	"  .word 0x52567027\n"
	"  add a2, a2, t4\n"
	"  addi t0, t0, 1\n"
	"  addi t2, t2, -1\n"
	"  j 3b\n"
	"4:\n"
	"  li a0, 0\n"
	"  ret\n"
	".size xsfmm_kernel_bf16, .-xsfmm_kernel_bf16\n");

extern int xsfmm_kernel_bf16(const u16 *a, const u16 *b, void *c,
			      size_t m, size_t n, size_t k);

static unsigned long parse_request_address(const char *value)
{
	unsigned long result = 0;
	unsigned int digit;

	if (!value)
		return 0;
	if (value[0] == '0' && (value[1] == 'x' || value[1] == 'X'))
		value += 2;
	while (*value) {
		if (*value >= '0' && *value <= '9')
			digit = (unsigned int)(*value - '0');
		else if (*value >= 'a' && *value <= 'f')
			digit = (unsigned int)(*value - 'a' + 10);
		else if (*value >= 'A' && *value <= 'F')
			digit = (unsigned int)(*value - 'A' + 10);
		else
			break;
		if (result > (~0UL - digit) / 16)
			return 0;
		result = result * 16 + digit;
		value++;
	}
	return result;
}

static int xsfmm_run_set(const char *value, const struct kernel_param *kp)
{
	struct xsfmm_request *request;
	unsigned long address;
	unsigned long saved_status;
	unsigned long execution_status;
	u64 cycle_start;
	u64 cycle_end;
	u64 completed = 0;
	u64 batch_count;
	u64 a_batch_stride;
	u64 b_batch_stride;
	u64 c_batch_stride;
	int status;

	(void)kp;
	address = parse_request_address(value);
	if (!address)
		return -EINVAL;
	saved_status = csr_read(CSR_STATUS);
	execution_status =
		(saved_status & ~(SR_IE | SR_SUM | SR_FS | SR_VS | SR_MS_MASK)) |
		SR_SUM | SR_FS_DIRTY | SR_VS_DIRTY | SR_MS_VALUE;
	csr_write(CSR_STATUS, execution_status);
	request = (struct xsfmm_request *)address;
	batch_count = request->batch_count ? request->batch_count : 1;
	a_batch_stride = request->a_batch_stride ?
		request->a_batch_stride : request->m * request->k;
	b_batch_stride = request->b_batch_stride;
	c_batch_stride = request->c_batch_stride ?
		request->c_batch_stride : request->m * request->n;
	if (request->magic != XSFMM_REQUEST_MAGIC ||
	    !request->a || !request->b || !request->c ||
	    !request->m || request->m > XSFMM_MAX_M ||
	    !request->n || request->n > 32 ||
	    !request->k || (request->k & 1) ||
	    !request->repeats || request->repeats > XSFMM_MAX_REPEATS ||
	    !batch_count || batch_count > XSFMM_MAX_BATCHES ||
	    (batch_count > 1 && (!a_batch_stride || !c_batch_stride))) {
		status = -EINVAL;
	} else {
		asm volatile("rdcycle %0" : "=r"(cycle_start));
		do {
			u64 batch;

			for (batch = 0; batch < batch_count; batch++) {
				status = xsfmm_kernel_bf16(
					(const u16 *)(unsigned long)request->a +
						batch * a_batch_stride,
					(const u16 *)(unsigned long)request->b +
						batch * b_batch_stride,
					(float *)(unsigned long)request->c +
						batch * c_batch_stride,
					(size_t)request->m,
					(size_t)request->n,
					(size_t)request->k);
				if (status)
					break;
			}
			if (status)
				break;
			completed++;
		} while (completed < request->repeats);
		asm volatile("rdcycle %0" : "=r"(cycle_end));
		request->cycles = cycle_end - cycle_start;
	}
	request->completed_repeats = completed;
	request->status = status;
	asm volatile("fence rw, rw" ::: "memory");
	csr_write(CSR_STATUS, saved_status);
	return 0;
}

static const struct kernel_param_ops xsfmm_run_ops = {
	.set = xsfmm_run_set,
};

module_param_cb(run, &xsfmm_run_ops, NULL, 0200);

static int __init xsfmm_ctx_init(void)
{
	return 0;
}

static void __exit xsfmm_ctx_exit(void)
{
}

module_init(xsfmm_ctx_init);
module_exit(xsfmm_ctx_exit);

MODULE_DESCRIPTION("Execute SiFive Xsfmm requests with kernel-managed context");
MODULE_LICENSE("GPL");
