// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/sched/task_stack.h>

#include <asm/csr.h>
#include <asm/ptrace.h>

#define SR_MS_SHIFT 29
#define SR_MS_MASK (3UL << SR_MS_SHIFT)
#define SR_MS_INITIAL (1UL << SR_MS_SHIFT)

static int __init xsfmm_ctx_init(void)
{
	struct pt_regs *regs = task_pt_regs(current);

	if (!regs)
		return -EINVAL;

	regs->status = (regs->status & ~SR_MS_MASK) | SR_MS_INITIAL;
	csr_set(CSR_STATUS, SR_MS_INITIAL);
	return 0;
}

static void __exit xsfmm_ctx_exit(void)
{
}

module_init(xsfmm_ctx_init);
module_exit(xsfmm_ctx_exit);

MODULE_DESCRIPTION("Enable SiFive Xsfmm tile context for the loading task");
MODULE_LICENSE("GPL");
