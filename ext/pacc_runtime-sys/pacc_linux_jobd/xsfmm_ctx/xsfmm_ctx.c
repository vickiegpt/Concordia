// SPDX-License-Identifier: GPL-2.0
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/sched/task_stack.h>

#include <asm/csr.h>
#include <asm/ptrace.h>

#define SR_MS_SHIFT 29
#define SR_MS_MASK (3UL << SR_MS_SHIFT)
#define SR_MS_INITIAL (1UL << SR_MS_SHIFT)

static int xsfmm_ctx_open(struct inode *inode, struct file *file)
{
	struct pt_regs *regs = task_pt_regs(current);

	if (!regs)
		return -EINVAL;

	regs->status = (regs->status & ~SR_MS_MASK) | SR_MS_INITIAL;
	csr_set(CSR_STATUS, SR_MS_INITIAL);
	return 0;
}

static const struct file_operations xsfmm_ctx_fops = {
	.owner = THIS_MODULE,
	.open = xsfmm_ctx_open,
	.llseek = no_llseek,
};

static struct miscdevice xsfmm_ctx_device = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "xsfmm_ctx",
	.fops = &xsfmm_ctx_fops,
	.mode = 0666,
};

static int __init xsfmm_ctx_init(void)
{
	return misc_register(&xsfmm_ctx_device);
}

static void __exit xsfmm_ctx_exit(void)
{
	misc_deregister(&xsfmm_ctx_device);
}

module_init(xsfmm_ctx_init);
module_exit(xsfmm_ctx_exit);

MODULE_DESCRIPTION("Enable SiFive Xsfmm tile context for a pinned userspace task");
MODULE_LICENSE("GPL");
