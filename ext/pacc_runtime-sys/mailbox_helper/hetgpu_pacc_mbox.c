// SPDX-License-Identifier: GPL-2.0
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/gfp.h>
#include <linux/io.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/moduleparam.h>
#include <linux/uaccess.h>

#define HETGPU_PACC_MBOX_DEV "hetgpu_pacc_mbox"
#define PACC_COUNT 4
#define PACC_HOST_MBOX_SRAM_OFF 0x210000ULL
#define MBOX_SIZE 0x2000UL
#define SHARED_DDR_USER_OFF 0x100000ULL
#define DOORBELL_SIZE 32UL

static unsigned long long shared_ddr_base;
static unsigned long shared_ddr_size = 0x01000000UL;
module_param(shared_ddr_base, ullong, 0444);
MODULE_PARM_DESC(shared_ddr_base, "Pcore/PACC-visible shared DDR physical base, or 0 to allocate");
module_param(shared_ddr_size, ulong, 0444);
MODULE_PARM_DESC(shared_ddr_size, "Pcore/PACC-visible shared DDR window size");

static const phys_addr_t g_pacc_base[PACC_COUNT] = {
	0x38100000ULL, 0x38500000ULL, 0x39100000ULL, 0x39500000ULL,
};
static dev_t g_dev;
static struct cdev g_cdev;
static struct class *g_class;
static DEFINE_MUTEX(g_lock);
static void __iomem *g_ap2pacc[PACC_COUNT];
static void __iomem *g_pacc2ap[PACC_COUNT];
static void __iomem *g_shared_ddr_io;
static void *g_shared_ddr_mem;
static bool g_shared_ddr_allocated;

static gfp_t shared_ddr_gfp_flags(void)
{
	gfp_t flags = GFP_KERNEL | __GFP_ZERO;
#ifdef GFP_DMA32
	flags |= GFP_DMA32;
#endif
	return flags;
}

static unsigned int mbox_minor(struct file *file)
{
	return iminor(file_inode(file));
}

static ssize_t mbox_write(struct file *file, const char __user *buf, size_t len, loff_t *ppos)
{
	u8 tmp[DOORBELL_SIZE];
	unsigned int minor = mbox_minor(file);
	size_t n;
	u64 ddr_off;

	if (minor >= PACC_COUNT)
		return -ENODEV;
	if (*ppos < 0)
		return -EINVAL;

	n = min_t(size_t, len, DOORBELL_SIZE);
	if (copy_from_user(tmp, buf, n))
		return -EFAULT;

	mutex_lock(&g_lock);
	if ((u64)*ppos >= SHARED_DDR_USER_OFF) {
		ddr_off = (u64)*ppos - SHARED_DDR_USER_OFF;
		if ((!g_shared_ddr_io && !g_shared_ddr_mem) || ddr_off + n > shared_ddr_size) {
			mutex_unlock(&g_lock);
			return -EINVAL;
		}
		if (g_shared_ddr_io)
			memcpy_toio((u8 __iomem *)g_shared_ddr_io + ddr_off, tmp, n);
		else
			memcpy((u8 *)g_shared_ddr_mem + ddr_off, tmp, n);
	} else {
		if (!g_ap2pacc[minor] || (size_t)*ppos + n > MBOX_SIZE) {
			mutex_unlock(&g_lock);
			return -EINVAL;
		}
		memcpy_toio(g_ap2pacc[minor] + *ppos, tmp, n);
	}
	mb();
	mutex_unlock(&g_lock);

	*ppos += n;
	return n;
}

static ssize_t mbox_read(struct file *file, char __user *buf, size_t len, loff_t *ppos)
{
	u8 tmp[DOORBELL_SIZE];
	unsigned int minor = mbox_minor(file);
	size_t n;
	u64 ddr_off;

	if (minor >= PACC_COUNT)
		return -ENODEV;
	if (*ppos < 0)
		return 0;

	n = min_t(size_t, len, DOORBELL_SIZE);

	mutex_lock(&g_lock);
	if ((u64)*ppos >= SHARED_DDR_USER_OFF) {
		ddr_off = (u64)*ppos - SHARED_DDR_USER_OFF;
		if ((!g_shared_ddr_io && !g_shared_ddr_mem) || ddr_off >= shared_ddr_size) {
			mutex_unlock(&g_lock);
			return 0;
		}
		if (ddr_off + n > shared_ddr_size)
			n = shared_ddr_size - ddr_off;
		if (g_shared_ddr_io)
			memcpy_fromio(tmp, (u8 __iomem *)g_shared_ddr_io + ddr_off, n);
		else
			memcpy(tmp, (u8 *)g_shared_ddr_mem + ddr_off, n);
	} else {
		if (!g_pacc2ap[minor] || *ppos >= MBOX_SIZE) {
			mutex_unlock(&g_lock);
			return 0;
		}
		if ((size_t)*ppos + n > MBOX_SIZE)
			n = MBOX_SIZE - (size_t)*ppos;
		memcpy_fromio(tmp, g_pacc2ap[minor] + *ppos, n);
	}
	mutex_unlock(&g_lock);

	if (copy_to_user(buf, tmp, n))
		return -EFAULT;
	*ppos += n;
	return n;
}

static loff_t mbox_llseek(struct file *file, loff_t off, int whence)
{
	loff_t next;

	switch (whence) {
	case SEEK_SET:
		next = off;
		break;
	case SEEK_CUR:
		next = file->f_pos + off;
		break;
	default:
		return -EINVAL;
	}
	if (next < 0 || (next > MBOX_SIZE && (u64)next > SHARED_DDR_USER_OFF + shared_ddr_size))
		return -EINVAL;
	file->f_pos = next;
	return next;
}

static const struct file_operations mbox_fops = {
	.owner = THIS_MODULE,
	.read = mbox_read,
	.write = mbox_write,
	.llseek = mbox_llseek,
};

static int __init hetgpu_pacc_mbox_init(void)
{
	int ret;
	unsigned int i;

	for (i = 0; i < PACC_COUNT; i++) {
		phys_addr_t ap2pacc = g_pacc_base[i] + PACC_HOST_MBOX_SRAM_OFF;
		phys_addr_t pacc2ap = ap2pacc + MBOX_SIZE;

		g_ap2pacc[i] = ioremap(ap2pacc, MBOX_SIZE);
		if (!g_ap2pacc[i]) {
			ret = -ENOMEM;
			goto err_unmap_all;
		}
		g_pacc2ap[i] = ioremap(pacc2ap, MBOX_SIZE);
		if (!g_pacc2ap[i]) {
			ret = -ENOMEM;
			goto err_unmap_all;
		}
	}
	if (shared_ddr_size) {
		if (shared_ddr_base) {
			g_shared_ddr_io = ioremap((phys_addr_t)shared_ddr_base, shared_ddr_size);
			if (!g_shared_ddr_io) {
				ret = -ENOMEM;
				goto err_unmap_all;
			}
		} else {
			g_shared_ddr_mem = alloc_pages_exact(shared_ddr_size, shared_ddr_gfp_flags());
			if (!g_shared_ddr_mem) {
				ret = -ENOMEM;
				goto err_unmap_all;
			}
			g_shared_ddr_allocated = true;
			shared_ddr_base = (unsigned long long)virt_to_phys(g_shared_ddr_mem);
		}
	}

	ret = alloc_chrdev_region(&g_dev, 0, PACC_COUNT, HETGPU_PACC_MBOX_DEV);
	if (ret)
		goto err_unmap_all;

	cdev_init(&g_cdev, &mbox_fops);
	ret = cdev_add(&g_cdev, g_dev, PACC_COUNT);
	if (ret)
		goto err_chrdev;

	g_class = class_create(HETGPU_PACC_MBOX_DEV);
	if (IS_ERR(g_class)) {
		ret = PTR_ERR(g_class);
		goto err_cdev;
	}
	for (i = 0; i < PACC_COUNT; i++) {
		device_create(g_class, NULL, MKDEV(MAJOR(g_dev), MINOR(g_dev) + i),
			      NULL, HETGPU_PACC_MBOX_DEV "%u", i);
	}
	pr_info("hetgpu_pacc_mbox: %u PACC mailboxes, SRAM off=0x%llx size=0x%lx shared_ddr=0x%llx+0x%lx user_off=0x%llx\n",
		PACC_COUNT, (unsigned long long)PACC_HOST_MBOX_SRAM_OFF,
		(unsigned long)MBOX_SIZE, shared_ddr_base, shared_ddr_size,
		(unsigned long long)SHARED_DDR_USER_OFF);
	return 0;

err_cdev:
	cdev_del(&g_cdev);
err_chrdev:
	unregister_chrdev_region(g_dev, PACC_COUNT);
err_unmap_all:
	if (g_shared_ddr_io)
		iounmap(g_shared_ddr_io);
	if (g_shared_ddr_allocated)
		free_pages_exact(g_shared_ddr_mem, shared_ddr_size);
	for (i = 0; i < PACC_COUNT; i++) {
		if (g_pacc2ap[i])
			iounmap(g_pacc2ap[i]);
		if (g_ap2pacc[i])
			iounmap(g_ap2pacc[i]);
	}
	return ret;
}

static void __exit hetgpu_pacc_mbox_exit(void)
{
	unsigned int i;

	for (i = 0; i < PACC_COUNT; i++)
		device_destroy(g_class, MKDEV(MAJOR(g_dev), MINOR(g_dev) + i));
	class_destroy(g_class);
	cdev_del(&g_cdev);
	unregister_chrdev_region(g_dev, PACC_COUNT);
	for (i = 0; i < PACC_COUNT; i++) {
		iounmap(g_pacc2ap[i]);
		iounmap(g_ap2pacc[i]);
	}
	if (g_shared_ddr_io)
		iounmap(g_shared_ddr_io);
	if (g_shared_ddr_allocated)
		free_pages_exact(g_shared_ddr_mem, shared_ddr_size);
}

module_init(hetgpu_pacc_mbox_init);
module_exit(hetgpu_pacc_mbox_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("hetGPU");
MODULE_DESCRIPTION("Tiny AP2PACC mailbox helper for hetGPU PACC job doorbells");
