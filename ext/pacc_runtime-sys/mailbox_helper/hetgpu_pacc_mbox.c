// SPDX-License-Identifier: GPL-2.0
#include <linux/cdev.h>
#include <linux/debugfs.h>
#include <linux/dma-mapping.h>
#include <linux/device.h>
#include <linux/err.h>
#include <linux/fs.h>
#include <linux/gfp.h>
#include <linux/io.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/moduleparam.h>
#include <linux/platform_device.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>

#define HETGPU_PACC_MBOX_DEV "hetgpu_pacc_mbox"
#define PACC_COUNT 4
#define PACC_HOST_MBOX_SRAM_OFF 0x210000ULL
#define MBOX_SIZE 0x2000UL
#define SHARED_DDR_USER_OFF 0x100000ULL
#define AP2PACC_READ_USER_OFF 0x02000000ULL
#define PACC2AP_RW_USER_OFF 0x02002000ULL
#define SHARED_DDR_BASE_INFO_OFF 0x02004000ULL
#define DOORBELL_SIZE 32UL

/* Allocated by the kernel helper and exported via debugfs for userspace. */
static u64 shared_ddr_base;
static unsigned long shared_ddr_size = 0x01000000UL;
module_param(shared_ddr_size, ulong, 0444);
MODULE_PARM_DESC(shared_ddr_size, "Pcore/PACC-visible shared DDR window size");

static const phys_addr_t g_pacc_base[PACC_COUNT] = {
	0x38100000ULL, 0x38500000ULL, 0x39100000ULL, 0x39500000ULL,
};
static dev_t g_dev;
static struct cdev g_cdev;
static struct class *g_class;
static struct dentry *g_dentry;
static struct platform_device *g_pdev;
static DEFINE_MUTEX(g_lock);
static void __iomem *g_ap2pacc[PACC_COUNT];
static void __iomem *g_pacc2ap[PACC_COUNT];
static void *g_shared_ddr_mem;
static bool g_shared_ddr_allocated;

static gfp_t shared_ddr_gfp_flags(void)
{
	return GFP_KERNEL | __GFP_ZERO;
}

static int shared_ddr_base_show(struct seq_file *m, void *unused)
{
	seq_printf(m, "0x%llx\n", (unsigned long long)shared_ddr_base);
	return 0;
}

DEFINE_SHOW_ATTRIBUTE(shared_ddr_base);

static int shared_ddr_size_show(struct seq_file *m, void *unused)
{
	seq_printf(m, "0x%lx\n", shared_ddr_size);
	return 0;
}

DEFINE_SHOW_ATTRIBUTE(shared_ddr_size);

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

	if (!mutex_trylock(&g_lock))
		return -EBUSY;
	if ((u64)*ppos >= SHARED_DDR_USER_OFF &&
	    (u64)*ppos < SHARED_DDR_USER_OFF + shared_ddr_size) {
		ddr_off = (u64)*ppos - SHARED_DDR_USER_OFF;
		if (!g_shared_ddr_mem || ddr_off + n > shared_ddr_size) {
			mutex_unlock(&g_lock);
			return -EINVAL;
		}
		memcpy((u8 *)g_shared_ddr_mem + ddr_off, tmp, n);
	} else if ((u64)*ppos >= PACC2AP_RW_USER_OFF &&
		   (u64)*ppos < PACC2AP_RW_USER_OFF + MBOX_SIZE) {
		u64 off = (u64)*ppos - PACC2AP_RW_USER_OFF;
		if (!g_pacc2ap[minor] || off + n > MBOX_SIZE) {
			mutex_unlock(&g_lock);
			return -EINVAL;
		}
		memcpy_toio(g_pacc2ap[minor] + off, tmp, n);
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

	if (!mutex_trylock(&g_lock))
		return -EBUSY;
	if ((u64)*ppos >= SHARED_DDR_USER_OFF &&
	    (u64)*ppos < SHARED_DDR_USER_OFF + shared_ddr_size) {
		ddr_off = (u64)*ppos - SHARED_DDR_USER_OFF;
		if (!g_shared_ddr_mem || ddr_off >= shared_ddr_size) {
			mutex_unlock(&g_lock);
			return 0;
		}
		if (ddr_off + n > shared_ddr_size)
			n = shared_ddr_size - ddr_off;
		memcpy(tmp, (u8 *)g_shared_ddr_mem + ddr_off, n);
	} else if ((u64)*ppos == SHARED_DDR_BASE_INFO_OFF) {
		memcpy(tmp, &shared_ddr_base, min_t(size_t, n, sizeof(shared_ddr_base)));
		n = min_t(size_t, n, sizeof(shared_ddr_base));
	} else if ((u64)*ppos >= AP2PACC_READ_USER_OFF &&
		   (u64)*ppos < AP2PACC_READ_USER_OFF + MBOX_SIZE) {
		u64 off = (u64)*ppos - AP2PACC_READ_USER_OFF;
		if (!g_ap2pacc[minor] || off >= MBOX_SIZE) {
			mutex_unlock(&g_lock);
			return 0;
		}
		if (off + n > MBOX_SIZE)
			n = MBOX_SIZE - off;
		memcpy_fromio(tmp, g_ap2pacc[minor] + off, n);
	} else if ((u64)*ppos >= PACC2AP_RW_USER_OFF &&
		   (u64)*ppos < PACC2AP_RW_USER_OFF + MBOX_SIZE) {
		u64 off = (u64)*ppos - PACC2AP_RW_USER_OFF;
		if (!g_pacc2ap[minor] || off >= MBOX_SIZE) {
			mutex_unlock(&g_lock);
			return 0;
		}
		if (off + n > MBOX_SIZE)
			n = MBOX_SIZE - off;
		memcpy_fromio(tmp, g_pacc2ap[minor] + off, n);
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
	if (next < 0 ||
	    ((u64)next > MBOX_SIZE &&
	     !((u64)next >= SHARED_DDR_USER_OFF &&
	       (u64)next <= SHARED_DDR_USER_OFF + shared_ddr_size) &&
	     !((u64)next >= AP2PACC_READ_USER_OFF &&
	       (u64)next <= AP2PACC_READ_USER_OFF + MBOX_SIZE) &&
	     !((u64)next >= PACC2AP_RW_USER_OFF &&
	       (u64)next <= PACC2AP_RW_USER_OFF + MBOX_SIZE) &&
	     !((u64)next >= SHARED_DDR_BASE_INFO_OFF &&
	       (u64)next <= SHARED_DDR_BASE_INFO_OFF + sizeof(shared_ddr_base))))
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
		g_pdev = platform_device_register_simple(HETGPU_PACC_MBOX_DEV "_dma",
							 PLATFORM_DEVID_NONE, NULL, 0);
		if (IS_ERR(g_pdev)) {
			ret = PTR_ERR(g_pdev);
			g_pdev = NULL;
			goto err_unmap_all;
		}

		ret = dma_set_mask_and_coherent(&g_pdev->dev, DMA_BIT_MASK(64));
		if (ret)
			goto err_unmap_all;

		g_shared_ddr_mem = dma_alloc_noncoherent(
			&g_pdev->dev,
			shared_ddr_size,
			(dma_addr_t *)&shared_ddr_base,
			DMA_BIDIRECTIONAL,
			shared_ddr_gfp_flags());
		if (!g_shared_ddr_mem) {
			ret = -ENOMEM;
			goto err_unmap_all;
		}
		g_shared_ddr_allocated = true;

		g_dentry = debugfs_create_dir(HETGPU_PACC_MBOX_DEV, NULL);
		if (IS_ERR_OR_NULL(g_dentry)) {
			pr_warn("hetgpu_pacc_mbox: failed to create debugfs dir for shared_ddr_base export\n");
			g_dentry = NULL;
		} else {
			debugfs_create_file("shared_ddr_base", 0444, g_dentry, NULL,
					    &shared_ddr_base_fops);
			debugfs_create_file("shared_ddr_size", 0444, g_dentry, NULL,
					    &shared_ddr_size_fops);
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
	pr_info("hetgpu_pacc_mbox: %u PACC mailboxes, SRAM off=0x%llx size=0x%lx shared_ddr=0x%llx+0x%lx user_off=0x%llx debugfs=/sys/kernel/debug/%s/{shared_ddr_base,shared_ddr_size}\n",
		PACC_COUNT, (unsigned long long)PACC_HOST_MBOX_SRAM_OFF,
		(unsigned long)MBOX_SIZE, (unsigned long long)shared_ddr_base, shared_ddr_size,
		(unsigned long long)SHARED_DDR_USER_OFF, HETGPU_PACC_MBOX_DEV);
	return 0;

err_cdev:
	cdev_del(&g_cdev);
err_chrdev:
	unregister_chrdev_region(g_dev, PACC_COUNT);
err_unmap_all:
	if (g_shared_ddr_allocated) {
		dma_free_noncoherent(
			&g_pdev->dev,
			shared_ddr_size,
			g_shared_ddr_mem,
			(dma_addr_t)shared_ddr_base,
			DMA_BIDIRECTIONAL);
		debugfs_remove(g_dentry);
	}
	if (g_pdev) {
		platform_device_unregister(g_pdev);
		g_pdev = NULL;
	}
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

	if (g_shared_ddr_allocated) {
		dma_free_noncoherent(
			&g_pdev->dev,
			shared_ddr_size,
			g_shared_ddr_mem,
			(dma_addr_t)shared_ddr_base,
			DMA_BIDIRECTIONAL);
		debugfs_remove(g_dentry);
	}
	if (g_pdev) {
		platform_device_unregister(g_pdev);
		g_pdev = NULL;
	}
	for (i = 0; i < PACC_COUNT; i++)
		device_destroy(g_class, MKDEV(MAJOR(g_dev), MINOR(g_dev) + i));
	class_destroy(g_class);
	cdev_del(&g_cdev);
	unregister_chrdev_region(g_dev, PACC_COUNT);
	for (i = 0; i < PACC_COUNT; i++) {
		iounmap(g_pacc2ap[i]);
		iounmap(g_ap2pacc[i]);
	}
}

module_init(hetgpu_pacc_mbox_init);
module_exit(hetgpu_pacc_mbox_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("hetGPU");
MODULE_DESCRIPTION("Tiny AP2PACC mailbox helper with kernel-allocated shared DDR export");
