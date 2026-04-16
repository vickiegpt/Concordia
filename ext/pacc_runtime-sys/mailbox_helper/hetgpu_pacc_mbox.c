// SPDX-License-Identifier: GPL-2.0
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/uaccess.h>

#define HETGPU_PACC_MBOX_DEV "hetgpu_pacc_mbox"
#define PACC_COUNT 4
#define PACC_HOST_MBOX_SRAM_OFF 0x210000ULL
#define MBOX_SIZE 0x2000UL
#define DOORBELL_SIZE 32UL

static const phys_addr_t g_pacc_base[PACC_COUNT] = {
	0x38100000ULL, 0x38500000ULL, 0x39100000ULL, 0x39500000ULL,
};
static dev_t g_dev;
static struct cdev g_cdev;
static struct class *g_class;
static DEFINE_MUTEX(g_lock);
static void __iomem *g_ap2pacc[PACC_COUNT];
static void __iomem *g_pacc2ap[PACC_COUNT];

static unsigned int mbox_minor(struct file *file)
{
	return iminor(file_inode(file));
}

static ssize_t mbox_write(struct file *file, const char __user *buf, size_t len, loff_t *ppos)
{
	u8 tmp[DOORBELL_SIZE];
	unsigned int minor = mbox_minor(file);
	size_t n;

	if (minor >= PACC_COUNT || !g_ap2pacc[minor])
		return -ENODEV;
	if (*ppos < 0 || *ppos >= MBOX_SIZE)
		return -EINVAL;

	n = min_t(size_t, len, DOORBELL_SIZE);
	if ((size_t)*ppos + n > MBOX_SIZE)
		return -EINVAL;
	if (copy_from_user(tmp, buf, n))
		return -EFAULT;

	mutex_lock(&g_lock);
	memcpy_toio(g_ap2pacc[minor] + *ppos, tmp, n);
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

	if (minor >= PACC_COUNT || !g_pacc2ap[minor])
		return -ENODEV;
	if (*ppos < 0 || *ppos >= MBOX_SIZE)
		return 0;

	n = min_t(size_t, len, DOORBELL_SIZE);
	if ((size_t)*ppos + n > MBOX_SIZE)
		n = MBOX_SIZE - (size_t)*ppos;

	mutex_lock(&g_lock);
	memcpy_fromio(tmp, g_pacc2ap[minor] + *ppos, n);
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
	if (next < 0 || next > MBOX_SIZE)
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
	pr_info("hetgpu_pacc_mbox: %u PACC mailboxes, SRAM off=0x%llx size=0x%lx\n",
		PACC_COUNT, (unsigned long long)PACC_HOST_MBOX_SRAM_OFF,
		(unsigned long)MBOX_SIZE);
	return 0;

err_cdev:
	cdev_del(&g_cdev);
err_chrdev:
	unregister_chrdev_region(g_dev, PACC_COUNT);
err_unmap_all:
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
}

module_init(hetgpu_pacc_mbox_init);
module_exit(hetgpu_pacc_mbox_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("hetGPU");
MODULE_DESCRIPTION("Tiny AP2PACC mailbox helper for hetGPU PACC job doorbells");
