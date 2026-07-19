// SPDX-License-Identifier: GPL-2.0
#include <linux/firmware.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/module.h>
#include <linux/overflow.h>
#include <linux/string.h>
#include <linux/types.h>

#define PACC_COUNT 4
#define PACC_CORE_COUNT 4

#define LX500_PACC_TOP_ABP_CRG_BASE 0x200000ULL
#define LX500_PACC_TOP_ABP_CFG_BASE 0x201000ULL
#define LX500_PACC_TOP_ABP_MBOX_RAM_BASE 0x210000ULL
#define LX500_PACC_TOP_ABP_MBOX_DB_BASE 0x214000ULL
#define LX500_PACC_CFG_CORE_RESET0  0x14ULL
#define LX500_PACC_CFG_SYS_RESET    0x24ULL
#define LX500_PACC_CFG_FORCE_RELOAD 0x28ULL
#define LX500_PACC_CFG_RST_VEC_LOW  0x6cULL
#define LX500_PACC_CFG_RST_VEC_HIGH 0x70ULL
#define LX500_PACC_CFG_SECURE_TIEOFF 0xbcULL
#define LX500_PACC_CFG_ADDR_WINDOW  0xc4ULL
#define LX500_PACC_MBOX_RAM_SIZE    0x4000ULL

static const u64 pacc_base[PACC_COUNT] = {
	0x38100000ULL, 0x38500000ULL, 0x39100000ULL, 0x39500000ULL,
};

struct pacc_base_cmd {
	u64 reserved0;
	u64 reserved1;
	u64 base;
	u64 size;
	u64 reserved2;
} __packed;

static char *firmware_name = "lanxin/lx500_pacc.bin";
module_param(firmware_name, charp, 0444);
MODULE_PARM_DESC(firmware_name, "PACC firmware path under /lib/firmware");

static ulong load_addr = 0xc0000000UL;
module_param(load_addr, ulong, 0444);
MODULE_PARM_DESC(load_addr, "Physical address where the PACC image is copied");

static ulong entry = 0xc0000000UL;
module_param(entry, ulong, 0444);
MODULE_PARM_DESC(entry, "PACC reset-vector entry address");

static ulong reserved_size = 0x08000000UL;
module_param(reserved_size, ulong, 0444);
MODULE_PARM_DESC(reserved_size, "PACC reserved-memory window size passed in the boot base command");

static ulong base_cmd_addr;
module_param(base_cmd_addr, ulong, 0444);
MODULE_PARM_DESC(base_cmd_addr,
		 "Address placed in the lx500 boot base command; 0 keeps the AP-visible load address");

static bool per_pacc_load = true;
module_param(per_pacc_load, bool, 0444);
MODULE_PARM_DESC(per_pacc_load, "Use load_addr + pacc_id * reserved_size for per-PACC reserved-memory windows");

static bool per_pacc_entry = true;
module_param(per_pacc_entry, bool, 0444);
MODULE_PARM_DESC(per_pacc_entry, "Use load_addr + pacc_id * reserved_size as each PACC reset-vector entry");

static bool local_doorbell_bit;
module_param(local_doorbell_bit, bool, 0444);
MODULE_PARM_DESC(local_doorbell_bit,
		 "Use bit0 for each per-PACC local mailbox doorbell; default uses AP-visible bit 1<<pacc_id");

static bool send_base_cmd_param = true;
module_param_named(send_base_cmd, send_base_cmd_param, bool, 0444);
MODULE_PARM_DESC(send_base_cmd, "Send the lx500 base command before releasing PACC cores");

static bool send_base_after_core_release;
module_param(send_base_after_core_release, bool, 0444);
MODULE_PARM_DESC(send_base_after_core_release,
		 "Release selected PACC cores before sending the lx500 base command");

static uint send_base_delay_ms;
module_param(send_base_delay_ms, uint, 0444);
MODULE_PARM_DESC(send_base_delay_ms,
		 "Delay after releasing cores and before sending the base command when send_base_after_core_release=1");

static uint pacc_mask = 0x1;
module_param(pacc_mask, uint, 0444);
MODULE_PARM_DESC(pacc_mask, "PACC cluster mask");

static uint core_mask = 0xf;
module_param(core_mask, uint, 0444);
MODULE_PARM_DESC(core_mask, "PACC core mask within each selected cluster");

static bool sys_cold_reset = true;
module_param(sys_cold_reset, bool, 0444);
MODULE_PARM_DESC(sys_cold_reset, "Pulse SYS_RESET before releasing selected cores");

static bool force_resetpc_reload = true;
module_param(force_resetpc_reload, bool, 0444);
MODULE_PARM_DESC(force_resetpc_reload, "Pulse FORCE_RESETPC_RELOAD after programming reset vectors");

static bool cores_are_wfi;
module_param(cores_are_wfi, bool, 0444);
MODULE_PARM_DESC(cores_are_wfi, "Required safety acknowledgement before reset/release");

static bool boot_on_load;
module_param(boot_on_load, bool, 0444);
MODULE_PARM_DESC(boot_on_load, "Load firmware and release PACC cores when module loads");

static bool reset_only_on_load;
module_param(reset_only_on_load, bool, 0444);
MODULE_PARM_DESC(reset_only_on_load, "Assert SYS/core reset for selected PACC clusters without releasing them");

static bool staged_boot = true;
module_param(staged_boot, bool, 0444);
MODULE_PARM_DESC(staged_boot, "Prepare all selected PACC clusters before releasing any cores");

static bool diag_on_load;
module_param(diag_on_load, bool, 0444);
MODULE_PARM_DESC(diag_on_load, "Only dump PACC CFG/CRG/mailbox registers selected by pacc_mask");

static bool set_nonsecure;
module_param(set_nonsecure, bool, 0444);
MODULE_PARM_DESC(set_nonsecure, "Set CFG+0xbc nonsecure bits 26:27 before release");

static bool clear_nonsecure_bits;
module_param(clear_nonsecure_bits, bool, 0444);
MODULE_PARM_DESC(clear_nonsecure_bits, "Clear CFG+0xbc nonsecure bits 26:27 before release");

static bool clear_addr_window_bit23;
module_param(clear_addr_window_bit23, bool, 0444);
MODULE_PARM_DESC(clear_addr_window_bit23,
		 "Clear CFG+0xc4 bit23 before release, matching the vendor lx500 init path");

static bool clear_db_status;
module_param(clear_db_status, bool, 0444);
MODULE_PARM_DESC(clear_db_status, "Write all-ones to DB status/clear registers before sending base command");

static bool clear_db_on_load;
module_param(clear_db_on_load, bool, 0444);
MODULE_PARM_DESC(clear_db_on_load, "Only clear mailbox doorbell/status registers for selected PACC devices");

static bool patch_pacc_id_env;
module_param(patch_pacc_id_env, bool, 0444);
MODULE_PARM_DESC(patch_pacc_id_env,
		 "Patch HETGPU_PACC_ID=0 in the copied firmware image to the selected PACC id");

static void __iomem *map_cfg(unsigned int pacc_id)
{
	return ioremap(pacc_base[pacc_id] + LX500_PACC_TOP_ABP_CFG_BASE, 0x1000);
}

static void __iomem *map_crg(unsigned int pacc_id)
{
	return ioremap(pacc_base[pacc_id] + LX500_PACC_TOP_ABP_CRG_BASE, 0x1000);
}

static void __iomem *map_mbox_ram(unsigned int pacc_id)
{
	return ioremap(pacc_base[pacc_id] + LX500_PACC_TOP_ABP_MBOX_RAM_BASE,
		       LX500_PACC_MBOX_RAM_SIZE);
}

static void __iomem *map_mbox_db(unsigned int pacc_id)
{
	return ioremap(pacc_base[pacc_id] + LX500_PACC_TOP_ABP_MBOX_DB_BASE, 0x1000);
}

static ulong pacc_load_addr(unsigned int pacc_id)
{
	if (!per_pacc_load)
		return load_addr;
	return load_addr + (ulong)pacc_id * reserved_size;
}

static ulong pacc_base_cmd_addr(unsigned int pacc_id)
{
	if (base_cmd_addr)
		return base_cmd_addr;
	return pacc_load_addr(pacc_id);
}

static void write32(void __iomem *cfg, u64 off, u32 value)
{
	iowrite32(value, cfg + off);
	wmb();
}

static u32 read32(void __iomem *base, u64 off)
{
	u32 value = ioread32(base + off);

	rmb();
	return value;
}

static void set32(void __iomem *base, u64 off, u32 mask)
{
	write32(base, off, read32(base, off) | mask);
}

static void clear32(void __iomem *base, u64 off, u32 mask)
{
	write32(base, off, read32(base, off) & ~mask);
}

static void init_crg_1100_ref2pll(unsigned int pacc_id)
{
	void __iomem *crg;
	u32 value;
	unsigned int i;

	crg = map_crg(pacc_id);
	if (!crg) {
		pr_err("hetgpu_pacc_boot: failed to map CRG for pacc%u\n", pacc_id);
		return;
	}

	/*
	 * Mirrors the vendor driver's pacc_lx500_init() path:
	 * crg_cfg_pll_freq(..., 1100) followed by crg_clk_init_ref2pll().
	 */
	write32(crg, 0x8, 0x00dc0a12);
	write32(crg, 0x0, 0x00000110);
	write32(crg, 0x0, 0x00000111);
	for (i = 0; i < 200; i++) {
		value = read32(crg, 0x0);
		if (value & BIT(24))
			break;
		udelay(1000);
	}
	value = read32(crg, 0x0);
	write32(crg, 0x0, value & ~BIT(4));
	value = read32(crg, 0x0);
	write32(crg, 0x0, value | BIT(12));
	value = (1U << 21) - 1;
	write32(crg, 0x130, value);
	write32(crg, 0x134, value);
	pr_info("hetgpu_pacc_boot: pacc%u CRG init done pll_status=0x%08x\n",
		pacc_id, read32(crg, 0x0));

	iounmap(crg);
}

static int send_base_cmd(unsigned int pacc_id)
{
	ulong base = pacc_base_cmd_addr(pacc_id);
	struct pacc_base_cmd cmd = {
		.base = (u64)base,
		.size = (u64)reserved_size,
	};
	void __iomem *ram;
	void __iomem *db;

	ram = map_mbox_ram(pacc_id);
	if (!ram)
		return -ENOMEM;
	db = map_mbox_db(pacc_id);
	if (!db) {
		iounmap(ram);
		return -ENOMEM;
	}

	memset_io(ram, 0, LX500_PACC_MBOX_RAM_SIZE);
	if (clear_db_status) {
		write32(db, 0x4, 0xffffffff);
		write32(db, 0x8, 0xffffffff);
	}
	write32(db, 0x10, 0xffffffff);
	memcpy_toio(ram, &cmd, sizeof(cmd));
	wmb();
	write32(db, 0x0, local_doorbell_bit ? 1U : (1U << pacc_id));
	pr_info("hetgpu_pacc_boot: pacc%u base command base=0x%llx size=0x%llx db_mask=0x%x\n",
		pacc_id, cmd.base, cmd.size,
		local_doorbell_bit ? 1U : (1U << pacc_id));

	iounmap(db);
	iounmap(ram);
	return 0;
}

static void release_selected_cores(void __iomem *cfg)
{
	unsigned int core;

	for (core = 0; core < PACC_CORE_COUNT; core++) {
		if (!(core_mask & BIT(core)))
			continue;
		clear32(cfg, LX500_PACC_CFG_CORE_RESET0 + 4ULL * core, 0x3);
	}
}

static void clear_one_pacc_db(unsigned int pacc_id)
{
	void __iomem *db;

	db = map_mbox_db(pacc_id);
	if (!db) {
		pr_err("hetgpu_pacc_boot: failed to map mailbox DB for pacc%u\n", pacc_id);
		return;
	}

	write32(db, 0x4, 0xffffffff);
	write32(db, 0x8, 0xffffffff);
	write32(db, 0x10, 0xffffffff);
	pr_info("hetgpu_pacc_boot: pacc%u cleared mailbox DB status\n", pacc_id);
	iounmap(db);
}

static void clear_selected_pacc_dbs(void)
{
	unsigned int pacc_id;

	for (pacc_id = 0; pacc_id < PACC_COUNT; pacc_id++) {
		if (pacc_mask & BIT(pacc_id))
			clear_one_pacc_db(pacc_id);
	}
}

static int reset_one_pacc(unsigned int pacc_id)
{
	void __iomem *cfg;
	unsigned int core;

	cfg = map_cfg(pacc_id);
	if (!cfg)
		return -ENOMEM;

	for (core = 0; core < PACC_CORE_COUNT; core++) {
		if (!(core_mask & BIT(core)))
			continue;
		set32(cfg, LX500_PACC_CFG_CORE_RESET0 + 4ULL * core, 0x3);
	}
	set32(cfg, LX500_PACC_CFG_SYS_RESET, 0x1);
	iounmap(cfg);

	clear_one_pacc_db(pacc_id);
	pr_info("hetgpu_pacc_boot: pacc%u held in reset core_mask=0x%x\n",
		pacc_id, core_mask);
	return 0;
}

static int reset_selected_paccs(void)
{
	unsigned int pacc_id;
	int ret;

	if (!cores_are_wfi) {
		pr_err("hetgpu_pacc_boot: refusing reset-only without cores_are_wfi=1\n");
		return -EPERM;
	}
	if ((pacc_mask & ~0xfU) || (core_mask & ~0xfU) || !pacc_mask || !core_mask)
		return -EINVAL;

	for (pacc_id = 0; pacc_id < PACC_COUNT; pacc_id++) {
		if (!(pacc_mask & BIT(pacc_id)))
			continue;
		ret = reset_one_pacc(pacc_id);
		if (ret)
			return ret;
	}
	return 0;
}

static void patch_copied_firmware_pacc_id(void *dst, size_t size, unsigned int pacc_id)
{
	static const char needle[] = "HETGPU_PACC_ID=0";
	char *cursor = dst;
	size_t remaining = size;
	unsigned int patched = 0;

	if (!patch_pacc_id_env || pacc_id >= PACC_COUNT || size < sizeof(needle) - 1)
		return;

	while (remaining >= sizeof(needle) - 1) {
		char *hit = memchr(cursor, needle[0], remaining - (sizeof(needle) - 2));
		if (!hit)
			break;
		remaining -= (size_t)(hit - cursor);
		cursor = hit;
		if (!memcmp(cursor, needle, sizeof(needle) - 1)) {
			cursor[sizeof(needle) - 2] = '0' + pacc_id;
			wmb();
			patched++;
		}
		cursor++;
		remaining--;
	}

	if (patched) {
		pr_info("hetgpu_pacc_boot: patched %u firmware HETGPU_PACC_ID markers to %u\n",
			patched, pacc_id);
	} else {
		pr_warn("hetgpu_pacc_boot: HETGPU_PACC_ID=0 marker not found for pacc%u\n",
			pacc_id);
	}
}

static int copy_firmware_to_phys(const struct firmware *fw, unsigned int pacc_id)
{
	void *dst;
	void __iomem *iodst;
	phys_addr_t end;
	ulong base = pacc_load_addr(pacc_id);

	if (base < 0x80000000UL) {
		pr_err("hetgpu_pacc_boot: refusing suspicious load_addr=0x%lx\n",
		       base);
		return -EINVAL;
	}
	if (check_add_overflow((phys_addr_t)base, (phys_addr_t)fw->size,
			       &end)) {
		pr_err("hetgpu_pacc_boot: firmware range overflows: load_addr=0x%lx size=%zu\n",
		       base, fw->size);
		return -EINVAL;
	}

	dst = memremap((phys_addr_t)base, fw->size, MEMREMAP_WC);
	if (dst) {
		memcpy(dst, fw->data, fw->size);
		patch_copied_firmware_pacc_id(dst, fw->size, pacc_id);
		wmb();
		memunmap(dst);
		pr_info("hetgpu_pacc_boot: copied %zu bytes for pacc%u to phys 0x%lx via MEMREMAP_WC\n",
			fw->size, pacc_id, base);
		return 0;
	}

	iodst = ioremap((phys_addr_t)base, fw->size);
	if (!iodst)
		return -ENOMEM;
	memcpy_toio(iodst, fw->data, fw->size);
	wmb();
	iounmap(iodst);
	pr_info("hetgpu_pacc_boot: copied %zu bytes for pacc%u to phys 0x%lx via ioremap\n",
		fw->size, pacc_id, base);
	return 0;
}

static int prepare_one_pacc(unsigned int pacc_id)
{
	void __iomem *cfg;
	ulong pacc_entry = per_pacc_entry ? pacc_load_addr(pacc_id) : entry;
	u32 lo = (u32)(pacc_entry & 0xffffffffUL);
	u32 hi = (u32)((u64)pacc_entry >> 32);
	unsigned int core;

	cfg = map_cfg(pacc_id);
	if (!cfg)
		return -ENOMEM;

	if (pacc_entry < 0x30000000UL) {
		pr_err("hetgpu_pacc_boot: refusing suspicious entry=0x%lx\n",
		       pacc_entry);
		iounmap(cfg);
		return -EINVAL;
	}

	pr_info("hetgpu_pacc_boot: boot pacc%u entry=0x%lx core_mask=0x%x sys_cold_reset=%d\n",
		pacc_id, pacc_entry, core_mask, sys_cold_reset ? 1 : 0);

	for (core = 0; core < PACC_CORE_COUNT; core++) {
		if (!(core_mask & BIT(core)))
			continue;
		set32(cfg, LX500_PACC_CFG_CORE_RESET0 + 4ULL * core, 0x3);
	}

	if (sys_cold_reset)
		set32(cfg, LX500_PACC_CFG_SYS_RESET, 0x1);

	init_crg_1100_ref2pll(pacc_id);

	for (core = 0; core < PACC_CORE_COUNT; core++) {
		if (!(core_mask & BIT(core)))
			continue;
		write32(cfg, LX500_PACC_CFG_RST_VEC_LOW + 8ULL * core, lo);
		write32(cfg, LX500_PACC_CFG_RST_VEC_HIGH + 8ULL * core, hi);
	}

	/*
	 * Match the vendor lx500 init path: pacc_dev_info_lx500 requests that
	 * CFG[0xbc].bit0 is cleared before SYS_RESET is released.
	 */
	clear32(cfg, LX500_PACC_CFG_SECURE_TIEOFF, BIT(0));
	if (clear_nonsecure_bits)
		clear32(cfg, LX500_PACC_CFG_SECURE_TIEOFF, 0x3U << 26);
	if (clear_addr_window_bit23)
		clear32(cfg, LX500_PACC_CFG_ADDR_WINDOW, BIT(23));
	if (set_nonsecure)
		set32(cfg, LX500_PACC_CFG_SECURE_TIEOFF, 0x3U << 26);

	if (force_resetpc_reload) {
		write32(cfg, LX500_PACC_CFG_FORCE_RELOAD, core_mask & 0xfU);
		write32(cfg, LX500_PACC_CFG_FORCE_RELOAD, 0);
	}

	iounmap(cfg);
	return 0;
}

static int release_one_pacc(unsigned int pacc_id)
{
	void __iomem *cfg;
	int ret;

	cfg = map_cfg(pacc_id);
	if (!cfg)
		return -ENOMEM;

	clear32(cfg, LX500_PACC_CFG_SYS_RESET, 0x1);

	if (send_base_cmd_param && !send_base_after_core_release) {
		ret = send_base_cmd(pacc_id);
		if (ret) {
			iounmap(cfg);
			return ret;
		}
	} else if (!send_base_cmd_param) {
		pr_info("hetgpu_pacc_boot: pacc%u skipping base command\n", pacc_id);
	}

	release_selected_cores(cfg);

	if (send_base_cmd_param && send_base_after_core_release) {
		if (send_base_delay_ms)
			msleep(send_base_delay_ms);
		ret = send_base_cmd(pacc_id);
		if (ret) {
			iounmap(cfg);
			return ret;
		}
	}

	iounmap(cfg);
	return 0;
}

static int boot_one_pacc(unsigned int pacc_id)
{
	int ret;

	ret = prepare_one_pacc(pacc_id);
	if (ret)
		return ret;
	return release_one_pacc(pacc_id);
}

static int boot_selected_paccs(void)
{
	const struct firmware *fw;
	unsigned int pacc_id;
	int ret;

	if (!cores_are_wfi) {
		pr_err("hetgpu_pacc_boot: refusing reset without cores_are_wfi=1\n");
		return -EPERM;
	}
	if ((pacc_mask & ~0xfU) || (core_mask & ~0xfU) || !pacc_mask || !core_mask)
		return -EINVAL;

	ret = request_firmware(&fw, firmware_name, NULL);
	if (ret) {
		pr_err("hetgpu_pacc_boot: request_firmware(%s) failed: %d\n",
		       firmware_name, ret);
		return ret;
	}

	for (pacc_id = 0; pacc_id < PACC_COUNT; pacc_id++) {
		if (!(pacc_mask & BIT(pacc_id)))
			continue;
		ret = copy_firmware_to_phys(fw, pacc_id);
		if (ret) {
			release_firmware(fw);
			return ret;
		}
		if (!staged_boot) {
			ret = boot_one_pacc(pacc_id);
			if (ret) {
				release_firmware(fw);
				return ret;
			}
		}
	}
	if (staged_boot) {
		pr_info("hetgpu_pacc_boot: staged boot prepare mask=0x%x\n", pacc_mask);
		for (pacc_id = 0; pacc_id < PACC_COUNT; pacc_id++) {
			if (!(pacc_mask & BIT(pacc_id)))
				continue;
			ret = prepare_one_pacc(pacc_id);
			if (ret) {
				release_firmware(fw);
				return ret;
			}
		}
		pr_info("hetgpu_pacc_boot: staged boot release mask=0x%x\n", pacc_mask);
		for (pacc_id = 0; pacc_id < PACC_COUNT; pacc_id++) {
			if (!(pacc_mask & BIT(pacc_id)))
				continue;
			ret = release_one_pacc(pacc_id);
			if (ret) {
				release_firmware(fw);
				return ret;
			}
		}
	}
	release_firmware(fw);
	return 0;
}

static void dump_words(const char *tag, void __iomem *base, u64 off, unsigned int words)
{
	unsigned int i;

	for (i = 0; i < words; i += 4) {
		pr_info("hetgpu_pacc_boot: %s+0x%03x %08x %08x %08x %08x\n",
			tag, i * 4,
			read32(base, off + (i + 0) * 4),
			read32(base, off + (i + 1) * 4),
			read32(base, off + (i + 2) * 4),
			read32(base, off + (i + 3) * 4));
	}
}

static void diag_one_pacc(unsigned int pacc_id)
{
	void __iomem *cfg;
	void __iomem *crg;
	void __iomem *ram;
	void __iomem *db;
	void *fwmem;
	unsigned int core;

	cfg = map_cfg(pacc_id);
	crg = map_crg(pacc_id);
	ram = map_mbox_ram(pacc_id);
	db = map_mbox_db(pacc_id);
	if (!cfg || !crg || !ram || !db) {
		pr_err("hetgpu_pacc_boot: pacc%u diag map failed cfg=%p crg=%p ram=%p db=%p\n",
		       pacc_id, cfg, crg, ram, db);
		goto out;
	}

	pr_info("hetgpu_pacc_boot: pacc%u diag base=0x%llx load=0x%lx entry=0x%lx\n",
		pacc_id, pacc_base[pacc_id], pacc_load_addr(pacc_id),
		per_pacc_entry ? pacc_load_addr(pacc_id) : entry);
	pr_info("hetgpu_pacc_boot: pacc%u crg0=0x%08x pll=0x%08x gate=0x%08x sel=0x%08x\n",
		pacc_id, read32(crg, 0x0), read32(crg, 0x8),
		read32(crg, 0x130), read32(crg, 0x134));
	pr_info("hetgpu_pacc_boot: pacc%u sys_reset=0x%08x force_reload=0x%08x secure=0x%08x db0=0x%08x db4=0x%08x db8=0x%08x db10=0x%08x\n",
		pacc_id, read32(cfg, LX500_PACC_CFG_SYS_RESET),
		read32(cfg, LX500_PACC_CFG_FORCE_RELOAD),
		read32(cfg, LX500_PACC_CFG_SECURE_TIEOFF),
		read32(db, 0x0), read32(db, 0x4), read32(db, 0x8), read32(db, 0x10));
	pr_info("hetgpu_pacc_boot: pacc%u addr_window=0x%08x\n",
		pacc_id, read32(cfg, LX500_PACC_CFG_ADDR_WINDOW));
	for (core = 0; core < PACC_CORE_COUNT; core++) {
		pr_info("hetgpu_pacc_boot: pacc%u core%u_reset=0x%08x reset_vec=0x%08x_%08x\n",
			pacc_id, core,
			read32(cfg, LX500_PACC_CFG_CORE_RESET0 + 4ULL * core),
			read32(cfg, LX500_PACC_CFG_RST_VEC_HIGH + 8ULL * core),
			read32(cfg, LX500_PACC_CFG_RST_VEC_LOW + 8ULL * core));
	}
	dump_words("ap2pacc", ram, 0, 16);
	dump_words("pacc2ap", ram, 0x2000, 16);
	fwmem = memremap((phys_addr_t)pacc_load_addr(pacc_id), 0x80, MEMREMAP_WB);
	if (!fwmem)
		fwmem = memremap((phys_addr_t)pacc_load_addr(pacc_id), 0x80, MEMREMAP_WT);
	if (fwmem) {
		u32 *words = fwmem;

		pr_info("hetgpu_pacc_boot: pacc%u fw@0x%lx %08x %08x %08x %08x %08x %08x %08x %08x\n",
			pacc_id, pacc_load_addr(pacc_id),
			words[0], words[1], words[2], words[3],
			words[4], words[5], words[6], words[7]);
		memunmap(fwmem);
	} else {
		pr_err("hetgpu_pacc_boot: pacc%u failed to map fw load window 0x%lx\n",
		       pacc_id, pacc_load_addr(pacc_id));
	}

out:
	if (db)
		iounmap(db);
	if (ram)
		iounmap(ram);
	if (crg)
		iounmap(crg);
	if (cfg)
		iounmap(cfg);
}

static void diag_selected_paccs(void)
{
	unsigned int pacc_id;

	for (pacc_id = 0; pacc_id < PACC_COUNT; pacc_id++) {
		if (pacc_mask & BIT(pacc_id))
			diag_one_pacc(pacc_id);
	}
}

static int __init hetgpu_pacc_boot_init(void)
{
	if (clear_db_on_load)
		clear_selected_pacc_dbs();
	if (diag_on_load)
		diag_selected_paccs();
	if (reset_only_on_load)
		return reset_selected_paccs();
	if (!boot_on_load) {
		pr_info("hetgpu_pacc_boot: loaded with boot_on_load=0\n");
		return 0;
	}
	return boot_selected_paccs();
}

static void __exit hetgpu_pacc_boot_exit(void)
{
	pr_info("hetgpu_pacc_boot: unloaded\n");
}

module_init(hetgpu_pacc_boot_init);
module_exit(hetgpu_pacc_boot_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("One-shot LX500 PACC firmware boot helper");
