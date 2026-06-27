use crate::r#impl::concordia_delta::{AofDiskLog, AofRecord};
use crate::r#impl::concordia_runtime::mpi_scoped_aof_path;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

const PAGE_SIZE: usize = 4096;
const DEFAULT_MIN_ALLOCATION_BYTES: usize = 1 << 20;
const DEFAULT_MAX_REGIONS: usize = 128;
const DEFAULT_CHECKPOINT_EVERY: u64 = 1;
const DEFAULT_AOF_PATH: &str = "/tmp/concordia/kimi.aof";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KimiConcordiaConfig {
    enabled: bool,
    restore_enabled: bool,
    min_allocation_bytes: usize,
    max_regions: usize,
    checkpoint_every_launches: u64,
    aof_path: PathBuf,
    restore_aof_path: PathBuf,
    logs_enabled: bool,
}

impl KimiConcordiaConfig {
    pub(crate) fn from_env() -> Self {
        let pairs: Vec<(String, String)> = CONFIG_KEYS
            .iter()
            .filter_map(|key| {
                std::env::var(key)
                    .ok()
                    .map(|value| ((*key).to_string(), value))
            })
            .collect();
        let refs: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        Self::from_pairs(&refs)
    }

    pub(crate) fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        let enabled = read_bool_from_pairs(pairs, &["HETGPU_KIMI_CONCORDIA"], false);
        let restore_enabled =
            read_bool_from_pairs(pairs, &["HETGPU_KIMI_CONCORDIA_RESTORE"], false);
        let min_allocation_bytes = read_usize_from_pairs(
            pairs,
            &["HETGPU_KIMI_CONCORDIA_ALLOC_MIN_BYTES"],
            DEFAULT_MIN_ALLOCATION_BYTES,
        )
        .max(PAGE_SIZE);
        let max_regions = read_usize_from_pairs(
            pairs,
            &["HETGPU_KIMI_CONCORDIA_MAX_REGIONS"],
            DEFAULT_MAX_REGIONS,
        );
        let checkpoint_every_launches = read_u64_from_pairs(
            pairs,
            &["HETGPU_KIMI_CONCORDIA_CHECKPOINT_EVERY"],
            DEFAULT_CHECKPOINT_EVERY,
        )
        .max(1);
        let aof_path = read_path_from_pairs(
            pairs,
            &["HETGPU_KIMI_CONCORDIA_AOF_PATH", "CONCORDIA_AOF_PATH"],
            DEFAULT_AOF_PATH,
        );
        let restore_aof_path = read_path_from_pairs(
            pairs,
            &[
                "HETGPU_KIMI_CONCORDIA_RESTORE_AOF",
                "CONCORDIA_RESTORE_AOF_PATH",
                "HETGPU_KIMI_CONCORDIA_AOF_PATH",
                "CONCORDIA_AOF_PATH",
            ],
            aof_path.to_string_lossy().as_ref(),
        );
        let logs_enabled = read_bool_from_pairs(
            pairs,
            &["HETGPU_KIMI_CONCORDIA_LOGS", "CONCORDIA_LOGS"],
            false,
        );

        Self {
            enabled,
            restore_enabled,
            min_allocation_bytes,
            max_regions,
            checkpoint_every_launches,
            aof_path,
            restore_aof_path,
            logs_enabled,
        }
    }

    fn should_track_allocation(&self, size: usize) -> bool {
        self.enabled && tracked_len(size) >= self.min_allocation_bytes
    }
}

const CONFIG_KEYS: &[&str] = &[
    "HETGPU_KIMI_CONCORDIA",
    "HETGPU_KIMI_CONCORDIA_RESTORE",
    "HETGPU_KIMI_CONCORDIA_ALLOC_MIN_BYTES",
    "HETGPU_KIMI_CONCORDIA_MAX_REGIONS",
    "HETGPU_KIMI_CONCORDIA_CHECKPOINT_EVERY",
    "HETGPU_KIMI_CONCORDIA_AOF_PATH",
    "CONCORDIA_AOF_PATH",
    "HETGPU_KIMI_CONCORDIA_RESTORE_AOF",
    "CONCORDIA_RESTORE_AOF_PATH",
    "HETGPU_KIMI_CONCORDIA_LOGS",
    "CONCORDIA_LOGS",
];

#[derive(Debug)]
struct KimiTrackedRegion {
    ptr: u64,
    requested_size: usize,
    tracked_size: usize,
    region_id: u64,
    initialized: bool,
    restored: bool,
    #[cfg(all(
        feature = "nvidia",
        not(feature = "amd"),
        not(feature = "intel"),
        not(feature = "tenstorrent")
    ))]
    device: Option<DeviceCheckpointResources>,
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
#[derive(Debug)]
struct DeviceCheckpointResources {
    shadow: u64,
    bitmap: u64,
    bitmap_words: usize,
}

pub(crate) struct KimiConcordiaManager {
    config: KimiConcordiaConfig,
    next_region_id: u64,
    next_epoch: u64,
    stateful_launches: u64,
    allocations: BTreeMap<u64, KimiTrackedRegion>,
    aof: Option<AofDiskLog>,
    #[cfg(all(
        feature = "nvidia",
        not(feature = "amd"),
        not(feature = "intel"),
        not(feature = "tenstorrent")
    ))]
    worker_handle: Option<i64>,
}

impl KimiConcordiaManager {
    fn new(config: KimiConcordiaConfig) -> Self {
        Self {
            config,
            next_region_id: 1,
            next_epoch: 1,
            stateful_launches: 0,
            allocations: BTreeMap::new(),
            aof: None,
            #[cfg(all(
                feature = "nvidia",
                not(feature = "amd"),
                not(feature = "intel"),
                not(feature = "tenstorrent")
            ))]
            worker_handle: None,
        }
    }

    pub(crate) fn new_for_test(config: KimiConcordiaConfig) -> Self {
        Self::new(config)
    }

    fn note_allocation_inner(&mut self, ptr: u64, size: usize) -> Option<u64> {
        if ptr == 0 || !self.config.should_track_allocation(size) {
            return None;
        }
        if self.allocations.contains_key(&ptr) || self.allocations.len() >= self.config.max_regions
        {
            return None;
        }

        let region_id = self.next_region_id;
        self.next_region_id += 1;
        let tracked_size = tracked_len(size);
        self.allocations.insert(
            ptr,
            KimiTrackedRegion {
                ptr,
                requested_size: size,
                tracked_size,
                region_id,
                initialized: false,
                restored: false,
                #[cfg(all(
                    feature = "nvidia",
                    not(feature = "amd"),
                    not(feature = "intel"),
                    not(feature = "tenstorrent")
                ))]
                device: None,
            },
        );
        self.log(&format!(
            "registered allocation ptr=0x{ptr:x} size={size} tracked={tracked_size} region={region_id}"
        ));
        Some(region_id)
    }

    pub(crate) fn note_allocation_for_test(&mut self, ptr: u64, size: usize) -> Option<u64> {
        self.note_allocation_inner(ptr, size)
    }

    fn should_checkpoint_after_launch_inner(&mut self, kernel_name: &str) -> bool {
        if !self.config.enabled || !is_kimi_stateful_kernel_name(kernel_name) {
            return false;
        }
        self.stateful_launches = self.stateful_launches.saturating_add(1);
        self.stateful_launches % self.config.checkpoint_every_launches == 0
    }

    pub(crate) fn should_checkpoint_after_launch_for_test(&mut self, kernel_name: &str) -> bool {
        self.should_checkpoint_after_launch_inner(kernel_name)
    }

    fn log(&self, message: &str) {
        if self.config.logs_enabled {
            eprintln!("[hetGPU Kimi Concordia] {message}");
        }
    }

    fn open_aof(&mut self) -> Result<&mut AofDiskLog, String> {
        if self.aof.is_none() {
            let path = mpi_scoped_aof_path(self.config.aof_path.clone());
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|err| {
                        format!("create Concordia AOF dir {}: {err}", parent.display())
                    })?;
                }
            }
            self.aof = Some(
                AofDiskLog::open_append(&path)
                    .map_err(|err| format!("open Concordia AOF {}: {err}", path.display()))?,
            );
        }
        self.aof
            .as_mut()
            .ok_or_else(|| "Concordia AOF log was not initialized".to_string())
    }

    fn append_record(
        &mut self,
        epoch: u64,
        region_id: u64,
        offset: usize,
        payload: Vec<u8>,
    ) -> Result<(), String> {
        self.open_aof()?
            .append_record(&AofRecord {
                epoch,
                region_id,
                offset,
                payload,
            })
            .map_err(|err| format!("append Concordia AOF record for region {region_id}: {err}"))
    }
}

pub(crate) fn is_kimi_stateful_kernel_name(kernel_name: &str) -> bool {
    let name = kernel_name.to_ascii_lowercase();
    if name.contains("kv")
        || name.contains("k_cache")
        || name.contains("v_cache")
        || name.contains("cache_update")
        || name.contains("paged_cache")
        || name.contains("pagedattention")
    {
        return true;
    }

    let attention_like =
        name.contains("attention") || name.contains("flash_attn") || name.contains("_attn");
    let decode_like = name.contains("decode")
        || name.contains("rope")
        || name.contains("neox")
        || name.contains("qkv")
        || name.contains("kq");
    attention_like && decode_like
}

pub(crate) fn dirty_page_offsets_from_words(bitmap_words: &[u64], page_count: usize) -> Vec<usize> {
    let mut offsets = Vec::new();
    for page in 0..page_count {
        let word = bitmap_words.get(page / 64).copied().unwrap_or(0);
        if word & (1_u64 << (page % 64)) != 0 {
            offsets.push(page * PAGE_SIZE);
        }
    }
    offsets
}

pub(crate) fn note_allocation(ptr: u64, size: usize) {
    let Ok(mut manager) = global_manager().lock() else {
        return;
    };
    let Some(region_id) = manager.note_allocation_inner(ptr, size) else {
        return;
    };

    #[cfg(all(
        feature = "nvidia",
        not(feature = "amd"),
        not(feature = "intel"),
        not(feature = "tenstorrent")
    ))]
    if manager.config.restore_enabled {
        if let Err(err) = manager.restore_device_region(ptr) {
            manager.log(&format!(
                "restore skipped for ptr=0x{ptr:x} region={region_id}: {err}"
            ));
        }
    }
}

pub(crate) fn note_deallocation(ptr: u64) {
    let Ok(mut manager) = global_manager().lock() else {
        return;
    };
    let Some(region) = manager.allocations.remove(&ptr) else {
        return;
    };

    #[cfg(all(
        feature = "nvidia",
        not(feature = "amd"),
        not(feature = "intel"),
        not(feature = "tenstorrent")
    ))]
    manager.release_device_resources(region);
}

fn global_manager() -> &'static Mutex<KimiConcordiaManager> {
    static MANAGER: OnceLock<Mutex<KimiConcordiaManager>> = OnceLock::new();
    MANAGER.get_or_init(|| Mutex::new(KimiConcordiaManager::new(KimiConcordiaConfig::from_env())))
}

fn tracked_len(size: usize) -> usize {
    size / PAGE_SIZE * PAGE_SIZE
}

fn read_bool_from_pairs(pairs: &[(&str, &str)], keys: &[&str], default: bool) -> bool {
    keys.iter()
        .find_map(|key| {
            pairs
                .iter()
                .find(|(candidate, _)| candidate == key)
                .map(|(_, value)| parse_bool(value))
        })
        .unwrap_or(default)
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn read_usize_from_pairs(pairs: &[(&str, &str)], keys: &[&str], default: usize) -> usize {
    keys.iter()
        .find_map(|key| {
            pairs
                .iter()
                .find(|(candidate, _)| candidate == key)
                .and_then(|(_, value)| value.trim().parse::<usize>().ok())
        })
        .unwrap_or(default)
}

fn read_u64_from_pairs(pairs: &[(&str, &str)], keys: &[&str], default: u64) -> u64 {
    keys.iter()
        .find_map(|key| {
            pairs
                .iter()
                .find(|(candidate, _)| candidate == key)
                .and_then(|(_, value)| value.trim().parse::<u64>().ok())
        })
        .unwrap_or(default)
}

fn read_path_from_pairs(pairs: &[(&str, &str)], keys: &[&str], default: &str) -> PathBuf {
    keys.iter()
        .find_map(|key| {
            pairs
                .iter()
                .find(|(candidate, _)| candidate == key)
                .map(|(_, value)| PathBuf::from(value))
        })
        .unwrap_or_else(|| PathBuf::from(default))
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
mod nvidia_device {
    use super::*;
    use crate::r#impl::nvidia_runtime_sys;
    use cuda_types::cuda::{CUdeviceptr, CUdeviceptr_v2, CUstream};
    use std::ffi::c_void;
    use std::ptr;

    impl KimiConcordiaManager {
        pub(crate) fn observe_kernel_launch(&mut self, kernel_name: &str, stream: CUstream) {
            if !self.should_checkpoint_after_launch_inner(kernel_name) {
                return;
            }

            let rc = nvidia_runtime_sys::cuStreamSynchronize_ckpt(stream);
            if rc != 0 {
                self.log(&format!(
                    "stream synchronize before checkpoint failed for {kernel_name}: {rc}"
                ));
                return;
            }

            if let Err(err) = self.checkpoint_device_regions(kernel_name) {
                self.log(&format!("checkpoint after {kernel_name} failed: {err}"));
            }
        }

        fn checkpoint_device_regions(&mut self, kernel_name: &str) -> Result<(), String> {
            if self.allocations.is_empty() {
                return Ok(());
            }
            let ptrs: Vec<u64> = self.allocations.keys().copied().collect();
            for ptr in ptrs {
                if !self.allocations.contains_key(&ptr) {
                    continue;
                }
                let needs_init = self
                    .allocations
                    .get(&ptr)
                    .map(|region| !region.initialized)
                    .unwrap_or(false);
                if needs_init {
                    self.initialize_device_region(ptr)?;
                }
                self.scan_device_region(ptr)?;
            }
            self.log(&format!(
                "checkpointed {} region(s) after {kernel_name}",
                self.allocations.len()
            ));
            Ok(())
        }

        fn initialize_device_region(&mut self, ptr: u64) -> Result<(), String> {
            let (region_id, tracked_size) = {
                let region = self
                    .allocations
                    .get(&ptr)
                    .ok_or_else(|| format!("unknown Kimi Concordia allocation 0x{ptr:x}"))?;
                (region.region_id, region.tracked_size)
            };
            let page_count = tracked_size / PAGE_SIZE;
            let bitmap_words = page_count.div_ceil(64).max(1);
            self.shutdown_worker();
            let shadow = raw_device_alloc(tracked_size)?;
            let bitmap = raw_device_alloc(bitmap_words * std::mem::size_of::<u64>())?;
            raw_memset(bitmap, 0, bitmap_words * std::mem::size_of::<u64>())?;

            let epoch = self.next_epoch;
            self.next_epoch = self.next_epoch.saturating_add(1);
            for offset in (0..tracked_size).step_by(PAGE_SIZE) {
                let payload = raw_device_to_host(ptr + offset as u64, PAGE_SIZE)?;
                raw_host_to_device(shadow + offset as u64, &payload)?;
                self.append_record(epoch, region_id, offset, payload)?;
            }

            let region = self
                .allocations
                .get_mut(&ptr)
                .ok_or_else(|| format!("unknown Kimi Concordia allocation 0x{ptr:x}"))?;
            region.initialized = true;
            region.device = Some(DeviceCheckpointResources {
                shadow,
                bitmap,
                bitmap_words,
            });
            self.log(&format!(
                "base snapshot region={region_id} ptr=0x{ptr:x} pages={page_count}"
            ));
            Ok(())
        }

        fn scan_device_region(&mut self, ptr: u64) -> Result<(), String> {
            let (region_id, tracked_size, shadow, bitmap, bitmap_words) = {
                let region = self
                    .allocations
                    .get(&ptr)
                    .ok_or_else(|| format!("unknown Kimi Concordia allocation 0x{ptr:x}"))?;
                let resources = region.device.as_ref().ok_or_else(|| {
                    format!("region {} has no checkpoint resources", region.region_id)
                })?;
                (
                    region.region_id,
                    region.tracked_size,
                    resources.shadow,
                    resources.bitmap,
                    resources.bitmap_words,
                )
            };
            let page_count = tracked_size / PAGE_SIZE;
            if page_count == 0 {
                return Ok(());
            }

            let bitmap_bytes = bitmap_words * std::mem::size_of::<u64>();
            raw_memset(bitmap, 0, bitmap_bytes)?;
            let handle = self.ensure_worker()?;
            let seq = unsafe {
                crate::r#impl::concordia_gpu::concordia_gpu_scan_dirty_pages(
                    handle,
                    ptr,
                    shadow,
                    bitmap,
                    page_count as i64,
                )
            };
            if seq < 0 {
                return Err(format!("enqueue dirty scan failed for region {region_id}"));
            }
            unsafe {
                crate::r#impl::concordia_gpu::concordia_gpu_sync(handle);
            }

            let bitmap_data = raw_device_to_host(bitmap, bitmap_bytes)?;
            let mut words = Vec::with_capacity(bitmap_words);
            for chunk in bitmap_data.chunks_exact(std::mem::size_of::<u64>()) {
                words.push(u64::from_le_bytes(
                    chunk.try_into().expect("chunk size checked"),
                ));
            }
            let offsets = dirty_page_offsets_from_words(&words, page_count);
            if offsets.is_empty() {
                return Ok(());
            }

            let epoch = self.next_epoch;
            self.next_epoch = self.next_epoch.saturating_add(1);
            for offset in offsets {
                let payload = raw_device_to_host(ptr + offset as u64, PAGE_SIZE)?;
                self.append_record(epoch, region_id, offset, payload)?;
            }
            Ok(())
        }

        fn ensure_worker(&mut self) -> Result<i64, String> {
            if let Some(handle) = self.worker_handle {
                if handle >= 0 {
                    return Ok(handle);
                }
            }
            let capacity = std::env::var("CONCORDIA_PERSISTENT_CAPACITY")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(1024);
            let handle = crate::r#impl::concordia_gpu::concordia_gpu_init(
                crate::r#impl::persistent_router::persistent_device_id(),
                capacity,
            );
            if handle < 0 {
                return Err("persistent checkpoint worker init failed".to_string());
            }
            self.worker_handle = Some(handle);
            Ok(handle)
        }

        fn shutdown_worker(&mut self) {
            if let Some(handle) = self.worker_handle.take() {
                unsafe {
                    crate::r#impl::concordia_gpu::concordia_gpu_shutdown(handle);
                }
            }
        }

        pub(crate) fn restore_device_region(&mut self, ptr: u64) -> Result<usize, String> {
            let (region_id, tracked_size) = {
                let region = self
                    .allocations
                    .get(&ptr)
                    .ok_or_else(|| format!("unknown Kimi Concordia allocation 0x{ptr:x}"))?;
                (region.region_id, region.tracked_size)
            };
            let path = mpi_scoped_aof_path(self.config.restore_aof_path.clone());
            let records = AofDiskLog::read_committed(&path)
                .map_err(|err| format!("read Concordia restore AOF {}: {err}", path.display()))?;
            let mut applied = 0usize;
            for record in records
                .iter()
                .filter(|record| record.region_id == region_id)
            {
                let end = record
                    .offset
                    .checked_add(record.payload.len())
                    .ok_or_else(|| {
                        format!(
                            "restore offset overflow for region {region_id} at {}",
                            record.offset
                        )
                    })?;
                if end > tracked_size {
                    return Err(format!(
                        "restore record for region {region_id} writes {}..{} beyond {}",
                        record.offset, end, tracked_size
                    ));
                }
                raw_host_to_device(ptr + record.offset as u64, &record.payload)?;
                applied += 1;
            }
            if let Some(region) = self.allocations.get_mut(&ptr) {
                region.restored = applied > 0;
            }
            self.log(&format!(
                "restored region={region_id} ptr=0x{ptr:x} records={applied}"
            ));
            Ok(applied)
        }

        pub(crate) fn release_device_resources(&mut self, region: KimiTrackedRegion) {
            if let Some(resources) = region.device {
                self.shutdown_worker();
                let _ = raw_device_free(resources.shadow);
                let _ = raw_device_free(resources.bitmap);
                self.log(&format!(
                    "released checkpoint resources region={} ptr=0x{:x}",
                    region.region_id, region.ptr
                ));
            }
        }
    }

    pub(crate) fn observe_kernel_launch(kernel_name: &str, stream: CUstream) {
        let Ok(mut manager) = super::global_manager().lock() else {
            return;
        };
        manager.observe_kernel_launch(kernel_name, stream);
    }

    fn raw_device_alloc(bytes: usize) -> Result<u64, String> {
        let mut dptr: CUdeviceptr = CUdeviceptr_v2(ptr::null_mut());
        let rc = nvidia_runtime_sys::cuMemAlloc_v2(&mut dptr, bytes);
        if rc != 0 {
            return Err(format!("cuMemAlloc_v2({bytes}) failed: {rc}"));
        }
        Ok(dptr.0 as u64)
    }

    fn raw_device_free(ptr_value: u64) -> Result<(), String> {
        if ptr_value == 0 {
            return Ok(());
        }
        let rc = nvidia_runtime_sys::cuMemFree_v2(CUdeviceptr_v2(ptr_value as *mut c_void));
        if rc != 0 {
            return Err(format!("cuMemFree_v2(0x{ptr_value:x}) failed: {rc}"));
        }
        Ok(())
    }

    fn raw_device_to_host(ptr_value: u64, len: usize) -> Result<Vec<u8>, String> {
        let mut data = vec![0u8; len];
        if len == 0 {
            return Ok(data);
        }
        let rc = nvidia_runtime_sys::cuMemcpyDtoH_v2(
            data.as_mut_ptr().cast::<c_void>(),
            CUdeviceptr_v2(ptr_value as *mut c_void),
            len,
        );
        if rc != 0 {
            return Err(format!(
                "cuMemcpyDtoH_v2(0x{ptr_value:x}, {len}) failed: {rc}"
            ));
        }
        Ok(data)
    }

    fn raw_host_to_device(ptr_value: u64, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        let rc = nvidia_runtime_sys::cuMemcpyHtoD_v2(
            CUdeviceptr_v2(ptr_value as *mut c_void),
            data.as_ptr().cast::<c_void>(),
            data.len(),
        );
        if rc != 0 {
            return Err(format!(
                "cuMemcpyHtoD_v2(0x{ptr_value:x}, {}) failed: {rc}",
                data.len()
            ));
        }
        Ok(())
    }

    fn raw_memset(ptr_value: u64, value: u8, len: usize) -> Result<(), String> {
        if len == 0 {
            return Ok(());
        }
        let rc =
            nvidia_runtime_sys::cuMemsetD8_v2(CUdeviceptr_v2(ptr_value as *mut c_void), value, len);
        if rc != 0 {
            return Err(format!(
                "cuMemsetD8_v2(0x{ptr_value:x}, {len}) failed: {rc}"
            ));
        }
        Ok(())
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) use nvidia_device::observe_kernel_launch;

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled_config() -> KimiConcordiaConfig {
        KimiConcordiaConfig::from_pairs(&[
            ("HETGPU_KIMI_CONCORDIA", "1"),
            ("HETGPU_KIMI_CONCORDIA_ALLOC_MIN_BYTES", "4096"),
            ("HETGPU_KIMI_CONCORDIA_MAX_REGIONS", "2"),
            ("HETGPU_KIMI_CONCORDIA_CHECKPOINT_EVERY", "3"),
            ("CONCORDIA_AOF_PATH", "/tmp/kimi.aof"),
        ])
    }

    #[test]
    fn classifies_kimi_stateful_kernel_names() {
        assert!(is_kimi_stateful_kernel_name("ggml_cuda_op_mul_mat_q_kv"));
        assert!(is_kimi_stateful_kernel_name("rope_neox_attention_decode"));
        assert!(is_kimi_stateful_kernel_name("paged_kv_cache_update"));
        assert!(!is_kimi_stateful_kernel_name("layer_03_ffn_gate_mul_mat"));
        assert!(!is_kimi_stateful_kernel_name("tensor_add_relu"));
    }

    #[test]
    fn allocation_policy_tracks_large_allocations_only_when_enabled() {
        let mut manager = KimiConcordiaManager::new_for_test(enabled_config());

        assert_eq!(manager.note_allocation_for_test(0x1000, 1024), None);
        assert_eq!(manager.note_allocation_for_test(0x2000, 4096), Some(1));
        assert_eq!(manager.note_allocation_for_test(0x3000, 8192), Some(2));
        assert_eq!(manager.note_allocation_for_test(0x4000, 16384), None);

        let disabled = KimiConcordiaConfig::from_pairs(&[("HETGPU_KIMI_CONCORDIA", "0")]);
        let mut manager = KimiConcordiaManager::new_for_test(disabled);
        assert_eq!(manager.note_allocation_for_test(0x5000, 1 << 20), None);
    }

    #[test]
    fn checkpoint_cadence_is_driven_by_stateful_kernels() {
        let mut manager = KimiConcordiaManager::new_for_test(enabled_config());

        assert!(!manager.should_checkpoint_after_launch_for_test("layer_0_ffn_mul_mat"));
        assert!(!manager.should_checkpoint_after_launch_for_test("paged_kv_cache_update"));
        assert!(!manager.should_checkpoint_after_launch_for_test("attention_decode"));
        assert!(manager.should_checkpoint_after_launch_for_test("rope_neox_attention"));
        assert!(!manager.should_checkpoint_after_launch_for_test("tensor_add_relu"));
    }

    #[test]
    fn bitmap_words_expand_to_dirty_page_offsets() {
        let bitmap = [0b1010_u64, 1_u64 << 1];
        assert_eq!(
            dirty_page_offsets_from_words(&bitmap, 70),
            vec![4096, 3 * 4096, 65 * 4096]
        );
    }
}
