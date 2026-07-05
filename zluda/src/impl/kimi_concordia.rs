use crate::r#impl::concordia_delta::{AofDiskLog, AofRecord};
use crate::r#impl::concordia_runtime::mpi_scoped_aof_path;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

const PAGE_SIZE: usize = 4096;
#[cfg(test)]
const DEFAULT_KIMI_PAGE_SIZE: usize = PAGE_SIZE;
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
    manifest_path: PathBuf,
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
        let manifest_default = default_manifest_path(&aof_path);
        let manifest_path = mpi_scoped_aof_path(read_path_from_pairs(
            pairs,
            &["HETGPU_KIMI_CONCORDIA_MANIFEST", "CONCORDIA_MANIFEST_PATH"],
            manifest_default.to_string_lossy().as_ref(),
        ));
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
            manifest_path,
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
    "HETGPU_KIMI_CONCORDIA_MANIFEST",
    "CONCORDIA_MANIFEST_PATH",
    "HETGPU_KIMI_CONCORDIA_RESTORE_AOF",
    "CONCORDIA_RESTORE_AOF_PATH",
    "HETGPU_KIMI_CONCORDIA_LOGS",
    "CONCORDIA_LOGS",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum KimiRegionKind {
    OpaqueMutable,
    KvCache,
    Adapter,
}

impl KimiRegionKind {
    fn stable_prefix(self) -> &'static str {
        match self {
            KimiRegionKind::OpaqueMutable => "opaque",
            KimiRegionKind::KvCache => "kv-cache",
            KimiRegionKind::Adapter => "adapter",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct KimiRegionSpec {
    pub region_id: u64,
    pub stable_key: String,
    pub kind: KimiRegionKind,
    pub ptr: u64,
    pub requested_size: usize,
    pub tracked_size: usize,
    pub allocation_index: u64,
    pub last_kernel: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allocator: Option<KimiAllocatorMetadata>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct KimiRegionManifest {
    pub regions: Vec<KimiRegionSpec>,
}

impl KimiRegionManifest {
    fn upsert(&mut self, spec: KimiRegionSpec) {
        match self
            .regions
            .iter_mut()
            .find(|region| region.ptr == spec.ptr)
        {
            Some(existing) => *existing = spec,
            None => self.regions.push(spec),
        }
        self.regions
            .sort_by_key(|region| (region.allocation_index, region.region_id));
    }

    fn remove_ptr(&mut self, ptr: u64) {
        self.regions.retain(|region| region.ptr != ptr);
    }

    fn find_region_id(
        &self,
        kind: KimiRegionKind,
        stable_key: &str,
        tracked_size: usize,
    ) -> Option<u64> {
        self.regions
            .iter()
            .find(|region| {
                region.kind == kind
                    && region.stable_key == stable_key
                    && region.tracked_size == tracked_size
            })
            .map(|region| region.region_id)
    }

    fn max_region_id(&self) -> u64 {
        self.regions
            .iter()
            .map(|region| region.region_id)
            .max()
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct KimiAllocatorMetadata {
    pub block_table_ptr: u64,
    pub block_count: usize,
    pub dirty_bitmap_ptr: u64,
    pub dirty_bitmap_bytes: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub block_table_snapshot: Vec<u32>,
}

#[derive(Debug)]
struct KimiTrackedRegion {
    ptr: u64,
    requested_size: usize,
    tracked_size: usize,
    region_id: u64,
    allocation_index: u64,
    kind: KimiRegionKind,
    stable_key: String,
    last_kernel: Option<String>,
    initialized: bool,
    restored: bool,
    allocator: Option<KimiAllocatorMetadata>,
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
    aof_staging_host: u64,
    aof_staging_device: u64,
    aof_staging_bytes: usize,
}

pub(crate) struct KimiConcordiaManager {
    config: KimiConcordiaConfig,
    next_region_id: u64,
    next_allocation_index: u64,
    next_epoch: u64,
    stateful_launches: u64,
    allocations: BTreeMap<u64, KimiTrackedRegion>,
    manifest: KimiRegionManifest,
    restore_manifest: Option<KimiRegionManifest>,
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
        let restore_manifest = if config.restore_enabled {
            read_manifest(&config.manifest_path).ok()
        } else {
            None
        };
        let next_region_id = restore_manifest
            .as_ref()
            .map(|manifest| manifest.max_region_id().saturating_add(1).max(1))
            .unwrap_or(1);
        Self {
            config,
            next_region_id,
            next_allocation_index: 1,
            next_epoch: 1,
            stateful_launches: 0,
            allocations: BTreeMap::new(),
            manifest: KimiRegionManifest::default(),
            restore_manifest,
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
        let allocation_index = self.next_allocation_index;
        self.next_allocation_index += 1;
        let tracked_size = tracked_len(size);
        let kind = KimiRegionKind::OpaqueMutable;
        let stable_key = format!(
            "{}:{allocation_index:04}:{tracked_size}",
            kind.stable_prefix()
        );
        self.allocations.insert(
            ptr,
            KimiTrackedRegion {
                ptr,
                requested_size: size,
                tracked_size,
                region_id,
                allocation_index,
                kind,
                stable_key,
                last_kernel: None,
                initialized: false,
                restored: false,
                allocator: None,
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
        self.upsert_manifest_for_ptr(ptr);
        self.persist_manifest_best_effort();
        Some(region_id)
    }

    pub(crate) fn note_allocation_for_test(&mut self, ptr: u64, size: usize) -> Option<u64> {
        self.note_allocation_inner(ptr, size)
    }

    pub(crate) fn note_kernel_pointers_for_test(&mut self, kernel_name: &str, ptrs: &[u64]) {
        self.note_kernel_pointers_inner(kernel_name, ptrs);
    }

    pub(crate) fn note_allocator_metadata_for_test(
        &mut self,
        ptr: u64,
        block_table_ptr: u64,
        block_count: usize,
        dirty_bitmap_ptr: u64,
        dirty_bitmap_bytes: usize,
    ) -> Result<(), String> {
        self.note_allocator_metadata_inner(
            ptr,
            block_table_ptr,
            block_count,
            dirty_bitmap_ptr,
            dirty_bitmap_bytes,
        )
    }

    pub(crate) fn manifest_snapshot_for_test(&self) -> KimiRegionManifest {
        self.manifest.clone()
    }

    pub(crate) fn set_restore_manifest_for_test(&mut self, manifest: KimiRegionManifest) {
        self.next_region_id = manifest.max_region_id().saturating_add(1).max(1);
        self.restore_manifest = Some(manifest);
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

    fn note_kernel_pointers_inner(&mut self, kernel_name: &str, ptrs: &[u64]) {
        let Some(kind) = region_kind_for_kernel(kernel_name) else {
            return;
        };
        for ptr in ptrs.iter().copied().filter(|ptr| *ptr != 0) {
            if !self.allocations.contains_key(&ptr) {
                continue;
            }
            let tracked_size = self
                .allocations
                .get(&ptr)
                .map(|region| region.tracked_size)
                .unwrap_or(0);
            let stable_key = self.stable_key_for_region(ptr, kind, tracked_size);
            let region_id = self
                .restore_manifest
                .as_ref()
                .and_then(|manifest| manifest.find_region_id(kind, &stable_key, tracked_size))
                .unwrap_or_else(|| {
                    self.allocations
                        .get(&ptr)
                        .map(|region| region.region_id)
                        .unwrap_or(0)
                });

            if let Some(region) = self.allocations.get_mut(&ptr) {
                region.kind = kind;
                region.stable_key = stable_key;
                region.region_id = region_id;
                region.last_kernel = Some(kernel_name.to_string());
            }
            self.upsert_manifest_for_ptr(ptr);
        }
        self.persist_manifest_best_effort();
    }

    fn stable_key_for_region(&self, ptr: u64, kind: KimiRegionKind, tracked_size: usize) -> String {
        if let Some(existing) = self
            .manifest
            .regions
            .iter()
            .find(|region| region.ptr == ptr && region.kind == kind)
        {
            return existing.stable_key.clone();
        }
        let ordinal = self
            .manifest
            .regions
            .iter()
            .filter(|region| region.ptr != ptr && region.kind == kind)
            .count()
            + 1;
        format!("{}:{ordinal:04}:{tracked_size}", kind.stable_prefix())
    }

    fn upsert_manifest_for_ptr(&mut self, ptr: u64) {
        let Some(region) = self.allocations.get(&ptr) else {
            return;
        };
        self.manifest.upsert(KimiRegionSpec {
            region_id: region.region_id,
            stable_key: region.stable_key.clone(),
            kind: region.kind,
            ptr: region.ptr,
            requested_size: region.requested_size,
            tracked_size: region.tracked_size,
            allocation_index: region.allocation_index,
            last_kernel: region.last_kernel.clone(),
            allocator: region.allocator.clone(),
        });
    }

    fn note_allocator_metadata_inner(
        &mut self,
        ptr: u64,
        block_table_ptr: u64,
        block_count: usize,
        dirty_bitmap_ptr: u64,
        dirty_bitmap_bytes: usize,
    ) -> Result<(), String> {
        if ptr == 0 {
            return Err("Kimi allocator metadata region pointer is null".to_string());
        }
        if dirty_bitmap_ptr == 0 || dirty_bitmap_bytes == 0 {
            return Err(format!(
                "allocator metadata for 0x{ptr:x} needs a non-empty dirty bitmap"
            ));
        }
        let region = self
            .allocations
            .get_mut(&ptr)
            .ok_or_else(|| format!("unknown Kimi Concordia allocation 0x{ptr:x}"))?;
        region.allocator = Some(KimiAllocatorMetadata {
            block_table_ptr,
            block_count,
            dirty_bitmap_ptr,
            dirty_bitmap_bytes,
            block_table_snapshot: Vec::new(),
        });
        self.upsert_manifest_for_ptr(ptr);
        self.persist_manifest_best_effort();
        Ok(())
    }

    fn persist_manifest_best_effort(&self) {
        if !self.config.enabled {
            return;
        }
        if let Err(err) = write_manifest(&self.config.manifest_path, &self.manifest) {
            self.log(&format!(
                "failed to write Kimi Concordia manifest {}: {err}",
                self.config.manifest_path.display()
            ));
        }
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

fn region_kind_for_kernel(kernel_name: &str) -> Option<KimiRegionKind> {
    let name = kernel_name.to_ascii_lowercase();
    if name.contains("lora") || name.contains("adapter") {
        return Some(KimiRegionKind::Adapter);
    }
    is_kimi_stateful_kernel_name(kernel_name).then_some(KimiRegionKind::KvCache)
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

pub(crate) fn dirty_page_offsets_from_bitmap_bytes(bitmap: &[u8], page_count: usize) -> Vec<usize> {
    let mut offsets = Vec::new();
    for page in 0..page_count {
        let byte = bitmap.get(page / 8).copied().unwrap_or(0);
        if byte & (1_u8 << (page % 8)) != 0 {
            offsets.push(page * PAGE_SIZE);
        }
    }
    offsets
}

fn decode_block_table_snapshot(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("chunk size checked")))
        .collect()
}

fn translate_kv_restore_offset(
    old_offset: usize,
    old_allocator: &KimiAllocatorMetadata,
    new_allocator: &KimiAllocatorMetadata,
) -> Result<usize, String> {
    if old_allocator.block_table_snapshot.is_empty()
        || new_allocator.block_table_snapshot.is_empty()
    {
        return Ok(old_offset);
    }

    let old_physical_page = old_offset / PAGE_SIZE;
    let intra_page = old_offset % PAGE_SIZE;
    let logical_page = old_allocator
        .block_table_snapshot
        .iter()
        .position(|page| *page as usize == old_physical_page)
        .ok_or_else(|| {
            format!("old KV physical page {old_physical_page} is absent from saved block table")
        })?;
    let new_physical_page = new_allocator
        .block_table_snapshot
        .get(logical_page)
        .copied()
        .ok_or_else(|| format!("new KV block table has no logical page {logical_page}"))?
        as usize;
    new_physical_page
        .checked_mul(PAGE_SIZE)
        .and_then(|base| base.checked_add(intra_page))
        .ok_or_else(|| {
            format!("translated KV restore offset overflow for physical page {new_physical_page}")
        })
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
    if manager.config.restore_enabled && manager.restore_manifest.is_none() {
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
    manager.manifest.remove_ptr(ptr);
    manager.persist_manifest_best_effort();

    #[cfg(all(
        feature = "nvidia",
        not(feature = "amd"),
        not(feature = "intel"),
        not(feature = "tenstorrent")
    ))]
    manager.release_device_resources(region);
}

pub(crate) fn note_allocator_metadata(
    ptr: u64,
    block_table_ptr: u64,
    block_count: usize,
    dirty_bitmap_ptr: u64,
    dirty_bitmap_bytes: usize,
) -> Result<(), String> {
    let mut manager = global_manager()
        .lock()
        .map_err(|_| "Kimi Concordia manager lock poisoned".to_string())?;
    manager.note_allocator_metadata_inner(
        ptr,
        block_table_ptr,
        block_count,
        dirty_bitmap_ptr,
        dirty_bitmap_bytes,
    )
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_kimi_concordia_register_allocator_metadata(
    ptr: u64,
    block_table_ptr: u64,
    block_count: usize,
    dirty_bitmap_ptr: u64,
    dirty_bitmap_bytes: usize,
) -> i32 {
    match note_allocator_metadata(
        ptr,
        block_table_ptr,
        block_count,
        dirty_bitmap_ptr,
        dirty_bitmap_bytes,
    ) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

fn global_manager() -> &'static Mutex<KimiConcordiaManager> {
    static MANAGER: OnceLock<Mutex<KimiConcordiaManager>> = OnceLock::new();
    MANAGER.get_or_init(|| Mutex::new(KimiConcordiaManager::new(KimiConcordiaConfig::from_env())))
}

fn tracked_len(size: usize) -> usize {
    size / PAGE_SIZE * PAGE_SIZE
}

fn default_manifest_path(aof_path: &std::path::Path) -> PathBuf {
    let mut path = aof_path.to_path_buf();
    path.set_extension("manifest.json");
    path
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

fn read_manifest(path: &std::path::Path) -> Result<KimiRegionManifest, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("read Kimi Concordia manifest {}: {err}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|err| format!("parse Kimi Concordia manifest {}: {err}", path.display()))
}

fn write_manifest(path: &std::path::Path, manifest: &KimiRegionManifest) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "create Kimi Concordia manifest dir {}: {err}",
                    parent.display()
                )
            })?;
        }
    }
    let text = serde_json::to_string_pretty(manifest)
        .map_err(|err| format!("serialize Kimi Concordia manifest: {err}"))?;
    std::fs::write(path, text)
        .map_err(|err| format!("write Kimi Concordia manifest {}: {err}", path.display()))
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
        fn allocator_with_current_snapshot(
            &mut self,
            ptr: u64,
            metadata: &KimiAllocatorMetadata,
        ) -> Result<KimiAllocatorMetadata, String> {
            if metadata.block_table_ptr == 0 || metadata.block_count == 0 {
                return Ok(metadata.clone());
            }
            let table_bytes = metadata
                .block_count
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| {
                    format!(
                        "KV block table byte length overflow for {} entries",
                        metadata.block_count
                    )
                })?;
            let snapshot = decode_block_table_snapshot(&raw_device_to_host(
                metadata.block_table_ptr,
                table_bytes,
            )?);
            let mut updated = metadata.clone();
            updated.block_table_snapshot = snapshot;
            if let Some(region) = self.allocations.get_mut(&ptr) {
                if let Some(region_allocator) = region.allocator.as_mut() {
                    region_allocator.block_table_snapshot = updated.block_table_snapshot.clone();
                }
            }
            self.upsert_manifest_for_ptr(ptr);
            self.persist_manifest_best_effort();
            Ok(updated)
        }

        pub(crate) fn prepare_kernel_launch(&mut self, kernel_name: &str, kernel_ptrs: &[u64]) {
            self.note_kernel_pointers_inner(kernel_name, kernel_ptrs);
            if !self.config.restore_enabled {
                return;
            }

            let ptrs = kernel_ptrs
                .iter()
                .copied()
                .filter(|ptr| {
                    self.allocations
                        .get(ptr)
                        .map(|region| {
                            region.kind != KimiRegionKind::OpaqueMutable && !region.restored
                        })
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>();
            for ptr in ptrs {
                if let Err(err) = self.restore_device_region(ptr) {
                    self.log(&format!(
                        "prelaunch restore skipped for {kernel_name} ptr=0x{ptr:x}: {err}"
                    ));
                }
            }
        }

        pub(crate) fn observe_kernel_launch(
            &mut self,
            kernel_name: &str,
            stream: CUstream,
            kernel_ptrs: &[u64],
        ) {
            self.note_kernel_pointers_inner(kernel_name, kernel_ptrs);
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
            let (aof_staging_host, aof_staging_device) = match raw_host_mapped_alloc(PAGE_SIZE) {
                Ok(staging) => staging,
                Err(err) => {
                    let _ = raw_device_free(shadow);
                    let _ = raw_device_free(bitmap);
                    return Err(err);
                }
            };

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
                aof_staging_host,
                aof_staging_device,
                aof_staging_bytes: PAGE_SIZE,
            });
            self.log(&format!(
                "base snapshot region={region_id} ptr=0x{ptr:x} pages={page_count}"
            ));
            Ok(())
        }

        fn scan_device_region(&mut self, ptr: u64) -> Result<(), String> {
            let (region_id, tracked_size, shadow, bitmap, bitmap_words, allocator) = {
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
                    region.allocator.clone(),
                )
            };
            let page_count = tracked_size / PAGE_SIZE;
            if page_count == 0 {
                return Ok(());
            }

            let (offsets, refresh_shadow_from_payload) = if let Some(metadata) = allocator {
                let metadata = self.allocator_with_current_snapshot(ptr, &metadata)?;
                let bitmap_data =
                    raw_device_to_host(metadata.dirty_bitmap_ptr, metadata.dirty_bitmap_bytes)?;
                (
                    dirty_page_offsets_from_bitmap_bytes(&bitmap_data, page_count),
                    true,
                )
            } else {
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
                (dirty_page_offsets_from_words(&words, page_count), false)
            };
            if offsets.is_empty() {
                return Ok(());
            }

            let epoch = self.next_epoch;
            self.next_epoch = self.next_epoch.saturating_add(1);
            let handle = self.ensure_worker()?;
            for offset in offsets {
                let payload =
                    self.stage_aof_payload_via_worker(handle, ptr, ptr + offset as u64, PAGE_SIZE)?;
                if refresh_shadow_from_payload {
                    raw_host_to_device(shadow + offset as u64, &payload)?;
                }
                self.append_record(epoch, region_id, offset, payload)?;
            }
            Ok(())
        }

        fn stage_aof_payload_via_worker(
            &self,
            handle: i64,
            region_ptr: u64,
            source: u64,
            len: usize,
        ) -> Result<Vec<u8>, String> {
            let (region_id, staging_host, staging_device, staging_bytes) = {
                let region = self
                    .allocations
                    .get(&region_ptr)
                    .ok_or_else(|| format!("unknown Kimi Concordia allocation 0x{region_ptr:x}"))?;
                let resources = region.device.as_ref().ok_or_else(|| {
                    format!("region {} has no checkpoint resources", region.region_id)
                })?;
                (
                    region.region_id,
                    resources.aof_staging_host,
                    resources.aof_staging_device,
                    resources.aof_staging_bytes,
                )
            };
            if len > staging_bytes {
                return Err(format!(
                    "AOF staging buffer for region {region_id} is {staging_bytes} bytes, need {len}"
                ));
            }
            let seq = unsafe {
                crate::r#impl::concordia_gpu::concordia_gpu_stage_aof_bytes(
                    handle,
                    source,
                    staging_device,
                    len as i64,
                )
            };
            if seq < 0 {
                return Err(format!(
                    "enqueue AOF staging copy failed for region {region_id}"
                ));
            }
            unsafe {
                crate::r#impl::concordia_gpu::concordia_gpu_sync(handle);
            }
            raw_host_mapped_to_vec(staging_host, len)
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
            let (region_id, tracked_size, old_allocator, current_allocator) = {
                let region = self
                    .allocations
                    .get(&ptr)
                    .ok_or_else(|| format!("unknown Kimi Concordia allocation 0x{ptr:x}"))?;
                let old_allocator = self.restore_manifest.as_ref().and_then(|manifest| {
                    manifest
                        .regions
                        .iter()
                        .find(|saved| saved.region_id == region.region_id)
                        .and_then(|saved| saved.allocator.clone())
                });
                (
                    region.region_id,
                    region.tracked_size,
                    old_allocator,
                    region.allocator.clone(),
                )
            };
            let current_allocator = match current_allocator.as_ref() {
                Some(metadata) => Some(self.allocator_with_current_snapshot(ptr, metadata)?),
                None => None,
            };
            let path = mpi_scoped_aof_path(self.config.restore_aof_path.clone());
            let records = AofDiskLog::read_committed(&path)
                .map_err(|err| format!("read Concordia restore AOF {}: {err}", path.display()))?;
            let region_records = records
                .iter()
                .filter(|record| record.region_id == region_id)
                .collect::<Vec<_>>();
            let max_payload = region_records
                .iter()
                .map(|record| record.payload.len())
                .max()
                .unwrap_or(0);
            let staging = if max_payload > 0 {
                Some(raw_host_mapped_alloc(max_payload)?)
            } else {
                None
            };
            let mut applied = 0usize;
            let result = (|| -> Result<(), String> {
                for record in region_records {
                    let target_offset = match (&old_allocator, &current_allocator) {
                        (Some(old), Some(new)) => {
                            translate_kv_restore_offset(record.offset, old, new)?
                        }
                        _ => record.offset,
                    };
                    let end = target_offset
                        .checked_add(record.payload.len())
                        .ok_or_else(|| {
                            format!(
                                "restore offset overflow for region {region_id} at {}",
                                target_offset
                            )
                        })?;
                    if end > tracked_size {
                        return Err(format!(
                            "restore record for region {region_id} writes {}..{} beyond {}",
                            target_offset, end, tracked_size
                        ));
                    }
                    let (staging_host, staging_device) = staging.ok_or_else(|| {
                        format!("restore region {region_id} has no staging buffer")
                    })?;
                    raw_host_mapped_copy_from_slice(staging_host, &record.payload)?;
                    let handle = self.ensure_worker()?;
                    let seq = unsafe {
                        crate::r#impl::concordia_gpu::concordia_gpu_restore_bytes(
                            handle,
                            staging_device,
                            ptr + target_offset as u64,
                            record.payload.len() as i64,
                        )
                    };
                    if seq < 0 {
                        return Err(format!(
                            "enqueue restore copy failed for region {region_id}"
                        ));
                    }
                    unsafe {
                        crate::r#impl::concordia_gpu::concordia_gpu_sync(handle);
                    }
                    applied += 1;
                }
                Ok(())
            })();
            if let Some((staging_host, _)) = staging {
                self.shutdown_worker();
                let _ = raw_host_mapped_free(staging_host);
            }
            result?;
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
                let _ = raw_host_mapped_free(resources.aof_staging_host);
                let _ = raw_device_free(resources.shadow);
                let _ = raw_device_free(resources.bitmap);
                self.log(&format!(
                    "released checkpoint resources region={} ptr=0x{:x}",
                    region.region_id, region.ptr
                ));
            }
        }
    }

    pub(crate) fn prepare_kernel_launch(kernel_name: &str, kernel_ptrs: &[u64]) {
        let Ok(mut manager) = super::global_manager().lock() else {
            return;
        };
        manager.prepare_kernel_launch(kernel_name, kernel_ptrs);
    }

    pub(crate) fn observe_kernel_launch(kernel_name: &str, stream: CUstream, kernel_ptrs: &[u64]) {
        let Ok(mut manager) = super::global_manager().lock() else {
            return;
        };
        manager.observe_kernel_launch(kernel_name, stream, kernel_ptrs);
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

    fn raw_host_mapped_alloc(bytes: usize) -> Result<(u64, u64), String> {
        let bytes = bytes.max(1);
        let mut host = ptr::null_mut();
        let rc = nvidia_runtime_sys::cuMemAllocHost_v2(&mut host, bytes);
        if rc != 0 || host.is_null() {
            return Err(format!("cuMemAllocHost_v2({bytes}) failed: {rc}"));
        }
        let mut device = CUdeviceptr_v2(ptr::null_mut());
        let rc = nvidia_runtime_sys::cuMemHostGetDevicePointer_v2(&mut device, host, 0);
        if rc != 0 {
            unsafe {
                nvidia_runtime_sys::cuMemFreeHost(host);
            }
            return Err(format!(
                "cuMemHostGetDevicePointer_v2({bytes}) failed: {rc}"
            ));
        }
        Ok((host as u64, device.0 as u64))
    }

    fn raw_host_mapped_free(host: u64) -> Result<(), String> {
        if host == 0 {
            return Ok(());
        }
        let rc = unsafe { nvidia_runtime_sys::cuMemFreeHost(host as *mut c_void) };
        if rc != 0 {
            return Err(format!("cuMemFreeHost(0x{host:x}) failed: {rc}"));
        }
        Ok(())
    }

    fn raw_host_mapped_to_vec(host: u64, len: usize) -> Result<Vec<u8>, String> {
        if host == 0 && len != 0 {
            return Err("host-mapped staging pointer is null".to_string());
        }
        let mut out = vec![0u8; len];
        if len != 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(host as *const u8, out.as_mut_ptr(), len);
            }
        }
        Ok(out)
    }

    fn raw_host_mapped_copy_from_slice(host: u64, data: &[u8]) -> Result<(), String> {
        if host == 0 && !data.is_empty() {
            return Err("host-mapped staging pointer is null".to_string());
        }
        if !data.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), host as *mut u8, data.len());
            }
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
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) use nvidia_device::prepare_kernel_launch;

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        previous: Vec<(&'static str, Option<String>)>,
    }

    impl EnvGuard {
        fn set(vars: &[(&'static str, Option<&str>)]) -> Self {
            let lock = crate::r#impl::test_env::lock();
            let previous = vars
                .iter()
                .map(|(name, _)| (*name, std::env::var(name).ok()))
                .collect::<Vec<_>>();
            for (name, value) in vars {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
            Self {
                _lock: lock,
                previous,
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in self.previous.drain(..) {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

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
    fn manifest_path_is_mpi_rank_scoped_with_default_from_aof_path() {
        let _guard = EnvGuard::set(&[
            ("OMPI_COMM_WORLD_RANK", Some("2")),
            ("OMPI_COMM_WORLD_SIZE", Some("4")),
            ("OMPI_COMM_WORLD_LOCAL_RANK", Some("1")),
        ]);
        let config = KimiConcordiaConfig::from_pairs(&[
            ("HETGPU_KIMI_CONCORDIA", "1"),
            ("CONCORDIA_AOF_PATH", "/tmp/kimi.aof"),
        ]);

        assert_eq!(
            config.manifest_path,
            PathBuf::from("/tmp/kimi.manifest.rank0002-of-0004.json")
        );
    }

    #[test]
    fn explicit_manifest_path_expands_mpi_rank_template() {
        let _guard = EnvGuard::set(&[
            ("OMPI_COMM_WORLD_RANK", Some("3")),
            ("OMPI_COMM_WORLD_SIZE", Some("8")),
            ("OMPI_COMM_WORLD_LOCAL_RANK", Some("1")),
        ]);
        let config = KimiConcordiaConfig::from_pairs(&[
            ("HETGPU_KIMI_CONCORDIA", "1"),
            ("CONCORDIA_AOF_PATH", "/tmp/kimi.aof"),
            (
                "HETGPU_KIMI_CONCORDIA_MANIFEST",
                "/tmp/kimi-r{rank}-w{world}-l{local_rank}.manifest.json",
            ),
        ]);

        assert_eq!(
            config.manifest_path,
            PathBuf::from("/tmp/kimi-r3-w8-l1.manifest.json")
        );
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

    #[test]
    fn bitmap_bytes_expand_to_dirty_page_offsets() {
        let bitmap = [0b0000_1010_u8, 0b0000_0001_u8];
        assert_eq!(
            dirty_page_offsets_from_bitmap_bytes(&bitmap, 10),
            vec![4096, 3 * 4096, 8 * 4096]
        );
    }

    #[test]
    fn manifest_marks_kernel_pointer_regions_as_kv_cache() {
        let mut manager = KimiConcordiaManager::new_for_test(enabled_config());

        assert_eq!(manager.note_allocation_for_test(0x2000, 8192), Some(1));
        assert_eq!(manager.note_allocation_for_test(0x8000, 4096), Some(2));
        manager.note_kernel_pointers_for_test("paged_kv_cache_update", &[0x2000, 0xdead]);

        let manifest = manager.manifest_snapshot_for_test();
        let kv = manifest
            .regions
            .iter()
            .find(|region| region.ptr == 0x2000)
            .expect("tracked KV region should be present");
        assert_eq!(kv.kind, KimiRegionKind::KvCache);
        assert_eq!(kv.stable_key, "kv-cache:0001:8192");
        assert_eq!(kv.last_kernel.as_deref(), Some("paged_kv_cache_update"));

        let untouched = manifest
            .regions
            .iter()
            .find(|region| region.ptr == 0x8000)
            .expect("untouched tracked region should be present");
        assert_eq!(untouched.kind, KimiRegionKind::OpaqueMutable);
    }

    #[test]
    fn manifest_persists_allocator_metadata_for_kv_region() {
        let mut manager = KimiConcordiaManager::new_for_test(enabled_config());

        assert_eq!(manager.note_allocation_for_test(0x2000, 8192), Some(1));
        manager.note_kernel_pointers_for_test("paged_kv_cache_update", &[0x2000]);
        manager
            .note_allocator_metadata_for_test(0x2000, 0xabc0, 128, 0xdef0, 16)
            .unwrap();

        let manifest = manager.manifest_snapshot_for_test();
        let kv = manifest
            .regions
            .iter()
            .find(|region| region.ptr == 0x2000)
            .expect("tracked KV region should be present");
        let allocator = kv
            .allocator
            .as_ref()
            .expect("allocator metadata should be persisted");
        assert_eq!(allocator.block_table_ptr, 0xabc0);
        assert_eq!(allocator.block_count, 128);
        assert_eq!(allocator.dirty_bitmap_ptr, 0xdef0);
        assert_eq!(allocator.dirty_bitmap_bytes, 16);
    }

    #[test]
    fn kv_restore_offset_uses_logical_block_table_mapping() {
        let old_allocator = KimiAllocatorMetadata {
            block_table_ptr: 0xabc0,
            block_count: 4,
            dirty_bitmap_ptr: 0xdef0,
            dirty_bitmap_bytes: 1,
            block_table_snapshot: vec![4, 7, 9, 11],
        };
        let new_allocator = KimiAllocatorMetadata {
            block_table_ptr: 0xabc0,
            block_count: 4,
            dirty_bitmap_ptr: 0xdef0,
            dirty_bitmap_bytes: 1,
            block_table_snapshot: vec![5, 2, 3, 8],
        };

        assert_eq!(
            translate_kv_restore_offset(
                7 * DEFAULT_KIMI_PAGE_SIZE + 128,
                &old_allocator,
                &new_allocator
            )
            .unwrap(),
            2 * DEFAULT_KIMI_PAGE_SIZE + 128
        );
    }

    #[test]
    fn restore_manifest_maps_kv_region_without_allocation_order() {
        let mut writer = KimiConcordiaManager::new_for_test(enabled_config());
        assert_eq!(writer.note_allocation_for_test(0x2000, 8192), Some(1));
        writer.note_kernel_pointers_for_test("paged_kv_cache_update", &[0x2000]);
        let saved = writer.manifest_snapshot_for_test();

        let mut restore = KimiConcordiaManager::new_for_test(enabled_config());
        restore.set_restore_manifest_for_test(saved);
        assert_eq!(restore.note_allocation_for_test(0x9000, 4096), Some(2));
        assert_eq!(restore.note_allocation_for_test(0xa000, 8192), Some(3));
        restore.note_kernel_pointers_for_test("paged_kv_cache_update", &[0xa000]);

        let manifest = restore.manifest_snapshot_for_test();
        let kv = manifest
            .regions
            .iter()
            .find(|region| region.ptr == 0xa000)
            .expect("restored KV region should be present");
        assert_eq!(kv.kind, KimiRegionKind::KvCache);
        assert_eq!(kv.region_id, 1);
        assert_eq!(kv.stable_key, "kv-cache:0001:8192");
    }
}
