use crate::r#impl::concordia_delta::{apply_records_to_region, AofDiskLog, DeltaCheckpointState};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CheckpointSummary {
    pub epoch: u64,
    pub region_id: u64,
    pub dirty_pages: usize,
    pub dirty_bytes: usize,
    pub boundary: String,
}

pub(crate) struct ConcordiaRuntime {
    state: DeltaCheckpointState,
    aof: Option<AofDiskLog>,
    boundary_count: u64,
    mpi_info: MpiInfo,
}

impl ConcordiaRuntime {
    pub(crate) fn new(page_size: usize) -> Self {
        Self {
            state: DeltaCheckpointState::new(page_size),
            aof: None,
            boundary_count: 0,
            mpi_info: mpi_info_from_env(),
        }
    }

    pub(crate) fn set_aof_path(&mut self, path: PathBuf) {
        match AofDiskLog::create(&path) {
            Ok(log) => self.aof = Some(log),
            Err(err) => {
                eprintln!(
                    "[hetGPU Concordia] failed to create AOF log {}: {err}",
                    path.display()
                );
                self.aof = None;
            }
        }
    }

    pub(crate) fn register_opaque_host_region(&mut self, base_addr: u64, initial: &[u8]) -> u64 {
        self.state.register_opaque_host_region(base_addr, initial)
    }

    pub(crate) fn register_allocator_bitmap_region(&mut self, base_addr: u64, len: usize) -> u64 {
        self.state.register_allocator_bitmap_region(base_addr, len)
    }

    pub(crate) fn checkpoint_host_region(
        &mut self,
        region_id: u64,
        current: &[u8],
        boundary: &str,
    ) -> Result<CheckpointSummary, String> {
        let delta = self.state.create_host_delta(region_id, current)?;
        if let Some(aof) = self.aof.as_mut() {
            aof.append_delta(&delta)
                .map_err(|err| format!("append Concordia AOF {}: {err}", aof.path().display()))?;
        }
        let dirty_pages = delta.dirty_pages.len();
        let dirty_bytes = delta.dirty_pages.iter().map(|page| page.data.len()).sum();
        Ok(CheckpointSummary {
            epoch: delta.epoch,
            region_id: delta.region_id,
            dirty_pages,
            dirty_bytes,
            boundary: mpi_scoped_boundary(boundary, self.mpi_info),
        })
    }

    pub(crate) fn checkpoint_allocator_bitmap_region(
        &mut self,
        region_id: u64,
        current: &[u8],
        dirty_bitmap: &[u8],
        boundary: &str,
    ) -> Result<CheckpointSummary, String> {
        let delta = self
            .state
            .create_bitmap_delta(region_id, current, dirty_bitmap)?;
        if let Some(aof) = self.aof.as_mut() {
            aof.append_delta(&delta)
                .map_err(|err| format!("append Concordia AOF {}: {err}", aof.path().display()))?;
        }
        let dirty_pages = delta.dirty_pages.len();
        let dirty_bytes = delta.dirty_pages.iter().map(|page| page.data.len()).sum();
        Ok(CheckpointSummary {
            epoch: delta.epoch,
            region_id: delta.region_id,
            dirty_pages,
            dirty_bytes,
            boundary: mpi_scoped_boundary(boundary, self.mpi_info),
        })
    }

    pub(crate) fn checkpoint_boundary(&mut self, boundary: &str) {
        self.boundary_count = self.boundary_count.saturating_add(1);
        if runtime_logs_enabled() {
            eprintln!(
                "[hetGPU Concordia] checkpoint boundary #{}: {}",
                self.boundary_count,
                mpi_scoped_boundary(boundary, self.mpi_info)
            );
        }
    }
}

static CONCORDIA_RUNTIME: OnceLock<Mutex<ConcordiaRuntime>> = OnceLock::new();

fn global_runtime() -> &'static Mutex<ConcordiaRuntime> {
    CONCORDIA_RUNTIME.get_or_init(|| {
        let mut runtime = ConcordiaRuntime::new(4096);
        if let Ok(path) = std::env::var("CONCORDIA_AOF_PATH")
            .or_else(|_| std::env::var("HETGPU_CONCORDIA_AOF_PATH"))
        {
            runtime.aof = AofDiskLog::open_append(mpi_scoped_aof_path(PathBuf::from(path))).ok();
        }
        Mutex::new(runtime)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MpiInfo {
    pub rank: u32,
    pub world_size: u32,
    pub local_rank: u32,
}

impl MpiInfo {
    pub(crate) fn is_parallel(self) -> bool {
        self.world_size > 1
    }
}

const MPI_RANK_KEYS: &[&str] = &[
    "CONCORDIA_MPI_RANK",
    "HETGPU_CONCORDIA_MPI_RANK",
    "OMPI_COMM_WORLD_RANK",
    "PMIX_RANK",
    "PMI_RANK",
    "MV2_COMM_WORLD_RANK",
    "SLURM_PROCID",
    "RANK",
];

const MPI_WORLD_KEYS: &[&str] = &[
    "CONCORDIA_MPI_WORLD_SIZE",
    "HETGPU_CONCORDIA_MPI_WORLD_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "PMIX_SIZE",
    "PMI_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
    "WORLD_SIZE",
];

const MPI_LOCAL_RANK_KEYS: &[&str] = &[
    "CONCORDIA_MPI_LOCAL_RANK",
    "HETGPU_CONCORDIA_MPI_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "SLURM_LOCALID",
    "PMI_LOCAL_RANK",
    "LOCAL_RANK",
];

pub(crate) fn mpi_info_from_env() -> MpiInfo {
    let pairs: Vec<(String, String)> = MPI_RANK_KEYS
        .iter()
        .chain(MPI_WORLD_KEYS.iter())
        .chain(MPI_LOCAL_RANK_KEYS.iter())
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
    mpi_info_from_pairs(&refs)
}

pub(crate) fn mpi_info_from_pairs(pairs: &[(&str, &str)]) -> MpiInfo {
    let rank = read_u32_from_pairs(pairs, MPI_RANK_KEYS, 0);
    let world_size = read_u32_from_pairs(pairs, MPI_WORLD_KEYS, 1).max(1);
    let local_rank = read_u32_from_pairs(pairs, MPI_LOCAL_RANK_KEYS, rank);
    MpiInfo {
        rank,
        world_size,
        local_rank,
    }
}

fn read_u32_from_pairs(pairs: &[(&str, &str)], keys: &[&str], fallback: u32) -> u32 {
    keys.iter()
        .find_map(|key| {
            pairs
                .iter()
                .find(|(candidate, _)| candidate == key)
                .and_then(|(_, value)| value.trim().parse::<i64>().ok())
                .and_then(|value| u32::try_from(value).ok())
        })
        .unwrap_or(fallback)
}

pub(crate) fn mpi_scoped_aof_path(path: PathBuf) -> PathBuf {
    mpi_scoped_aof_path_for_info(path, mpi_info_from_env())
}

pub(crate) fn mpi_scoped_aof_path_for_info(path: PathBuf, info: MpiInfo) -> PathBuf {
    if !info.is_parallel() {
        return path;
    }

    let text = path.to_string_lossy();
    if text.contains("{rank}") || text.contains("{world}") || text.contains("{local_rank}") {
        return PathBuf::from(
            text.replace("{rank}", &info.rank.to_string())
                .replace("{world}", &info.world_size.to_string())
                .replace("{local_rank}", &info.local_rank.to_string()),
        );
    }

    let ranked_suffix = format!("rank{:04}-of-{:04}", info.rank, info.world_size);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("concordia.aof");
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(file_name);
    let ranked_name = match path.extension().and_then(|extension| extension.to_str()) {
        Some(extension) if !extension.is_empty() => format!("{stem}.{ranked_suffix}.{extension}"),
        _ => format!("{file_name}.{ranked_suffix}"),
    };

    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(ranked_name),
        _ => PathBuf::from(ranked_name),
    }
}

fn mpi_scoped_boundary(boundary: &str, info: MpiInfo) -> String {
    if info.is_parallel() {
        format!(
            "rank={}/{} local_rank={} {}",
            info.rank, info.world_size, info.local_rank, boundary
        )
    } else {
        boundary.to_string()
    }
}

fn runtime_enabled() -> bool {
    env_enabled("CONCORDIA_CHECKPOINT_ON_BOUNDARY") || env_enabled("HETGPU_CONCORDIA_BOUNDARY")
}

fn runtime_logs_enabled() -> bool {
    env_enabled("CONCORDIA_LOGS") || env_enabled("HETGPU_CONCORDIA_LOGS")
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub(crate) fn restore_region_from_aof_path(
    path: impl AsRef<std::path::Path>,
    region_id: u64,
    target: &mut [u8],
) -> Result<usize, String> {
    let path = path.as_ref();
    let records = AofDiskLog::read_committed(path)
        .map_err(|err| format!("read Concordia AOF {}: {err}", path.display()))?;
    apply_records_to_region(region_id, target, &records)
}

unsafe fn slice_from_raw<'a>(data: *const u8, len: usize) -> Option<&'a [u8]> {
    if data.is_null() && len != 0 {
        return None;
    }
    Some(std::slice::from_raw_parts(data, len))
}

unsafe fn slice_from_raw_mut<'a>(data: *mut u8, len: usize) -> Option<&'a mut [u8]> {
    if data.is_null() && len != 0 {
        return None;
    }
    Some(std::slice::from_raw_parts_mut(data, len))
}

unsafe fn cstr_from_raw<'a>(ptr: *const c_char, fallback: &'a str) -> String {
    if ptr.is_null() {
        return fallback.to_string();
    }
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_concordia_register_host_region(
    base_addr: u64,
    data: *const u8,
    len: usize,
) -> u64 {
    let Some(initial) = slice_from_raw(data, len) else {
        return 0;
    };
    match global_runtime().lock() {
        Ok(mut runtime) => runtime.register_opaque_host_region(base_addr, initial),
        Err(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_concordia_register_bitmap_region(
    base_addr: u64,
    len: usize,
) -> u64 {
    match global_runtime().lock() {
        Ok(mut runtime) => runtime.register_allocator_bitmap_region(base_addr, len),
        Err(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_concordia_checkpoint_host_region(
    region_id: u64,
    data: *const u8,
    len: usize,
    boundary: *const c_char,
) -> i32 {
    let Some(current) = slice_from_raw(data, len) else {
        return -1;
    };
    let boundary = cstr_from_raw(boundary, "ffi");
    match global_runtime().lock() {
        Ok(mut runtime) => runtime
            .checkpoint_host_region(region_id, current, &boundary)
            .map(|_| 0)
            .unwrap_or(-2),
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_concordia_checkpoint_bitmap_region(
    region_id: u64,
    data: *const u8,
    len: usize,
    dirty_bitmap: *const u8,
    dirty_bitmap_len: usize,
    boundary: *const c_char,
) -> i32 {
    let Some(current) = slice_from_raw(data, len) else {
        return -1;
    };
    let Some(bitmap) = slice_from_raw(dirty_bitmap, dirty_bitmap_len) else {
        return -1;
    };
    let boundary = cstr_from_raw(boundary, "ffi-bitmap");
    match global_runtime().lock() {
        Ok(mut runtime) => runtime
            .checkpoint_allocator_bitmap_region(region_id, current, bitmap, &boundary)
            .map(|_| 0)
            .unwrap_or(-2),
        Err(_) => -3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_concordia_checkpoint_boundary(boundary: *const c_char) -> i32 {
    if !runtime_enabled() {
        return 0;
    }
    let boundary = cstr_from_raw(boundary, "boundary");
    match global_runtime().lock() {
        Ok(mut runtime) => {
            runtime.checkpoint_boundary(&boundary);
            0
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_concordia_restore_region_from_aof(
    path: *const c_char,
    region_id: u64,
    data: *mut u8,
    len: usize,
) -> i64 {
    if path.is_null() {
        return -1;
    }
    let Some(target) = slice_from_raw_mut(data, len) else {
        return -1;
    };
    let path = CStr::from_ptr(path).to_string_lossy().into_owned();
    match restore_region_from_aof_path(PathBuf::from(path), region_id, target) {
        Ok(applied) => i64::try_from(applied).unwrap_or(-3),
        Err(err) => {
            if runtime_logs_enabled() {
                eprintln!("[hetGPU Concordia] AOF restore failed: {err}");
            }
            -2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#impl::concordia_delta::AofDiskLog;

    #[test]
    fn runtime_checkpoints_registered_host_region_to_aof() {
        let path = std::env::temp_dir().join(format!(
            "hetgpu_concordia_runtime_{}.aof",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let mut backing = vec![0u8; 8192];
        let mut runtime = ConcordiaRuntime::new(4096);
        runtime.set_aof_path(path.clone());
        let region_id = runtime.register_opaque_host_region(0x4000, &backing);

        backing[4096..4100].copy_from_slice(&[9, 8, 7, 6]);
        let summary = runtime
            .checkpoint_host_region(region_id, &backing, "unit-test")
            .unwrap();

        assert_eq!(summary.dirty_pages, 1);
        assert_eq!(summary.dirty_bytes, 4096);
        assert_eq!(summary.boundary, "unit-test");

        let records = AofDiskLog::read_committed(&path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].region_id, region_id);
        assert_eq!(records[0].offset, 4096);
        assert_eq!(records[0].payload[..4], [9, 8, 7, 6]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mpi_info_detects_common_rank_environment() {
        let info = mpi_info_from_pairs(&[
            ("OMPI_COMM_WORLD_RANK", "2"),
            ("OMPI_COMM_WORLD_SIZE", "8"),
            ("OMPI_COMM_WORLD_LOCAL_RANK", "1"),
        ]);

        assert_eq!(info.rank, 2);
        assert_eq!(info.world_size, 8);
        assert_eq!(info.local_rank, 1);
        assert!(info.is_parallel());
    }

    #[test]
    fn mpi_aof_path_is_rank_scoped_without_template_tokens() {
        let info = MpiInfo {
            rank: 3,
            world_size: 16,
            local_rank: 1,
        };
        let path = mpi_scoped_aof_path_for_info(PathBuf::from("/tmp/concordia/session.aof"), info);

        assert_eq!(
            path,
            PathBuf::from("/tmp/concordia/session.rank0003-of-0016.aof")
        );
    }

    #[test]
    fn mpi_aof_path_expands_explicit_template_tokens() {
        let info = MpiInfo {
            rank: 5,
            world_size: 32,
            local_rank: 2,
        };
        let path = mpi_scoped_aof_path_for_info(
            PathBuf::from("/tmp/concordia/r{rank}-w{world}-l{local_rank}.aof"),
            info,
        );

        assert_eq!(path, PathBuf::from("/tmp/concordia/r5-w32-l2.aof"));
    }

    #[test]
    fn runtime_restores_region_from_committed_aof() {
        let path = std::env::temp_dir().join(format!(
            "hetgpu_concordia_restore_{}.aof",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let mut log = AofDiskLog::create(&path).unwrap();
        log.append_record(&crate::r#impl::concordia_delta::AofRecord {
            epoch: 1,
            region_id: 11,
            offset: 4096,
            payload: vec![4, 3, 2, 1],
        })
        .unwrap();
        drop(log);

        let mut replacement = vec![0u8; 8192];
        let applied = restore_region_from_aof_path(&path, 11, &mut replacement).unwrap();

        assert_eq!(applied, 1);
        assert_eq!(replacement[4096..4100], [4, 3, 2, 1]);
        let _ = std::fs::remove_file(&path);
    }
}
