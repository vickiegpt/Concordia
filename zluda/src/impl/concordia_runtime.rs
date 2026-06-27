use crate::r#impl::concordia_delta::{AofDiskLog, DeltaCheckpointState};
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
}

impl ConcordiaRuntime {
    pub(crate) fn new(page_size: usize) -> Self {
        Self {
            state: DeltaCheckpointState::new(page_size),
            aof: None,
            boundary_count: 0,
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
            boundary: boundary.to_string(),
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
            boundary: boundary.to_string(),
        })
    }

    pub(crate) fn checkpoint_boundary(&mut self, boundary: &str) {
        self.boundary_count = self.boundary_count.saturating_add(1);
        if runtime_logs_enabled() {
            eprintln!(
                "[hetGPU Concordia] checkpoint boundary #{}: {boundary}",
                self.boundary_count
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
            runtime.aof = AofDiskLog::open_append(PathBuf::from(path)).ok();
        }
        Mutex::new(runtime)
    })
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

unsafe fn slice_from_raw<'a>(data: *const u8, len: usize) -> Option<&'a [u8]> {
    if data.is_null() && len != 0 {
        return None;
    }
    Some(std::slice::from_raw_parts(data, len))
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
}
