const DEFAULT_PAGE_SIZE: usize = 4096;

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DirtyPage {
    pub region_id: u64,
    pub offset: usize,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeltaRecord {
    pub epoch: u64,
    pub region_id: u64,
    pub base_addr: u64,
    pub dirty_pages: Vec<DirtyPage>,
}

#[derive(Debug, Clone)]
struct TrackedRegion {
    id: u64,
    base_addr: u64,
    len: usize,
    shadow: Option<Vec<u8>>,
    kind: RegionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegionKind {
    OpaqueShadow,
    AllocatorBitmap,
}

#[derive(Debug, Clone)]
pub(crate) struct DeltaCheckpointState {
    page_size: usize,
    next_region_id: u64,
    next_epoch: u64,
    regions: Vec<TrackedRegion>,
}

impl Default for DeltaCheckpointState {
    fn default() -> Self {
        Self::new(DEFAULT_PAGE_SIZE)
    }
}

impl DeltaCheckpointState {
    pub(crate) fn new(page_size: usize) -> Self {
        let page_size = page_size.max(1);
        Self {
            page_size,
            next_region_id: 1,
            next_epoch: 1,
            regions: Vec::new(),
        }
    }

    pub(crate) fn register_opaque_host_region(&mut self, base_addr: u64, initial: &[u8]) -> u64 {
        let id = self.next_region_id;
        self.next_region_id += 1;
        self.regions.push(TrackedRegion {
            id,
            base_addr,
            len: initial.len(),
            shadow: Some(initial.to_vec()),
            kind: RegionKind::OpaqueShadow,
        });
        id
    }

    pub(crate) fn register_allocator_bitmap_region(&mut self, base_addr: u64, len: usize) -> u64 {
        let id = self.next_region_id;
        self.next_region_id += 1;
        self.regions.push(TrackedRegion {
            id,
            base_addr,
            len,
            shadow: None,
            kind: RegionKind::AllocatorBitmap,
        });
        id
    }

    pub(crate) fn create_host_delta(
        &mut self,
        region_id: u64,
        current: &[u8],
    ) -> Result<DeltaRecord, String> {
        let region = self
            .regions
            .iter_mut()
            .find(|region| region.id == region_id)
            .ok_or_else(|| format!("unknown Concordia checkpoint region {region_id}"))?;
        if region.kind != RegionKind::OpaqueShadow {
            return Err(format!(
                "region {region_id} is not an opaque shadow checkpoint region"
            ));
        }
        if region.len != current.len() {
            return Err(format!(
                "region {region_id} size changed from {} to {} bytes",
                region.len,
                current.len()
            ));
        }
        let shadow = region
            .shadow
            .as_mut()
            .ok_or_else(|| format!("region {region_id} has no shadow buffer"))?;

        let epoch = self.next_epoch;
        self.next_epoch += 1;
        let mut dirty_pages = Vec::new();
        for offset in (0..current.len()).step_by(self.page_size) {
            let end = (offset + self.page_size).min(current.len());
            if current[offset..end] != shadow[offset..end] {
                dirty_pages.push(DirtyPage {
                    region_id,
                    offset,
                    data: current[offset..end].to_vec(),
                });
                shadow[offset..end].copy_from_slice(&current[offset..end]);
            }
        }

        Ok(DeltaRecord {
            epoch,
            region_id,
            base_addr: region.base_addr,
            dirty_pages,
        })
    }

    pub(crate) fn create_bitmap_delta(
        &mut self,
        region_id: u64,
        current: &[u8],
        dirty_bitmap: &[u8],
    ) -> Result<DeltaRecord, String> {
        let region = self
            .regions
            .iter()
            .find(|region| region.id == region_id)
            .ok_or_else(|| format!("unknown Concordia checkpoint region {region_id}"))?;
        if region.kind != RegionKind::AllocatorBitmap {
            return Err(format!(
                "region {region_id} is not an allocator-bitmap checkpoint region"
            ));
        }
        if region.len != current.len() {
            return Err(format!(
                "region {region_id} size changed from {} to {} bytes",
                region.len,
                current.len()
            ));
        }

        let epoch = self.next_epoch;
        self.next_epoch += 1;
        let page_count = current.len().div_ceil(self.page_size);
        let mut dirty_pages = Vec::new();
        for page_index in 0..page_count {
            let byte = dirty_bitmap.get(page_index / 8).copied().unwrap_or(0);
            if byte & (1 << (page_index % 8)) == 0 {
                continue;
            }
            let offset = page_index * self.page_size;
            let end = (offset + self.page_size).min(current.len());
            dirty_pages.push(DirtyPage {
                region_id,
                offset,
                data: current[offset..end].to_vec(),
            });
        }

        Ok(DeltaRecord {
            epoch,
            region_id,
            base_addr: region.base_addr,
            dirty_pages,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AofRecord {
    pub epoch: u64,
    pub region_id: u64,
    pub offset: usize,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AofEntry {
    record: AofRecord,
    committed: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct AofLog {
    entries: Vec<AofEntry>,
}

impl AofLog {
    pub(crate) fn append_committed(
        &mut self,
        epoch: u64,
        region_id: u64,
        offset: usize,
        payload: &[u8],
    ) {
        self.entries.push(AofEntry {
            record: AofRecord {
                epoch,
                region_id,
                offset,
                payload: payload.to_vec(),
            },
            committed: true,
        });
    }

    pub(crate) fn append_delta(&mut self, delta: &DeltaRecord) {
        for page in &delta.dirty_pages {
            self.append_committed(delta.epoch, page.region_id, page.offset, &page.data);
        }
    }

    pub(crate) fn replay_committed(&self) -> Vec<AofRecord> {
        self.entries
            .iter()
            .take_while(|entry| entry.committed)
            .map(|entry| entry.record.clone())
            .collect()
    }

    #[cfg(test)]
    fn append_uncommitted_for_test(
        &mut self,
        epoch: u64,
        region_id: u64,
        offset: usize,
        payload: &[u8],
    ) {
        self.entries.push(AofEntry {
            record: AofRecord {
                epoch,
                region_id,
                offset,
                payload: payload.to_vec(),
            },
            committed: false,
        });
    }
}

const AOF_MAGIC: &[u8; 4] = b"CONC";
const AOF_COMMIT: &[u8; 4] = b"CMIT";
const AOF_HEADER_LEN: usize = 36;

pub(crate) struct AofDiskLog {
    path: PathBuf,
    file: File,
}

impl AofDiskLog {
    pub(crate) fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&path)?;
        Ok(Self { path, file })
    }

    pub(crate) fn open_append(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        Ok(Self { path, file })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn append_delta(&mut self, delta: &DeltaRecord) -> io::Result<()> {
        for page in &delta.dirty_pages {
            self.append_record(&AofRecord {
                epoch: delta.epoch,
                region_id: page.region_id,
                offset: page.offset,
                payload: page.data.clone(),
            })?;
        }
        Ok(())
    }

    pub(crate) fn append_record(&mut self, record: &AofRecord) -> io::Result<()> {
        let offset = u64::try_from(record.offset).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "AOF offset does not fit in u64",
            )
        })?;
        let payload_len = u64::try_from(record.payload.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "AOF payload length does not fit in u64",
            )
        })?;

        self.file.write_all(AOF_MAGIC)?;
        self.file.write_all(&record.epoch.to_le_bytes())?;
        self.file.write_all(&record.region_id.to_le_bytes())?;
        self.file.write_all(&offset.to_le_bytes())?;
        self.file.write_all(&payload_len.to_le_bytes())?;
        self.file.write_all(&record.payload)?;
        self.file.write_all(AOF_COMMIT)?;
        self.file.flush()
    }

    pub(crate) fn read_committed(path: impl AsRef<Path>) -> io::Result<Vec<AofRecord>> {
        let mut file = File::open(path)?;
        let mut records = Vec::new();

        loop {
            let mut header = [0u8; AOF_HEADER_LEN];
            match file.read_exact(&mut header) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(err) => return Err(err),
            }
            if &header[0..4] != AOF_MAGIC {
                break;
            }

            let epoch = read_le_u64(&header[4..12]);
            let region_id = read_le_u64(&header[12..20]);
            let offset_u64 = read_le_u64(&header[20..28]);
            let payload_len_u64 = read_le_u64(&header[28..36]);
            let offset = match usize::try_from(offset_u64) {
                Ok(offset) => offset,
                Err(_) => break,
            };
            let payload_len = match usize::try_from(payload_len_u64) {
                Ok(payload_len) => payload_len,
                Err(_) => break,
            };
            let mut payload = vec![0u8; payload_len];
            if file.read_exact(&mut payload).is_err() {
                break;
            }
            let mut commit = [0u8; 4];
            if file.read_exact(&mut commit).is_err() || &commit != AOF_COMMIT {
                break;
            }
            records.push(AofRecord {
                epoch,
                region_id,
                offset,
                payload,
            });
        }

        Ok(records)
    }
}

fn read_le_u64(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(bytes);
    u64::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_region_delta_records_only_changed_pages() {
        let mut backing = vec![0u8; 8192];
        let mut state = DeltaCheckpointState::new(4096);

        let region_id = state.register_opaque_host_region(0x1000, &backing);
        let first = state.create_host_delta(region_id, &backing).unwrap();
        assert_eq!(first.dirty_pages.len(), 0);

        backing[4096..4100].copy_from_slice(&[1, 2, 3, 4]);
        let second = state.create_host_delta(region_id, &backing).unwrap();

        assert_eq!(second.dirty_pages.len(), 1);
        assert_eq!(second.dirty_pages[0].offset, 4096);
        assert_eq!(second.dirty_pages[0].data[..4], [1, 2, 3, 4]);
    }

    #[test]
    fn aof_replay_ignores_uncommitted_suffix() {
        let mut log = AofLog::default();
        log.append_committed(7, 0x1000, 0, &[1, 2, 3]);
        log.append_uncommitted_for_test(8, 0x1000, 4096, &[9, 9, 9]);

        let records = log.replay_committed();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].epoch, 7);
        assert_eq!(records[0].payload, vec![1, 2, 3]);
    }

    #[test]
    fn allocator_dirty_bitmap_records_only_marked_pages() {
        let mut backing = vec![0u8; 3 * 4096];
        backing[0..4].copy_from_slice(&[1, 2, 3, 4]);
        backing[8192..8196].copy_from_slice(&[5, 6, 7, 8]);
        let mut state = DeltaCheckpointState::new(4096);
        let region_id = state.register_allocator_bitmap_region(0x8000, backing.len());

        let delta = state
            .create_bitmap_delta(region_id, &backing, &[0b101])
            .unwrap();

        assert_eq!(delta.dirty_pages.len(), 2);
        assert_eq!(delta.dirty_pages[0].offset, 0);
        assert_eq!(delta.dirty_pages[0].data[..4], [1, 2, 3, 4]);
        assert_eq!(delta.dirty_pages[1].offset, 8192);
        assert_eq!(delta.dirty_pages[1].data[..4], [5, 6, 7, 8]);
    }

    #[test]
    fn disk_aof_replay_ignores_truncated_suffix() {
        let path =
            std::env::temp_dir().join(format!("hetgpu_concordia_aof_{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut log = AofDiskLog::create(&path).unwrap();
        log.append_record(&AofRecord {
            epoch: 1,
            region_id: 7,
            offset: 4096,
            payload: vec![1, 2, 3, 4],
        })
        .unwrap();
        drop(log);

        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&[0x43, 0x4f, 0x4e, 0x43, 0, 0, 0]);
        std::fs::write(&path, bytes).unwrap();

        let records = AofDiskLog::read_committed(&path).unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].epoch, 1);
        assert_eq!(records[0].region_id, 7);
        assert_eq!(records[0].offset, 4096);
        assert_eq!(records[0].payload, vec![1, 2, 3, 4]);
        let _ = std::fs::remove_file(&path);
    }
}
