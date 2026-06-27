const DEFAULT_PAGE_SIZE: usize = 4096;

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
    shadow: Vec<u8>,
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
            shadow: initial.to_vec(),
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
        if region.shadow.len() != current.len() {
            return Err(format!(
                "region {region_id} size changed from {} to {} bytes",
                region.shadow.len(),
                current.len()
            ));
        }

        let epoch = self.next_epoch;
        self.next_epoch += 1;
        let mut dirty_pages = Vec::new();
        for offset in (0..current.len()).step_by(self.page_size) {
            let end = (offset + self.page_size).min(current.len());
            if current[offset..end] != region.shadow[offset..end] {
                dirty_pages.push(DirtyPage {
                    region_id,
                    offset,
                    data: current[offset..end].to_vec(),
                });
                region.shadow[offset..end].copy_from_slice(&current[offset..end]);
            }
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
}
