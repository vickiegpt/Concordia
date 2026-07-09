fn main() {
    concordia_oob::run_demo().expect("demo should complete");
}

mod concordia_oob {
    use std::fs::{self, File, OpenOptions};
    use std::io::{self, ErrorKind, Read, Write};
    use std::path::{Path, PathBuf};
    use std::sync::{mpsc, Arc, Mutex};
    use std::thread::{self, JoinHandle};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub const PAGE_SIZE: usize = 4096;

    const MAGIC: &[u8; 4] = b"CAOF";
    const VERSION: u32 = 1;
    const COMMIT: &[u8; 7] = b"COMMIT\n";

    pub type RegionHandle = Arc<Mutex<Region>>;

    #[derive(Debug)]
    pub struct Region {
        id: u64,
        bytes: Vec<u8>,
        shadow: Vec<u8>,
        epoch: u64,
    }

    impl Region {
        pub fn write(&mut self, offset: usize, data: &[u8]) -> io::Result<()> {
            let end = offset
                .checked_add(data.len())
                .ok_or_else(|| invalid_data("region write offset overflow"))?;
            if end > self.bytes.len() {
                return Err(invalid_data("region write is out of bounds"));
            }
            self.bytes[offset..end].copy_from_slice(data);
            Ok(())
        }

        pub fn bytes(&self) -> &[u8] {
            &self.bytes
        }

        fn scan_delta(&mut self) -> AofRecord {
            let mut pages = Vec::new();
            for page_index in 0..self.page_count() {
                let start = page_index * PAGE_SIZE;
                let end = self.bytes.len().min(start + PAGE_SIZE);
                if self.bytes[start..end] != self.shadow[start..end] {
                    let mut payload = vec![0; PAGE_SIZE];
                    payload[..end - start].copy_from_slice(&self.bytes[start..end]);
                    self.shadow[start..end].copy_from_slice(&self.bytes[start..end]);
                    pages.push(DirtyPage {
                        index: page_index as u32,
                        bytes: payload,
                    });
                }
            }
            self.epoch += 1;
            AofRecord {
                epoch: self.epoch,
                region_id: self.id,
                pages,
            }
        }

        fn page_count(&self) -> usize {
            self.bytes.len().div_ceil(PAGE_SIZE)
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct DirtyPage {
        index: u32,
        bytes: Vec<u8>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct AofRecord {
        epoch: u64,
        region_id: u64,
        pages: Vec<DirtyPage>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CheckpointSummary {
        pub epoch: u64,
        pub dirty_pages: usize,
        pub dirty_bytes: usize,
        pub aof_bytes: u64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct RestoreSummary {
        pub applied_records: usize,
        pub applied_pages: usize,
    }

    enum Task {
        Checkpoint {
            region: RegionHandle,
            ack: mpsc::Sender<io::Result<CheckpointSummary>>,
        },
        Restore {
            region: RegionHandle,
            ack: mpsc::Sender<io::Result<RestoreSummary>>,
        },
        Shutdown {
            ack: mpsc::Sender<io::Result<()>>,
        },
    }

    pub struct ConcordiaRuntime {
        tx: mpsc::Sender<Task>,
        worker: Mutex<Option<JoinHandle<()>>>,
    }

    impl ConcordiaRuntime {
        pub fn start(log_path: impl Into<PathBuf>) -> io::Result<Self> {
            let log_path = log_path.into();
            if let Some(parent) = log_path.parent() {
                fs::create_dir_all(parent)?;
            }

            let (tx, rx) = mpsc::channel();
            let worker = thread::Builder::new()
                .name("concordia-oob-persistent-worker".to_string())
                .spawn(move || persistent_worker(rx, log_path))
                .map_err(|err| io::Error::new(ErrorKind::Other, err))?;

            Ok(Self {
                tx,
                worker: Mutex::new(Some(worker)),
            })
        }

        pub fn register_region(&self, id: u64, bytes: Vec<u8>) -> io::Result<RegionHandle> {
            if bytes.is_empty() {
                return Err(invalid_data("checkpoint region must not be empty"));
            }
            Ok(Arc::new(Mutex::new(Region {
                id,
                shadow: bytes.clone(),
                bytes,
                epoch: 0,
            })))
        }

        pub fn checkpoint(&self, region: &RegionHandle) -> io::Result<CheckpointSummary> {
            let (ack, done) = mpsc::channel();
            self.tx
                .send(Task::Checkpoint {
                    region: Arc::clone(region),
                    ack,
                })
                .map_err(|_| broken_pipe("persistent worker has exited"))?;
            done.recv()
                .map_err(|_| broken_pipe("persistent worker closed checkpoint ack"))?
        }

        pub fn restore(&self, region: &RegionHandle) -> io::Result<RestoreSummary> {
            let (ack, done) = mpsc::channel();
            self.tx
                .send(Task::Restore {
                    region: Arc::clone(region),
                    ack,
                })
                .map_err(|_| broken_pipe("persistent worker has exited"))?;
            done.recv()
                .map_err(|_| broken_pipe("persistent worker closed restore ack"))?
        }

        pub fn shutdown(&self) -> io::Result<()> {
            let mut worker = self.worker.lock().unwrap();
            if worker.is_none() {
                return Ok(());
            }

            let (ack, done) = mpsc::channel();
            self.tx
                .send(Task::Shutdown { ack })
                .map_err(|_| broken_pipe("persistent worker has exited"))?;
            done.recv()
                .map_err(|_| broken_pipe("persistent worker closed shutdown ack"))??;

            if let Some(worker) = worker.take() {
                worker
                    .join()
                    .map_err(|_| io::Error::new(ErrorKind::Other, "persistent worker panicked"))?;
            }
            Ok(())
        }
    }

    impl Drop for ConcordiaRuntime {
        fn drop(&mut self) {
            let _ = self.shutdown();
        }
    }

    pub fn run_demo() -> io::Result<()> {
        let out_dir = PathBuf::from("target");
        fs::create_dir_all(&out_dir)?;
        let log_path = out_dir.join("concordia_oob_demo.aof");
        match fs::remove_file(&log_path) {
            Ok(()) => {}
            Err(err) if err.kind() == ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }

        let runtime = ConcordiaRuntime::start(&log_path)?;
        let region = runtime.register_region(2606, vec![0; PAGE_SIZE * 4])?;

        {
            let mut region = region.lock().unwrap();
            region.write(PAGE_SIZE + 64, b"kv-page-one")?;
            region.write(PAGE_SIZE * 3 + 128, b"adapter-page-three")?;
        }

        let checkpoint = runtime.checkpoint(&region)?;
        runtime.shutdown()?;

        let replay = ConcordiaRuntime::start(&log_path)?;
        let restored = replay.register_region(2606, vec![0; PAGE_SIZE * 4])?;
        let restore = replay.restore(&restored)?;
        replay.shutdown()?;

        let restored = restored.lock().unwrap();
        let ok = &restored.bytes()[PAGE_SIZE + 64..PAGE_SIZE + 75] == b"kv-page-one"
            && &restored.bytes()[PAGE_SIZE * 3 + 128..PAGE_SIZE * 3 + 146]
                == b"adapter-page-three";

        println!("concordia_oob={}", if ok { "pass" } else { "fail" });
        println!(
            "checkpoint epoch={} dirty_pages={} dirty_bytes={} aof_bytes={}",
            checkpoint.epoch, checkpoint.dirty_pages, checkpoint.dirty_bytes, checkpoint.aof_bytes
        );
        println!(
            "restore applied_records={} applied_pages={}",
            restore.applied_records, restore.applied_pages
        );
        println!("aof_path={}", log_path.display());

        if ok {
            Ok(())
        } else {
            Err(invalid_data("restored bytes did not match checkpointed delta"))
        }
    }

    fn persistent_worker(rx: mpsc::Receiver<Task>, log_path: PathBuf) {
        while let Ok(task) = rx.recv() {
            match task {
                Task::Checkpoint { region, ack } => {
                    let result = checkpoint_region(&region, &log_path);
                    let _ = ack.send(result);
                }
                Task::Restore { region, ack } => {
                    let result = restore_region(&region, &log_path);
                    let _ = ack.send(result);
                }
                Task::Shutdown { ack } => {
                    let _ = ack.send(Ok(()));
                    break;
                }
            }
        }
    }

    fn checkpoint_region(region: &RegionHandle, log_path: &Path) -> io::Result<CheckpointSummary> {
        let mut region = region.lock().unwrap();
        let record = region.scan_delta();
        let dirty_pages = record.pages.len();
        let dirty_bytes = dirty_pages * PAGE_SIZE;
        let epoch = record.epoch;
        let aof_bytes = if dirty_pages == 0 {
            0
        } else {
            append_record(log_path, &record)?
        };

        Ok(CheckpointSummary {
            epoch,
            dirty_pages,
            dirty_bytes,
            aof_bytes,
        })
    }

    fn restore_region(region: &RegionHandle, log_path: &Path) -> io::Result<RestoreSummary> {
        let mut region = region.lock().unwrap();
        let records = read_committed_records(log_path)?;
        let mut applied_records = 0;
        let mut applied_pages = 0;

        for record in records {
            if record.region_id != region.id {
                continue;
            }
            applied_records += 1;
            for page in record.pages {
                let start = page.index as usize * PAGE_SIZE;
                let end = (start + PAGE_SIZE).min(region.bytes.len());
                if start >= region.bytes.len() {
                    return Err(invalid_data("AOF page index is out of region bounds"));
                }
                let payload = &page.bytes[..end - start];
                region.bytes[start..end].copy_from_slice(payload);
                region.shadow[start..end].copy_from_slice(payload);
                applied_pages += 1;
            }
            region.epoch = region.epoch.max(record.epoch);
        }

        Ok(RestoreSummary {
            applied_records,
            applied_pages,
        })
    }

    fn append_record(log_path: &Path, record: &AofRecord) -> io::Result<u64> {
        let before = fs::metadata(log_path).map(|m| m.len()).unwrap_or(0);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        file.write_all(MAGIC)?;
        write_u32(&mut file, VERSION)?;
        write_u64(&mut file, record.epoch)?;
        write_u64(&mut file, record.region_id)?;
        write_u32(&mut file, PAGE_SIZE as u32)?;
        write_u32(&mut file, record.pages.len() as u32)?;

        let mut checksum = 0u64;
        for page in &record.pages {
            write_u32(&mut file, page.index)?;
            write_u32(&mut file, page.bytes.len() as u32)?;
            file.write_all(&page.bytes)?;
            checksum = checksum.wrapping_add(page_checksum(page));
        }
        write_u64(&mut file, checksum)?;
        file.write_all(COMMIT)?;
        file.flush()?;

        let after = fs::metadata(log_path)?.len();
        Ok(after.saturating_sub(before))
    }

    fn read_committed_records(log_path: &Path) -> io::Result<Vec<AofRecord>> {
        let mut file = match File::open(log_path) {
            Ok(file) => file,
            Err(err) if err.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(err),
        };

        let mut records = Vec::new();
        loop {
            let mut magic = [0; 4];
            match file.read_exact(&mut magic) {
                Ok(()) => {}
                Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
                Err(err) => return Err(err),
            }
            if &magic != MAGIC {
                break;
            }

            let version = match read_u32(&mut file) {
                Ok(version) => version,
                Err(err) if err.kind() == ErrorKind::UnexpectedEof => break,
                Err(err) => return Err(err),
            };
            if version != VERSION {
                break;
            }

            let epoch = read_u64_or_break(&mut file)?;
            let Some(epoch) = epoch else { break };
            let region_id = read_u64_or_break(&mut file)?;
            let Some(region_id) = region_id else { break };
            let page_size = read_u32_or_break(&mut file)?;
            let Some(page_size) = page_size else { break };
            let page_count = read_u32_or_break(&mut file)?;
            let Some(page_count) = page_count else { break };
            if page_size != PAGE_SIZE as u32 || page_count > 1_000_000 {
                break;
            }

            let mut pages = Vec::with_capacity(page_count as usize);
            let mut checksum = 0u64;
            let mut complete = true;
            for _ in 0..page_count {
                let page_index = match read_u32_or_break(&mut file)? {
                    Some(page_index) => page_index,
                    None => {
                        complete = false;
                        break;
                    }
                };
                let byte_len = match read_u32_or_break(&mut file)? {
                    Some(byte_len) => byte_len,
                    None => {
                        complete = false;
                        break;
                    }
                };
                if byte_len as usize != PAGE_SIZE {
                    complete = false;
                    break;
                }
                let mut bytes = vec![0; byte_len as usize];
                if file.read_exact(&mut bytes).is_err() {
                    complete = false;
                    break;
                }
                let page = DirtyPage {
                    index: page_index,
                    bytes,
                };
                checksum = checksum.wrapping_add(page_checksum(&page));
                pages.push(page);
            }
            if !complete {
                break;
            }

            let expected = match read_u64_or_break(&mut file)? {
                Some(expected) => expected,
                None => break,
            };
            let mut commit = [0; COMMIT.len()];
            if file.read_exact(&mut commit).is_err() || commit != *COMMIT || checksum != expected {
                break;
            }

            records.push(AofRecord {
                epoch,
                region_id,
                pages,
            });
        }

        Ok(records)
    }

    fn page_checksum(page: &DirtyPage) -> u64 {
        page.bytes.iter().fold(page.index as u64, |acc, byte| {
            acc.wrapping_mul(16_777_619).wrapping_add(*byte as u64)
        })
    }

    fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
        writer.write_all(&value.to_le_bytes())
    }

    fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
        writer.write_all(&value.to_le_bytes())
    }

    fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
        let mut bytes = [0; 8];
        reader.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_u32_or_break(reader: &mut impl Read) -> io::Result<Option<u32>> {
        match read_u32(reader) {
            Ok(value) => Ok(Some(value)),
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn read_u64_or_break(reader: &mut impl Read) -> io::Result<Option<u64>> {
        match read_u64(reader) {
            Ok(value) => Ok(Some(value)),
            Err(err) if err.kind() == ErrorKind::UnexpectedEof => Ok(None),
            Err(err) => Err(err),
        }
    }

    pub fn append_bytes(path: &Path, bytes: &[u8]) -> io::Result<()> {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?
            .write_all(bytes)
    }

    pub struct TempLog {
        path: PathBuf,
    }

    impl Drop for TempLog {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    pub fn temp_log(name: &str) -> (TempLog, PathBuf) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "concordia-oob-{name}-{}-{nanos}.aof",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        (TempLog { path: path.clone() }, path)
    }

    fn invalid_data(message: &'static str) -> io::Error {
        io::Error::new(ErrorKind::InvalidData, message)
    }

    fn broken_pipe(message: &'static str) -> io::Error {
        io::Error::new(ErrorKind::BrokenPipe, message)
    }
}

#[cfg(test)]
mod tests {
    use super::concordia_oob::*;

    #[test]
    fn checkpoint_detects_dirty_pages() {
        let (_tmp, log_path) = temp_log("dirty-pages");
        let runtime = ConcordiaRuntime::start(log_path).unwrap();
        let region = runtime.register_region(7, vec![0; PAGE_SIZE * 4]).unwrap();

        {
            let mut region = region.lock().unwrap();
            region.write(1 * PAGE_SIZE + 17, &[1, 2, 3, 4]).unwrap();
            region.write(3 * PAGE_SIZE, &[9]).unwrap();
        }

        let summary = runtime.checkpoint(&region).unwrap();
        runtime.shutdown().unwrap();

        assert_eq!(summary.dirty_pages, 2);
        assert_eq!(summary.dirty_bytes, PAGE_SIZE * 2);
        assert!(summary.aof_bytes > 0);
    }

    #[test]
    fn restore_applies_committed_records() {
        let (_tmp, log_path) = temp_log("restore");
        let runtime = ConcordiaRuntime::start(log_path.clone()).unwrap();
        let region = runtime.register_region(9, vec![0; PAGE_SIZE * 2]).unwrap();

        {
            let mut region = region.lock().unwrap();
            region.write(PAGE_SIZE + 3, &[42, 43]).unwrap();
        }
        runtime.checkpoint(&region).unwrap();
        runtime.shutdown().unwrap();

        let replay = ConcordiaRuntime::start(log_path).unwrap();
        let target = replay.register_region(9, vec![0; PAGE_SIZE * 2]).unwrap();
        let restore = replay.restore(&target).unwrap();
        replay.shutdown().unwrap();

        let target = target.lock().unwrap();
        assert_eq!(restore.applied_records, 1);
        assert_eq!(restore.applied_pages, 1);
        assert_eq!(&target.bytes()[PAGE_SIZE + 3..PAGE_SIZE + 5], &[42, 43]);
    }

    #[test]
    fn recovery_ignores_incomplete_suffix() {
        let (_tmp, log_path) = temp_log("suffix");
        {
            let runtime = ConcordiaRuntime::start(log_path.clone()).unwrap();
            let region = runtime.register_region(11, vec![0; PAGE_SIZE]).unwrap();
            region.lock().unwrap().write(0, &[88]).unwrap();
            runtime.checkpoint(&region).unwrap();
            runtime.shutdown().unwrap();
        }
        append_bytes(&log_path, b"CAOF this record has no commit marker").unwrap();

        let runtime = ConcordiaRuntime::start(log_path).unwrap();
        let target = runtime.register_region(11, vec![0; PAGE_SIZE]).unwrap();
        let restore = runtime.restore(&target).unwrap();
        runtime.shutdown().unwrap();

        assert_eq!(restore.applied_records, 1);
        assert_eq!(target.lock().unwrap().bytes()[0], 88);
    }

    #[test]
    fn worker_roundtrip_reports_noop_checkpoint() {
        let (_tmp, log_path) = temp_log("noop");
        let runtime = ConcordiaRuntime::start(log_path).unwrap();
        let region = runtime.register_region(3, vec![5; PAGE_SIZE]).unwrap();
        let summary = runtime.checkpoint(&region).unwrap();
        runtime.shutdown().unwrap();

        assert_eq!(summary.dirty_pages, 0);
        assert_eq!(summary.dirty_bytes, 0);
    }
}
