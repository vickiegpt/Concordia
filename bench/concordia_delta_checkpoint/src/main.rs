use cuda_core::{memory, sys, CudaContext, DeviceBuffer, IntoResult, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use std::cmp;
use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

const PAGE_SIZE: usize = 4096;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn diff_pages_kernel(current: &[u8], shadow: &[u8], mut dirty_flags: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let page = idx.get();

        if let Some(flag) = dirty_flags.get_mut(idx) {
            let mut dirty = 0u32;
            let mut offset = 0usize;
            while offset < PAGE_SIZE {
                let byte = page * PAGE_SIZE + offset;
                if current[byte] != shadow[byte] {
                    dirty = 1;
                }
                offset += 1;
            }
            *flag = dirty;
        }
    }

    #[kernel]
    pub fn touch_kernel(mut sink: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(value) = sink.get_mut(idx) {
            *value += 1;
        }
    }
}

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone)]
struct Config {
    self_test: bool,
    launch_only: bool,
    warmup: usize,
    iters: usize,
    device_override: Option<usize>,
}

#[derive(Debug, Clone)]
struct CaseResult {
    region_mb: f64,
    dirty_pages: usize,
    cpu_dtoh_ms: f64,
    cpu_diff_ms: f64,
    gpu_diff_ms: f64,
    gpu_append_ms: f64,
    observed_dirty: usize,
}

impl CaseResult {
    fn cpu_total_ms(&self) -> f64 {
        self.cpu_dtoh_ms + self.cpu_diff_ms
    }

    fn gpu_total_ms(&self) -> f64 {
        self.gpu_diff_ms + self.gpu_append_ms
    }

    fn speedup(&self) -> f64 {
        self.cpu_total_ms() / self.gpu_total_ms().max(f64::EPSILON)
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> BenchResult<()> {
    let config = parse_args()?;
    let device_count = cuda_device_count()?;
    if device_count == 0 {
        return Err("no CUDA devices visible".into());
    }

    let local_rank = local_rank();
    let ordinal = config
        .device_override
        .unwrap_or_else(|| local_rank.unwrap_or(0) % device_count);

    let ctx = CudaContext::new(ordinal)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let (cc_major, cc_minor) = ctx.compute_capability()?;
    eprintln!(
        "# device={} ordinal={} device_count={} cc={}.{} rank={}",
        ctx.device_name()?,
        ctx.ordinal(),
        device_count,
        cc_major,
        cc_minor,
        local_rank
            .map(|rank| rank.to_string())
            .unwrap_or_else(|| "none".to_string())
    );

    if config.self_test {
        run_self_test(&module, &stream)?;
        return Ok(());
    }

    if config.launch_only {
        print_launch_overhead(&module, &stream, config.iters)?;
        return Ok(());
    }

    println!("# table5_sparse_delta_checkpoint");
    print_result_header();
    for mb in [16usize, 50, 128, 256] {
        let result = run_case(
            &module,
            &stream,
            mb * 1024 * 1024,
            1,
            config.warmup,
            config.iters,
        )?;
        print_result(&result);
    }

    println!("# table6_dirty_scaling_256mb");
    print_result_header();
    for dirty_pages in [1usize, 4, 10, 32] {
        let result = run_case(
            &module,
            &stream,
            256 * 1024 * 1024,
            dirty_pages,
            config.warmup,
            config.iters,
        )?;
        print_result(&result);
    }

    print_launch_overhead(&module, &stream, config.iters)?;
    Ok(())
}

fn run_self_test(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
) -> BenchResult<()> {
    let result = run_case(module, stream, 4 * 1024 * 1024, 2, 1, 2)?;
    if result.observed_dirty != 2 {
        return Err(format!(
            "self-test expected 2 dirty pages, observed {}",
            result.observed_dirty
        )
        .into());
    }
    println!(
        "self-test passed: dirty_pages={} cpu_copy_ms={:.6} gpu_diff_ms={:.6}",
        result.observed_dirty, result.cpu_dtoh_ms, result.gpu_diff_ms
    );
    Ok(())
}

fn run_case(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    region_bytes: usize,
    dirty_pages: usize,
    warmup: usize,
    iters: usize,
) -> BenchResult<CaseResult> {
    if region_bytes == 0 || region_bytes % PAGE_SIZE != 0 {
        return Err(format!("region size must be a non-zero multiple of {PAGE_SIZE}").into());
    }
    let page_count = region_bytes / PAGE_SIZE;
    if dirty_pages == 0 || dirty_pages > page_count {
        return Err(format!("dirty_pages must be in 1..={page_count}, got {dirty_pages}").into());
    }
    if page_count > u32::MAX as usize {
        return Err("page_count exceeds cuda-oxide LaunchConfig u32 size".into());
    }

    let selected_pages = selected_dirty_pages(page_count, dirty_pages);
    let host_shadow = vec![0u8; region_bytes];
    let mut host_current = host_shadow.clone();
    apply_payloads(&mut host_current, &selected_pages);

    let current_dev = DeviceBuffer::from_host(stream, &host_current)?;
    let shadow_dev = DeviceBuffer::from_host(stream, &host_shadow)?;
    let mut dirty_flags = DeviceBuffer::<u32>::zeroed(stream, page_count)?;
    let mut cpu_current = vec![0u8; region_bytes];

    let total = warmup + iters;
    let mut cpu_dtoh = Vec::with_capacity(iters);
    let mut cpu_diff = Vec::with_capacity(iters);
    let mut gpu_diff = Vec::with_capacity(iters);
    let mut gpu_append = Vec::with_capacity(iters);
    let mut observed_dirty = 0usize;

    for iter in 0..total {
        let dtoh_start = Instant::now();
        current_dev.copy_to_host(stream, &mut cpu_current)?;
        let dtoh_ms = ms(dtoh_start.elapsed());

        let diff_start = Instant::now();
        let cpu_dirty = cpu_page_diff(&cpu_current, &host_shadow, PAGE_SIZE);
        let diff_ms = ms(diff_start.elapsed());

        dirty_flags.zero_async(stream)?;
        stream.synchronize()?;

        let gpu_diff_start = Instant::now();
        module.diff_pages_kernel(
            stream,
            LaunchConfig::for_num_elems(page_count as u32),
            &current_dev,
            &shadow_dev,
            &mut dirty_flags,
        )?;
        stream.synchronize()?;
        let gpu_diff_ms = ms(gpu_diff_start.elapsed());

        let append_start = Instant::now();
        let gpu_dirty = append_dirty_payloads(&current_dev, &dirty_flags, stream)?;
        let gpu_append_ms = ms(append_start.elapsed());

        if iter >= warmup {
            cpu_dtoh.push(dtoh_ms);
            cpu_diff.push(diff_ms);
            gpu_diff.push(gpu_diff_ms);
            gpu_append.push(gpu_append_ms);
            observed_dirty = gpu_dirty.len();
        }

        if cpu_dirty.len() != dirty_pages || gpu_dirty.len() != dirty_pages {
            return Err(format!(
                "dirty-page mismatch: expected {dirty_pages}, cpu={}, gpu={}",
                cpu_dirty.len(),
                gpu_dirty.len()
            )
            .into());
        }
    }

    Ok(CaseResult {
        region_mb: region_bytes as f64 / (1024.0 * 1024.0),
        dirty_pages,
        cpu_dtoh_ms: median(&mut cpu_dtoh),
        cpu_diff_ms: median(&mut cpu_diff),
        gpu_diff_ms: median(&mut gpu_diff),
        gpu_append_ms: median(&mut gpu_append),
        observed_dirty,
    })
}

fn append_dirty_payloads(
    current: &DeviceBuffer<u8>,
    dirty_flags: &DeviceBuffer<u32>,
    stream: &cuda_core::CudaStream,
) -> BenchResult<Vec<usize>> {
    let flags = dirty_flags.to_host_vec(stream)?;
    let dirty_pages: Vec<usize> = flags
        .iter()
        .enumerate()
        .filter_map(|(page, &flag)| (flag != 0).then_some(page))
        .collect();
    let mut append_log = vec![0u8; dirty_pages.len() * PAGE_SIZE];

    for (slot, page) in dirty_pages.iter().copied().enumerate() {
        let payload = &mut append_log[slot * PAGE_SIZE..(slot + 1) * PAGE_SIZE];
        let offset = (page * PAGE_SIZE) as sys::CUdeviceptr;
        let src = current.cu_deviceptr() + offset;
        unsafe {
            memory::memcpy_dtoh_async(payload.as_mut_ptr(), src, PAGE_SIZE, stream.cu_stream())?;
        }
    }
    stream.synchronize()?;
    Ok(dirty_pages)
}

fn apply_payloads(current: &mut [u8], selected_pages: &[usize]) {
    for (slot, &page) in selected_pages.iter().enumerate() {
        let start = page * PAGE_SIZE;
        let end = start + PAGE_SIZE;
        for (offset, byte) in current[start..end].iter_mut().enumerate() {
            *byte = ((slot + offset + 1) % 251) as u8;
        }
    }
}

fn cpu_page_diff(current: &[u8], shadow: &[u8], page_size: usize) -> Vec<usize> {
    let pages = current.len() / page_size;
    let mut dirty = Vec::new();
    let mut append_log = Vec::new();
    for page in 0..pages {
        let start = page * page_size;
        let end = start + page_size;
        if current[start..end] != shadow[start..end] {
            dirty.push(page);
            append_log.extend_from_slice(&current[start..end]);
        }
    }
    dirty
}

fn print_launch_overhead(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    iters: usize,
) -> BenchResult<()> {
    let iters = cmp::max(1, iters);
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut sink = DeviceBuffer::<u32>::zeroed(stream, 1)?;

    let mut sync_samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        module.touch_kernel(stream, config, &mut sink)?;
        stream.synchronize()?;
        sync_samples.push(us(start.elapsed()));
    }

    let batch = 64usize;
    let mut batch_samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        for _ in 0..batch {
            module.touch_kernel(stream, config, &mut sink)?;
        }
        stream.synchronize()?;
        batch_samples.push(us(start.elapsed()) / batch as f64);
    }

    println!("# launch_overhead_us");
    println!("mode,p50_us");
    println!("sync,{:.3}", median(&mut sync_samples));
    println!("batch_per_launch,{:.3}", median(&mut batch_samples));
    Ok(())
}

fn print_result_header() {
    println!(
        "region_mb,requested_dirty_pages,observed_dirty_pages,cpu_dtoh_ms,cpu_diff_ms,gpu_diff_ms,gpu_append_ms,cpu_total_ms,gpu_total_ms,speedup"
    );
}

fn print_result(result: &CaseResult) {
    println!(
        "{:.0},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
        result.region_mb,
        result.dirty_pages,
        result.observed_dirty,
        result.cpu_dtoh_ms,
        result.cpu_diff_ms,
        result.gpu_diff_ms,
        result.gpu_append_ms,
        result.cpu_total_ms(),
        result.gpu_total_ms(),
        result.speedup()
    );
}

fn selected_dirty_pages(page_count: usize, dirty_pages: usize) -> Vec<usize> {
    if dirty_pages == 1 {
        return vec![page_count / 2];
    }

    let step = cmp::max(1, page_count / dirty_pages);
    (0..dirty_pages)
        .map(|idx| cmp::min(page_count - 1, idx * step))
        .collect()
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config {
        self_test: env_flag("CONCORDIA_BENCH_SELF_TEST"),
        launch_only: env_flag("CONCORDIA_BENCH_LAUNCH_ONLY"),
        warmup: env_usize("CONCORDIA_BENCH_WARMUP", 3)?,
        iters: env_usize("CONCORDIA_BENCH_ITERS", 10)?,
        device_override: env_usize_opt("CONCORDIA_BENCH_DEVICE")?,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--self-test" => config.self_test = true,
            "--launch-only" => config.launch_only = true,
            "--warmup" => {
                config.warmup = parse_next_usize(&mut args, "--warmup")?;
            }
            "--iters" => {
                config.iters = parse_next_usize(&mut args, "--iters")?;
            }
            "--device" => {
                config.device_override = Some(parse_next_usize(&mut args, "--device")?);
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    if config.iters == 0 {
        return Err("iters must be > 0".to_string());
    }
    Ok(config)
}

fn parse_next_usize(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))?
        .parse::<usize>()
        .map_err(|err| format!("{flag} expects usize: {err}"))
}

fn print_help() {
    println!(
        "Concordia delta checkpoint benchmark using NVlabs cuda-oxide\n\
         \n\
         Options:\n\
           --self-test       run a 4 MiB correctness smoke\n\
           --launch-only     measure typed cuda-oxide launch overhead only\n\
           --warmup N        warmup iterations before sampling\n\
           --iters N         measured iterations\n\
           --device N        CUDA ordinal; overrides MPI local rank\n\
         \n\
         Env equivalents for cargo oxide runs:\n\
           CONCORDIA_BENCH_SELF_TEST=1\n\
           CONCORDIA_BENCH_LAUNCH_ONLY=1\n\
           CONCORDIA_BENCH_WARMUP=N\n\
           CONCORDIA_BENCH_ITERS=N\n\
           CONCORDIA_BENCH_DEVICE=N"
    );
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    env_usize_opt(name).map(|value| value.unwrap_or(default))
}

fn env_usize_opt(name: &str) -> Result<Option<usize>, String> {
    match env::var(name) {
        Ok(value) if value.is_empty() => Ok(None),
        Ok(value) => value
            .parse::<usize>()
            .map(Some)
            .map_err(|err| format!("{name} expects usize: {err}")),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(format!("{name}: {err}")),
    }
}

fn local_rank() -> Option<usize> {
    [
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
        "SLURM_LOCALID",
    ]
    .iter()
    .find_map(|name| env::var(name).ok()?.parse::<usize>().ok())
}

fn cuda_device_count() -> Result<usize, cuda_core::DriverError> {
    unsafe {
        cuda_core::init(0)?;
        let mut count = 0;
        sys::cuDeviceGetCount(&mut count).result()?;
        Ok(count as usize)
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn us(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}
