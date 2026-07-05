use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig, sys};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use std::cmp;
use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

const WORKER_TPB: u32 = 128;
const DEFAULT_ELEMENTS: usize = 64 * 1024 * 1024;
const DEFAULT_COMPUTE_ITERS: u32 = 256;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn persistent_worker(
        stop: &[u32],
        max_batches: u64,
        mut counters: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d();
        let lane = idx.get();
        let stop_flag = unsafe { DeviceAtomicU32::from_ptr(stop.as_ptr() as *mut u32) };
        let mut acc = (lane as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut batches = 0u64;

        while stop_flag.load(AtomicOrdering::Acquire) == 0 && batches < max_batches {
            let mut inner = 0u32;
            while inner < 512 {
                acc = acc
                    .wrapping_mul(2_862_933_555_777_941_757)
                    .wrapping_add(3_037_000_493);
                inner += 1;
            }
            batches += 1;
        }

        if let Some(counter) = counters.get_mut(idx) {
            *counter = acc ^ batches;
        }
    }

    #[kernel]
    pub fn copy_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < input.len() {
            if let Some(out) = output.get_mut(idx) {
                *out = input[i] + 1.0;
            }
        }
    }

    #[kernel]
    pub fn compute_kernel(input: &[f32], mut output: DisjointSlice<f32>, inner_iters: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i < input.len() {
            let mut x = input[i];
            let mut k = 0u32;
            while k < inner_iters {
                x = x * 1.000_000_1 + 0.000_000_119;
                k += 1;
            }
            if let Some(out) = output.get_mut(idx) {
                *out = x;
            }
        }
    }
}

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone)]
struct Config {
    self_test: bool,
    warmup: usize,
    iters: usize,
    elements: usize,
    compute_iters: u32,
    worker_blocks: Vec<u32>,
    device_override: Option<usize>,
}

#[derive(Debug, Clone)]
struct RawCase {
    worker_blocks: u32,
    theoretical_sm_pct: f64,
    copy_ms: f64,
    compute_ms: f64,
    worker_active_threads: usize,
    worker_counter_xor: u64,
}

#[derive(Debug, Clone)]
struct PrintedCase {
    worker_blocks: u32,
    theoretical_sm_pct: f64,
    copy_ms: f64,
    copy_gbs: f64,
    copy_overhead_pct: f64,
    compute_ms: f64,
    compute_gops: f64,
    compute_overhead_pct: f64,
    worker_active_threads: usize,
    worker_counter_xor: u64,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> BenchResult<()> {
    let mut config = parse_args()?;
    let device_count = cuda_device_count()?;
    if device_count == 0 {
        return Err("no CUDA devices visible".into());
    }

    if config.self_test {
        config.elements = cmp::min(config.elements, 1 << 20);
        config.compute_iters = cmp::min(config.compute_iters, 32);
        config.warmup = cmp::min(config.warmup, 1);
        config.iters = cmp::min(config.iters, 2);
        config.worker_blocks = vec![0, 1];
    }

    let local_rank = local_rank();
    let ordinal = config
        .device_override
        .unwrap_or_else(|| local_rank.unwrap_or(0) % device_count);

    let ctx = CudaContext::new(ordinal)?;
    let work_stream = ctx.new_stream()?;
    let worker_stream = ctx.new_stream()?;
    let control_stream = ctx.new_stream()?;
    let module = kernels::load(&ctx)?;
    let sm_count = sm_count(&ctx)?;
    let (cc_major, cc_minor) = ctx.compute_capability()?;

    eprintln!(
        "# device={} ordinal={} device_count={} sm_count={} cc={}.{} elements={} compute_iters={} warmup={} iters={} rank={}",
        ctx.device_name()?,
        ctx.ordinal(),
        device_count,
        sm_count,
        cc_major,
        cc_minor,
        config.elements,
        config.compute_iters,
        config.warmup,
        config.iters,
        local_rank
            .map(|rank| rank.to_string())
            .unwrap_or_else(|| "none".to_string())
    );

    let input_host: Vec<f32> = (0..config.elements)
        .map(|i| ((i % 1024) as f32) * 0.001)
        .collect();
    let input_dev = DeviceBuffer::from_host(&work_stream, &input_host)?;
    let mut copy_out = DeviceBuffer::<f32>::zeroed(&work_stream, config.elements)?;
    let mut compute_out = DeviceBuffer::<f32>::zeroed(&work_stream, config.elements)?;

    let mut raw_cases = Vec::new();
    for &worker_blocks in &config.worker_blocks {
        raw_cases.push(run_case(
            &module,
            &work_stream,
            &worker_stream,
            &control_stream,
            &input_dev,
            &mut copy_out,
            &mut compute_out,
            config.elements,
            config.compute_iters,
            config.warmup,
            config.iters,
            sm_count,
            worker_blocks,
        )?);
    }

    let printed = finalize_cases(&raw_cases, config.elements, config.compute_iters)?;
    print_cases(&ctx.device_name()?, sm_count, &printed);

    if config.self_test {
        if printed.len() != 2 || printed[1].worker_active_threads == 0 {
            return Err("self-test worker did not report active threads".into());
        }
        println!(
            "self-test passed: worker_blocks={} active_threads={}",
            printed[1].worker_blocks, printed[1].worker_active_threads
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    module: &kernels::LoadedModule,
    work_stream: &cuda_core::CudaStream,
    worker_stream: &cuda_core::CudaStream,
    control_stream: &cuda_core::CudaStream,
    input_dev: &DeviceBuffer<f32>,
    copy_out: &mut DeviceBuffer<f32>,
    compute_out: &mut DeviceBuffer<f32>,
    elements: usize,
    compute_iters: u32,
    warmup: usize,
    iters: usize,
    sm_count: i32,
    worker_blocks: u32,
) -> BenchResult<RawCase> {
    let mut stop = DeviceBuffer::<u32>::from_host(control_stream, &[0])?;
    let worker_threads = worker_blocks as usize * WORKER_TPB as usize;
    let mut counters = DeviceBuffer::<u64>::zeroed(worker_stream, cmp::max(1, worker_threads))?;

    if worker_blocks > 0 {
        let worker_config = LaunchConfig {
            grid_dim: (worker_blocks, 1, 1),
            block_dim: (WORKER_TPB, 1, 1),
            shared_mem_bytes: 0,
        };
        module.persistent_worker(
            worker_stream,
            worker_config,
            &stop,
            u64::MAX,
            &mut counters,
        )?;
        std::thread::sleep(Duration::from_millis(20));
    }

    let copy_ms = measure_copy(
        module,
        work_stream,
        input_dev,
        copy_out,
        elements,
        warmup,
        iters,
    )?;
    let compute_ms = measure_compute(
        module,
        work_stream,
        input_dev,
        compute_out,
        elements,
        compute_iters,
        warmup,
        iters,
    )?;

    let mut worker_active_threads = 0usize;
    let mut worker_counter_xor = 0u64;
    if worker_blocks > 0 {
        stop.copy_from_host(control_stream, &[1])?;
        control_stream.synchronize()?;
        worker_stream.synchronize()?;
        let counters_host = counters.to_host_vec(worker_stream)?;
        worker_active_threads = counters_host.iter().filter(|&&value| value != 0).count();
        worker_counter_xor = counters_host.iter().fold(0u64, |acc, &value| acc ^ value);
        if worker_active_threads == 0 {
            return Err(format!("worker_blocks={worker_blocks} did not update counters").into());
        }
    }

    Ok(RawCase {
        worker_blocks,
        theoretical_sm_pct: if sm_count > 0 {
            100.0 * worker_blocks as f64 / sm_count as f64
        } else {
            0.0
        },
        copy_ms,
        compute_ms,
        worker_active_threads,
        worker_counter_xor,
    })
}

fn measure_copy(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    input_dev: &DeviceBuffer<f32>,
    output_dev: &mut DeviceBuffer<f32>,
    elements: usize,
    warmup: usize,
    iters: usize,
) -> BenchResult<f64> {
    let config = LaunchConfig::for_num_elems(elements as u32);
    let mut samples = Vec::with_capacity(iters);
    for iter in 0..(warmup + iters) {
        let start = Instant::now();
        module.copy_kernel(stream, config, input_dev, output_dev)?;
        stream.synchronize()?;
        if iter >= warmup {
            samples.push(ms(start.elapsed()));
        }
    }
    Ok(median(&mut samples))
}

fn measure_compute(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    input_dev: &DeviceBuffer<f32>,
    output_dev: &mut DeviceBuffer<f32>,
    elements: usize,
    compute_iters: u32,
    warmup: usize,
    iters: usize,
) -> BenchResult<f64> {
    let config = LaunchConfig::for_num_elems(elements as u32);
    let mut samples = Vec::with_capacity(iters);
    for iter in 0..(warmup + iters) {
        let start = Instant::now();
        module.compute_kernel(stream, config, input_dev, output_dev, compute_iters)?;
        stream.synchronize()?;
        if iter >= warmup {
            samples.push(ms(start.elapsed()));
        }
    }
    Ok(median(&mut samples))
}

fn finalize_cases(
    raw_cases: &[RawCase],
    elements: usize,
    compute_iters: u32,
) -> BenchResult<Vec<PrintedCase>> {
    let baseline = raw_cases
        .iter()
        .find(|case| case.worker_blocks == 0)
        .ok_or("worker block list must include 0 baseline")?;
    let copy_bytes = elements as f64 * std::mem::size_of::<f32>() as f64 * 2.0;
    let compute_ops = elements as f64 * compute_iters as f64 * 2.0;

    Ok(raw_cases
        .iter()
        .map(|case| PrintedCase {
            worker_blocks: case.worker_blocks,
            theoretical_sm_pct: case.theoretical_sm_pct,
            copy_ms: case.copy_ms,
            copy_gbs: throughput(copy_bytes, case.copy_ms),
            copy_overhead_pct: pct_delta(case.copy_ms, baseline.copy_ms),
            compute_ms: case.compute_ms,
            compute_gops: throughput(compute_ops, case.compute_ms),
            compute_overhead_pct: pct_delta(case.compute_ms, baseline.compute_ms),
            worker_active_threads: case.worker_active_threads,
            worker_counter_xor: case.worker_counter_xor,
        })
        .collect())
}

fn print_cases(device_name: &str, sm_count: i32, cases: &[PrintedCase]) {
    println!("# persistent_kernel_overhead_ablation");
    println!("# device={device_name} sm_count={sm_count} worker_tpb={WORKER_TPB}");
    println!(
        "worker_blocks,theoretical_sm_pct,copy_ms,copy_gbs,copy_overhead_pct,compute_ms,compute_gops,compute_overhead_pct,worker_active_threads,worker_counter_xor"
    );
    for case in cases {
        println!(
            "{},{:.3},{:.6},{:.3},{:.3},{:.6},{:.3},{:.3},{},{}",
            case.worker_blocks,
            case.theoretical_sm_pct,
            case.copy_ms,
            case.copy_gbs,
            case.copy_overhead_pct,
            case.compute_ms,
            case.compute_gops,
            case.compute_overhead_pct,
            case.worker_active_threads,
            case.worker_counter_xor
        );
    }
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config {
        self_test: env_flag("CONCORDIA_PERSISTENT_OVERHEAD_SELF_TEST"),
        warmup: env_usize("CONCORDIA_PERSISTENT_OVERHEAD_WARMUP", 3)?,
        iters: env_usize("CONCORDIA_PERSISTENT_OVERHEAD_ITERS", 10)?,
        elements: env_usize("CONCORDIA_PERSISTENT_OVERHEAD_ELEMENTS", DEFAULT_ELEMENTS)?,
        compute_iters: env_u32("CONCORDIA_PERSISTENT_OVERHEAD_COMPUTE_ITERS", DEFAULT_COMPUTE_ITERS)?,
        worker_blocks: env_worker_blocks("CONCORDIA_PERSISTENT_OVERHEAD_WORKER_BLOCKS")?
            .unwrap_or_else(|| vec![0, 1, 2, 4, 8]),
        device_override: env_usize_opt("CONCORDIA_PERSISTENT_OVERHEAD_DEVICE")?,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--self-test" => config.self_test = true,
            "--warmup" => config.warmup = parse_next_usize(&mut args, "--warmup")?,
            "--iters" => config.iters = parse_next_usize(&mut args, "--iters")?,
            "--elements" => config.elements = parse_next_usize(&mut args, "--elements")?,
            "--compute-iters" => {
                config.compute_iters = parse_next_u32(&mut args, "--compute-iters")?;
            }
            "--worker-blocks" => {
                config.worker_blocks = parse_worker_blocks(&parse_next_string(&mut args, "--worker-blocks")?)?;
            }
            "--device" => config.device_override = Some(parse_next_usize(&mut args, "--device")?),
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
    if config.elements == 0 || config.elements > u32::MAX as usize {
        return Err(format!("elements must be in 1..={}", u32::MAX).to_string());
    }
    config.worker_blocks.sort_unstable();
    config.worker_blocks.dedup();
    if !config.worker_blocks.contains(&0) {
        config.worker_blocks.insert(0, 0);
    }
    Ok(config)
}

fn print_help() {
    println!(
        "Concordia persistent-kernel overhead ablation using NVlabs cuda-oxide\n\
         \n\
         Options:\n\
           --self-test           run a small live correctness smoke\n\
           --warmup N            warmup iterations before sampling\n\
           --iters N             measured iterations\n\
           --elements N          f32 elements in copy/compute kernels\n\
           --compute-iters N     inner FMA iterations per element\n\
           --worker-blocks LIST  comma-separated resident worker block counts\n\
           --device N            CUDA ordinal; overrides MPI local rank\n\
         \n\
         Env equivalents:\n\
           CONCORDIA_PERSISTENT_OVERHEAD_SELF_TEST=1\n\
           CONCORDIA_PERSISTENT_OVERHEAD_WARMUP=N\n\
           CONCORDIA_PERSISTENT_OVERHEAD_ITERS=N\n\
           CONCORDIA_PERSISTENT_OVERHEAD_ELEMENTS=N\n\
           CONCORDIA_PERSISTENT_OVERHEAD_COMPUTE_ITERS=N\n\
           CONCORDIA_PERSISTENT_OVERHEAD_WORKER_BLOCKS=0,1,2,4,8\n\
           CONCORDIA_PERSISTENT_OVERHEAD_DEVICE=N"
    );
}

fn cuda_device_count() -> Result<usize, cuda_core::DriverError> {
    unsafe {
        cuda_core::init(0)?;
        let mut count = 0;
        sys::cuDeviceGetCount(&mut count).result()?;
        Ok(count as usize)
    }
}

fn sm_count(ctx: &CudaContext) -> Result<i32, cuda_core::DriverError> {
    ctx.bind_to_thread()?;
    let mut sm_count = 0;
    unsafe {
        sys::cuDeviceGetAttribute(
            &mut sm_count,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            ctx.cu_device(),
        )
        .result()?;
    }
    Ok(sm_count)
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

fn parse_next_string(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_next_usize(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    parse_next_string(args, flag)?
        .parse::<usize>()
        .map_err(|err| format!("{flag} expects usize: {err}"))
}

fn parse_next_u32(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<u32, String> {
    parse_next_string(args, flag)?
        .parse::<u32>()
        .map_err(|err| format!("{flag} expects u32: {err}"))
}

fn env_flag(name: &str) -> bool {
    env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    env_usize_opt(name).map(|value| value.unwrap_or(default))
}

fn env_u32(name: &str, default: u32) -> Result<u32, String> {
    match env::var(name) {
        Ok(value) if value.is_empty() => Ok(default),
        Ok(value) => value
            .parse::<u32>()
            .map_err(|err| format!("{name} expects u32: {err}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(format!("{name}: {err}")),
    }
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

fn env_worker_blocks(name: &str) -> Result<Option<Vec<u32>>, String> {
    match env::var(name) {
        Ok(value) if value.is_empty() => Ok(None),
        Ok(value) => parse_worker_blocks(&value).map(Some),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(format!("{name}: {err}")),
    }
}

fn parse_worker_blocks(value: &str) -> Result<Vec<u32>, String> {
    let mut blocks = Vec::new();
    for part in value.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        blocks.push(
            trimmed
                .parse::<u32>()
                .map_err(|err| format!("worker block count '{trimmed}' expects u32: {err}"))?,
        );
    }
    if blocks.is_empty() {
        return Err("worker block list must not be empty".to_string());
    }
    Ok(blocks)
}

fn pct_delta(value: f64, baseline: f64) -> f64 {
    if baseline.abs() <= f64::EPSILON {
        0.0
    } else {
        (value / baseline - 1.0) * 100.0
    }
}

fn throughput(work_units: f64, ms: f64) -> f64 {
    work_units / (ms / 1_000.0) / 1_000_000_000.0
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
