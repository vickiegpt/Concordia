use cuda_types::cuda::*;
use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

const TASK_SIZE: usize = 64;

const PERSISTENT_KERNEL_CUDA: &str = r#"
extern "C" {

struct Task {
    int op;
    int flags;
    long long numel;
    void* in0;
    void* in1;
    void* out0;
    int num_params;
    int _pad[5];
};

__device__ void do_add(const Task& t) {
    float* a = (float*)t.in0;
    float* b = (float*)t.in1;
    float* c = (float*)t.out0;
    for (long long i = threadIdx.x + (long long)blockIdx.x * blockDim.x;
         i < t.numel; i += (long long)blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

__device__ void do_mul(const Task& t) {
    float* a = (float*)t.in0;
    float* b = (float*)t.in1;
    float* c = (float*)t.out0;
    for (long long i = threadIdx.x + (long long)blockIdx.x * blockDim.x;
         i < t.numel; i += (long long)blockDim.x * gridDim.x) {
        c[i] = a[i] * b[i];
    }
}

__device__ void do_sub(const Task& t) {
    float* a = (float*)t.in0;
    float* b = (float*)t.in1;
    float* c = (float*)t.out0;
    for (long long i = threadIdx.x + (long long)blockIdx.x * blockDim.x;
         i < t.numel; i += (long long)blockDim.x * gridDim.x) {
        c[i] = a[i] - b[i];
    }
}

__device__ void do_silu(const Task& t) {
    float* a = (float*)t.in0;
    float* c = (float*)t.out0;
    for (long long i = threadIdx.x + (long long)blockIdx.x * blockDim.x;
         i < t.numel; i += (long long)blockDim.x * gridDim.x) {
        float x = a[i];
        c[i] = x / (1.0f + expf(-x));
    }
}

__device__ void do_relu(const Task& t) {
    float* a = (float*)t.in0;
    float* c = (float*)t.out0;
    for (long long i = threadIdx.x + (long long)blockIdx.x * blockDim.x;
         i < t.numel; i += (long long)blockDim.x * gridDim.x) {
        c[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

__device__ void do_scale(const Task& t) {
    float* a = (float*)t.in0;
    float* scale = (float*)t.in1;
    float* c = (float*)t.out0;
    float s = scale ? scale[0] : 1.0f;
    for (long long i = threadIdx.x + (long long)blockIdx.x * blockDim.x;
         i < t.numel; i += (long long)blockDim.x * gridDim.x) {
        c[i] = a[i] * s;
    }
}

__device__ void do_add_relu(const Task& t) {
    float* a = (float*)t.in0;
    float* b = (float*)t.in1;
    float* c = (float*)t.out0;
    for (long long i = threadIdx.x + (long long)blockIdx.x * blockDim.x;
         i < t.numel; i += (long long)blockDim.x * gridDim.x) {
        float v = a[i] + b[i];
        c[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void persistent_worker(
    Task* tasks,
    int capacity,
    int* head,
    int* tail,
    int* quit,
    unsigned long long* processed
) {
    __shared__ Task task;
    __shared__ int has_work;

    while (atomicAdd(quit, 0) == 0) {
        if (threadIdx.x == 0) {
            has_work = 0;
            int h = atomicAdd(head, 0);
            int t = atomicAdd(tail, 0);
            if (h < t) {
                task = tasks[h % capacity];
                __threadfence_system();
                atomicExch(head, h + 1);
                has_work = 1;
            }
        }
        __syncthreads();

        if (!has_work) {
#if __CUDA_ARCH__ >= 700
            if (threadIdx.x == 0) __nanosleep(1000);
#endif
            __syncthreads();
            continue;
        }

        switch (task.op) {
            case 0: do_add(task); break;
            case 1: do_mul(task); break;
            case 2: do_sub(task); break;
            case 3: do_silu(task); break;
            case 4: do_relu(task); break;
            case 5: do_scale(task); break;
            case 6: do_add_relu(task); break;
            default: break;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            __threadfence_system();
            atomicAdd_system(processed, 1ULL);
        }
        __syncthreads();
    }
}

}
"#;

pub(crate) struct GpuPersistentKernel {
    module: CUmodule,
    worker_func: CUfunction,
    tasks_host: *mut c_void,
    tasks_device: u64,
    capacity: u32,
    ctrl_host: *mut c_void,
    head_host: *mut i32,
    tail_host: *mut i32,
    quit_host: *mut i32,
    processed_host: *mut u64,
    head_device: u64,
    tail_device: u64,
    quit_device: u64,
    processed_device: u64,
    stream: CUstream,
    submitted: AtomicU64,
    running: bool,
    num_blocks: u32,
    threads_per_block: u32,
}

impl GpuPersistentKernel {
    fn init(device_id: i32, capacity: u32) -> Result<Self, String> {
        let capacity = capacity.max(1);
        nvidia_runtime_sys::init()?;

        let rc = nvidia_runtime_sys::cuInit(0);
        if rc != 0 {
            return Err(format!("cuInit failed: {rc}"));
        }

        let mut device: CUdevice = 0;
        let rc = nvidia_runtime_sys::cuDeviceGet(&mut device, device_id);
        if rc != 0 {
            return Err(format!("cuDeviceGet({device_id}) failed: {rc}"));
        }

        let mut ctx = CUcontext(ptr::null_mut());
        let rc = nvidia_runtime_sys::cuDevicePrimaryCtxRetain(&mut ctx, device);
        if rc != 0 {
            return Err(format!("cuDevicePrimaryCtxRetain failed: {rc}"));
        }
        let rc = nvidia_runtime_sys::cuCtxSetCurrent(ctx);
        if rc != 0 {
            return Err(format!("cuCtxSetCurrent failed: {rc}"));
        }

        let ptx = Self::compile_cuda_to_ptx(PERSISTENT_KERNEL_CUDA)?;
        let ptx_cstr = CString::new(ptx).map_err(|err| format!("PTX CString: {err}"))?;
        let mut module = CUmodule(ptr::null_mut());
        let rc = nvidia_runtime_sys::cuModuleLoadData(&mut module, ptx_cstr.as_ptr().cast());
        if rc != 0 {
            return Err(format!("cuModuleLoadData persistent worker failed: {rc}"));
        }

        let worker_name = CString::new("persistent_worker").unwrap();
        let mut worker_func = CUfunction(ptr::null_mut());
        let rc =
            nvidia_runtime_sys::cuModuleGetFunction(&mut worker_func, module, worker_name.as_ptr());
        if rc != 0 {
            return Err(format!(
                "cuModuleGetFunction(persistent_worker) failed: {rc}"
            ));
        }

        let task_bytes = (capacity as usize)
            .checked_mul(TASK_SIZE)
            .ok_or_else(|| "persistent queue size overflow".to_string())?;
        let mut tasks_host = ptr::null_mut();
        let rc = nvidia_runtime_sys::cuMemAllocHost_v2(&mut tasks_host, task_bytes);
        if rc != 0 || tasks_host.is_null() {
            return Err(format!("cuMemAllocHost task ring failed: {rc}"));
        }
        unsafe { ptr::write_bytes(tasks_host.cast::<u8>(), 0, task_bytes) };

        let mut tasks_device = CUdeviceptr_v2(ptr::null_mut());
        let rc = nvidia_runtime_sys::cuMemHostGetDevicePointer_v2(&mut tasks_device, tasks_host, 0);
        if rc != 0 {
            unsafe {
                nvidia_runtime_sys::cuMemFreeHost(tasks_host);
            }
            return Err(format!("cuMemHostGetDevicePointer task ring failed: {rc}"));
        }

        let mut ctrl_host = ptr::null_mut();
        let ctrl_bytes = 32usize;
        let rc = nvidia_runtime_sys::cuMemAllocHost_v2(&mut ctrl_host, ctrl_bytes);
        if rc != 0 || ctrl_host.is_null() {
            unsafe {
                nvidia_runtime_sys::cuMemFreeHost(tasks_host);
            }
            return Err(format!("cuMemAllocHost control block failed: {rc}"));
        }
        unsafe { ptr::write_bytes(ctrl_host.cast::<u8>(), 0, ctrl_bytes) };

        let mut ctrl_device = CUdeviceptr_v2(ptr::null_mut());
        let rc = nvidia_runtime_sys::cuMemHostGetDevicePointer_v2(&mut ctrl_device, ctrl_host, 0);
        if rc != 0 {
            unsafe {
                nvidia_runtime_sys::cuMemFreeHost(tasks_host);
                nvidia_runtime_sys::cuMemFreeHost(ctrl_host);
            }
            return Err(format!(
                "cuMemHostGetDevicePointer control block failed: {rc}"
            ));
        }

        let mut sm_count = 1;
        let _ = nvidia_runtime_sys::cuDeviceGetAttribute(
            &mut sm_count,
            CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            device,
        );
        let num_blocks = std::env::var("CONCORDIA_PERSISTENT_BLOCKS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(1)
            .clamp(1, sm_count.max(1) as u32);

        let mut stream = CUstream(ptr::null_mut());
        let rc = nvidia_runtime_sys::cuStreamCreate_ckpt(&mut stream, 1);
        if rc != 0 {
            unsafe {
                nvidia_runtime_sys::cuMemFreeHost(tasks_host);
                nvidia_runtime_sys::cuMemFreeHost(ctrl_host);
            }
            return Err(format!("cuStreamCreate failed: {rc}"));
        }

        let ctrl_base = ctrl_host as usize;
        let ctrl_device_base = ctrl_device.0 as usize;
        let mut kernel = Self {
            module,
            worker_func,
            tasks_host,
            tasks_device: tasks_device.0 as u64,
            capacity,
            ctrl_host,
            head_host: ctrl_base as *mut i32,
            tail_host: (ctrl_base + 4) as *mut i32,
            quit_host: (ctrl_base + 8) as *mut i32,
            processed_host: (ctrl_base + 16) as *mut u64,
            head_device: ctrl_device_base as u64,
            tail_device: (ctrl_device_base + 4) as u64,
            quit_device: (ctrl_device_base + 8) as u64,
            processed_device: (ctrl_device_base + 16) as u64,
            stream,
            submitted: AtomicU64::new(0),
            running: false,
            num_blocks,
            threads_per_block: 128,
        };
        kernel.launch()?;
        Ok(kernel)
    }

    fn compile_cuda_to_ptx(cuda_src: &str) -> Result<String, String> {
        let src_path = std::env::temp_dir().join("hetgpu_concordia_persistent.cu");
        let ptx_path = std::env::temp_dir().join("hetgpu_concordia_persistent.ptx");
        std::fs::write(&src_path, cuda_src).map_err(|err| format!("write CUDA source: {err}"))?;

        let nvcc = std::env::var("CONCORDIA_NVCC")
            .or_else(|_| std::env::var("HETGPU_CONCORDIA_NVCC"))
            .unwrap_or_else(|_| "nvcc".to_string());
        let arch = std::env::var("CONCORDIA_ARCH")
            .or_else(|_| std::env::var("HETGPU_CONCORDIA_ARCH"))
            .unwrap_or_else(|_| "sm_80".to_string());
        let output = std::process::Command::new(nvcc)
            .arg("--ptx")
            .arg(format!("-arch={arch}"))
            .arg("--std=c++17")
            .arg("-o")
            .arg(&ptx_path)
            .arg(&src_path)
            .output()
            .map_err(|err| format!("run nvcc: {err}"))?;
        if !output.status.success() {
            return Err(format!(
                "nvcc persistent worker failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        std::fs::read_to_string(&ptx_path).map_err(|err| format!("read PTX: {err}"))
    }

    fn launch(&mut self) -> Result<(), String> {
        let mut tasks = self.tasks_device;
        let mut capacity = self.capacity as i32;
        let mut head = self.head_device;
        let mut tail = self.tail_device;
        let mut quit = self.quit_device;
        let mut processed = self.processed_device;
        let mut params = [
            (&mut tasks as *mut u64).cast::<c_void>(),
            (&mut capacity as *mut i32).cast::<c_void>(),
            (&mut head as *mut u64).cast::<c_void>(),
            (&mut tail as *mut u64).cast::<c_void>(),
            (&mut quit as *mut u64).cast::<c_void>(),
            (&mut processed as *mut u64).cast::<c_void>(),
        ];
        let rc = nvidia_runtime_sys::cuLaunchKernel(
            self.worker_func,
            self.num_blocks,
            1,
            1,
            self.threads_per_block,
            1,
            1,
            0,
            self.stream,
            params.as_mut_ptr(),
            ptr::null_mut(),
        );
        if rc != 0 {
            return Err(format!("cuLaunchKernel persistent worker failed: {rc}"));
        }
        self.running = true;
        Ok(())
    }

    fn enqueue_task(
        &self,
        op: i32,
        numel: i64,
        in0: u64,
        in1: u64,
        out0: u64,
    ) -> Result<u64, String> {
        if !self.running {
            return Err("persistent worker is not running".to_string());
        }
        let head = unsafe { ptr::read_volatile(self.head_host) };
        let tail = unsafe { ptr::read_volatile(self.tail_host) };
        if tail - head >= self.capacity as i32 {
            return Err("persistent worker ring is full".to_string());
        }

        let slot = (tail as u32 % self.capacity) as usize;
        let slot_ptr = unsafe { self.tasks_host.cast::<u8>().add(slot * TASK_SIZE) };
        let mut task = [0u8; TASK_SIZE];
        unsafe {
            ptr::write_unaligned(task.as_mut_ptr().add(0).cast::<i32>(), op);
            ptr::write_unaligned(task.as_mut_ptr().add(4).cast::<i32>(), 0);
            ptr::write_unaligned(task.as_mut_ptr().add(8).cast::<i64>(), numel);
            ptr::write_unaligned(task.as_mut_ptr().add(16).cast::<u64>(), in0);
            ptr::write_unaligned(task.as_mut_ptr().add(24).cast::<u64>(), in1);
            ptr::write_unaligned(task.as_mut_ptr().add(32).cast::<u64>(), out0);
            ptr::write_unaligned(task.as_mut_ptr().add(40).cast::<i32>(), 3);
            ptr::copy_nonoverlapping(task.as_ptr(), slot_ptr, TASK_SIZE);
            std::sync::atomic::fence(Ordering::Release);
            ptr::write_volatile(self.tail_host, tail + 1);
        }
        Ok(self.submitted.fetch_add(1, Ordering::Relaxed))
    }

    fn sync(&self) -> u64 {
        let target = self.submitted.load(Ordering::Acquire);
        loop {
            let processed = unsafe { ptr::read_volatile(self.processed_host) };
            if processed >= target {
                return processed;
            }
            std::thread::yield_now();
        }
    }

    fn shutdown(&mut self) {
        if !self.running {
            return;
        }
        unsafe {
            ptr::write_volatile(self.quit_host, 1);
        }
        let _ = nvidia_runtime_sys::cuStreamSynchronize_ckpt(self.stream);
        self.running = false;
    }
}

impl Drop for GpuPersistentKernel {
    fn drop(&mut self) {
        self.shutdown();
        let _ = nvidia_runtime_sys::cuModuleUnload(self.module);
        unsafe {
            if !self.tasks_host.is_null() {
                nvidia_runtime_sys::cuMemFreeHost(self.tasks_host);
            }
            if !self.ctrl_host.is_null() {
                nvidia_runtime_sys::cuMemFreeHost(self.ctrl_host);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn concordia_gpu_init(device_id: i32, capacity: u32) -> i64 {
    match GpuPersistentKernel::init(device_id, capacity) {
        Ok(kernel) => Box::into_raw(Box::new(kernel)) as i64,
        Err(err) => {
            eprintln!("[hetGPU Concordia] persistent worker init failed: {err}");
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn concordia_gpu_enqueue(
    handle: i64,
    op: i32,
    numel: i64,
    in0: u64,
    in1: u64,
    out0: u64,
) -> i64 {
    if handle <= 0 {
        return -1;
    }
    let kernel = &*(handle as *const GpuPersistentKernel);
    kernel
        .enqueue_task(op, numel, in0, in1, out0)
        .map(|seq| seq as i64)
        .unwrap_or(-1)
}

#[no_mangle]
pub unsafe extern "C" fn concordia_gpu_sync(handle: i64) -> u64 {
    if handle <= 0 {
        return 0;
    }
    let kernel = &*(handle as *const GpuPersistentKernel);
    kernel.sync()
}

#[no_mangle]
pub unsafe extern "C" fn concordia_gpu_shutdown(handle: i64) {
    if handle <= 0 {
        return;
    }
    let mut kernel = Box::from_raw(handle as *mut GpuPersistentKernel);
    kernel.shutdown();
}
