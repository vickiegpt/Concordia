use std::sync::OnceLock;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PersistentOp {
    Add = 0,
    Mul = 1,
    Sub = 2,
    Silu = 3,
    Relu = 4,
    Scale = 5,
    AddRelu = 6,
    DirtyScan = 7,
    AofAppend = 8,
    RestoreCopy = 9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Routing {
    Persistent { op: PersistentOp, numel: i64 },
    Passthrough,
}

pub(crate) fn is_persistent_enabled_for_value(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES") | Some("on")
    )
}

pub(crate) fn is_persistent_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        is_persistent_enabled_for_value(std::env::var("CONCORDIA_PERSISTENT").ok().as_deref())
    })
}

pub(crate) fn persistent_device_id() -> i32 {
    let pairs: Vec<(String, String)> = PERSISTENT_DEVICE_KEYS
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
    persistent_device_id_from_pairs(&refs)
}

pub(crate) fn persistent_device_id_from_pairs(pairs: &[(&str, &str)]) -> i32 {
    read_i32_from_pairs(pairs, &["CONCORDIA_PERSISTENT_DEVICE"], None)
        .or_else(|| read_i32_from_pairs(pairs, MPI_LOCAL_RANK_KEYS, None))
        .unwrap_or(0)
        .max(0)
}

const PERSISTENT_DEVICE_KEYS: &[&str] = &[
    "CONCORDIA_PERSISTENT_DEVICE",
    "CONCORDIA_MPI_LOCAL_RANK",
    "HETGPU_CONCORDIA_MPI_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "SLURM_LOCALID",
    "PMI_LOCAL_RANK",
    "LOCAL_RANK",
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

fn read_i32_from_pairs(
    pairs: &[(&str, &str)],
    keys: &[&str],
    fallback: Option<i32>,
) -> Option<i32> {
    keys.iter()
        .find_map(|key| {
            pairs
                .iter()
                .find(|(candidate, _)| candidate == key)
                .and_then(|(_, value)| value.trim().parse::<i32>().ok())
        })
        .or(fallback)
}

pub(crate) fn classify(
    kernel_name: &str,
    grid_dims: (u32, u32, u32),
    block_dims: (u32, u32, u32),
    num_params: usize,
) -> Routing {
    if grid_dims.1 != 1 || grid_dims.2 != 1 || block_dims.1 != 1 || block_dims.2 != 1 {
        return Routing::Passthrough;
    }
    if num_params < 2 || num_params > 8 {
        return Routing::Passthrough;
    }

    let name = kernel_name.to_ascii_lowercase();
    let is_add = name.contains("add") || name.contains("plus");
    let is_mul = name.contains("mul") || name.contains("multiply");
    let is_sub = name.contains("sub") || name.contains("minus");
    let is_relu = name.contains("relu");
    let is_silu = name.contains("silu") || name.contains("swish");

    let op = if is_silu && !is_add {
        PersistentOp::Silu
    } else if is_relu && is_add {
        PersistentOp::AddRelu
    } else if is_relu {
        PersistentOp::Relu
    } else if is_add {
        PersistentOp::Add
    } else if is_mul {
        PersistentOp::Mul
    } else if is_sub {
        PersistentOp::Sub
    } else {
        return Routing::Passthrough;
    };

    let numel = (grid_dims.0 as i64) * (block_dims.0 as i64);
    if !(32..=(1_i64 << 30)).contains(&numel) {
        return Routing::Passthrough;
    }

    Routing::Persistent { op, numel }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
mod nvidia_router {
    use super::{classify, is_persistent_enabled, persistent_device_id, PersistentOp, Routing};
    use std::os::raw::c_void;
    use std::sync::{Mutex, OnceLock};

    struct PersistentRouter {
        kernel_handle: i64,
    }

    impl PersistentRouter {
        fn new() -> Option<Self> {
            if !is_persistent_enabled() {
                return None;
            }
            let capacity = std::env::var("CONCORDIA_PERSISTENT_CAPACITY")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1024);
            let kernel_handle =
                crate::r#impl::concordia_gpu::concordia_gpu_init(persistent_device_id(), capacity);
            (kernel_handle >= 0).then_some(Self { kernel_handle })
        }

        fn route(&self, op: PersistentOp, numel: i64, in0: u64, in1: u64, out0: u64) -> bool {
            if self.kernel_handle < 0 {
                return false;
            }
            let seq = unsafe {
                crate::r#impl::concordia_gpu::concordia_gpu_enqueue(
                    self.kernel_handle,
                    op as i32,
                    numel,
                    in0,
                    in1,
                    out0,
                )
            };
            seq >= 0
        }
    }

    static ROUTER: OnceLock<Mutex<Option<PersistentRouter>>> = OnceLock::new();

    fn get_router() -> &'static Mutex<Option<PersistentRouter>> {
        ROUTER.get_or_init(|| Mutex::new(PersistentRouter::new()))
    }

    pub(crate) fn try_route(
        kernel_name: &str,
        kernel_params: *mut *mut c_void,
        num_params: usize,
        grid_dims: (u32, u32, u32),
        block_dims: (u32, u32, u32),
    ) -> bool {
        if !is_persistent_enabled() || kernel_params.is_null() {
            return false;
        }

        let (op, numel) = match classify(kernel_name, grid_dims, block_dims, num_params) {
            Routing::Persistent { op, numel } => (op, numel),
            Routing::Passthrough => return false,
        };

        let (in0, in1, out0) = unsafe {
            let param = |index: usize| -> u64 {
                if index >= num_params {
                    return 0;
                }
                let ptr = *kernel_params.add(index);
                if ptr.is_null() {
                    0
                } else {
                    std::ptr::read_unaligned(ptr as *const u64)
                }
            };
            (param(1), param(2), param(0))
        };
        if in0 == 0 || out0 == 0 {
            return false;
        }

        match get_router().lock() {
            Ok(guard) => guard
                .as_ref()
                .map(|router| router.route(op, numel, in0, in1, out0))
                .unwrap_or(false),
            Err(_) => false,
        }
    }
}

#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
))]
pub(crate) use nvidia_router::try_route;

#[cfg(not(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent"),
    not(feature = "tmatmul")
)))]
pub(crate) fn try_route(
    _kernel_name: &str,
    _kernel_params: *mut *mut std::os::raw::c_void,
    _num_params: usize,
    _grid_dims: (u32, u32, u32),
    _block_dims: (u32, u32, u32),
) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_routes_simple_1d_add_kernel() {
        match classify(
            "at::native::vectorized_elementwise_kernel<add>",
            (128, 1, 1),
            (256, 1, 1),
            3,
        ) {
            Routing::Persistent { op, numel } => {
                assert_eq!(op, PersistentOp::Add);
                assert_eq!(numel, 32768);
            }
            Routing::Passthrough => panic!("add kernel should be routed"),
        }
    }

    #[test]
    fn classify_rejects_non_1d_launches() {
        assert!(matches!(
            classify("relu_kernel", (2, 2, 1), (128, 1, 1), 2),
            Routing::Passthrough
        ));
    }

    #[test]
    fn persistent_routing_is_opt_in() {
        let _guard = crate::r#impl::test_env::lock();
        std::env::remove_var("CONCORDIA_PERSISTENT");
        assert!(!is_persistent_enabled_for_value(None));
        assert!(is_persistent_enabled_for_value(Some("1")));
        assert!(is_persistent_enabled_for_value(Some("true")));
        assert!(!is_persistent_enabled_for_value(Some("0")));
    }

    #[test]
    fn persistent_device_defaults_to_mpi_local_rank() {
        assert_eq!(
            persistent_device_id_from_pairs(&[("OMPI_COMM_WORLD_LOCAL_RANK", "3")]),
            3
        );
        assert_eq!(
            persistent_device_id_from_pairs(&[
                ("CONCORDIA_PERSISTENT_DEVICE", "7"),
                ("OMPI_COMM_WORLD_LOCAL_RANK", "3"),
            ]),
            7
        );
    }

    #[test]
    fn checkpoint_worker_ops_have_stable_task_ids() {
        assert_eq!(PersistentOp::DirtyScan as i32, 7);
        assert_eq!(PersistentOp::AofAppend as i32, 8);
        assert_eq!(PersistentOp::RestoreCopy as i32, 9);
    }

    #[test]
    fn user_kernel_classifier_does_not_route_aof_worker_names() {
        assert!(matches!(
            classify("concordia_aof_append_task", (1, 1, 1), (128, 1, 1), 3),
            Routing::Passthrough
        ));
        assert!(matches!(
            classify("concordia_restore_copy_task", (1, 1, 1), (128, 1, 1), 3),
            Routing::Passthrough
        ));
    }
}
