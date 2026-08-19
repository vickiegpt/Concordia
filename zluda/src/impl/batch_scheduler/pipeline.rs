// /home/victoryang00/hetGPU/zluda/src/impl/batch_scheduler/pipeline.rs
use super::{aggregator::Batch, config::PipelineConfig};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Unique identifier for tracking operations through the pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationId(u64);

impl OperationId {
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

/// Assignment of operations to a specific FPGA instance
#[derive(Debug, Clone)]
pub struct InstanceAssignment {
    pub instance_id: usize,
    pub operations: Vec<(OperationId, usize)>, // (op_id, batch_index)
}

/// Memory region allocated for staging data
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub device_ptr: usize,
    pub bytes: usize,
    pub instance_id: usize,
}

/// Allocator for managing staging memory regions per instance
#[derive(Debug)]
pub struct StagingAllocator {
    free_pools: Vec<Vec<MemoryRegion>>,
    active_regions: HashMap<usize, MemoryRegion>,
    next_region_id: AtomicUsize,
}

impl StagingAllocator {
    pub fn new(num_instances: usize) -> Self {
        let free_pools = (0..num_instances)
            .map(|_| Vec::new())
            .collect();

        Self {
            free_pools,
            active_regions: HashMap::new(),
            next_region_id: AtomicUsize::new(0),
        }
    }

    pub fn allocate(&mut self, instance_id: usize, bytes: usize) -> MemoryRegion {
        // Try to reuse from free pool first
        if let Some(region) = self.free_pools.get_mut(instance_id).and_then(|pool| pool.pop()) {
            if region.bytes >= bytes {
                self.active_regions.insert(region.device_ptr, region.clone());
                return region;
            }
            // Put back if too small and allocate new
            if let Some(pool) = self.free_pools.get_mut(instance_id) {
                pool.push(region);
            }
        }

        // Allocate new region (simplified)
        let region_id = self.next_region_id.fetch_add(1, Ordering::SeqCst);
        let region = MemoryRegion {
            device_ptr: 0x1000_0000 + (region_id * 0x1000),
            bytes: bytes.next_power_of_two(),
            instance_id,
        };

        self.active_regions.insert(region.device_ptr, region.clone());
        region
    }

    pub fn deallocate(&mut self, region: MemoryRegion) {
        if let Some(active) = self.active_regions.remove(&region.device_ptr) {
            if let Some(pool) = self.free_pools.get_mut(active.instance_id) {
                pool.push(active);
            }
        }
    }
}

/// Double-buffered prefetch for overlapping memory transfers with computation
#[derive(Debug)]
pub struct PrefetchBuffer {
    current_batch: Option<(Batch, Vec<InstanceAssignment>)>,
    next_batch: Option<(Batch, Vec<InstanceAssignment>)>,
    is_prefetching: AtomicBool,
}

impl PrefetchBuffer {
    pub fn new() -> Self {
        Self {
            current_batch: None,
            next_batch: None,
            is_prefetching: AtomicBool::new(false),
        }
    }

    pub fn set_current(&mut self, batch: Batch, assignments: Vec<InstanceAssignment>) {
        self.current_batch = Some((batch, assignments));
    }

    pub fn set_next(&mut self, batch: Batch, assignments: Vec<InstanceAssignment>) {
        self.next_batch = Some((batch, assignments));
    }

    pub fn swap_batches(&mut self) -> Option<(Batch, Vec<InstanceAssignment>)> {
        let current = self.current_batch.take();
        self.current_batch = self.next_batch.take();
        current
    }

    pub fn is_prefetching(&self) -> bool {
        self.is_prefetching.load(Ordering::Relaxed)
    }

    pub fn set_prefetching(&self, value: bool) {
        self.is_prefetching.store(value, Ordering::Relaxed);
    }
}

/// Main memory pipeline for managing data movement and execution
#[derive(Debug)]
pub struct MemoryPipeline {
    config: PipelineConfig,
    allocator: StagingAllocator,
    prefetch_buffer: PrefetchBuffer,
    active_transfers: HashMap<OperationId, Arc<AtomicBool>>,
}

impl MemoryPipeline {
    pub fn new(config: PipelineConfig, num_instances: usize) -> Self {
        Self {
            config,
            allocator: StagingAllocator::new(num_instances),
            prefetch_buffer: PrefetchBuffer::new(),
            active_transfers: HashMap::new(),
        }
    }

    /// Stage memory allocations for the next batch (prefetch)
    pub fn stage_next_batch(&mut self, batch: &Batch, assignments: &[InstanceAssignment]) {
        if !self.config.enable_prefetch {
            return;
        }

        // Allocate memory for next batch
        for assignment in assignments {
            for (op_id, _batch_idx) in &assignment.operations {
                let region = self.allocator.allocate(assignment.instance_id, 4096);
                self.active_transfers.insert(*op_id, Arc::new(AtomicBool::new(false)));
                // In real implementation, would initiate async transfer here
                let _ = region; // Suppress unused warning
            }
        }

        self.prefetch_buffer.set_next(batch.clone(), assignments.to_vec());
    }

    /// Execute the current batch on FPGA instances
    pub fn execute_on_instances(&mut self, batch: &Batch, assignments: Vec<InstanceAssignment>) {
        // Set current batch and start execution
        self.prefetch_buffer.set_current(batch.clone(), assignments.clone());

        for assignment in &assignments {
            for (op_id, _batch_idx) in &assignment.operations {
                if let Some(transfer_flag) = self.active_transfers.get(op_id) {
                    transfer_flag.store(true, Ordering::Relaxed);
                }
            }
        }
    }

    /// Collect completed operations from the pipeline
    pub fn collect_results(&mut self) -> Vec<CompletedOperation> {
        let mut completed = Vec::new();

        if let Some((batch, assignments)) = &self.prefetch_buffer.current_batch {
            for assignment in assignments {
                for (op_id, batch_idx) in &assignment.operations {
                    // Simulate completion check
                    if let Some(transfer_flag) = self.active_transfers.get(op_id) {
                        if transfer_flag.load(Ordering::Relaxed) {
                            completed.push(CompletedOperation {
                                operation_id: *op_id,
                                batch_index: *batch_idx,
                                success: true,
                                latency_us: 500,
                            });
                        }
                    }
                }
            }
            // Track batch operations for completion
            let _ = batch;
        }

        completed
    }

    /// Cleanup resources after batch completion
    pub fn cleanup_batch(&mut self, assignments: &[InstanceAssignment]) {
        for assignment in assignments {
            for (op_id, _) in &assignment.operations {
                self.active_transfers.remove(op_id);
            }

            // Return allocated regions to free pool
            let instance_id = assignment.instance_id;
            let regions_to_cleanup: Vec<_> = self.allocator.free_pools
                .get_mut(instance_id)
                .map(|pool| std::mem::take(pool))
                .unwrap_or_default();

            for region in regions_to_cleanup {
                self.allocator.deallocate(region);
            }
        }
    }
}

/// Result of a completed operation
#[derive(Debug, Clone)]
pub struct CompletedOperation {
    pub operation_id: OperationId,
    pub batch_index: usize,
    pub success: bool,
    pub latency_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(
        feature = "nvidia",
        not(feature = "amd"),
        not(feature = "intel"),
        not(feature = "tenstorrent")
    ))]
    use crate::r#impl::nvint4_tmatmul::Nvint4Launch;

    #[test]
    fn test_allocator_basic() {
        let mut allocator = StagingAllocator::new(4);

        let region1 = allocator.allocate(0, 1024);
        assert_eq!(region1.instance_id, 0);
        assert!(region1.bytes >= 1024);

        let region2 = allocator.allocate(1, 2048);
        assert_eq!(region2.instance_id, 1);

        allocator.deallocate(region1);
        assert_eq!(allocator.free_pools[0].len(), 1);
    }

    #[test]
    fn test_allocator_reuse() {
        let mut allocator = StagingAllocator::new(4);

        let region1 = allocator.allocate(0, 1024);
        allocator.deallocate(region1.clone());

        // Should reuse the same region
        let region2 = allocator.allocate(0, 512);
        assert_eq!(region1.device_ptr, region2.device_ptr);
        assert_eq!(allocator.free_pools[0].len(), 0);
    }

    #[test]
    fn test_prefetch_buffer() {
        let mut buffer = PrefetchBuffer::new();

        let op_id = OperationId::new();
        let assignment = InstanceAssignment {
            instance_id: 0,
            operations: vec![(op_id, 0)],
        };

        let batch = Batch {
            id: super::super::aggregator::BatchId::new(),
            operations: vec![],
            created_at: std::time::Instant::now(),
            size_bytes: 4096,
        };

        let assignments = vec![assignment.clone()];

        buffer.set_next(batch.clone(), assignments.clone());
        assert!(buffer.next_batch.is_some());

        buffer.swap_batches();
        assert!(buffer.current_batch.is_some());
        assert!(buffer.next_batch.is_none());
    }

    #[test]
    fn test_pipeline_execute() {
        let config = PipelineConfig::default();
        let mut pipeline = MemoryPipeline::new(config, 4);

        let op_id = OperationId::new();
        let assignment = InstanceAssignment {
            instance_id: 0,
            operations: vec![(op_id, 0)],
        };

        let batch = Batch {
            id: super::super::aggregator::BatchId::new(),
            operations: vec![],
            created_at: std::time::Instant::now(),
            size_bytes: 4096,
        };

        let assignments = vec![assignment.clone()];

        pipeline.execute_on_instances(&batch, assignments);
        let results = pipeline.collect_results();

        // Should have one completed operation
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].operation_id, op_id);
        assert!(results[0].success);
    }

    #[test]
    fn test_pipeline_prefetch_disabled() {
        let config = PipelineConfig {
            enable_prefetch: false,
            ..Default::default()
        };
        let mut pipeline = MemoryPipeline::new(config, 4);

        let op_id = OperationId::new();
        let assignment = InstanceAssignment {
            instance_id: 0,
            operations: vec![(op_id, 0)],
        };

        let batch = Batch {
            id: super::super::aggregator::BatchId::new(),
            operations: vec![],
            created_at: std::time::Instant::now(),
            size_bytes: 4096,
        };

        let assignments = vec![assignment];

        // Should not stage anything when prefetch disabled
        pipeline.stage_next_batch(&batch, &assignments);
        assert!(pipeline.prefetch_buffer.next_batch.is_none());
    }

    #[test]
    fn test_pipeline_cleanup() {
        let config = PipelineConfig::default();
        let mut pipeline = MemoryPipeline::new(config, 4);

        let op_id = OperationId::new();
        let assignment = InstanceAssignment {
            instance_id: 0,
            operations: vec![(op_id, 0)],
        };

        let batch = Batch {
            id: super::super::aggregator::BatchId::new(),
            operations: vec![],
            created_at: std::time::Instant::now(),
            size_bytes: 4096,
        };

        let assignments = vec![assignment.clone()];

        pipeline.execute_on_instances(&batch, assignments.clone());

        // Should have active transfers
        assert!(!pipeline.active_transfers.is_empty());

        pipeline.cleanup_batch(&assignments);

        // Should be cleaned up
        assert!(pipeline.active_transfers.is_empty());
    }
}