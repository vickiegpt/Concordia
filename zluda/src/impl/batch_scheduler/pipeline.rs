//! Memory Pipeline implementation with double buffering and prefetching
//!
//! This module provides the memory management infrastructure for the batch scheduler,
//! including staging allocation, prefetch buffering, and concurrent transfer management.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use super::{scheduler::InstanceAssignment, aggregator::Batch, config::PipelineConfig};

/// A memory region allocated on a specific FPGA instance
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Device pointer to the allocated memory
    pub device_ptr: u64,
    /// Size of the region in bytes
    pub bytes: usize,
    /// FPGA instance ID where this region is allocated
    pub instance_id: u32,
    /// Unique region identifier for tracking
    pub region_id: u64,
}

/// Staging allocator with per-instance memory pools for efficient reuse
pub struct StagingAllocator {
    /// Free memory pools per FPGA instance (instance_id -> vec of reusable regions)
    free_pools: Vec<Vec<MemoryRegion>>,
    /// Currently active regions (region_id -> region)
    active_regions: HashMap<u64, MemoryRegion>,
    /// Next unique region ID
    next_region_id: AtomicU64,
}

impl StagingAllocator {
    /// Create a new staging allocator with per-instance free pools
    pub fn new(num_instances: usize) -> Self {
        let mut free_pools = Vec::with_capacity(num_instances);
        for _ in 0..num_instances {
            free_pools.push(Vec::new());
        }

        Self {
            free_pools,
            active_regions: HashMap::new(),
            next_region_id: AtomicU64::new(1),
        }
    }

    /// Allocate memory for a specific instance, reusing from free pool if available
    pub fn allocate(&mut self, instance_id: u32, bytes: usize) -> MemoryRegion {
        let instance_idx = instance_id as usize;

        // Try to reuse from free pool
        if let Some(free_pool) = self.free_pools.get_mut(instance_idx) {
            // Find a suitable region (first-fit strategy)
            if let Some(pos) = free_pool.iter().position(|region| region.bytes >= bytes) {
                let mut region = free_pool.remove(pos);
                region.region_id = self.next_region_id.fetch_add(1, Ordering::SeqCst);
                self.active_regions.insert(region.region_id, region.clone());
                return region;
            }
        }

        // Allocate new region
        let region_id = self.next_region_id.fetch_add(1, Ordering::SeqCst);
        let region = MemoryRegion {
            device_ptr: 0x1000 + (region_id * 0x1000), // Simulated device pointer
            bytes,
            instance_id,
            region_id,
        };

        self.active_regions.insert(region_id, region.clone());
        region
    }

    /// Deallocate a region and return it to the appropriate free pool
    pub fn deallocate(&mut self, region: MemoryRegion) {
        // Remove from active regions
        self.active_regions.remove(&region.region_id);

        // Return to appropriate free pool
        let instance_idx = region.instance_id as usize;
        if let Some(free_pool) = self.free_pools.get_mut(instance_idx) {
            free_pool.push(region);
        }
    }

    /// Get the number of active regions
    pub fn active_count(&self) -> usize {
        self.active_regions.len()
    }

    /// Get the total size of free pools per instance
    pub fn free_pool_sizes(&self) -> Vec<usize> {
        self.free_pools.iter().map(|pool| pool.len()).collect()
    }
}

/// Double-buffered prefetch buffer for concurrent execution and prefetching
pub struct PrefetchBuffer {
    /// Current batch being executed
    current_batch: Option<(Batch, Vec<InstanceAssignment>)>,
    /// Next batch being prefetched
    next_batch: Option<(Batch, Vec<InstanceAssignment>)>,
    /// Whether prefetching is in progress
    is_prefetching: Arc<AtomicBool>,
}

impl PrefetchBuffer {
    /// Create a new empty prefetch buffer
    pub fn new() -> Self {
        Self {
            current_batch: None,
            next_batch: None,
            is_prefetching: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Set the current processing batch
    pub fn set_current(&mut self, batch: Batch, assignments: Vec<InstanceAssignment>) {
        self.current_batch = Some((batch, assignments));
    }

    /// Set the next batch for prefetching
    pub fn set_next(&mut self, batch: Batch, assignments: Vec<InstanceAssignment>) {
        self.next_batch = Some((batch, assignments));
    }

    /// Exchange current and next batches (double buffering)
    pub fn swap_batches(&mut self) {
        std::mem::swap(&mut self.current_batch, &mut self.next_batch);
    }

    /// Check if prefetching is in progress
    pub fn is_prefetching(&self) -> bool {
        self.is_prefetching.load(Ordering::Relaxed)
    }

    /// Set prefetching state
    pub fn set_prefetching(&self, value: bool) {
        self.is_prefetching.store(value, Ordering::Relaxed);
    }

    /// Get reference to current batch
    pub fn get_current(&self) -> Option<&(Batch, Vec<InstanceAssignment>)> {
        self.current_batch.as_ref()
    }

    /// Get reference to next batch
    pub fn get_next(&self) -> Option<&(Batch, Vec<InstanceAssignment>)> {
        self.next_batch.as_ref()
    }

    /// Clear both buffers
    pub fn clear(&mut self) {
        self.current_batch = None;
        self.next_batch = None;
        self.set_prefetching(false);
    }
}

/// Result of a completed operation
#[derive(Debug, Clone)]
pub struct CompletedOperation {
    /// Unique operation identifier
    pub operation_id: u64,
    /// Index in the batch
    pub batch_index: usize,
    /// Whether the operation succeeded
    pub success: bool,
    /// Execution latency in microseconds
    pub latency_us: u64,
}

/// Main memory pipeline orchestrating allocation, prefetching, and execution
pub struct MemoryPipeline {
    /// Pipeline configuration
    config: PipelineConfig,
    /// Staging allocator for memory management
    allocator: StagingAllocator,
    /// Prefetch buffer for double buffering
    prefetch_buffer: PrefetchBuffer,
    /// Active memory transfers (transfer_id -> metadata)
    active_transfers: HashMap<u64, (u32, usize)>, // (instance_id, batch_index)
    /// Next transfer ID
    next_transfer_id: AtomicU64,
}

impl MemoryPipeline {
    /// Create a new memory pipeline with given configuration
    pub fn new(config: PipelineConfig, num_instances: usize) -> Self {
        Self {
            config,
            allocator: StagingAllocator::new(num_instances),
            prefetch_buffer: PrefetchBuffer::new(),
            active_transfers: HashMap::new(),
            next_transfer_id: AtomicU64::new(1),
        }
    }

    /// Stage the next batch for prefetching (allocate memory and start transfers)
    pub fn stage_next_batch(&mut self, batch: &Batch, assignments: &[InstanceAssignment]) -> bool {
        if self.prefetch_buffer.is_prefetching() {
            return false; // Already prefetching
        }

        // Allocate memory for each assignment
        for assignment in assignments {
            let region = self.allocator.allocate(
                assignment.instance_id as u32,
                batch.size_bytes
            );

            // Simulate memory transfer start
            let transfer_id = self.next_transfer_id.fetch_add(1, Ordering::SeqCst);

            // Extract the batch index from the first operation in the assignment
            let batch_index = assignment.operations.first()
                .map(|(_, idx)| *idx)
                .unwrap_or(0);

            self.active_transfers.insert(transfer_id, (assignment.instance_id as u32, batch_index));

            // Track the region for cleanup
            let _ = region;
        }

        // Store in prefetch buffer
        self.prefetch_buffer.set_next(batch.clone(), assignments.to_vec());
        self.prefetch_buffer.set_prefetching(true);

        true
    }

    /// Execute the current batch on FPGA instances
    pub fn execute_on_instances(&mut self, batch: &Batch, assignments: &[InstanceAssignment]) -> Vec<CompletedOperation> {
        let mut results = Vec::new();

        for assignment in assignments {
            // Process each operation in the assignment
            for (operation_id, batch_index) in &assignment.operations {
                // Simulate execution latency based on batch size
                let latency_us = (batch.size_bytes as f64 / 1024.0 * 2.0) as u64; // 2us per KB

                let operation = CompletedOperation {
                    operation_id: operation_id.0, // Extract the u64 from OperationId
                    batch_index: *batch_index,
                    success: latency_us < 10000, // Default timeout of 10ms
                    latency_us,
                };

                results.push(operation);
            }
        }

        // Update current buffer
        self.prefetch_buffer.set_current(batch.clone(), assignments.to_vec());

        results
    }

    /// Collect results from completed operations
    pub fn collect_results(&mut self) -> Vec<CompletedOperation> {
        let mut results = Vec::new();

        // Simulate collecting from active transfers
        let transfer_ids: Vec<u64> = self.active_transfers.keys().cloned().collect();
        for transfer_id in transfer_ids {
            if let Some((instance_id, batch_index)) = self.active_transfers.remove(&transfer_id) {
                let operation = CompletedOperation {
                    operation_id: transfer_id,
                    batch_index,
                    success: true,
                    latency_us: 100, // Simulated transfer completion
                };
                results.push(operation);
                let _ = instance_id; // Suppress unused warning
            }
        }

        // Clear prefetching flag if no active transfers
        if self.active_transfers.is_empty() {
            self.prefetch_buffer.set_prefetching(false);
        }

        results
    }

    /// Clean up resources after batch completion
    pub fn cleanup_batch(&mut self, assignments: &[InstanceAssignment]) {
        // In a real implementation, we would deallocate regions here
        // For now, we keep regions in the free pool for reuse

        // Clear completed transfers
        self.active_transfers.retain(|_, _| false);
    }

    /// Swap buffers and start processing next batch
    pub fn swap_to_next_batch(&mut self) {
        self.prefetch_buffer.swap_batches();
    }

    /// Get reference to prefetch buffer
    pub fn get_prefetch_buffer(&self) -> &PrefetchBuffer {
        &self.prefetch_buffer
    }

    /// Get mutable reference to allocator
    pub fn get_allocator(&self) -> &StagingAllocator {
        &self.allocator
    }

    /// Get current number of active transfers
    pub fn active_transfer_count(&self) -> usize {
        self.active_transfers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::impl::batch_scheduler::aggregator::{Batch, BatchId};
    use crate::impl::batch_scheduler::config::PipelineConfig;

    fn create_test_config() -> PipelineConfig {
        PipelineConfig {
            enable_double_buffer: true,
            enable_prefetch: true,
            max_concurrent_transfers: 8,
        }
    }

    fn create_test_batch(size: usize) -> Batch {
        Batch {
            id: BatchId::new(),
            operations: vec![],
            created_at: std::time::Instant::now(),
            size_bytes: size * 1024,
        }
    }

    fn create_test_assignments(count: usize, instance_count: usize) -> Vec<InstanceAssignment> {
        use crate::impl::batch_scheduler::scheduler::OperationId;

        (0..count).map(|i| InstanceAssignment {
            instance_id: i % instance_count,
            operations: vec![(OperationId::new(), i)],
        }).collect()
    }

    #[test]
    fn test_allocator_basic() {
        let mut allocator = StagingAllocator::new(4);

        // Test initial allocation
        let region1 = allocator.allocate(0, 1024);
        assert_eq!(region1.instance_id, 0);
        assert_eq!(region1.bytes, 1024);
        assert_eq!(allocator.active_count(), 1);

        // Test allocation for different instance
        let region2 = allocator.allocate(2, 2048);
        assert_eq!(region2.instance_id, 2);
        assert_eq!(region2.bytes, 2048);
        assert_eq!(allocator.active_count(), 2);

        // Test deallocation and reuse
        allocator.deallocate(region1);
        assert_eq!(allocator.active_count(), 1);
        assert_eq!(allocator.free_pool_sizes()[0], 1); // Instance 0 has 1 free region

        // Allocate again - should reuse the deallocated region
        let region3 = allocator.allocate(0, 512);
        assert_eq!(region3.instance_id, 0);
        assert_eq!(allocator.active_count(), 2);
        assert_eq!(allocator.free_pool_sizes()[0], 0); // Region was reused

        // Verify region IDs are unique
        assert_ne!(region1.region_id, region2.region_id);
        assert_ne!(region1.region_id, region3.region_id);
    }

    #[test]
    fn test_prefetch_buffer() {
        let mut buffer = PrefetchBuffer::new();

        // Test initial state
        assert!(buffer.get_current().is_none());
        assert!(buffer.get_next().is_none());
        assert!(!buffer.is_prefetching());

        // Test setting batches
        let batch = create_test_batch(3);
        let assignments = create_test_assignments(3, 4);

        buffer.set_current(batch.clone(), assignments.clone());
        assert!(buffer.get_current().is_some());
        assert!(!buffer.is_prefetching());

        buffer.set_next(batch.clone(), assignments.clone());
        assert!(buffer.get_next().is_some());

        // Test prefetching flag
        buffer.set_prefetching(true);
        assert!(buffer.is_prefetching());

        // Test buffer swap
        buffer.swap_batches();
        let current = buffer.get_current();
        let next = buffer.get_next();

        assert!(current.is_some());
        assert!(next.is_some());

        // After swap, previous next should be current
        assert_eq!(current.unwrap().0.id, batch.id);

        // Test clear
        buffer.clear();
        assert!(buffer.get_current().is_none());
        assert!(buffer.get_next().is_none());
        assert!(!buffer.is_prefetching());
    }

    #[test]
    fn test_pipeline_execute() {
        let config = create_test_config();
        let mut pipeline = MemoryPipeline::new(config, 4);

        // Create test data
        let batch = create_test_batch(4);
        let assignments = create_test_assignments(4, 4);

        // Test staging next batch
        let staged = pipeline.stage_next_batch(&batch, &assignments);
        assert!(staged);
        assert!(pipeline.get_prefetch_buffer().is_prefetching());
        assert_eq!(pipeline.active_transfer_count(), 4);

        // Test execution
        let results = pipeline.execute_on_instances(&batch, &assignments);
        assert_eq!(results.len(), 4);

        for result in results {
            assert!(result.success);
            assert!(result.latency_us > 0);
            assert!(result.latency_us < 10000); // Less than default timeout
        }

        // Test result collection
        let completed = pipeline.collect_results();
        assert_eq!(completed.len(), 4);

        // Test cleanup
        pipeline.cleanup_batch(&assignments);
        assert_eq!(pipeline.active_transfer_count(), 0);

        // Test buffer swap
        pipeline.swap_to_next_batch();
        assert!(pipeline.get_prefetch_buffer().get_current().is_some());

        // Verify allocator state
        let allocator = pipeline.get_allocator();
        assert_eq!(allocator.active_count(), 4); // Regions should still be allocated
    }
}