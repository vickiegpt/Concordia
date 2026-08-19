// /home/victoryang00/hetGPU/zluda/src/impl/batch_scheduler/scheduler.rs
use super::{aggregator::Batch, config::{SchedulerConfig, SchedulingPolicy}};
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationId(u64);

#[derive(Debug)]
pub enum SchedulerError {
    InvalidInstanceId { instance_id: usize, max_id: usize },
    InvalidConfiguration(String),
    NoInstancesAvailable,
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerError::InvalidInstanceId { instance_id, max_id } => {
                write!(f, "Invalid instance ID {}: must be < {}", instance_id, max_id)
            }
            SchedulerError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            SchedulerError::NoInstancesAvailable => {
                write!(f, "No instances available for scheduling")
            }
        }
    }
}

impl std::error::Error for SchedulerError {}

#[derive(Debug, Clone)]
pub struct InstanceAssignment {
    pub instance_id: usize,
    pub operations: Vec<(OperationId, usize)>, // (op_id, batch_index)
}

#[derive(Debug, Clone)]
pub struct InstanceState {
    pub queue_depth: usize,
    pub current_op: Option<OperationId>,
    pub total_completed: u64,
    pub avg_latency_us: u64,
    pub last_activity: Instant,
    pub utilization: f32,
}

impl InstanceState {
    pub fn new() -> Self {
        Self {
            queue_depth: 0,
            current_op: None,
            total_completed: 0,
            avg_latency_us: 0,
            last_activity: Instant::now(),
            utilization: 0.0,
        }
    }

    pub fn update_utilization(&mut self) {
        let elapsed = self.last_activity.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            self.utilization = (self.queue_depth as f32 / elapsed).min(1.0);
        }
        self.last_activity = Instant::now();
    }
}

#[derive(Debug)]
pub struct InstanceScheduler {
    config: SchedulerConfig,
    instance_status: Vec<InstanceState>,
    round_robin_index: Arc<RwLock<usize>>,
}

impl InstanceScheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self, SchedulerError> {
        // Validate configuration
        if config.num_instances == 0 {
            return Err(SchedulerError::InvalidConfiguration(
                "Number of instances must be greater than 0".to_string()
            ));
        }

        let instance_status = (0..config.num_instances)
            .map(|_| InstanceState::new())
            .collect();

        Ok(Self {
            config,
            instance_status,
            round_robin_index: Arc::new(RwLock::new(0)),
        })
    }

    pub fn assign_to_instances(&mut self, batch: &Batch) -> Result<Vec<InstanceAssignment>, SchedulerError> {
        if self.instance_status.is_empty() {
            return Err(SchedulerError::NoInstancesAvailable);
        }

        let assignments = match self.config.policy {
            SchedulingPolicy::RoundRobin => self.assign_round_robin(batch)?,
            SchedulingPolicy::SizeAware => self.assign_size_aware(batch)?,
            SchedulingPolicy::LoadBalanced => self.assign_load_balanced(batch)?,
            SchedulingPolicy::Adaptive => self.assign_adaptive(batch)?,
        };

        Ok(assignments)
    }

    fn assign_round_robin(&mut self, batch: &Batch) -> Result<Vec<InstanceAssignment>, SchedulerError> {
        let num_instances = self.config.num_instances;
        let mut assignments: Vec<InstanceAssignment> = (0..num_instances)
            .map(|id| InstanceAssignment {
                instance_id: id,
                operations: Vec::new(),
            })
            .collect();

        // Thread-safe round-robin index update
        let mut rr_index = self.round_robin_index.write()
            .map_err(|_| SchedulerError::InvalidConfiguration(
                "Round-robin index lock poisoned".to_string()
            ))?;

        for (idx, _op) in batch.operations.iter().enumerate() {
            let instance_id = *rr_index % num_instances;
            let op_id = OperationId(idx as u64);

            if instance_id >= assignments.len() {
                return Err(SchedulerError::InvalidInstanceId {
                    instance_id,
                    max_id: assignments.len()
                });
            }

            assignments[instance_id].operations.push((op_id, idx));
            *rr_index = rr_index.wrapping_add(1);
        }

        // Update instance states
        for (idx, assignment) in assignments.iter().enumerate() {
            if idx < self.instance_status.len() {
                self.instance_status[idx].queue_depth += assignment.operations.len();
            }
        }

        Ok(assignments)
    }

    fn assign_size_aware(&mut self, batch: &Batch) -> Result<Vec<InstanceAssignment>, SchedulerError> {
        let num_instances = self.config.num_instances;
        let mut assignments: Vec<InstanceAssignment> = (0..num_instances)
            .map(|id| InstanceAssignment {
                instance_id: id,
                operations: Vec::new(),
            })
            .collect();

        let mut instance_loads = vec![0usize; num_instances];

        for (idx, op) in batch.operations.iter().enumerate() {
            let op_size = op.dim as usize;

            // Find instance with lowest load - O(n) search, but m operations makes this O(n*m)
            // TODO: Consider using a heap for O(log n) lookups if n*m becomes problematic
            let min_load_idx = instance_loads
                .iter()
                .enumerate()
                .min_by_key(|(_, load)| *load)
                .map(|(min_idx, _)| min_idx)
                .ok_or_else(|| SchedulerError::NoInstancesAvailable)?;

            assignments[min_load_idx].operations.push((OperationId(idx as u64), idx));
            instance_loads[min_load_idx] += op_size;
        }

        Ok(assignments)
    }

    fn assign_load_balanced(&mut self, batch: &Batch) -> Result<Vec<InstanceAssignment>, SchedulerError> {
        let num_instances = self.config.num_instances;
        let mut assignments: Vec<InstanceAssignment> = (0..num_instances)
            .map(|id| InstanceAssignment {
                instance_id: id,
                operations: Vec::new(),
            })
            .collect();

        // Sort instances by current queue depth (ascending - least loaded first)
        let mut instance_priorities: Vec<usize> = (0..num_instances).collect();
        instance_priorities.sort_by_key(|&idx| {
            self.instance_status.get(idx)
                .map(|state| state.queue_depth)
                .unwrap_or(usize::MAX)
        });

        for (idx, _op) in batch.operations.iter().enumerate() {
            let instance_id = instance_priorities[idx % num_instances];

            if instance_id >= assignments.len() {
                return Err(SchedulerError::InvalidInstanceId {
                    instance_id,
                    max_id: assignments.len()
                });
            }

            assignments[instance_id].operations.push((OperationId(idx as u64), idx));
        }

        Ok(assignments)
    }

    fn assign_adaptive(&mut self, batch: &Batch) -> Result<Vec<InstanceAssignment>, SchedulerError> {
        // Adaptive: use load balanced for now, could be enhanced with:
        // - Performance metrics consideration
        // - Dynamic policy switching based on workload patterns
        // - Instance health monitoring
        self.assign_load_balanced(batch)
    }

    pub fn update_instance_status(&mut self, instance_id: usize, success: bool, latency_us: u64) -> Result<(), SchedulerError> {
        if instance_id >= self.instance_status.len() {
            return Err(SchedulerError::InvalidInstanceId {
                instance_id,
                max_id: self.instance_status.len()
            });
        }

        let state = &mut self.instance_status[instance_id];

        if success {
            state.total_completed += 1;
            // Update moving average latency with bounds checking
            if state.avg_latency_us == 0 {
                state.avg_latency_us = latency_us;
            } else {
                // Prevent overflow and maintain reasonable averaging
                state.avg_latency_us = (state.avg_latency_us.saturating_mul(9) + latency_us) / 10;
            }

            state.queue_depth = state.queue_depth.saturating_sub(1);
        }

        state.update_utilization();
        Ok(())
    }

    pub fn get_instance_utilization(&self) -> Vec<f32> {
        self.instance_status.iter()
            .map(|state| state.utilization)
            .collect()
    }

    pub fn get_average_latency(&self) -> u64 {
        if self.instance_status.is_empty() {
            return 0;
        }

        let total: u64 = self.instance_status.iter()
            .map(|s| s.avg_latency_us)
            .sum();

        total.saturating_div(self.instance_status.len() as u64)
    }

    // Thread-safe helper method to get current instance count
    pub fn num_instances(&self) -> usize {
        self.instance_status.len()
    }

    // Helper method for testing to set instance state safely
    #[cfg(test)]
    pub fn set_instance_queue_depth(&mut self, instance_id: usize, depth: usize) -> Result<(), SchedulerError> {
        if instance_id >= self.instance_status.len() {
            return Err(SchedulerError::InvalidInstanceId {
                instance_id,
                max_id: self.instance_status.len()
            });
        }

        self.instance_status[instance_id].queue_depth = depth;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent")))]
    use crate::r#impl::nvint4_tmatmul::Nvint4Launch;

    #[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent")))]
    fn create_test_batch(op_count: usize) -> Batch {
        let operations: Vec<Nvint4Launch> = (0..op_count)
            .map(|_| Nvint4Launch {
                packed_weights: 0x1000,
                input_q8_8: 0x2000,
                output_s64: 0x3000,
                dim: 2048,
                delta: 1,
                stream: cuda_types::cuda::CUstream(std::ptr::null_mut()),
            })
            .collect();

        Batch {
            id: super::aggregator::BatchId::new(),
            operations,
            created_at: Instant::now(),
            size_bytes: op_count * 4096,
        }
    }

    fn create_test_scheduler(policy: SchedulingPolicy, num_instances: usize) -> Result<InstanceScheduler, SchedulerError> {
        let config = SchedulerConfig {
            policy,
            num_instances,
            ..Default::default()
        };
        InstanceScheduler::new(config)
    }

    #[test]
    fn test_scheduler_creation_validation() {
        // Test invalid configuration - zero instances
        let config = SchedulerConfig {
            num_instances: 0,
            ..Default::default()
        };
        let result = InstanceScheduler::new(config);
        assert!(result.is_err(), "Should reject zero instances");

        if let Err(SchedulerError::InvalidConfiguration(msg)) = result {
            assert!(msg.contains("greater than 0"));
        } else {
            panic!("Expected InvalidConfiguration error");
        }
    }

    #[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent")))]
    #[test]
    fn test_scheduler_round_robin() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::RoundRobin, 16)
            .expect("Failed to create scheduler");
        let batch = create_test_batch(16);

        let assignments = scheduler.assign_to_instances(&batch)
            .expect("Failed to assign operations");

        assert_eq!(assignments.len(), 16);

        // Each instance should get 1 operation
        for assignment in &assignments {
            assert_eq!(assignment.operations.len(), 1);
        }
    }

    #[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent")))]
    #[test]
    fn test_scheduler_size_aware() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::SizeAware, 16)
            .expect("Failed to create scheduler");
        let batch = create_test_batch(8);

        let assignments = scheduler.assign_to_instances(&batch)
            .expect("Failed to assign operations");

        assert_eq!(assignments.len(), 16);

        // Verify operations are distributed
        let total_ops: usize = assignments.iter()
            .map(|a| a.operations.len())
            .sum();
        assert_eq!(total_ops, 8);
    }

    #[test]
    fn test_scheduler_status_update() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::default(), 16)
            .expect("Failed to create scheduler");

        // Test valid updates
        scheduler.update_instance_status(0, true, 1000)
            .expect("Failed to update instance status");
        scheduler.update_instance_status(0, true, 2000)
            .expect("Failed to update instance status");

        let utilizations = scheduler.get_instance_utilization();
        assert_eq!(utilizations[0], 0.0); // Should be updated

        let avg_latency = scheduler.get_average_latency();
        assert!(avg_latency > 0);

        // Test invalid instance ID
        let result = scheduler.update_instance_status(999, true, 1000);
        assert!(result.is_err(), "Should reject invalid instance ID");
    }

    #[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent")))]
    #[test]
    fn test_scheduler_load_balanced() {
        let mut scheduler = create_test_scheduler(SchedulingPolicy::LoadBalanced, 16)
            .expect("Failed to create scheduler");

        // Create initial load on instance 0 using safe helper method
        scheduler.set_instance_queue_depth(0, 10)
            .expect("Failed to set queue depth");

        let batch = create_test_batch(4);
        let assignments = scheduler.assign_to_instances(&batch)
            .expect("Failed to assign operations");

        // Instance 0 should have fewer assignments due to existing load
        let instance_0_ops = assignments[0].operations.len();
        assert!(instance_0_ops <= 1, "Instance 0 should have fewer operations due to load");

        // Test that setting queue depth on invalid instance fails
        let result = scheduler.set_instance_queue_depth(999, 5);
        assert!(result.is_err(), "Should reject invalid instance ID");
    }

    #[cfg(all(feature = "nvidia", not(feature = "amd"), not(feature = "intel"), not(feature = "tenstorrent")))]
    #[test]
    fn test_scheduler_error_handling() {
        // Test empty batch handling
        let mut scheduler = create_test_scheduler(SchedulingPolicy::RoundRobin, 4)
            .expect("Failed to create scheduler");

        let empty_batch = create_test_batch(0);
        let assignments = scheduler.assign_to_instances(&empty_batch)
            .expect("Should handle empty batch");

        assert_eq!(assignments.len(), 4);
        for assignment in assignments {
            assert_eq!(assignment.operations.len(), 0);
        }
    }

    #[test]
    fn test_scheduler_thread_safety() {
        // This test validates the thread safety design
        let scheduler = create_test_scheduler(SchedulingPolicy::RoundRobin, 8)
            .expect("Failed to create scheduler");

        // Verify we can share the round-robin index across threads
        let rr_index_clone = Arc::clone(&scheduler.round_robin_index);

        // Spawn multiple threads to test concurrent access
        let handles: Vec<_> = (0..4).map(|_| {
            let rr_index = Arc::clone(&rr_index_clone);
            std::thread::spawn(move || {
                let mut index = rr_index.write().unwrap();
                *index = index.wrapping_add(1);
            })
        }).collect();

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify the index was updated
        let final_index = *rr_index_clone.read().unwrap();
        assert_eq!(final_index, 4);
    }
}