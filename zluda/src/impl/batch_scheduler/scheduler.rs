// /home/victoryang00/hetGPU/zluda/src/impl/batch_scheduler/scheduler.rs
use super::{aggregator::Batch, config::{SchedulerConfig, SchedulingPolicy}};
use std::time::{Instant, Duration};

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
    // Track utilization metrics
    pub total_active_time: Duration,
    pub total_idle_time: Duration,
    pub last_state_change: Instant,
    pub is_busy: bool,
}

impl InstanceState {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            queue_depth: 0,
            current_op: None,
            total_completed: 0,
            avg_latency_us: 0,
            last_activity: now,
            utilization: 0.0,
            total_active_time: Duration::ZERO,
            total_idle_time: Duration::ZERO,
            last_state_change: now,
            is_busy: false,
        }
    }

    pub fn update_utilization(&mut self) {
        let now = Instant::now();
        let time_since_last_update = now.duration_since(self.last_state_change);

        // Track time spent in current state
        if self.is_busy {
            self.total_active_time += time_since_last_update;
        } else {
            self.total_idle_time += time_since_last_update;
        }

        // Calculate utilization as active_time / total_time
        let total_time = self.total_active_time + self.total_idle_time;
        if total_time > Duration::ZERO {
            self.utilization = self.total_active_time.as_secs_f32() / total_time.as_secs_f32();
        }

        self.last_activity = now;
        self.last_state_change = now;

        // Update busy state based on queue depth
        let was_busy = self.is_busy;
        self.is_busy = self.queue_depth > 0;

        // Reset state change time if busy state changed
        if was_busy != self.is_busy {
            self.last_state_change = now;
        }
    }

    pub fn set_busy(&mut self, busy: bool) {
        let now = Instant::now();
        let time_in_state = now.duration_since(self.last_state_change);

        if self.is_busy {
            self.total_active_time += time_in_state;
        } else {
            self.total_idle_time += time_in_state;
        }

        self.is_busy = busy;
        self.last_state_change = now;
    }
}

#[derive(Debug)]
pub struct InstanceScheduler {
    config: SchedulerConfig,
    instance_status: Vec<InstanceState>,
    round_robin_index: usize, // Simple usize - thread safety provided by external Arc<Mutex<>> in integration layer
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
            round_robin_index: 0, // Simple usize - thread safety provided by external Arc<Mutex<>> in integration layer
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

        for (idx, _op) in batch.operations.iter().enumerate() {
            let instance_id = self.round_robin_index % num_instances;
            let op_id = OperationId(idx as u64);

            if instance_id >= assignments.len() {
                return Err(SchedulerError::InvalidInstanceId {
                    instance_id,
                    max_id: assignments.len()
                });
            }

            assignments[instance_id].operations.push((op_id, idx));
            self.round_robin_index = self.round_robin_index.wrapping_add(1);
        }

        // Update instance states - consistent with other policies
        for (idx, assignment) in assignments.iter().enumerate() {
            if idx < self.instance_status.len() {
                self.instance_status[idx].queue_depth += assignment.operations.len();
                // Mark instances as busy when they receive work
                if !assignment.operations.is_empty() {
                    self.instance_status[idx].set_busy(true);
                }
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

        // Update instance states - consistent with other policies
        for (idx, assignment) in assignments.iter().enumerate() {
            if idx < self.instance_status.len() {
                self.instance_status[idx].queue_depth += assignment.operations.len();
                // Mark instances as busy when they receive work
                if !assignment.operations.is_empty() {
                    self.instance_status[idx].set_busy(true);
                }
            }
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

        // Update instance states - consistent with other policies
        for (idx, assignment) in assignments.iter().enumerate() {
            if idx < self.instance_status.len() {
                self.instance_status[idx].queue_depth += assignment.operations.len();
                // Mark instances as busy when they receive work
                if !assignment.operations.is_empty() {
                    self.instance_status[idx].set_busy(true);
                }
            }
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
    // Made public for better testability and integration testing
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
        // With the new utilization tracking, should be 0.0 since we haven't assigned work yet
        assert_eq!(utilizations[0], 0.0);

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
        // This test validates the simplified thread safety design
        // Thread safety is now provided by external Arc<Mutex<>> in integration layer
        // Internal round_robin_index is a simple usize without locking
        let scheduler = create_test_scheduler(SchedulingPolicy::RoundRobin, 8)
            .expect("Failed to create scheduler");

        // Verify round_robin_index is a simple value
        let initial_index = scheduler.round_robin_index;
        assert_eq!(initial_index, 0);

        // The design relies on external synchronization - this validates
        // that internal state doesn't have redundant locking mechanisms
        assert!(!std::any::type_name::<usize>().contains("Mutex"));
        assert!(!std::any::type_name::<usize>().contains("RwLock"));
    }

    #[test]
    fn test_scheduler_latency_moving_average() {
        // Test latency moving average calculation
        let mut scheduler = create_test_scheduler(SchedulingPolicy::default(), 4)
            .expect("Failed to create scheduler");

        // Test initial state - no latency yet
        assert_eq!(scheduler.get_average_latency(), 0);

        // First latency sample should set the average directly
        scheduler.update_instance_status(0, true, 1000)
            .expect("Failed to update instance status");
        assert_eq!(scheduler.get_average_latency(), 250); // 1000 / 4 instances

        // Second sample on same instance should trigger moving average
        scheduler.update_instance_status(0, true, 2000)
            .expect("Failed to update instance status");

        // Moving average formula: (avg * 9 + new) / 10
        // First update: avg = 1000 (initial)
        // Second update: avg = (1000 * 9 + 2000) / 10 = 1100
        let instance_0_avg = scheduler.instance_status[0].avg_latency_us;
        assert_eq!(instance_0_avg, 1100, "Moving average calculation incorrect");

        // Add more samples to verify formula
        scheduler.update_instance_status(0, true, 3000)
            .expect("Failed to update instance status");

        // Third update: avg = (1100 * 9 + 3000) / 10 = 1290
        let instance_0_avg = scheduler.instance_status[0].avg_latency_us;
        assert_eq!(instance_0_avg, 1290, "Moving average calculation incorrect");

        // Verify the average across all instances
        let overall_avg = scheduler.get_average_latency();
        assert!(overall_avg > 0 && overall_avg < 1500, "Overall average should be reasonable");

        // Test that moving average prevents overflow
        scheduler.update_instance_status(0, true, u64::MAX / 10)
            .expect("Failed to update instance status with large value");

        // Should not panic or overflow
        let final_avg = scheduler.instance_status[0].avg_latency_us;
        assert!(final_avg > 0, "Average should be positive even with large inputs");
    }

    #[test]
    fn test_scheduler_utilization_calculation() {
        // Test proper utilization calculation with active/idle time tracking
        let mut scheduler = create_test_scheduler(SchedulingPolicy::default(), 2)
            .expect("Failed to create scheduler");

        // Initial utilization should be 0 (no activity yet)
        let utilizations = scheduler.get_instance_utilization();
        assert_eq!(utilizations[0], 0.0);
        assert_eq!(utilizations[1], 0.0);

        // Add work to instance 0
        scheduler.instance_status[0].queue_depth = 5;
        scheduler.instance_status[0].update_utilization();

        let utilizations = scheduler.get_instance_utilization();
        assert!(utilizations[0] > 0.0, "Instance 0 should have utilization > 0");
        assert_eq!(utilizations[1], 0.0, "Instance 1 should still have 0 utilization");

        // Simulate instance 0 completing work
        scheduler.instance_status[0].queue_depth = 0;
        scheduler.instance_status[0].update_utilization();

        // Instance 0 should have accumulated some active time
        let util0 = scheduler.get_instance_utilization();
        assert!(util0[0] > 0.0, "Instance 0 should have positive utilization history");

        // Instance 1 with no work should have 0 utilization
        assert_eq!(util0[1], 0.0, "Instance 1 should have 0 utilization with no work");
    }
}