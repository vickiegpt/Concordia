// /home/victoryang00/hetGPU/zluda/src/impl/batch_scheduler/error_handling.rs
use std::collections::HashMap;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureType {
    TransientTimeout,
    DmaError,
    HardwareFault,
    Corruption,
}

#[derive(Debug, Clone)]
pub enum RecoveryAction {
    Retry { max_attempts: u32, backoff_ms: u32 },
    FallbackToGPU,
    SkipAndLog,
    AbortRequest,
}

#[derive(Debug, Clone)]
pub struct InstanceHealth {
    pub success_count: u64,
    pub failure_count: u64,
    pub last_error: Option<Instant>,
    pub health_score: f32,
    pub error_types: HashMap<FailureType, u64>,
}

impl InstanceHealth {
    pub fn new() -> Self {
        Self {
            success_count: 0,
            failure_count: 0,
            last_error: None,
            health_score: 1.0,
            error_types: HashMap::new(),
        }
    }

    pub fn record_success(&mut self) {
        self.success_count += 1;
        self.update_health_score();
    }

    pub fn record_failure(&mut self, error_type: FailureType) {
        self.failure_count += 1;
        self.last_error = Some(Instant::now());
        *self.error_types.entry(error_type).or_insert(0) += 1;
        self.update_health_score();
    }

    fn update_health_score(&mut self) {
        let total = self.success_count + self.failure_count;
        if total > 0 {
            self.health_score = self.success_count as f32 / total as f32;
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.health_score > 0.5
    }

    pub fn should_blacklist(&self) -> bool {
        // Blacklist if health score is very low
        self.health_score < 0.1 || self.failure_count > 10
    }
}

#[derive(Debug)]
pub struct HealthMonitor {
    instance_health: Vec<InstanceHealth>,
    fallback_count: u64,
    blacklist: Vec<usize>,
}

impl HealthMonitor {
    pub fn new(num_instances: usize) -> Self {
        let instance_health = (0..num_instances)
            .map(|_| InstanceHealth::new())
            .collect();

        Self {
            instance_health,
            fallback_count: 0,
            blacklist: Vec::new(),
        }
    }

    pub fn record_instance_result(&mut self, instance_id: usize, success: bool, error: Option<FailureType>) {
        if instance_id >= self.instance_health.len() {
            return;
        }

        let health = &mut self.instance_health[instance_id];
        if success {
            health.record_success();
        } else if let Some(error_type) = error {
            health.record_failure(error_type);

            if health.should_blacklist() && !self.blacklist.contains(&instance_id) {
                self.blacklist.push(instance_id);
            }
        }
    }

    pub fn get_healthy_instances(&self) -> Vec<usize> {
        self.instance_health.iter()
            .enumerate()
            .filter(|(idx, health)| {
                !self.blacklist.contains(idx) && health.is_healthy()
            })
            .map(|(idx, _)| idx)
            .collect()
    }

    pub fn record_fallback(&mut self) {
        self.fallback_count += 1;
    }

    pub fn get_fallback_count(&self) -> u64 {
        self.fallback_count
    }

    pub fn get_instance_health(&self, instance_id: usize) -> Option<&InstanceHealth> {
        self.instance_health.get(instance_id)
    }

    pub fn get_blacklist(&self) -> &[usize] {
        &self.blacklist
    }

    pub fn clear_blacklist(&mut self, instance_id: usize) {
        self.blacklist.retain(|&id| id != instance_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor_success() {
        let mut monitor = HealthMonitor::new(4);
        monitor.record_instance_result(0, true, None);

        let health = monitor.get_instance_health(0).unwrap();
        assert_eq!(health.success_count, 1);
        assert_eq!(health.health_score, 1.0);
        assert!(health.is_healthy());
    }

    #[test]
    fn test_health_monitor_failure() {
        let mut monitor = HealthMonitor::new(4);

        for _ in 0..5 {
            monitor.record_instance_result(0, false, Some(FailureType::TransientTimeout));
        }

        let health = monitor.get_instance_health(0).unwrap();
        assert_eq!(health.failure_count, 5);
        assert!(!health.is_healthy());
    }

    #[test]
    fn test_health_monitor_blacklist() {
        let mut monitor = HealthMonitor::new(4);

        for _ in 0..15 {
            monitor.record_instance_result(0, false, Some(FailureType::HardwareFault));
        }

        let blacklist = monitor.get_blacklist();
        assert!(blacklist.contains(&0));

        let healthy = monitor.get_healthy_instances();
        assert!(!healthy.contains(&0));
    }

    #[test]
    fn test_recovery_action_retry() {
        let action = RecoveryAction::Retry {
            max_attempts: 3,
            backoff_ms: 100,
        };

        assert!(matches!(action, RecoveryAction::Retry { .. }));
    }
}