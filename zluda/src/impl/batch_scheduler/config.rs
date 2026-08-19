// /home/victoryang00/hetGPU/zluda/src/impl/batch_scheduler/config.rs
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub min_batch_size: usize,
    pub timeout_ms: u32,
    pub target_utilization: f32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            min_batch_size: 16,
            timeout_ms: 5,
            target_utilization: 0.85,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SchedulingPolicy {
    RoundRobin,
    SizeAware,
    LoadBalanced,
    Adaptive,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        Self::Adaptive
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub num_instances: usize,
    pub policy: SchedulingPolicy,
    pub health_check_interval: Duration,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_instances: 16,
            policy: SchedulingPolicy::Adaptive,
            health_check_interval: Duration::from_millis(100),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub enable_double_buffer: bool,
    pub enable_prefetch: bool,
    pub max_concurrent_transfers: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_double_buffer: true,
            enable_prefetch: true,
            max_concurrent_transfers: 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.min_batch_size, 16);
        assert_eq!(config.timeout_ms, 5);
        assert_eq!(config.target_utilization, 0.85);
    }

    #[test]
    fn test_scheduler_config_default() {
        let config = SchedulerConfig::default();
        assert_eq!(config.num_instances, 16);
        assert!(matches!(config.policy, SchedulingPolicy::Adaptive));
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.enable_double_buffer);
        assert!(config.enable_prefetch);
        assert_eq!(config.max_concurrent_transfers, 8);
    }
}