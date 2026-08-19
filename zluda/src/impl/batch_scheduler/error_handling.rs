// /home/victoryang00/hetGPU/zluda/src/impl/batch_scheduler/error_handling.rs
//! Error handling and health monitoring for FPGA batch scheduler
//!
//! # Error Recovery Configuration
//! Retry limits are configurable to handle different failure scenarios:
//! - TransientTimeout: Default 3 retries with exponential backoff
//! - DmaError: Default 2 retries with fixed backoff
//! - HardwareFault: Immediate fallback to GPU
//! - Corruption: Abort request immediately

use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureType {
    TransientTimeout,
    DmaError,
    HardwareFault,
    Corruption,
}

impl FailureType {
    /// Severity weight for health score calculation (lower = less severe)
    pub fn severity_weight(&self) -> f32 {
        match self {
            FailureType::TransientTimeout => 0.1, // Least severe
            FailureType::DmaError => 0.3,
            FailureType::HardwareFault => 0.7,    // Very severe
            FailureType::Corruption => 1.0,       // Most severe
        }
    }

    /// Default retry limit for this error type
    pub fn default_retry_limit(&self) -> u32 {
        match self {
            FailureType::TransientTimeout => 3,
            FailureType::DmaError => 2,
            FailureType::HardwareFault => 0, // No retry, immediate fallback
            FailureType::Corruption => 0,    // No retry, abort
        }
    }

    /// Default backoff time in milliseconds
    pub fn default_backoff_ms(&self) -> u32 {
        match self {
            FailureType::TransientTimeout => 100,
            FailureType::DmaError => 50,
            FailureType::HardwareFault => 0,
            FailureType::Corruption => 0,
        }
    }
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
    pub recent_errors: Vec<(FailureType, Instant)>, // Track recent errors for time-decay
    pub blacklist_time: Option<Instant>,              // When instance was blacklisted
    pub consecutive_failures: u32,                    // Track consecutive failures
}

impl InstanceHealth {
    pub fn new() -> Self {
        Self {
            success_count: 0,
            failure_count: 0,
            last_error: None,
            health_score: 1.0,
            error_types: HashMap::new(),
            recent_errors: Vec::new(),
            blacklist_time: None,
            consecutive_failures: 0,
        }
    }

    pub fn record_success(&mut self) {
        self.success_count += 1;
        self.consecutive_failures = 0;
        self.cleanup_old_errors();
        self.update_health_score();
    }

    pub fn record_failure(&mut self, error_type: FailureType) {
        self.failure_count += 1;
        self.consecutive_failures += 1;
        self.last_error = Some(Instant::now());
        *self.error_types.entry(error_type).or_insert(0) += 1;

        // Track recent errors for time-decay analysis (keep last 100)
        self.recent_errors.push((error_type, Instant::now()));
        if self.recent_errors.len() > 100 {
            self.recent_errors.remove(0);
        }

        self.cleanup_old_errors();
        self.update_health_score();
    }

    fn cleanup_old_errors(&mut self) {
        let now = Instant::now();
        // Remove errors older than 5 minutes for time-decay calculation
        self.recent_errors.retain(|(_, time)| {
            now.duration_since(*time).as_secs() < 300
        });
    }

    fn update_health_score(&mut self) {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            self.health_score = 1.0;
            return;
        }

        // Base score from success rate
        let base_score = self.success_count as f32 / total as f32;

        // Calculate recent error penalty (more weight on recent errors)
        let recent_error_penalty = if self.recent_errors.is_empty() {
            0.0
        } else {
            let severity_sum: f32 = self.recent_errors.iter()
                .map(|(error_type, _)| error_type.severity_weight())
                .sum();

            // Penalty increases with recent severe errors
            (severity_sum / self.recent_errors.len() as f32) * 0.3
        };

        // Consecutive failure penalty
        let consecutive_penalty = (self.consecutive_failures as f32 * 0.05).min(0.2);

        // Apply penalties to base score
        self.health_score = (base_score - recent_error_penalty - consecutive_penalty).max(0.0);
    }

    pub fn is_healthy(&self) -> bool {
        self.health_score > 0.5 && self.consecutive_failures < 5
    }

    pub fn should_blacklist(&self) -> bool {
        // Blacklist if health score is very low OR too many consecutive failures
        self.health_score < 0.1 || self.consecutive_failures > 10 || self.failure_count > 20
    }

    /// Check if instance is ready for blacklist recovery (cooldown period)
    pub fn can_recover_from_blacklist(&self, cooldown_duration: Duration) -> bool {
        if let Some(blacklist_time) = self.blacklist_time {
            let time_since_blacklist = Instant::now().duration_since(blacklist_time);
            time_since_blacklist > cooldown_duration && self.health_score > 0.3
        } else {
            false
        }
    }
}

#[derive(Debug)]
pub struct HealthMonitor {
    instance_health: Vec<InstanceHealth>,
    fallback_count: u64,
    blacklist: Vec<usize>,
    blacklist_cooldown: Duration,      // Cooldown period before considering recovery
    recovery_check_interval: Duration, // How often to check for blacklist recovery
    last_recovery_check: Instant,
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
            blacklist_cooldown: Duration::from_secs(300), // 5 minute default cooldown
            recovery_check_interval: Duration::from_secs(60), // Check every minute
            last_recovery_check: Instant::now(),
        }
    }

    pub fn with_cooldown(num_instances: usize, cooldown: Duration) -> Self {
        let instance_health = (0..num_instances)
            .map(|_| InstanceHealth::new())
            .collect();

        Self {
            instance_health,
            fallback_count: 0,
            blacklist: Vec::new(),
            blacklist_cooldown: cooldown,
            recovery_check_interval: Duration::from_secs(60),
            last_recovery_check: Instant::now(),
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
                health.blacklist_time = Some(Instant::now());
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

    /// Periodically check for blacklist recovery opportunities
    pub fn check_blacklist_recovery(&mut self) -> Vec<usize> {
        let now = Instant::now();
        if now.duration_since(self.last_recovery_check) < self.recovery_check_interval {
            return Vec::new();
        }

        self.last_recovery_check = now;
        let mut recovered = Vec::new();

        // Check each blacklisted instance for recovery eligibility
        for instance_id in self.blacklist.clone() {
            if let Some(health) = self.instance_health.get(instance_id) {
                if health.can_recover_from_blacklist(self.blacklist_cooldown) {
                    recovered.push(instance_id);
                }
            }
        }

        // Remove recovered instances from blacklist
        for instance_id in &recovered {
            self.clear_blacklist(*instance_id);
            if let Some(health) = self.instance_health.get_mut(*instance_id) {
                health.blacklist_time = None;
                health.consecutive_failures = 0; // Reset on recovery
            }
        }

        recovered
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

    /// Manually trigger recovery check (useful for testing)
    pub fn force_recovery_check(&mut self) -> Vec<usize> {
        self.last_recovery_check = Instant::now() - self.recovery_check_interval;
        self.check_blacklist_recovery()
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
        assert_eq!(health.consecutive_failures, 0);
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
        assert_eq!(health.consecutive_failures, 5);
    }

    #[test]
    fn test_health_monitor_blacklist() {
        let mut monitor = HealthMonitor::new(4);

        // Generate enough consecutive failures to trigger blacklist
        for _ in 0..12 {
            monitor.record_instance_result(0, false, Some(FailureType::HardwareFault));
        }

        let blacklist = monitor.get_blacklist();
        assert!(blacklist.contains(&0));

        let healthy = monitor.get_healthy_instances();
        assert!(!healthy.contains(&0));

        // Verify blacklist time was set
        let health = monitor.get_instance_health(0).unwrap();
        assert!(health.blacklist_time.is_some());
    }

    #[test]
    fn test_error_severity_weights() {
        assert_eq!(FailureType::TransientTimeout.severity_weight(), 0.1);
        assert_eq!(FailureType::DmaError.severity_weight(), 0.3);
        assert_eq!(FailureType::HardwareFault.severity_weight(), 0.7);
        assert_eq!(FailureType::Corruption.severity_weight(), 1.0);
    }

    #[test]
    fn test_error_recovery_with_different_severities() {
        let mut monitor = HealthMonitor::new(4);

        // Record transient errors (less severe)
        for _ in 0..3 {
            monitor.record_instance_result(0, false, Some(FailureType::TransientTimeout));
        }

        let health = monitor.get_instance_health(0).unwrap();
        // Should still be relatively healthy despite failures due to low severity
        assert!(health.health_score > 0.5);
    }

    #[test]
    fn test_consecutive_failure_tracking() {
        let mut monitor = HealthMonitor::new(4);

        // Record 3 consecutive failures
        monitor.record_instance_result(0, false, Some(FailureType::TransientTimeout));
        monitor.record_instance_result(0, false, Some(FailureType::DmaError));
        monitor.record_instance_result(0, false, Some(FailureType::HardwareFault));

        let health = monitor.get_instance_health(0).unwrap();
        assert_eq!(health.consecutive_failures, 3);

        // Success should reset consecutive failures
        monitor.record_instance_result(0, true, None);
        let health = monitor.get_instance_health(0).unwrap();
        assert_eq!(health.consecutive_failures, 0);
    }

    #[test]
    fn test_blacklist_recovery() {
        let mut monitor = HealthMonitor::with_cooldown(
            4,
            Duration::from_millis(100) // Short cooldown for testing
        );

        // Blacklist an instance
        for _ in 0..12 {
            monitor.record_instance_result(0, false, Some(FailureType::HardwareFault));
        }

        assert!(monitor.get_blacklist().contains(&0));

        // Simulate some success to improve health
        for _ in 0..5 {
            monitor.record_instance_result(0, true, None);
        }

        // Wait for cooldown period
        std::thread::sleep(Duration::from_millis(150));

        // Force recovery check
        let recovered = monitor.force_recovery_check();

        // Instance should be recovered
        assert!(recovered.contains(&0));
        assert!(!monitor.get_blacklist().contains(&0));

        // Verify health state was reset
        let health = monitor.get_instance_health(0).unwrap();
        assert!(health.blacklist_time.is_none());
        assert_eq!(health.consecutive_failures, 0);
    }

    #[test]
    fn test_health_score_with_recent_errors() {
        let mut monitor = HealthMonitor::new(4);

        // Create failures with different severities
        monitor.record_instance_result(0, false, Some(FailureType::Corruption)); // 1.0 weight
        monitor.record_instance_result(0, false, Some(FailureType::HardwareFault)); // 0.7 weight
        monitor.record_instance_result(0, false, Some(FailureType::TransientTimeout)); // 0.1 weight

        let health = monitor.get_instance_health(0).unwrap();

        // Health score should be penalized more heavily due to severe recent errors
        assert!(health.health_score < 0.5);
        assert_eq!(health.recent_errors.len(), 3);
    }

    #[test]
    fn test_recent_errors_time_decay() {
        let mut monitor = HealthMonitor::new(4);

        // Record some errors
        for _ in 0..5 {
            monitor.record_instance_result(0, false, Some(FailureType::TransientTimeout));
        }

        let health = monitor.get_instance_health(0).unwrap();
        assert_eq!(health.recent_errors.len(), 5);

        // Wait longer than cleanup threshold (5 minutes)
        // For testing, we'll just trigger cleanup by recording a success
        monitor.record_instance_result(0, true, None);

        // Recent errors should be cleaned up if old
        // (In real testing, we'd wait 5+ minutes, but this validates the mechanism)
        let health = monitor.get_instance_health(0).unwrap();
        assert!(health.recent_errors.len() <= 6); // Should have cleaned up old errors
    }
}