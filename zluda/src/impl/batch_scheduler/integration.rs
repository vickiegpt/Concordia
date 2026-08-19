// /home/victoryang00/hetGPU/zluda/src/impl/batch_scheduler/integration.rs
use super::{aggregator::RequestAggregator, scheduler::InstanceScheduler, pipeline::MemoryPipeline, demux::ResponseDemux, error_handling::HealthMonitor};
use super::config::{BatchConfig, SchedulerConfig, PipelineConfig};
use std::sync::{Arc, Mutex};
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
use crate::r#impl::nvint4_tmatmul::Nvint4Launch;

#[derive(Clone)]
pub struct BatchSchedulerManager {
    aggregator: Arc<Mutex<RequestAggregator>>,
    scheduler: Arc<Mutex<InstanceScheduler>>,
    pipeline: Arc<Mutex<MemoryPipeline>>,
    demux: Arc<Mutex<ResponseDemux>>,
    health_monitor: Arc<Mutex<HealthMonitor>>,
}

impl BatchSchedulerManager {
    pub fn new() -> Self {
        let num_instances = 16;

        Self {
            aggregator: Arc::new(Mutex::new(RequestAggregator::new(BatchConfig::default()))),
            scheduler: Arc::new(Mutex::new(InstanceScheduler::new(SchedulerConfig::default()))),
            pipeline: Arc::new(Mutex::new(MemoryPipeline::new(PipelineConfig::default(), num_instances))),
            demux: Arc::new(Mutex::new(ResponseDemux::new(super::demux::OrderingMode::Relaxed))),
            health_monitor: Arc::new(Mutex::new(HealthMonitor::new(num_instances))),
        }
    }

    #[cfg(all(
        feature = "nvidia",
        not(feature = "amd"),
        not(feature = "intel"),
        not(feature = "tenstorrent")
    ))]
    pub fn submit_launch(&self, launch: Nvint4Launch) -> Result<(), String> {
        let mut aggregator = self.aggregator.lock()
            .map_err(|e| format!("Aggregator lock failed: {}", e))?;

        aggregator.submit_request(launch);
        Ok(())
    }

    pub fn process_pending(&self) -> Result<Vec<super::pipeline::CompletedOperation>, String> {
        // Try to build a batch
        let batch = {
            let mut aggregator = self.aggregator.lock()
                .map_err(|e| format!("Aggregator lock failed: {}", e))?;
            aggregator.try_build_batch()
        };

        let mut completed_operations = Vec::new();

        if let Some(batch) = batch {
            // Assign to instances
            let assignments = {
                let mut scheduler = self.scheduler.lock()
                    .map_err(|e| format!("Scheduler lock failed: {}", e))?;
                scheduler.assign_to_instances(&batch)
            };

            // Execute on instances
            {
                let mut pipeline = self.pipeline.lock()
                    .map_err(|e| format!("Pipeline lock failed: {}", e))?;
                pipeline.execute_on_instances(&batch, assignments.clone());
            }

            // Collect results
            {
                let mut pipeline = self.pipeline.lock()
                    .map_err(|e| format!("Pipeline lock failed: {}", e))?;
                completed_operations = pipeline.collect_results();
                pipeline.cleanup_batch(&assignments);
            }
        }

        Ok(completed_operations)
    }

    pub fn get_statistics(&self) -> SchedulerStats {
        let (utilizations, avg_latency, fallback_count) = {
            let scheduler = self.scheduler.lock().unwrap();
            let health = self.health_monitor.lock().unwrap();

            let utilizations = scheduler.get_instance_utilization();
            let avg_latency = scheduler.get_average_latency();
            let fallback_count = health.get_fallback_count();

            (utilizations, avg_latency, fallback_count)
        };

        SchedulerStats {
            instance_utilizations: utilizations,
            average_latency_us: avg_latency,
            fallback_count,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub instance_utilizations: Vec<f32>,
    pub average_latency_us: u64,
    pub fallback_count: u64,
}

// Global instance
static GLOBAL_SCHEDULER: std::sync::OnceLock<BatchSchedulerManager> = std::sync::OnceLock::new();

pub fn get_global_scheduler() -> &'static BatchSchedulerManager {
    GLOBAL_SCHEDULER.get_or_init(|| BatchSchedulerManager::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_scheduler() {
        let scheduler = get_global_scheduler();
        let stats = scheduler.get_statistics();

        assert_eq!(stats.instance_utilizations.len(), 16);
    }

    #[test]
    fn test_scheduler_stats() {
        let stats = SchedulerStats {
            instance_utilizations: vec![0.5, 0.7, 0.9],
            average_latency_us: 1000,
            fallback_count: 5,
        };

        assert_eq!(stats.instance_utilizations.len(), 3);
        assert_eq!(stats.average_latency_us, 1000);
        assert_eq!(stats.fallback_count, 5);
    }

    #[test]
    fn test_process_pending_empty() {
        let scheduler = get_global_scheduler();
        let result = scheduler.process_pending();

        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}