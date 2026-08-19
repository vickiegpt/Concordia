// /home/victoryang00/hetGPU/zluda/src/impl/batch_scheduler/aggregator.rs
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
use crate::r#impl::nvint4_tmatmul::Nvint4Launch;
use super::config::BatchConfig;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchId(u64);

impl BatchId {
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

#[derive(Debug, Clone)]
pub struct Batch {
    pub id: BatchId,
    pub operations: Vec<Nvint4Launch>,
    pub created_at: Instant,
    pub size_bytes: usize,
}

#[derive(Debug)]
pub struct RequestAggregator {
    config: BatchConfig,
    pending_queue: VecDeque<Nvint4Launch>,
    active_batches: HashMap<BatchId, Batch>,
    last_flush: Instant,
}

impl RequestAggregator {
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            pending_queue: VecDeque::new(),
            active_batches: HashMap::new(),
            last_flush: Instant::now(),
        }
    }

    pub fn submit_request(&mut self, request: Nvint4Launch) {
        self.pending_queue.push_back(request);
    }

    pub fn try_build_batch(&mut self) -> Option<Batch> {
        let now = Instant::now();
        let timeout = Duration::from_millis(self.config.timeout_ms as u64);

        // Check if we should flush based on size or timeout
        let should_flush = self.pending_queue.len() >= self.config.max_batch_size ||
                          (self.pending_queue.len() >= self.config.min_batch_size &&
                           now.duration_since(self.last_flush) >= timeout);

        if !should_flush || self.pending_queue.is_empty() {
            return None;
        }

        let operations: Vec<Nvint4Launch> = self.pending_queue.drain(..).collect();
        let size_bytes = operations.iter()
            .map(|op| op.dim as usize * 2) // Estimate input size
            .sum();

        let batch = Batch {
            id: BatchId::new(),
            operations,
            created_at: now,
            size_bytes,
        };

        self.active_batches.insert(batch.id, batch.clone());
        self.last_flush = now;

        Some(batch)
    }

    pub fn flush_all(&mut self) -> Vec<Batch> {
        let remaining: Vec<Nvint4Launch> = self.pending_queue.drain(..).collect();

        if remaining.is_empty() {
            return Vec::new();
        }

        let size_bytes = remaining.iter()
            .map(|op| op.dim as usize * 2)
            .sum();

        let batch = Batch {
            id: BatchId::new(),
            operations: remaining,
            created_at: Instant::now(),
            size_bytes,
        };

        self.active_batches.insert(batch.id, batch.clone());
        vec![batch]
    }

    pub fn complete_batch(&mut self, batch_id: BatchId) {
        self.active_batches.remove(&batch_id);
    }

    pub fn pending_count(&self) -> usize {
        self.pending_queue.len()
    }

    pub fn active_batch_count(&self) -> usize {
        self.active_batches.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_launch(dim: u32) -> Nvint4Launch {
        Nvint4Launch {
            packed_weights: 0x1000,
            input_q8_8: 0x2000,
            output_s64: 0x3000,
            dim,
            delta: 1,
            stream: cuda_types::cuda::CUstream(std::ptr::null_mut()),
        }
    }

    #[test]
    fn test_aggregator_single_request() {
        let config = BatchConfig::default();
        let mut aggregator = RequestAggregator::new(config);

        aggregator.submit_request(create_test_launch(2048));

        // Should not form batch yet (below min_batch_size)
        assert!(aggregator.try_build_batch().is_none());
        assert_eq!(aggregator.pending_count(), 1);
    }

    #[test]
    fn test_aggregator_min_batch_size() {
        let config = BatchConfig {
            min_batch_size: 2,
            ..Default::default()
        };
        let mut aggregator = RequestAggregator::new(config);

        for _ in 0..2 {
            aggregator.submit_request(create_test_launch(2048));
        }

        let batch = aggregator.try_build_batch();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().operations.len(), 2);
        assert_eq!(aggregator.pending_count(), 0);
    }

    #[test]
    fn test_aggregator_max_batch_size() {
        let config = BatchConfig {
            max_batch_size: 4,
            min_batch_size: 2,
            ..Default::default()
        };
        let mut aggregator = RequestAggregator::new(config);

        // Submit 6 requests (exceeds max_batch_size)
        for _ in 0..6 {
            aggregator.submit_request(create_test_launch(2048));
        }

        let batch = aggregator.try_build_batch().unwrap();
        assert_eq!(batch.operations.len(), 4);
        assert_eq!(aggregator.pending_count(), 2);
    }

    #[test]
    fn test_aggregator_flush_all() {
        let config = BatchConfig::default();
        let mut aggregator = RequestAggregator::new(config);

        for _ in 0..3 {
            aggregator.submit_request(create_test_launch(2048));
        }

        let batches = aggregator.flush_all();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].operations.len(), 3);
        assert_eq!(aggregator.pending_count(), 0);
    }

    #[test]
    fn test_aggregator_complete_batch() {
        let config = BatchConfig::default();
        let mut aggregator = RequestAggregator::new(config);

        for _ in 0..16 {
            aggregator.submit_request(create_test_launch(2048));
        }

        let batch = aggregator.try_build_batch().unwrap();
        let batch_id = batch.id;
        assert_eq!(aggregator.active_batch_count(), 1);

        aggregator.complete_batch(batch_id);
        assert_eq!(aggregator.active_batch_count(), 0);
    }
}