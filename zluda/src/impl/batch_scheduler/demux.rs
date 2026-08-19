// /home/victoryang00/hetGPU/zluda/src/impl/batch_scheduler/demux.rs
use super::pipeline::OperationId;
use super::error_handling::{FailureType, RecoveryAction};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use cuda_types::cuda::CUstream;

#[derive(Debug, Clone, Copy)]
pub enum OrderingMode {
    Strict,
    Relaxed,
    Priority,
}

#[derive(Debug, Clone)]
pub struct PendingOperation {
    pub original_stream: CUstream,
    pub completion_callback: CompletionCallback,
    pub timestamp: Instant,
    pub retry_count: u32,
}

pub type CompletionCallback = fn(OperationId, Result<Vec<u8>, String>);

#[derive(Debug)]
pub struct ResponseDemux {
    pending_operations: HashMap<OperationId, PendingOperation>,
    completion_queue: VecDeque<CompletedResponse>,
    ordering_mode: OrderingMode,
}

#[derive(Debug, Clone)]
pub struct CompletedResponse {
    pub operation_id: OperationId,
    pub result: Result<Vec<u8>, String>,
    pub latency_us: u64,
}

impl ResponseDemux {
    pub fn new(ordering_mode: OrderingMode) -> Self {
        Self {
            pending_operations: HashMap::new(),
            completion_queue: VecDeque::new(),
            ordering_mode,
        }
    }

    pub fn register_operation(&mut self, operation_id: OperationId, pending: PendingOperation) {
        self.pending_operations.insert(operation_id, pending);
    }

    pub fn route_results(&mut self, completed_ops: Vec<super::pipeline::CompletedOperation>) {
        for op in completed_ops {
            if let Some(pending) = self.pending_operations.get(&op.operation_id) {
                let latency = pending.timestamp.elapsed().as_micros() as u64;

                let response = CompletedResponse {
                    operation_id: op.operation_id,
                    result: if op.success {
                        Ok(vec![0u8; 4096]) // Placeholder result
                    } else {
                        Err("Operation failed".to_string())
                    },
                    latency_us: latency,
                };

                self.completion_queue.push_back(response);
                self.pending_operations.remove(&op.operation_id);
            }
        }
    }

    pub fn handle_error(&mut self, operation_id: OperationId, error: FailureType) -> RecoveryAction {
        if let Some(pending) = self.pending_operations.get_mut(&operation_id) {
            pending.retry_count += 1;

            match error {
                FailureType::TransientTimeout => {
                    if pending.retry_count < 3 {
                        RecoveryAction::Retry {
                            max_attempts: 3,
                            backoff_ms: 100 * (1 << pending.retry_count),
                        }
                    } else {
                        RecoveryAction::FallbackToGPU
                    }
                }
                FailureType::DmaError => {
                    if pending.retry_count < 2 {
                        RecoveryAction::Retry {
                            max_attempts: 2,
                            backoff_ms: 50,
                        }
                    } else {
                        RecoveryAction::FallbackToGPU
                    }
                }
                FailureType::HardwareFault => RecoveryAction::FallbackToGPU,
                FailureType::Corruption => RecoveryAction::AbortRequest,
            }
        } else {
            RecoveryAction::SkipAndLog
        }
    }

    pub fn pop_completion(&mut self) -> Option<CompletedResponse> {
        match self.ordering_mode {
            OrderingMode::Relaxed => self.completion_queue.pop_front(),
            OrderingMode::Strict => {
                // Would need to implement ordering logic
                self.completion_queue.pop_front()
            }
            OrderingMode::Priority => {
                // Would need to implement priority sorting
                self.completion_queue.pop_front()
            }
        }
    }

    pub fn pending_count(&self) -> usize {
        self.pending_operations.len()
    }

    pub fn completion_count(&self) -> usize {
        self.completion_queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    fn dummy_callback(_: OperationId, _: Result<Vec<u8>, String>) {}

    #[test]
    fn test_demux_register_operation() {
        let mut demux = ResponseDemux::new(OrderingMode::Relaxed);
        let op_id = OperationId::new();

        let pending = PendingOperation {
            original_stream: ptr::null_mut(),
            completion_callback: dummy_callback,
            timestamp: Instant::now(),
            retry_count: 0,
        };

        demux.register_operation(op_id, pending);
        assert_eq!(demux.pending_count(), 1);
    }

    #[test]
    fn test_demux_error_handling() {
        let mut demux = ResponseDemux::new(OrderingMode::Relaxed);
        let op_id = OperationId::new();

        let pending = PendingOperation {
            original_stream: ptr::null_mut(),
            completion_callback: dummy_callback,
            timestamp: Instant::now(),
            retry_count: 0,
        };

        demux.register_operation(op_id, pending);

        let recovery = demux.handle_error(op_id, FailureType::TransientTimeout);
        assert!(matches!(recovery, RecoveryAction::Retry { .. }));
    }

    #[test]
    fn test_demux_route_results() {
        let mut demux = ResponseDemux::new(OrderingMode::Relaxed);
        let op_id = OperationId::new();

        let pending = PendingOperation {
            original_stream: ptr::null_mut(),
            completion_callback: dummy_callback,
            timestamp: Instant::now(),
            retry_count: 0,
        };

        demux.register_operation(op_id, pending.clone());

        let completed = vec![super::pipeline::CompletedOperation {
            operation_id: op_id,
            batch_index: 0,
            success: true,
            latency_us: 100,
        }];

        demux.route_results(completed);
        assert_eq!(demux.completion_count(), 1);
        assert_eq!(demux.pending_count(), 0);
    }
}