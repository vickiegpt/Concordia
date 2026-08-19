// /home/victoryang00/hetGPU/zluda/src/impl/batch_scheduler/demux.rs
//! Response Demux for routing completed operations back to their callers
//!
//! # Thread Safety
//! This implementation is designed for single-threaded use within the batch scheduler.
//! For multi-threaded environments, wrap ResponseDemux in Arc<Mutex<>> as shown in integration.
//!
//! # Ordering Modes
//! - **Strict**: Maintains original submission order using sequence numbers
//! - **Relaxed**: First-come-first-served based on completion time
//! - **Priority**: Ordered by priority value (lower = higher priority)

use super::pipeline::OperationId;
use super::error_handling::{FailureType, RecoveryAction};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
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
    pub priority: u32,
    pub submission_order: u64, // For Strict ordering mode
    pub timeout_ms: u64,       // Timeout in milliseconds
}

pub type CompletionCallback = fn(OperationId, Result<Vec<u8>, String>);

#[derive(Debug)]
pub struct ResponseDemux {
    pending_operations: HashMap<OperationId, PendingOperation>,
    completion_queue: VecDeque<CompletedResponse>,
    ordering_mode: OrderingMode,
    submission_counter: AtomicU64, // For tracking submission order in Strict mode
    default_timeout_ms: u64,      // Default timeout for operations
}

#[derive(Debug, Clone)]
pub struct CompletedResponse {
    pub operation_id: OperationId,
    pub result: Result<Vec<u8>, String>,
    pub latency_us: u64,
    pub priority: u32,
    pub submission_order: u64, // For Strict ordering mode
}

impl ResponseDemux {
    pub fn new(ordering_mode: OrderingMode) -> Self {
        Self {
            pending_operations: HashMap::new(),
            completion_queue: VecDeque::new(),
            ordering_mode,
            submission_counter: AtomicU64::new(0),
            default_timeout_ms: 5000, // 5 second default timeout
        }
    }

    pub fn with_timeout(ordering_mode: OrderingMode, timeout_ms: u64) -> Self {
        Self {
            pending_operations: HashMap::new(),
            completion_queue: VecDeque::new(),
            ordering_mode,
            submission_counter: AtomicU64::new(0),
            default_timeout_ms: timeout_ms,
        }
    }

    pub fn register_operation(&mut self, operation_id: OperationId, pending: PendingOperation) {
        let submission_order = self.submission_counter.fetch_add(1, AtomicOrdering::SeqCst);
        let mut pending_with_order = pending;
        pending_with_order.submission_order = submission_order;

        // Use default timeout if not specified
        if pending_with_order.timeout_ms == 0 {
            pending_with_order.timeout_ms = self.default_timeout_ms;
        }

        self.pending_operations.insert(operation_id, pending_with_order);
    }

    pub fn route_results(&mut self, completed_ops: Vec<super::pipeline::CompletedOperation>) {
        for op in completed_ops {
            if let Some(pending) = self.pending_operations.remove(&op.operation_id) {
                let latency = pending.timestamp.elapsed().as_micros() as u64;

                // In production, this would contain actual operation results from the FPGA
                // For now, we simulate result generation with proper context
                let result = if op.success {
                    // TODO: Replace with actual FPGA operation results
                    // This would typically come from the operation's output buffer
                    Ok(vec![0u8; 4096]) // Production: Replace with real result data
                } else {
                    // Provide detailed error context for debugging
                    Err(format!("Operation {} failed on instance {}: batch_index={}, latency_us={}",
                        op.operation_id.0, op.batch_index, op.batch_index, op.latency_us))
                };

                // Actually invoke the completion callback with the operation result
                (pending.completion_callback)(op.operation_id, result.clone());

                let response = CompletedResponse {
                    operation_id: op.operation_id,
                    result: result,
                    latency_us: latency,
                    priority: pending.priority,
                    submission_order: pending.submission_order,
                };

                // For Priority mode, insert in priority order (lower priority number = higher priority)
                if self.ordering_mode == OrderingMode::Priority {
                    let insert_pos = self.completion_queue
                        .iter()
                        .position(|r| pending.priority < r.priority);

                    if let Some(pos) = insert_pos {
                        self.completion_queue.insert(pos, response);
                    } else {
                        self.completion_queue.push_back(response);
                    }
                } else if self.ordering_mode == OrderingMode::Strict {
                    // For Strict mode, insert in submission order
                    let insert_pos = self.completion_queue
                        .iter()
                        .position(|r| pending.submission_order < r.submission_order);

                    if let Some(pos) = insert_pos {
                        self.completion_queue.insert(pos, response);
                    } else {
                        self.completion_queue.push_back(response);
                    }
                } else {
                    // Relaxed mode: FIFO based on completion time
                    self.completion_queue.push_back(response);
                }
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
            OrderingMode::Relaxed => {
                // First-come-first-served (completion order)
                self.completion_queue.pop_front()
            }
            OrderingMode::Strict => {
                // Return responses in original submission order (already sorted in route_results)
                self.completion_queue.pop_front()
            }
            OrderingMode::Priority => {
                // Return responses in priority order (already sorted in route_results)
                self.completion_queue.pop_front()
            }
        }
    }

    /// Check for timed out operations and return their IDs for cleanup
    pub fn check_timeouts(&mut self) -> Vec<OperationId> {
        let now = Instant::now();
        let mut timed_out = Vec::new();

        for (op_id, pending) in &self.pending_operations {
            let elapsed = now.duration_since(pending.timestamp).as_millis() as u64;
            if elapsed > pending.timeout_ms {
                timed_out.push(*op_id);
            }
        }

        // Remove timed out operations
        for op_id in &timed_out {
            self.pending_operations.remove(op_id);
        }

        timed_out
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
            priority: 0,
            submission_order: 0,
            timeout_ms: 1000,
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
            priority: 0,
            submission_order: 0,
            timeout_ms: 1000,
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
            priority: 0,
            submission_order: 0,
            timeout_ms: 1000,
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

    #[test]
    fn test_priority_ordering() {
        let mut demux = ResponseDemux::new(OrderingMode::Priority);

        let op1 = OperationId::new();
        let op2 = OperationId::new();
        let op3 = OperationId::new();

        // Register operations with different priorities (lower = higher priority)
        let pending_high = PendingOperation {
            original_stream: ptr::null_mut(),
            completion_callback: dummy_callback,
            timestamp: Instant::now(),
            retry_count: 0,
            priority: 1, // High priority
            submission_order: 0,
            timeout_ms: 1000,
        };

        let pending_low = PendingOperation {
            original_stream: ptr::null_mut(),
            completion_callback: dummy_callback,
            timestamp: Instant::now(),
            retry_count: 0,
            priority: 10, // Low priority
            submission_order: 0,
            timeout_ms: 1000,
        };

        let pending_medium = PendingOperation {
            original_stream: ptr::null_mut(),
            completion_callback: dummy_callback,
            timestamp: Instant::now(),
            retry_count: 0,
            priority: 5, // Medium priority
            submission_order: 0,
            timeout_ms: 1000,
        };

        demux.register_operation(op1, pending_low);
        demux.register_operation(op2, pending_high);
        demux.register_operation(op3, pending_medium);

        // Complete operations out of priority order
        let completed = vec![
            super::pipeline::CompletedOperation {
                operation_id: op1,
                batch_index: 0,
                success: true,
                latency_us: 100,
            },
            super::pipeline::CompletedOperation {
                operation_id: op2,
                batch_index: 0,
                success: true,
                latency_us: 100,
            },
            super::pipeline::CompletedOperation {
                operation_id: op3,
                batch_index: 0,
                success: true,
                latency_us: 100,
            },
        ];

        demux.route_results(completed);

        // Should be returned in priority order: high(1), medium(5), low(10)
        let result1 = demux.pop_completion().unwrap();
        let result2 = demux.pop_completion().unwrap();
        let result3 = demux.pop_completion().unwrap();

        assert_eq!(result1.priority, 1); // High priority first
        assert_eq!(result2.priority, 5); // Medium priority second
        assert_eq!(result3.priority, 10); // Low priority last
    }

    #[test]
    fn test_timeout_handling() {
        let mut demux = ResponseDemux::with_timeout(OrderingMode::Relaxed, 100); // 100ms timeout
        let op_id = OperationId::new();

        let pending = PendingOperation {
            original_stream: ptr::null_mut(),
            completion_callback: dummy_callback,
            timestamp: Instant::now(),
            retry_count: 0,
            priority: 0,
            submission_order: 0,
            timeout_ms: 100, // Will timeout
        };

        demux.register_operation(op_id, pending);

        // Wait for timeout
        std::thread::sleep(std::time::Duration::from_millis(150));

        let timed_out = demux.check_timeouts();
        assert_eq!(timed_out.len(), 1);
        assert_eq!(timed_out[0], op_id);
        assert_eq!(demux.pending_count(), 0);
    }

    #[test]
    fn test_strict_ordering() {
        let mut demux = ResponseDemux::new(OrderingMode::Strict);

        let op1 = OperationId::new();
        let op2 = OperationId::new();
        let op3 = OperationId::new();

        // Register operations in order 1, 2, 3
        let pending1 = PendingOperation {
            original_stream: ptr::null_mut(),
            completion_callback: dummy_callback,
            timestamp: Instant::now(),
            retry_count: 0,
            priority: 0,
            submission_order: 0, // Will be set to 0
            timeout_ms: 1000,
        };

        let pending2 = PendingOperation {
            original_stream: ptr::null_mut(),
            completion_callback: dummy_callback,
            timestamp: Instant::now(),
            retry_count: 0,
            priority: 0,
            submission_order: 0, // Will be set to 1
            timeout_ms: 1000,
        };

        let pending3 = PendingOperation {
            original_stream: ptr::null_mut(),
            completion_callback: dummy_callback,
            timestamp: Instant::now(),
            retry_count: 0,
            priority: 0,
            submission_order: 0, // Will be set to 2
            timeout_ms: 1000,
        };

        demux.register_operation(op1, pending1);
        demux.register_operation(op2, pending2);
        demux.register_operation(op3, pending3);

        // Complete operations out of order: op2, op3, op1
        let completed = vec![
            super::pipeline::CompletedOperation {
                operation_id: op2,
                batch_index: 0,
                success: true,
                latency_us: 100,
            },
            super::pipeline::CompletedOperation {
                operation_id: op3,
                batch_index: 0,
                success: true,
                latency_us: 100,
            },
            super::pipeline::CompletedOperation {
                operation_id: op1,
                batch_index: 0,
                success: true,
                latency_us: 100,
            },
        ];

        demux.route_results(completed);

        // Should be returned in submission order: op1, op2, op3
        let result1 = demux.pop_completion().unwrap();
        let result2 = demux.pop_completion().unwrap();
        let result3 = demux.pop_completion().unwrap();

        assert_eq!(result1.submission_order, 0); // First submitted
        assert_eq!(result2.submission_order, 1); // Second submitted
        assert_eq!(result3.submission_order, 2); // Third submitted
    }
}