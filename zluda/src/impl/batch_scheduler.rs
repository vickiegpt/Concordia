use crate::r#impl::cxl_tmatmul_v3::CompletedTaskV3;
use std::collections::HashSet;

const BATCH_LIMIT_ENV: &str = "HETGPU_FPGA_BATCH_LIMIT";

/// Software allocation ceiling for a single materialized plan. This is intentionally independent
/// of live `CapsV3.max_descriptors`: execution must submit the returned descriptors in
/// capability-sized windows, and callers needing more slices must plan logical work in windows.
const MAX_PLAN_SLICES_SOFTWARE_SAFETY_LIMIT: u32 = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchSlice {
    first: u32,
    count: u32,
}

impl BatchSlice {
    pub(crate) fn first(&self) -> u32 {
        self.first
    }

    pub(crate) fn count(&self) -> u32 {
        self.count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BatchPlan {
    slices: Vec<BatchSlice>,
}

impl BatchPlan {
    pub(crate) fn new(logical_batch: u32, max_batch: u32) -> Result<Self, String> {
        if logical_batch == 0 {
            return Err("logical_batch must be non-zero".into());
        }
        if max_batch == 0 {
            return Err("max_batch must be non-zero".into());
        }

        let slice_count = (logical_batch / max_batch)
            .checked_add(u32::from(logical_batch % max_batch != 0))
            .ok_or_else(|| "batch slice count overflow".to_string())?;
        if slice_count > MAX_PLAN_SLICES_SOFTWARE_SAFETY_LIMIT {
            return Err(format!(
                "batch plan requires {slice_count} slices, exceeding software safety limit {MAX_PLAN_SLICES_SOFTWARE_SAFETY_LIMIT}; plan logical work in windows"
            ));
        }
        let slice_count = usize::try_from(slice_count)
            .map_err(|_| "batch slice count does not fit in usize".to_string())?;
        let mut slices = Vec::new();
        slices.try_reserve_exact(slice_count).map_err(|error| {
            format!("unable to reserve storage for {slice_count} batch slices: {error}")
        })?;
        let mut first = 0u32;
        while first < logical_batch {
            let count = max_batch.min(logical_batch - first);
            slices.push(BatchSlice { first, count });
            first = first
                .checked_add(count)
                .ok_or_else(|| "batch slice offset overflow".to_string())?;
        }
        Ok(Self { slices })
    }

    pub(crate) fn slices(&self) -> &[BatchSlice] {
        &self.slices
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchSchedulerConfig {
    max_batch: u32,
}

impl BatchSchedulerConfig {
    pub(crate) fn parse(
        configured_limit: Option<&str>,
        live_max_batch: u32,
    ) -> Result<Self, String> {
        if live_max_batch == 0 {
            return Err("live max_batch must be non-zero".into());
        }

        let max_batch = match configured_limit {
            None => live_max_batch,
            Some(value) => {
                let value = value.parse::<u32>().map_err(|_| {
                    format!("{BATCH_LIMIT_ENV} must be an integer in 1..={live_max_batch}")
                })?;
                if value == 0 || value > live_max_batch {
                    return Err(format!(
                        "{BATCH_LIMIT_ENV} value {value} outside 1..={live_max_batch}"
                    ));
                }
                value
            }
        };

        Ok(Self { max_batch })
    }

    pub(crate) fn from_env(live_max_batch: u32) -> Result<Self, String> {
        match std::env::var(BATCH_LIMIT_ENV) {
            Ok(value) => Self::parse(Some(&value), live_max_batch),
            Err(std::env::VarError::NotPresent) => Self::parse(None, live_max_batch),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{BATCH_LIMIT_ENV} must be valid UTF-8"))
            }
        }
    }

    pub(crate) fn max_batch(&self) -> u32 {
        self.max_batch
    }

    pub(crate) fn plan(&self, logical_batch: u32) -> Result<BatchPlan, String> {
        BatchPlan::new(logical_batch, self.max_batch)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SchedulerReport {
    descriptor_count: u64,
    logical_items: u64,
    unique_submission_count: u64,
    lane_mask: u64,
    per_lane_completion_counts: Vec<u64>,
    total_accelerator_cycles: u64,
    total_matrix_bytes_read: u64,
    total_input_bytes_read: u64,
    total_output_bytes_written: u64,
}

impl SchedulerReport {
    pub(crate) fn from_completions(
        completed_tasks: &[CompletedTaskV3],
        num_instances: u32,
    ) -> Result<Self, String> {
        if num_instances == 0 {
            return Err("num_instances must be non-zero".into());
        }
        if num_instances > u64::BITS {
            return Err(format!(
                "num_instances {num_instances} exceeds 64-bit lane mask capacity"
            ));
        }

        let descriptor_count = u64::try_from(completed_tasks.len())
            .map_err(|_| "descriptor count does not fit in u64".to_string())?;
        let mut report = Self {
            descriptor_count,
            logical_items: 0,
            unique_submission_count: 0,
            lane_mask: 0,
            per_lane_completion_counts: vec![0; num_instances as usize],
            total_accelerator_cycles: 0,
            total_matrix_bytes_read: 0,
            total_input_bytes_read: 0,
            total_output_bytes_written: 0,
        };
        let mut submission_ids = HashSet::new();

        for completed in completed_tasks {
            validate_evidence(completed)?;
            if completed.task.batch == 0 {
                return Err(format!(
                    "submission_id={} has invalid zero task batch",
                    completed.submission_id
                ));
            }

            let lane = completed.completion.lane_used;
            if lane >= num_instances {
                return Err(format!("completion lane {lane} outside 0..{num_instances}"));
            }
            let lane_index = lane as usize;
            let lane_bit = 1u64
                .checked_shl(lane)
                .ok_or_else(|| format!("completion lane {lane} cannot fit in lane mask"))?;

            checked_add(
                &mut report.logical_items,
                u64::from(completed.task.batch),
                "logical item count",
            )?;
            checked_add(
                &mut report.per_lane_completion_counts[lane_index],
                1,
                "per-lane completion count",
            )?;
            checked_add(
                &mut report.total_accelerator_cycles,
                completed.completion.accelerator_cycles,
                "accelerator cycle total",
            )?;
            checked_add(
                &mut report.total_matrix_bytes_read,
                completed.completion.matrix_bytes_read,
                "matrix byte total",
            )?;
            checked_add(
                &mut report.total_input_bytes_read,
                completed.completion.input_bytes_read,
                "input byte total",
            )?;
            checked_add(
                &mut report.total_output_bytes_written,
                completed.completion.output_bytes_written,
                "output byte total",
            )?;
            report.lane_mask |= lane_bit;

            if submission_ids.insert(completed.submission_id) {
                checked_add(
                    &mut report.unique_submission_count,
                    1,
                    "unique submission count",
                )?;
            }
        }

        Ok(report)
    }

    pub(crate) fn descriptor_count(&self) -> u64 {
        self.descriptor_count
    }

    pub(crate) fn logical_items(&self) -> u64 {
        self.logical_items
    }

    pub(crate) fn unique_submission_count(&self) -> u64 {
        self.unique_submission_count
    }

    pub(crate) fn lane_mask(&self) -> u64 {
        self.lane_mask
    }

    pub(crate) fn per_lane_completion_counts(&self) -> &[u64] {
        &self.per_lane_completion_counts
    }

    pub(crate) fn total_accelerator_cycles(&self) -> u64 {
        self.total_accelerator_cycles
    }

    pub(crate) fn total_matrix_bytes_read(&self) -> u64 {
        self.total_matrix_bytes_read
    }

    pub(crate) fn total_input_bytes_read(&self) -> u64 {
        self.total_input_bytes_read
    }

    pub(crate) fn total_output_bytes_written(&self) -> u64 {
        self.total_output_bytes_written
    }
}

fn validate_evidence(completed: &CompletedTaskV3) -> Result<(), String> {
    if completed.submission_id == 0 {
        return Err("submission_id must be non-zero".into());
    }
    if completed.task.request_id == 0 {
        return Err(format!(
            "submission_id={} task request_id must be non-zero",
            completed.submission_id
        ));
    }
    if completed.completion.request_id != completed.task.request_id {
        return Err(format!(
            "submission_id={} completion request_id={} does not match task request_id={}",
            completed.submission_id, completed.completion.request_id, completed.task.request_id
        ));
    }
    if completed.completion.status != 0 {
        return Err(format!(
            "request_id={} completion status={}",
            completed.task.request_id, completed.completion.status
        ));
    }
    if completed.completion.end_cycle <= completed.completion.start_cycle {
        return Err(format!(
            "request_id={} invalid completion cycle range {}..{}",
            completed.task.request_id,
            completed.completion.start_cycle,
            completed.completion.end_cycle
        ));
    }
    let cycle_range = completed
        .completion
        .end_cycle
        .checked_sub(completed.completion.start_cycle)
        .ok_or_else(|| {
            format!(
                "request_id={} invalid completion cycle range",
                completed.task.request_id
            )
        })?;
    if completed.completion.accelerator_cycles != cycle_range {
        return Err(format!(
            "request_id={} invalid completion cycle range: accelerator_cycles={} range={cycle_range}",
            completed.task.request_id, completed.completion.accelerator_cycles
        ));
    }
    Ok(())
}

fn checked_add(total: &mut u64, value: u64, metric: &str) -> Result<(), String> {
    *total = total
        .checked_add(value)
        .ok_or_else(|| format!("{metric} overflow"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r#impl::cxl_tmatmul_v3::{CompletedTaskV3, CompletionV3, TaskV3};
    use std::ffi::OsString;

    struct EnvGuard {
        original: Option<OsString>,
    }

    impl EnvGuard {
        fn capture() -> Self {
            Self {
                original: std::env::var_os(BATCH_LIMIT_ENV),
            }
        }

        fn set(&self, value: Option<OsString>) {
            match value {
                Some(value) => std::env::set_var(BATCH_LIMIT_ENV, value),
                None => std::env::remove_var(BATCH_LIMIT_ENV),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.original.take() {
                Some(value) => std::env::set_var(BATCH_LIMIT_ENV, value),
                None => std::env::remove_var(BATCH_LIMIT_ENV),
            }
        }
    }

    fn completed_task(
        submission_id: u64,
        batch: u32,
        lane_used: u32,
        accelerator_cycles: u64,
        matrix_bytes_read: u64,
        input_bytes_read: u64,
        output_bytes_written: u64,
    ) -> CompletedTaskV3 {
        let mut task = TaskV3::default();
        task.request_id = submission_id;
        task.batch = batch;

        let mut completion = CompletionV3::default();
        completion.request_id = task.request_id;
        completion.lane_used = lane_used;
        completion.accelerator_cycles = accelerator_cycles;
        completion.matrix_bytes_read = matrix_bytes_read;
        completion.input_bytes_read = input_bytes_read;
        completion.output_bytes_written = output_bytes_written;
        completion.start_cycle = 0;
        completion.end_cycle = accelerator_cycles;

        CompletedTaskV3 {
            submission_id,
            task,
            completion,
        }
    }

    #[test]
    fn rejects_zero_logical_batch() {
        assert!(BatchPlan::new(0, 4).is_err());
    }

    #[test]
    fn slices_exact_multiple_in_logical_order() {
        let plan = BatchPlan::new(8, 4).unwrap();

        assert_eq!(
            plan.slices()
                .iter()
                .map(|slice| (slice.first(), slice.count()))
                .collect::<Vec<_>>(),
            vec![(0, 4), (4, 4)]
        );
    }

    #[test]
    fn slices_remainder_in_logical_order() {
        let plan = BatchPlan::new(10, 4).unwrap();

        assert_eq!(
            plan.slices()
                .iter()
                .map(|slice| (slice.first(), slice.count()))
                .collect::<Vec<_>>(),
            vec![(0, 4), (4, 4), (8, 2)]
        );
    }

    #[test]
    fn checked_ceiling_handles_u32_boundary() {
        let plan = BatchPlan::new(u32::MAX, u32::MAX).unwrap();

        assert_eq!(plan.slices().len(), 1);
        assert_eq!(plan.slices()[0].first(), 0);
        assert_eq!(plan.slices()[0].count(), u32::MAX);
    }

    #[test]
    fn rejects_plan_exceeding_software_slice_safety_limit() {
        let error = BatchPlan::new(u32::MAX, 1).unwrap_err();

        assert!(error.contains("software safety limit"), "{error}");
    }

    #[test]
    fn configured_limit_can_only_lower_live_max_batch() {
        assert_eq!(BatchSchedulerConfig::parse(None, 4).unwrap().max_batch(), 4);
        let configured = BatchSchedulerConfig::parse(Some("2"), 4).unwrap();
        assert_eq!(configured.max_batch(), 2);
        assert_eq!(
            configured
                .plan(5)
                .unwrap()
                .slices()
                .iter()
                .map(|slice| (slice.first(), slice.count()))
                .collect::<Vec<_>>(),
            vec![(0, 2), (2, 2), (4, 1)]
        );

        for value in [Some("0"), Some("not-a-number"), Some("5")] {
            assert!(BatchSchedulerConfig::parse(value, 4).is_err());
        }
        assert!(BatchSchedulerConfig::parse(None, 0).is_err());
    }

    #[test]
    fn env_wrapper_validates_all_supported_states_and_restores_with_raii() {
        let _lock = crate::r#impl::test_env::lock();
        let guard = EnvGuard::capture();

        guard.set(None);
        assert_eq!(BatchSchedulerConfig::from_env(4).unwrap().max_batch(), 4);

        guard.set(Some("3".into()));
        assert_eq!(BatchSchedulerConfig::from_env(4).unwrap().max_batch(), 3);

        for value in ["malformed", "0", "5"] {
            guard.set(Some(value.into()));
            assert!(BatchSchedulerConfig::from_env(4).is_err());
        }

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;

            guard.set(Some(OsString::from_vec(vec![0xff])));
            let error = BatchSchedulerConfig::from_env(4).unwrap_err();
            assert!(error.contains("valid UTF-8"), "{error}");
        }
    }

    #[test]
    fn completion_metrics_preserve_all_logical_work() {
        let completions = [
            completed_task(9, 4, 1, 100, 1_000, 100, 40),
            completed_task(12, 2, 0, 40, 500, 50, 20),
            completed_task(9, 4, 1, 60, 750, 75, 30),
        ];

        let report = SchedulerReport::from_completions(&completions, 4).unwrap();

        assert_eq!(report.descriptor_count(), 3);
        assert_eq!(report.logical_items(), 10);
        assert_eq!(report.unique_submission_count(), 2);
        assert_eq!(report.lane_mask(), 0b11);
        assert_eq!(report.per_lane_completion_counts(), &[1, 2, 0, 0]);
        assert_eq!(report.total_accelerator_cycles(), 200);
        assert_eq!(report.total_matrix_bytes_read(), 2_250);
        assert_eq!(report.total_input_bytes_read(), 225);
        assert_eq!(report.total_output_bytes_written(), 90);
    }

    #[test]
    fn completion_metrics_reject_invalid_shape() {
        let valid = completed_task(1, 1, 0, 1, 1, 1, 1);
        assert!(SchedulerReport::from_completions(&[valid], 0).is_err());
        assert!(SchedulerReport::from_completions(&[valid], 65).is_err());

        let out_of_range = completed_task(1, 1, 2, 1, 1, 1, 1);
        assert!(SchedulerReport::from_completions(&[out_of_range], 2).is_err());

        let zero_batch = completed_task(1, 0, 0, 1, 1, 1, 1);
        assert!(SchedulerReport::from_completions(&[zero_batch], 1).is_err());
    }

    #[test]
    fn completion_metrics_reject_invalid_evidence_records() {
        let zero_submission = completed_task(0, 1, 0, 1, 1, 1, 1);
        let error = SchedulerReport::from_completions(&[zero_submission], 1).unwrap_err();
        assert!(error.contains("submission_id"), "{error}");

        let mut zero_request = completed_task(1, 1, 0, 1, 1, 1, 1);
        zero_request.task.request_id = 0;
        let error = SchedulerReport::from_completions(&[zero_request], 1).unwrap_err();
        assert!(error.contains("task request_id"), "{error}");

        let mut mismatched_request = completed_task(1, 1, 0, 1, 1, 1, 1);
        mismatched_request.completion.request_id = 2;
        let error = SchedulerReport::from_completions(&[mismatched_request], 1).unwrap_err();
        assert!(error.contains("does not match"), "{error}");

        let mut failed = completed_task(1, 1, 0, 1, 1, 1, 1);
        failed.completion.status = -5;
        let error = SchedulerReport::from_completions(&[failed], 1).unwrap_err();
        assert!(error.contains("completion status"), "{error}");

        let zero_range = completed_task(1, 1, 0, 0, 1, 1, 1);
        let error = SchedulerReport::from_completions(&[zero_range], 1).unwrap_err();
        assert!(error.contains("cycle range"), "{error}");

        let mut reversed_range = completed_task(1, 1, 0, 1, 1, 1, 1);
        reversed_range.completion.start_cycle = 2;
        reversed_range.completion.end_cycle = 1;
        let error = SchedulerReport::from_completions(&[reversed_range], 1).unwrap_err();
        assert!(error.contains("cycle range"), "{error}");

        let mut mismatched_cycles = completed_task(1, 1, 0, 2, 1, 1, 1);
        mismatched_cycles.completion.end_cycle = 1;
        let error = SchedulerReport::from_completions(&[mismatched_cycles], 1).unwrap_err();
        assert!(error.contains("cycle range"), "{error}");
    }

    #[test]
    fn completion_metrics_reject_accelerator_cycle_total_overflow() {
        let max = completed_task(1, 1, 0, u64::MAX - 1, 1, 1, 1);
        let two = completed_task(2, 1, 0, 2, 1, 1, 1);

        let error = SchedulerReport::from_completions(&[max, two], 1).unwrap_err();
        assert_eq!(error, "accelerator cycle total overflow");
    }

    #[test]
    fn completion_metrics_reject_matrix_byte_total_overflow() {
        let max = completed_task(1, 1, 0, 1, u64::MAX, 1, 1);
        let one = completed_task(2, 1, 0, 1, 1, 1, 1);

        let error = SchedulerReport::from_completions(&[max, one], 1).unwrap_err();
        assert_eq!(error, "matrix byte total overflow");
    }

    #[test]
    fn completion_metrics_reject_input_byte_total_overflow() {
        let max = completed_task(1, 1, 0, 1, 1, u64::MAX, 1);
        let one = completed_task(2, 1, 0, 1, 1, 1, 1);

        let error = SchedulerReport::from_completions(&[max, one], 1).unwrap_err();
        assert_eq!(error, "input byte total overflow");
    }

    #[test]
    fn completion_metrics_reject_output_byte_total_overflow() {
        let max = completed_task(1, 1, 0, 1, 1, 1, u64::MAX);
        let one = completed_task(2, 1, 0, 1, 1, 1, 1);

        let error = SchedulerReport::from_completions(&[max, one], 1).unwrap_err();
        assert_eq!(error, "output byte total overflow");
    }
}
