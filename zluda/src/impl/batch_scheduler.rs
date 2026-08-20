use crate::r#impl::cxl_tmatmul_v3::CompletedTaskV3;
use std::collections::HashSet;

const BATCH_LIMIT_ENV: &str = "HETGPU_FPGA_BATCH_LIMIT";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchSlice {
    pub(crate) first: u32,
    pub(crate) count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BatchPlan {
    pub(crate) slices: Vec<BatchSlice>,
}

impl BatchPlan {
    pub(crate) fn new(logical_batch: u32, max_batch: u32) -> Result<Self, String> {
        if logical_batch == 0 {
            return Err("logical_batch must be non-zero".into());
        }
        if max_batch == 0 {
            return Err("max_batch must be non-zero".into());
        }

        let mut slices = Vec::new();
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BatchSchedulerConfig {
    pub(crate) max_batch: u32,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SchedulerReport {
    pub(crate) descriptor_count: u64,
    pub(crate) logical_items: u64,
    pub(crate) unique_submission_count: u64,
    pub(crate) lane_mask: u64,
    pub(crate) per_lane_completion_counts: Vec<u64>,
    pub(crate) total_accelerator_cycles: u64,
    pub(crate) total_matrix_bytes_read: u64,
    pub(crate) total_input_bytes_read: u64,
    pub(crate) total_output_bytes_written: u64,
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
        task.batch = batch;

        let mut completion = CompletionV3::default();
        completion.lane_used = lane_used;
        completion.accelerator_cycles = accelerator_cycles;
        completion.matrix_bytes_read = matrix_bytes_read;
        completion.input_bytes_read = input_bytes_read;
        completion.output_bytes_written = output_bytes_written;

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
            plan.slices,
            vec![
                BatchSlice { first: 0, count: 4 },
                BatchSlice { first: 4, count: 4 },
            ]
        );
    }

    #[test]
    fn slices_remainder_in_logical_order() {
        let plan = BatchPlan::new(10, 4).unwrap();

        assert_eq!(
            plan.slices,
            vec![
                BatchSlice { first: 0, count: 4 },
                BatchSlice { first: 4, count: 4 },
                BatchSlice { first: 8, count: 2 },
            ]
        );
    }

    #[test]
    fn configured_limit_can_only_lower_live_max_batch() {
        assert_eq!(BatchSchedulerConfig::parse(None, 4).unwrap().max_batch, 4);
        assert_eq!(
            BatchSchedulerConfig::parse(Some("2"), 4).unwrap().max_batch,
            2
        );

        for value in [Some("0"), Some("not-a-number"), Some("5")] {
            assert!(BatchSchedulerConfig::parse(value, 4).is_err());
        }
        assert!(BatchSchedulerConfig::parse(None, 0).is_err());
    }

    #[test]
    fn env_wrapper_reads_fpga_batch_limit() {
        let _lock = crate::r#impl::test_env::lock();
        let original = std::env::var_os("HETGPU_FPGA_BATCH_LIMIT");

        std::env::set_var("HETGPU_FPGA_BATCH_LIMIT", "3");
        let result = BatchSchedulerConfig::from_env(4);

        match original {
            Some(value) => std::env::set_var("HETGPU_FPGA_BATCH_LIMIT", value),
            None => std::env::remove_var("HETGPU_FPGA_BATCH_LIMIT"),
        }
        assert_eq!(result.unwrap().max_batch, 3);
    }

    #[test]
    fn completion_metrics_preserve_all_logical_work() {
        let completions = [
            completed_task(9, 4, 1, 100, 1_000, 100, 40),
            completed_task(12, 2, 0, 40, 500, 50, 20),
            completed_task(9, 4, 1, 60, 750, 75, 30),
        ];

        let report = SchedulerReport::from_completions(&completions, 4).unwrap();

        assert_eq!(report.descriptor_count, 3);
        assert_eq!(report.logical_items, 10);
        assert_eq!(report.unique_submission_count, 2);
        assert_eq!(report.lane_mask, 0b11);
        assert_eq!(report.per_lane_completion_counts, vec![1, 2, 0, 0]);
        assert_eq!(report.total_accelerator_cycles, 200);
        assert_eq!(report.total_matrix_bytes_read, 2_250);
        assert_eq!(report.total_input_bytes_read, 225);
        assert_eq!(report.total_output_bytes_written, 90);
    }

    #[test]
    fn completion_metrics_reject_invalid_shape_and_overflow() {
        let valid = completed_task(1, 1, 0, 1, 1, 1, 1);
        assert!(SchedulerReport::from_completions(&[valid], 0).is_err());
        assert!(SchedulerReport::from_completions(&[valid], 65).is_err());

        let out_of_range = completed_task(1, 1, 2, 1, 1, 1, 1);
        assert!(SchedulerReport::from_completions(&[out_of_range], 2).is_err());

        let zero_batch = completed_task(1, 0, 0, 1, 1, 1, 1);
        assert!(SchedulerReport::from_completions(&[zero_batch], 1).is_err());

        let max = completed_task(1, 1, 0, u64::MAX, u64::MAX, u64::MAX, u64::MAX);
        let one = completed_task(2, 1, 0, 1, 1, 1, 1);
        assert!(SchedulerReport::from_completions(&[max, one], 1).is_err());
    }
}
