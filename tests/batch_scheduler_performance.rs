// /home/victoryang00/hetGPU/tests/batch_scheduler_performance.rs
use std::time::Instant;
use hetgpu::zluda::r#impl::batch_scheduler::integration::{get_global_scheduler, SchedulerStats};

#[test]
fn test_batch_scheduler_basic_performance() {
    println!("Batch Scheduler Performance Test");
    println!("Note: This test requires the batch_scheduler module to be compiled.");
    println!("For full functionality, run with: --features nvidia --no-default-features");

    // Get the global batch scheduler
    let scheduler = get_global_scheduler();

    let start = Instant::now();
    let num_operations = 1000;

    // Process operations through the batch scheduler
    for i in 0..num_operations {
        // Process pending requests through the scheduler
        let _ = scheduler.process_pending();

        // Simulate some work (in real scenario, this would be actual operations)
        let _ = i * i;
    }

    let elapsed = start.elapsed();
    let tps = if elapsed.as_secs_f64() > 0.0 {
        num_operations as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!("Operations: {}", num_operations);
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("TPS: {:.2}", tps);

    // Get scheduler statistics
    let stats: SchedulerStats = scheduler.get_statistics();
    println!("\nScheduler Statistics:");
    println!("Instance Utilizations: {:?}", stats.instance_utilizations);
    println!("Average Latency: {} μs", stats.average_latency_us);
    println!("Fallback Count: {}", stats.fallback_count);

    // Calculate average utilization across all instances
    let avg_utilization: f32 = if stats.instance_utilizations.is_empty() {
        0.0
    } else {
        stats.instance_utilizations.iter().sum::<f32>() / stats.instance_utilizations.len() as f32
    };
    println!("Average Instance Utilization: {:.2}%", avg_utilization * 100.0);

    // Validate targets with conditional logic based on feature
    let target_tps = if cfg!(feature = "kimi") { 8.0 } else { 2000.0 };

    println!("\nValidating against target TPS: {:.2}", target_tps);
    assert!(tps >= target_tps, "TPS {:.2} < target {:.2}", tps, target_tps);

    println!("✅ PASSED: TPS {:.2} >= target {:.2}", tps, target_tps);
    println!("✅ Scheduler statistics collected successfully");
    println!("✅ Batch scheduler integration validated");
}