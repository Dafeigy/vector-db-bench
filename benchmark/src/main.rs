use clap::Parser;
use std::time::Instant;

mod anti_cheat;
mod loader;
mod runner;
mod scorer;

#[derive(Parser)]
#[command(name = "vector-db-benchmark")]
#[command(about = "Benchmark client for Vector DB evaluation")]
struct Cli {
    /// Server URL
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    server_url: String,

    /// Number of concurrent query threads
    #[arg(long, default_value_t = 4)]
    concurrency: usize,

    /// Number of warmup queries
    #[arg(long, default_value_t = 1000)]
    warmup: usize,

    /// Path to base vectors JSON file
    #[arg(long)]
    base_vectors: String,

    /// Path to query vectors JSON file
    #[arg(long)]
    query_vectors: String,

    /// Path to ground truth JSON file
    #[arg(long)]
    ground_truth: String,

    /// Recall threshold (0.0 to 1.0)
    #[arg(long, default_value_t = 0.95)]
    recall_threshold: f64,

    /// Random seed for query shuffling
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Batch size for bulk insert
    #[arg(long, default_value_t = 1000)]
    batch_size: usize,

    /// Maximum number of queries to run (0 = all)
    #[arg(long, default_value_t = 0)]
    max_queries: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let client = reqwest::Client::new();

    // 1. Load and insert base vectors
    eprintln!("Loading base vectors from {}...", cli.base_vectors);
    let base_vectors = loader::load_vectors_from_file(&cli.base_vectors).await?;
    eprintln!("Loaded {} base vectors, inserting...", base_vectors.len());
    let inserted = loader::bulk_insert_vectors(&client, &cli.server_url, &base_vectors, cli.batch_size).await?;
    eprintln!("Inserted {} vectors", inserted);

    // 2. Load query vectors and ground truth
    eprintln!("Loading query vectors from {}...", cli.query_vectors);
    let all_query_vectors = loader::load_vectors_from_file(&cli.query_vectors).await?;
    eprintln!("Loading ground truth from {}...", cli.ground_truth);
    let all_ground_truth = scorer::load_ground_truth(&cli.ground_truth).await?;

    // Apply max_queries limit if set
    let (query_vectors, ground_truth) = if cli.max_queries > 0 && cli.max_queries < all_query_vectors.len() {
        eprintln!("Limiting to {} queries (out of {})", cli.max_queries, all_query_vectors.len());
        let qv: Vec<_> = all_query_vectors.into_iter().take(cli.max_queries).collect();
        let gt: Vec<_> = all_ground_truth.into_iter().take(cli.max_queries).collect();
        (qv, gt)
    } else {
        (all_query_vectors, all_ground_truth)
    };

    // 3. Warmup
    eprintln!("Running {} warmup queries...", cli.warmup);
    runner::run_warmup(&client, &cli.server_url, &query_vectors, cli.warmup).await;

    // 4. Run benchmark queries
    eprintln!(
        "Running {} queries with concurrency {}...",
        query_vectors.len(),
        cli.concurrency
    );
    let start = Instant::now();
    let query_results = runner::run_queries(
        &client,
        &cli.server_url,
        &query_vectors,
        cli.concurrency,
        cli.seed,
    )
    .await;
    let duration_secs = start.elapsed().as_secs_f64();

    // 5. Compute results
    let result = scorer::compute_benchmark_result(
        &query_results,
        &ground_truth,
        duration_secs,
        cli.concurrency,
        cli.recall_threshold,
    );

    // 6. Anti-cheat detection
    let result_ids: Vec<Vec<u64>> = query_results
        .iter()
        .map(|qr| qr.results.iter().map(|r| r.id).collect())
        .collect();
    let anti_cheat = anti_cheat::detect_hardcoded_results(&result_ids);
    eprintln!("Anti-cheat: {}", anti_cheat.message);

    // 7. Output JSON to stdout
    let output = serde_json::json!({
        "benchmark": result,
        "anti_cheat": anti_cheat,
    });
    let json = serde_json::to_string_pretty(&output)?;
    println!("{}", json);

    Ok(())
}
