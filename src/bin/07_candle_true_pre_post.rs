#[cfg(feature = "candle")]
use clap::Parser;
#[cfg(feature = "candle")]
use mistral_fintune::candle_backend::{
    AggregateMetrics, ByteTokenizer, CandleRunManifest, TinyLmConfig, TrainHyperparameters,
    TrainSummary, build_model, cpu_device, eval_examples_from_rows, evaluate_pair, load_model,
    render_leaderboard_markdown, save_checkpoint, train_model, training_texts_from_rows,
    write_json, write_manifest,
};
#[cfg(feature = "candle")]
use mistral_fintune::{AppResult, LocalPath, load_and_validate_train_rows};
#[cfg(feature = "candle")]
use serde::Serialize;
#[cfg(feature = "candle")]
use std::path::PathBuf;

#[cfg(feature = "candle")]
#[derive(Parser, Debug)]
#[command(
    name = "07_candle_true_pre_post",
    about = "Run a true in-repo Candle pre-vs-post fine-tune benchmark"
)]
struct Cli {
    #[arg(long, default_value = "data/train_mistral.jsonl")]
    train_path: PathBuf,

    #[arg(long, default_value = "data/valid_mistral.jsonl")]
    valid_path: PathBuf,

    #[arg(long, default_value = "artifacts/candle-true-pre-post")]
    run_dir: PathBuf,

    #[arg(long, default_value_t = 512)]
    train_limit: usize,

    #[arg(long, default_value_t = 32)]
    eval_limit: usize,

    #[arg(long, default_value_t = 300)]
    steps: usize,

    #[arg(long, default_value_t = 0.0003)]
    learning_rate: f64,

    #[arg(long, default_value_t = 256)]
    max_seq_len: usize,

    #[arg(long, default_value_t = 192)]
    d_model: usize,

    #[arg(long, default_value_t = 4)]
    n_heads: usize,

    #[arg(long, default_value_t = 4)]
    n_layers: usize,

    #[arg(long, default_value_t = 768)]
    mlp_hidden: usize,

    #[arg(long, default_value_t = 200)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 25)]
    log_every: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[cfg(feature = "candle")]
#[derive(Serialize)]
struct TruePrePostReport {
    backend: String,
    dataset: String,
    train_summary: TrainSummary,
    base_checkpoint: String,
    finetuned_checkpoint: String,
    training_wall_ms: u128,
    eval_wall_ms: u128,
    base: AggregateMetrics,
    finetuned: AggregateMetrics,
    exact_match_delta: f64,
    rouge_l_delta: f64,
    response_len_delta: f64,
    latency_delta_ms: f64,
    rows: Vec<mistral_fintune::candle_backend::RowMetrics>,
}

#[cfg(feature = "candle")]
fn main() -> AppResult<()> {
    let cli = Cli::parse();

    std::fs::create_dir_all(&cli.run_dir)?;

    let train_rows =
        load_and_validate_train_rows(&LocalPath::new(cli.train_path.to_string_lossy()))?;
    let valid_rows =
        load_and_validate_train_rows(&LocalPath::new(cli.valid_path.to_string_lossy()))?;

    let train_texts = training_texts_from_rows(&train_rows, cli.train_limit)?;
    let eval_examples = eval_examples_from_rows(&valid_rows, cli.eval_limit)?;
    let tokenizer_corpus = train_texts
        .iter()
        .cloned()
        .chain(
            eval_examples
                .iter()
                .flat_map(|row| [row.prompt.clone(), row.target.clone()]),
        )
        .collect::<Vec<_>>();
    let tokenizer = ByteTokenizer::from_texts(&tokenizer_corpus)?;
    let config = TinyLmConfig {
        vocab_size: tokenizer.vocab_size(),
        max_seq_len: cli.max_seq_len,
        d_model: cli.d_model,
        n_heads: cli.n_heads,
        n_layers: cli.n_layers,
        mlp_hidden: cli.mlp_hidden,
    };

    let device = cpu_device();
    let base_checkpoint = cli.run_dir.join("base.safetensors");
    let finetuned_checkpoint = cli.run_dir.join("finetuned.safetensors");
    let manifest_path = cli.run_dir.join("run_manifest.json");
    let report_path = cli.run_dir.join("true_pre_post_report.json");
    let leaderboard_path = cli.run_dir.join("true_pre_post_leaderboard.md");

    let (var_map, model) = build_model(&config, &device)?;
    save_checkpoint(&var_map, &base_checkpoint)?;

    let train_started = std::time::Instant::now();
    let train_summary = train_model(
        &model,
        &var_map,
        &tokenizer,
        &train_texts,
        TrainHyperparameters {
            steps: cli.steps,
            learning_rate: cli.learning_rate,
            seed: cli.seed,
            log_every: cli.log_every,
        },
    )?;
    let training_wall_ms = train_started.elapsed().as_millis();
    save_checkpoint(&var_map, &finetuned_checkpoint)?;

    let (_, base_model) = load_model(&config, &base_checkpoint, &device)?;
    let (_, finetuned_model) = load_model(&config, &finetuned_checkpoint, &device)?;
    let eval_started = std::time::Instant::now();
    let (base, finetuned, rows) = evaluate_pair(
        &base_model,
        &finetuned_model,
        &tokenizer,
        &eval_examples,
        cli.max_new_tokens,
    )?;
    let eval_wall_ms = eval_started.elapsed().as_millis();

    let manifest = CandleRunManifest {
        model: config,
        tokenizer: tokenizer.clone(),
        train_limit: cli.train_limit,
        eval_limit: cli.eval_limit,
        steps: cli.steps,
        learning_rate: cli.learning_rate,
        seed: cli.seed,
        base_checkpoint: base_checkpoint.to_string_lossy().to_string(),
        finetuned_checkpoint: finetuned_checkpoint.to_string_lossy().to_string(),
    };
    write_manifest(&manifest_path, &manifest)?;

    let report = TruePrePostReport {
        backend: "candle".to_string(),
        dataset: cli.valid_path.to_string_lossy().to_string(),
        train_summary: train_summary.clone(),
        base_checkpoint: base_checkpoint.to_string_lossy().to_string(),
        finetuned_checkpoint: finetuned_checkpoint.to_string_lossy().to_string(),
        training_wall_ms,
        eval_wall_ms,
        exact_match_delta: finetuned.exact_match_rate - base.exact_match_rate,
        rouge_l_delta: finetuned.avg_rouge_l - base.avg_rouge_l,
        response_len_delta: finetuned.avg_response_len - base.avg_response_len,
        latency_delta_ms: finetuned.avg_latency_ms - base.avg_latency_ms,
        base,
        finetuned,
        rows,
    };
    write_json(&report_path, &report)?;

    let markdown = render_leaderboard_markdown(
        "True Pre/Post Fine-Tune Leaderboard",
        &report.dataset,
        &report.train_summary,
        &report.base,
        &report.finetuned,
    );
    std::fs::write(&leaderboard_path, markdown)?;

    println!("base checkpoint      : {}", report.base_checkpoint);
    println!("finetuned checkpoint : {}", report.finetuned_checkpoint);
    println!("base exact match     : {:.3}", report.base.exact_match_rate);
    println!(
        "ft exact match       : {:.3}",
        report.finetuned.exact_match_rate
    );
    println!("exact-match delta    : {:.3}", report.exact_match_delta);
    println!("ROUGE-L delta        : {:.3}", report.rouge_l_delta);
    println!("latency delta (ms)   : {:.3}", report.latency_delta_ms);
    println!("training wall (ms)   : {}", report.training_wall_ms);
    println!("eval wall (ms)       : {}", report.eval_wall_ms);
    println!("report written       : {}", report_path.to_string_lossy());
    println!(
        "leaderboard written  : {}",
        leaderboard_path.to_string_lossy()
    );

    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() -> mistral_fintune::AppResult<()> {
    Err(mistral_fintune::AppError::BackendUnavailable {
        backend: "candle".to_string(),
        reason: "enable with --features candle".to_string(),
    })
}
