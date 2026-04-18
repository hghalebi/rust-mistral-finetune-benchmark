use clap::{Parser, ValueEnum};
use mistral_fintune::{
    AppResult, LocalPath, backend_runtime::RuntimeCommand, load_and_validate_train_rows,
    summarize_train_rows,
};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Parser, Debug)]
#[command(
    name = "05_rust_train",
    about = "Run Rust-managed Candle/Burn fine-tune command"
)]
struct Cli {
    #[arg(
        long,
        default_value = "data/train_mistral.jsonl",
        help = "Training split path"
    )]
    train_path: PathBuf,

    #[arg(
        long,
        default_value = "data/valid_mistral.jsonl",
        help = "Validation split path"
    )]
    valid_path: PathBuf,

    #[arg(long, value_enum, default_value_t = Backend::Candle, help = "Backend target")]
    backend: Backend,

    #[arg(
        long,
        default_value = "mistralai/Ministral-3-3B-Base-2512",
        help = "HF identifier or local model path"
    )]
    model_id_or_path: String,

    #[arg(long, default_value_t = 64, help = "LoRA rank")]
    lora_rank: u32,

    #[arg(long, default_value_t = 300, help = "Training step budget")]
    max_steps: u64,

    #[arg(long, default_value_t = 1, help = "Per-step batch size")]
    batch_size: u32,

    #[arg(long, default_value_t = 3072, help = "Sequence length")]
    seq_len: u32,

    #[arg(long, default_value_t = 0.00006, help = "Learning rate")]
    learning_rate: f32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    seed: u64,

    #[arg(
        long,
        default_value = "artifacts/rust-finetune-run",
        help = "Output directory"
    )]
    run_dir: PathBuf,

    #[arg(
        long,
        default_value = "candle-train",
        help = "Backend command to execute"
    )]
    backend_train_binary: String,

    #[arg(
        long,
        value_name = "ARG",
        help = "Extra args passed to backend command. Supports placeholders like {model}, {train_path}, {valid_path}, {run_dir}"
    )]
    backend_train_arg: Vec<String>,

    #[arg(
        long,
        default_value_t = false,
        help = "Print resolved command and exit"
    )]
    dry_run: bool,
}

#[derive(ValueEnum, Clone, Debug)]
enum Backend {
    Candle,
    Burn,
}

#[derive(Serialize)]
struct TrainRunManifest {
    model_id_or_path: String,
    train_path: String,
    valid_path: String,
    run_dir: String,
    lora_rank: u32,
    max_steps: u64,
    batch_size: u32,
    seq_len: u32,
    learning_rate: f32,
    seed: u64,
    backend: String,
    resolved_command: Vec<String>,
    row_stats: TrainRowStats,
}

#[derive(Serialize)]
struct TrainRowStats {
    train_rows: usize,
    valid_rows: usize,
    train_messages: usize,
    valid_messages: usize,
    approx_train_tokens: usize,
    approx_valid_tokens: usize,
}

fn main() -> AppResult<()> {
    let cli = Cli::parse();

    assert_backend_enabled(&cli.backend)?;

    let train_path = LocalPath::new(cli.train_path.to_string_lossy());
    let valid_path = LocalPath::new(cli.valid_path.to_string_lossy());
    let run_dir = LocalPath::new(cli.run_dir.to_string_lossy());

    let train_rows = load_and_validate_train_rows(&train_path)?;
    let valid_rows = load_and_validate_train_rows(&valid_path)?;

    let (train_rows_count, train_messages, train_tokens) = summarize_train_rows(&train_rows);
    let (valid_rows_count, valid_messages, valid_tokens) = summarize_train_rows(&valid_rows);

    let default_args: Vec<String> = if cli.backend_train_arg.is_empty() {
        vec![
            "--backend".into(),
            "{backend}".into(),
            "--model".into(),
            "{model}".into(),
            "--train".into(),
            "{train_path}".into(),
            "--valid".into(),
            "{valid_path}".into(),
            "--out".into(),
            "{run_dir}".into(),
            "--lora-rank".into(),
            "{lora_rank}".into(),
            "--batch-size".into(),
            "{batch_size}".into(),
            "--seq-len".into(),
            "{seq_len}".into(),
            "--max-steps".into(),
            "{max_steps}".into(),
            "--learning-rate".into(),
            "{learning_rate}".into(),
            "--seed".into(),
            "{seed}".into(),
        ]
    } else {
        cli.backend_train_arg.clone()
    };

    let backend_s = match cli.backend {
        Backend::Candle => "candle",
        Backend::Burn => "burn",
    }
    .to_string();

    let vars: Vec<(&str, String)> = vec![
        ("backend", backend_s.clone()),
        ("model", cli.model_id_or_path.clone()),
        ("train_path", cli.train_path.to_string_lossy().to_string()),
        ("valid_path", cli.valid_path.to_string_lossy().to_string()),
        ("run_dir", run_dir.as_str().to_string()),
        ("lora_rank", cli.lora_rank.to_string()),
        ("batch_size", cli.batch_size.to_string()),
        ("seq_len", cli.seq_len.to_string()),
        ("max_steps", cli.max_steps.to_string()),
        ("learning_rate", cli.learning_rate.to_string()),
        ("seed", cli.seed.to_string()),
    ];

    let vars_map: HashMap<&str, &str> = vars
        .iter()
        .map(|(key, value)| (*key, value.as_str()))
        .collect();

    let command = RuntimeCommand {
        binary: cli.backend_train_binary.clone(),
        args: default_args,
    };

    let manifest = TrainRunManifest {
        model_id_or_path: cli.model_id_or_path,
        train_path: cli.train_path.to_string_lossy().to_string(),
        valid_path: cli.valid_path.to_string_lossy().to_string(),
        run_dir: run_dir.as_str().to_string(),
        lora_rank: cli.lora_rank,
        max_steps: cli.max_steps,
        batch_size: cli.batch_size,
        seq_len: cli.seq_len,
        learning_rate: cli.learning_rate,
        seed: cli.seed,
        backend: backend_s.clone(),
        resolved_command: resolved_command(&command, &vars_map),
        row_stats: TrainRowStats {
            train_rows: train_rows_count,
            valid_rows: valid_rows_count,
            train_messages,
            valid_messages,
            approx_train_tokens: train_tokens,
            approx_valid_tokens: valid_tokens,
        },
    };

    println!("backend      : {backend_s}");
    println!("train rows   : {}", manifest.row_stats.train_rows);
    println!("valid rows   : {}", manifest.row_stats.valid_rows);
    println!("run dir      : {}", manifest.run_dir);
    println!("command      : {}", manifest.resolved_command.join(" "));

    std::fs::create_dir_all(run_dir.as_str())?;

    let manifest_path = {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|v| v.as_secs())
            .unwrap_or(0);
        format!("{}/train_manifest_{}.json", run_dir.as_str(), ts)
    };

    if cli.dry_run {
        println!("dry_run enabled, command not executed");
        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, manifest_json)?;
        println!("manifest written: {manifest_path}");
        return Ok(());
    }

    let stdout = command.run_with_vars(&vars_map)?;

    let final_log = format!("backend command output\n--------------------\n{stdout}");

    let stdout_path = format!("{}/train_stdout.log", run_dir.as_str());
    let manifest_json = serde_json::to_string_pretty(&manifest)?;

    std::fs::write(&manifest_path, manifest_json)?;
    std::fs::write(&stdout_path, final_log)?;

    println!("manifest written: {manifest_path}");
    println!("stdout written  : {stdout_path}");

    Ok(())
}

fn resolved_command(command: &RuntimeCommand, vars: &HashMap<&str, &str>) -> Vec<String> {
    std::iter::once(command.binary.clone())
        .chain(
            command
                .args
                .iter()
                .map(|arg| replace_placeholders(arg, vars)),
        )
        .collect()
}

fn replace_placeholders(value: &str, vars: &HashMap<&str, &str>) -> String {
    let mut out = value.to_string();
    for (key, replacement) in vars {
        out = out.replace(&format!("{{{key}}}"), replacement);
    }
    out
}

fn assert_backend_enabled(backend: &Backend) -> AppResult<()> {
    match backend {
        Backend::Candle => {
            #[cfg(feature = "candle")]
            return Ok(());

            #[cfg(not(feature = "candle"))]
            return Err(mistral_fintune::AppError::BackendUnavailable {
                backend: "candle".to_string(),
                reason: "enable with --features candle".to_string(),
            });
        }
        Backend::Burn => {
            #[cfg(feature = "burn")]
            return Ok(());

            #[cfg(not(feature = "burn"))]
            return Err(mistral_fintune::AppError::BackendUnavailable {
                backend: "burn".to_string(),
                reason: "enable with --features burn".to_string(),
            });
        }
    }
}
