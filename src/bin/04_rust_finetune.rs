use clap::{Parser, ValueEnum};
use mistral_fintune::{AppResult, LocalPath, load_and_validate_train_rows, summarize_train_rows};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "04_rust_finetune",
    about = "Rust-only fine-tune manifest + preflight for Candle/Burn"
)]
struct Cli {
    #[arg(
        long,
        default_value = "data/train_mistral.jsonl",
        help = "Path to cleaned train split JSONL"
    )]
    train_path: PathBuf,

    #[arg(
        long,
        default_value = "data/valid_mistral.jsonl",
        help = "Path to cleaned validation split JSONL"
    )]
    valid_path: PathBuf,

    #[arg(long, value_enum, default_value_t = Backend::Candle, help = "Rust backend to target")]
    backend: Backend,

    #[arg(
        long,
        default_value = "mistralai/Ministral-3-3B-Base-2512",
        help = "HF model identifier or local path"
    )]
    model_id_or_path: String,

    #[arg(long, default_value_t = 64, help = "LoRA rank")]
    lora_rank: u32,

    #[arg(long, default_value_t = 300, help = "Approximate training step budget")]
    max_steps: u64,

    #[arg(long, default_value_t = 1, help = "Per-device batch size")]
    batch_size: u32,

    #[arg(long, default_value_t = 32768, help = "Sequence length")]
    seq_len: u32,

    #[arg(long, default_value_t = 42, help = "Random seed")]
    seed: u64,

    #[arg(long, default_value_t = 0.00006, help = "Learning rate")]
    learning_rate: f32,

    #[arg(
        long,
        default_value = "data/rust_finetune_manifest.yaml",
        help = "Where to write the generated manifest"
    )]
    manifest_path: PathBuf,

    #[arg(
        long,
        default_value = "artifacts/rust-finetune-run",
        help = "Directory where training outputs would be written"
    )]
    run_dir: PathBuf,
}

#[derive(ValueEnum, Clone, Debug)]
enum Backend {
    Candle,
    Burn,
}

#[derive(Serialize)]
struct FinetuneManifest {
    data: DataBlock,
    model_id_or_path: String,
    lora: LoraBlock,
    #[serde(rename = "seed")]
    seed: u64,
    batch_size: u32,
    max_steps: u64,
    seq_len: u32,
    optim: OptimBlock,
    backend: String,
    run_dir: String,
    rust_preflight: RustPreflight,
}

#[derive(Serialize)]
struct DataBlock {
    instruct_data: String,
    #[serde(rename = "eval_instruct_data")]
    eval_instruct_data: String,
}

#[derive(Serialize)]
struct LoraBlock {
    rank: u32,
}

#[derive(Serialize)]
struct OptimBlock {
    lr: f32,
    weight_decay: f32,
    pct_start: f32,
}

#[derive(Serialize)]
struct RustPreflight {
    approx_train_rows: usize,
    approx_train_tokens: usize,
    approx_eval_rows: usize,
    approx_eval_tokens: usize,
    total_messages: usize,
}

fn main() -> AppResult<()> {
    let cli = Cli::parse();

    let train_path = LocalPath::new(cli.train_path.to_string_lossy());
    let valid_path = LocalPath::new(cli.valid_path.to_string_lossy());

    let train_rows = load_and_validate_train_rows(&train_path)?;
    let valid_rows = load_and_validate_train_rows(&valid_path)?;

    let (train_rows_count, train_messages, train_tokens) = summarize_train_rows(&train_rows);
    let (valid_rows_count, _, valid_tokens) = summarize_train_rows(&valid_rows);
    let total_messages = train_messages + summarize_train_rows(&valid_rows).1;

    let backend_name = match cli.backend {
        Backend::Candle => "candle",
        Backend::Burn => "burn",
    }
    .to_string();

    let manifest = FinetuneManifest {
        data: DataBlock {
            instruct_data: cli.train_path.to_string_lossy().to_string(),
            eval_instruct_data: cli.valid_path.to_string_lossy().to_string(),
        },
        model_id_or_path: cli.model_id_or_path,
        lora: LoraBlock {
            rank: cli.lora_rank,
        },
        seed: cli.seed,
        batch_size: cli.batch_size,
        max_steps: cli.max_steps,
        seq_len: cli.seq_len,
        optim: OptimBlock {
            lr: cli.learning_rate,
            weight_decay: 0.1,
            pct_start: 0.05,
        },
        backend: backend_name.clone(),
        run_dir: cli.run_dir.to_string_lossy().to_string(),
        rust_preflight: RustPreflight {
            approx_train_rows: train_rows_count,
            approx_train_tokens: train_tokens,
            approx_eval_rows: valid_rows_count,
            approx_eval_tokens: valid_tokens,
            total_messages,
        },
    };

    let yaml = serde_yaml::to_string(&manifest)?;

    if let Some(parent) = cli.manifest_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(&cli.manifest_path)?;
    file.write_all(yaml.as_bytes())?;

    println!("manifest written: {}", cli.manifest_path.to_string_lossy());
    println!("train rows     : {}", train_rows_count);
    println!("valid rows     : {}", valid_rows_count);
    println!("train tokens≈  : {}", train_tokens);
    println!("valid tokens≈  : {}", valid_tokens);
    println!("total messages : {}", total_messages);
    println!("backend target : {}", backend_name);

    assert_backend_available(&cli.backend)?;
    print_next_step(cli.backend, &cli.run_dir);

    Ok(())
}

fn assert_backend_available(backend: &Backend) -> AppResult<()> {
    match backend {
        Backend::Candle => {
            #[cfg(feature = "candle")]
            return Ok(());

            #[cfg(not(feature = "candle"))]
            return Err(mistral_fintune::AppError::BackendUnavailable {
                backend: "candle".to_string(),
                reason: "enable with `--features candle` and add a Candle training loop module"
                    .to_string(),
            });
        }
        Backend::Burn => {
            #[cfg(feature = "burn")]
            return Ok(());

            #[cfg(not(feature = "burn"))]
            return Err(mistral_fintune::AppError::BackendUnavailable {
                backend: "burn".to_string(),
                reason: "enable with `--features burn` and add a Burn training loop module"
                    .to_string(),
            });
        }
    }
}

fn print_next_step(backend: Backend, run_dir: &PathBuf) {
    match backend {
        Backend::Candle => {
            println!(
                "next step: add Candle runtime adapter and launch step in this crate (manifest at: {:?})",
                run_dir
            );
        }
        Backend::Burn => {
            println!(
                "next step: add Burn LoRA training adapter and launch step in this crate (manifest at: {:?})",
                run_dir
            );
        }
    }
}
