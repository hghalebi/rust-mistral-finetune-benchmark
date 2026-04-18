use clap::{Parser, ValueEnum};
use mistral_fintune::{
    AppError, AppResult, LocalPath, backend_runtime::RuntimeCommand, eval_row_metrics,
    load_and_validate_train_rows, split_for_eval,
};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "06_rust_evaluate",
    about = "Run base vs fine-tuned inference and compute metrics (Rust-only)"
)]
struct Cli {
    #[arg(
        long,
        default_value = "data/valid_mistral.jsonl",
        help = "Held-out split"
    )]
    dataset_path: PathBuf,

    #[arg(long, value_enum, default_value_t = Backend::Candle, help = "Backend used for eval")]
    backend: Backend,

    #[arg(
        long,
        default_value = "mistralai/Ministral-3-3B-Base-2512",
        help = "Base model identifier or local path"
    )]
    base_model: String,

    #[arg(
        long,
        default_value = "artifacts/rust-finetune-run",
        help = "Fine-tuned model path / adapter checkpoint"
    )]
    finetune_model: String,

    #[arg(
        long,
        default_value_t = 512,
        help = "Generation max tokens for each inference"
    )]
    max_new_tokens: u32,

    #[arg(long, default_value_t = 0.2, help = "Sampling temperature")]
    temperature: f32,

    #[arg(
        long,
        default_value_t = 42,
        help = "Generation seed for reproducibility"
    )]
    seed: u64,

    #[arg(
        long,
        default_value = "candle-infer",
        help = "Backend command for single-sample generation"
    )]
    backend_infer_binary: String,

    #[arg(long, value_name = "ARG", help = "Extra args for inference backend")]
    backend_infer_arg: Vec<String>,

    #[arg(long, value_name = "N", help = "Evaluate only first N rows")]
    limit: Option<usize>,

    #[arg(
        long,
        default_value = "artifacts/rust_finetune_eval.json",
        help = "Output report path"
    )]
    report_path: PathBuf,

    #[arg(
        long,
        default_value_t = false,
        help = "Emit resolved command and skip inference"
    )]
    dry_run: bool,
}

#[derive(ValueEnum, Clone, Debug)]
enum Backend {
    Candle,
    Burn,
}

#[derive(Serialize)]
struct EvalRowReport {
    index: usize,
    prompt: String,
    target: String,
    base_output: String,
    ft_output: String,
    base_exact_match: bool,
    ft_exact_match: bool,
    base_rouge_l: f64,
    ft_rouge_l: f64,
    base_response_len: usize,
    ft_response_len: usize,
    base_latency_ms: u128,
    ft_latency_ms: u128,
}

#[derive(Serialize, Clone, Copy)]
struct Aggregate {
    exact_match_rate: f64,
    avg_rouge_l: f64,
    avg_response_len: f64,
    avg_latency_ms: f64,
    count: usize,
}

#[derive(Serialize)]
struct EvaluationReport {
    dataset: String,
    base_model: String,
    finetune_model: String,
    backend: String,
    base: Aggregate,
    finetune: Aggregate,
    exact_match_delta: f64,
    rouge_l_delta: f64,
    latency_delta_ms: f64,
    samples: Vec<EvalRowReport>,
}

fn main() -> AppResult<()> {
    let cli = Cli::parse();

    assert_backend_enabled(&cli.backend)?;

    let dataset_path = LocalPath::new(cli.dataset_path.to_string_lossy());
    let rows = load_and_validate_train_rows(&dataset_path)?;
    let eval_rows = if let Some(limit) = cli.limit {
        rows.into_iter().take(limit).collect::<Vec<_>>()
    } else {
        rows
    };

    if eval_rows.is_empty() {
        return Err(AppError::Validation(
            "dataset has no rows to evaluate".to_string(),
        ));
    }

    let args = if cli.backend_infer_arg.is_empty() {
        vec![
            "--backend".into(),
            "{backend}".into(),
            "--model".into(),
            "{model}".into(),
            "--prompt".into(),
            "{prompt}".into(),
            "--max-new-tokens".into(),
            "{max_new_tokens}".into(),
            "--temperature".into(),
            "{temperature}".into(),
            "--seed".into(),
            "{seed}".into(),
        ]
    } else {
        cli.backend_infer_arg.clone()
    };

    let command = RuntimeCommand {
        binary: cli.backend_infer_binary,
        args,
    };

    let backend_s = match cli.backend {
        Backend::Candle => "candle",
        Backend::Burn => "burn",
    };

    let backend_s_owned = backend_s.to_string();
    let base_model_s = cli.base_model.clone();
    let ft_model_s = cli.finetune_model.clone();
    let max_new_tokens_s = cli.max_new_tokens.to_string();
    let temperature_s = cli.temperature.to_string();
    let seed_s = cli.seed.to_string();

    let base_placeholder_vars: HashMap<&str, &str> = vec![
        ("backend", backend_s_owned.as_str()),
        ("model", base_model_s.as_str()),
        ("max_new_tokens", max_new_tokens_s.as_str()),
        ("temperature", temperature_s.as_str()),
        ("seed", seed_s.as_str()),
        ("prompt", "{prompt}"),
    ]
    .into_iter()
    .collect();

    if cli.dry_run {
        let base_cmd = resolved_command(&command, &base_placeholder_vars);
        let ft_placeholder_vars: HashMap<&str, &str> = vec![
            ("backend", backend_s_owned.as_str()),
            ("model", ft_model_s.as_str()),
            ("max_new_tokens", max_new_tokens_s.as_str()),
            ("temperature", temperature_s.as_str()),
            ("seed", seed_s.as_str()),
            ("prompt", "{prompt}"),
        ]
        .into_iter()
        .collect();

        let ft_cmd = resolved_command(&command, &ft_placeholder_vars);

        println!("backend          : {backend_s}");
        println!("base command     : {}", base_cmd.join(" "));
        println!("finetune command : {}", ft_cmd.join(" "));
        println!("dry-run active; no inference executed");
        let report = EvaluationReport {
            dataset: cli.dataset_path.to_string_lossy().to_string(),
            base_model: cli.base_model,
            finetune_model: cli.finetune_model,
            backend: backend_s.to_string(),
            base: Aggregate {
                exact_match_rate: 0.0,
                avg_rouge_l: 0.0,
                avg_response_len: 0.0,
                avg_latency_ms: 0.0,
                count: 0,
            },
            finetune: Aggregate {
                exact_match_rate: 0.0,
                avg_rouge_l: 0.0,
                avg_response_len: 0.0,
                avg_latency_ms: 0.0,
                count: 0,
            },
            exact_match_delta: 0.0,
            rouge_l_delta: 0.0,
            latency_delta_ms: 0.0,
            samples: Vec::new(),
        };

        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(&cli.report_path, json)?;
        println!(
            "dry-run report written: {}",
            cli.report_path.to_string_lossy()
        );
        return Ok(());
    }

    let mut samples = Vec::new();

    let mut base_exact = 0usize;
    let mut base_rouge = 0.0f64;
    let mut base_len = 0usize;
    let mut base_latency = 0u128;

    let mut ft_exact = 0usize;
    let mut ft_rouge = 0.0f64;
    let mut ft_len = 0usize;
    let mut ft_latency = 0u128;

    for (index, sample) in eval_rows.iter().enumerate() {
        let split = split_for_eval(sample)?;
        let prompt = split.prompt_text();
        let target = split.target.as_str().to_string();

        let static_vars: Vec<(&str, String)> = vec![
            ("backend", backend_s.to_string()),
            ("max_new_tokens", cli.max_new_tokens.to_string()),
            ("temperature", cli.temperature.to_string()),
            ("seed", cli.seed.to_string()),
        ];

        let mut base_vars_map: HashMap<&str, &str> = static_vars
            .iter()
            .map(|(key, value)| (*key, value.as_str()))
            .collect();
        base_vars_map.insert("model", cli.base_model.as_str());
        base_vars_map.insert("prompt", &prompt);

        let started = Instant::now();
        let base_output_raw = command.run_with_vars(&base_vars_map)?;
        let base_latency_ms = started.elapsed().as_millis();
        let base_output = extract_model_output(&base_output_raw);

        let base_metrics = eval_row_metrics(&base_output, &target, base_latency_ms);
        if base_metrics.exact_match {
            base_exact += 1;
        }
        base_rouge += base_metrics.rouge_l;
        base_len += base_metrics.response_len;
        base_latency += base_metrics.latency_ms;

        let mut ft_vars_map: HashMap<&str, &str> = static_vars
            .iter()
            .map(|(key, value)| (*key, value.as_str()))
            .collect();
        ft_vars_map.insert("model", cli.finetune_model.as_str());
        ft_vars_map.insert("prompt", &prompt);

        let started = Instant::now();
        let ft_output_raw = command.run_with_vars(&ft_vars_map)?;
        let ft_latency_ms = started.elapsed().as_millis();
        let ft_output = extract_model_output(&ft_output_raw);

        let ft_metrics = eval_row_metrics(&ft_output, &target, ft_latency_ms);
        if ft_metrics.exact_match {
            ft_exact += 1;
        }
        ft_rouge += ft_metrics.rouge_l;
        ft_len += ft_metrics.response_len;
        ft_latency += ft_metrics.latency_ms;

        samples.push(EvalRowReport {
            index: index + 1,
            prompt,
            target,
            base_output,
            ft_output,
            base_exact_match: base_metrics.exact_match,
            ft_exact_match: ft_metrics.exact_match,
            base_rouge_l: base_metrics.rouge_l,
            ft_rouge_l: ft_metrics.rouge_l,
            base_response_len: base_metrics.response_len,
            ft_response_len: ft_metrics.response_len,
            base_latency_ms: base_metrics.latency_ms,
            ft_latency_ms: ft_metrics.latency_ms,
        });
    }

    let n = samples.len() as f64;

    let base = Aggregate {
        exact_match_rate: base_exact as f64 / n,
        avg_rouge_l: base_rouge / n,
        avg_response_len: base_len as f64 / n,
        avg_latency_ms: base_latency as f64 / n,
        count: samples.len(),
    };

    let finetune = Aggregate {
        exact_match_rate: ft_exact as f64 / n,
        avg_rouge_l: ft_rouge / n,
        avg_response_len: ft_len as f64 / n,
        avg_latency_ms: ft_latency as f64 / n,
        count: samples.len(),
    };

    let report = EvaluationReport {
        dataset: cli.dataset_path.to_string_lossy().to_string(),
        base_model: cli.base_model,
        finetune_model: cli.finetune_model,
        backend: backend_s.to_string(),
        base,
        finetune,
        exact_match_delta: finetune.exact_match_rate - base.exact_match_rate,
        rouge_l_delta: finetune.avg_rouge_l - base.avg_rouge_l,
        latency_delta_ms: finetune.avg_latency_ms - base.avg_latency_ms,
        samples,
    };

    println!("base exact match   : {:.3}", report.base.exact_match_rate);
    println!(
        "ft exact match     : {:.3}",
        report.finetune.exact_match_rate
    );
    println!("exact-match delta  : {:.3}", report.exact_match_delta);
    println!("ROUGE-L delta      : {:.3}", report.rouge_l_delta);
    println!("latency delta (ms) : {:.3}", report.latency_delta_ms);

    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&cli.report_path, json)?;

    println!("report written: {}", cli.report_path.to_string_lossy());

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

fn extract_model_output(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    if trimmed.starts_with('{')
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(trimmed)
    {
        if let Some(text) = json.get("text")
            && let Some(value) = text.as_str()
        {
            return value.trim().to_string();
        }

        if let Some(text) = json.get("output")
            && let Some(value) = text.as_str()
        {
            return value.trim().to_string();
        }

        if let Some(text) = json.get("content")
            && let Some(value) = text.as_str()
        {
            return value.trim().to_string();
        }
    }

    trimmed.to_string()
}

fn assert_backend_enabled(backend: &Backend) -> AppResult<()> {
    match backend {
        Backend::Candle => {
            #[cfg(feature = "candle")]
            return Ok(());

            #[cfg(not(feature = "candle"))]
            return Err(AppError::BackendUnavailable {
                backend: "candle".to_string(),
                reason: "enable with --features candle".to_string(),
            });
        }
        Backend::Burn => {
            #[cfg(feature = "burn")]
            return Ok(());

            #[cfg(not(feature = "burn"))]
            return Err(AppError::BackendUnavailable {
                backend: "burn".to_string(),
                reason: "enable with --features burn".to_string(),
            });
        }
    }
}
