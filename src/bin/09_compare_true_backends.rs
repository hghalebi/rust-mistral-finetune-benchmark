use clap::Parser;
use mistral_fintune::AppResult;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "09_compare_true_backends",
    about = "Compare true Candle and Burn pre/post benchmark reports"
)]
struct Cli {
    #[arg(
        long,
        default_value = "artifacts/candle-true-pre-post/true_pre_post_report.json"
    )]
    candle_report: PathBuf,

    #[arg(
        long,
        default_value = "artifacts/burn-true-pre-post/true_pre_post_report.json"
    )]
    burn_report: PathBuf,

    #[arg(long, default_value = "artifacts/backend-comparison")]
    out_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AggregateMetrics {
    exact_match_rate: f64,
    avg_rouge_l: f64,
    avg_response_len: f64,
    avg_latency_ms: f64,
    count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainSummary {
    final_loss: f64,
    min_loss: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackendReport {
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BackendComparisonReport {
    candle: BackendReport,
    burn: BackendReport,
    best_rouge_backend: String,
    fastest_train_backend: String,
    fastest_finetuned_inference_backend: String,
    recommendation: String,
    visualization: String,
}

fn main() -> AppResult<()> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.out_dir)?;

    let candle: BackendReport = serde_json::from_str(&fs::read_to_string(&cli.candle_report)?)?;
    let burn: BackendReport = serde_json::from_str(&fs::read_to_string(&cli.burn_report)?)?;

    let best_rouge_backend = if candle.rouge_l_delta >= burn.rouge_l_delta {
        candle.backend.clone()
    } else {
        burn.backend.clone()
    };
    let fastest_train_backend = if candle.training_wall_ms <= burn.training_wall_ms {
        candle.backend.clone()
    } else {
        burn.backend.clone()
    };
    let fastest_finetuned_inference_backend =
        if candle.finetuned.avg_latency_ms <= burn.finetuned.avg_latency_ms {
            candle.backend.clone()
        } else {
            burn.backend.clone()
        };

    let recommendation = format!(
        "{} wins on held-out ROUGE-L delta, {} trains faster, and {} has lower fine-tuned latency.",
        best_rouge_backend, fastest_train_backend, fastest_finetuned_inference_backend
    );

    let svg_path = cli.out_dir.join("true_backend_comparison.svg");
    fs::write(&svg_path, render_svg(&candle, &burn))?;

    let report = BackendComparisonReport {
        candle: candle.clone(),
        burn: burn.clone(),
        best_rouge_backend,
        fastest_train_backend,
        fastest_finetuned_inference_backend,
        recommendation,
        visualization: svg_path.to_string_lossy().to_string(),
    };

    let report_path = cli.out_dir.join("true_backend_comparison_report.json");
    let leaderboard_path = cli.out_dir.join("true_backend_comparison_leaderboard.md");
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    fs::write(
        &leaderboard_path,
        render_markdown(&report.candle, &report.burn, &report.recommendation),
    )?;

    println!(
        "candle report        : {}",
        cli.candle_report.to_string_lossy()
    );
    println!(
        "burn report          : {}",
        cli.burn_report.to_string_lossy()
    );
    println!("comparison report    : {}", report_path.to_string_lossy());
    println!(
        "leaderboard          : {}",
        leaderboard_path.to_string_lossy()
    );
    println!("visualization        : {}", svg_path.to_string_lossy());

    Ok(())
}

fn render_markdown(candle: &BackendReport, burn: &BackendReport, recommendation: &str) -> String {
    format!(
        "# True Backend Comparison\n\n\
## Summary\n\
- recommendation: {recommendation}\n\n\
## Pre/Post Quality\n\n\
| Backend | Base EM | FT EM | EM Delta | Base ROUGE-L | FT ROUGE-L | ROUGE-L Delta |\n\
|---|---:|---:|---:|---:|---:|---:|\n\
| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n\
| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n\n\
## Runtime\n\n\
| Backend | Train Wall (s) | Eval Wall (s) | Base Latency (ms) | FT Latency (ms) | Latency Delta (ms) |\n\
|---|---:|---:|---:|---:|---:|\n\
| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n\
| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n\n\
## Response Shape\n\n\
| Backend | Base Avg Len | FT Avg Len | Len Delta |\n\
|---|---:|---:|---:|\n\
| {} | {:.2} | {:.2} | {:.2} |\n\
| {} | {:.2} | {:.2} | {:.2} |\n",
        candle.backend,
        candle.base.exact_match_rate,
        candle.finetuned.exact_match_rate,
        candle.exact_match_delta,
        candle.base.avg_rouge_l,
        candle.finetuned.avg_rouge_l,
        candle.rouge_l_delta,
        burn.backend,
        burn.base.exact_match_rate,
        burn.finetuned.exact_match_rate,
        burn.exact_match_delta,
        burn.base.avg_rouge_l,
        burn.finetuned.avg_rouge_l,
        burn.rouge_l_delta,
        candle.backend,
        candle.training_wall_ms as f64 / 1000.0,
        candle.eval_wall_ms as f64 / 1000.0,
        candle.base.avg_latency_ms,
        candle.finetuned.avg_latency_ms,
        candle.latency_delta_ms,
        burn.backend,
        burn.training_wall_ms as f64 / 1000.0,
        burn.eval_wall_ms as f64 / 1000.0,
        burn.base.avg_latency_ms,
        burn.finetuned.avg_latency_ms,
        burn.latency_delta_ms,
        candle.backend,
        candle.base.avg_response_len,
        candle.finetuned.avg_response_len,
        candle.response_len_delta,
        burn.backend,
        burn.base.avg_response_len,
        burn.finetuned.avg_response_len,
        burn.response_len_delta,
    )
}

fn render_svg(candle: &BackendReport, burn: &BackendReport) -> String {
    let width = 1200.0;
    let height = 760.0;
    let chart_width = 480.0;
    let chart_height = 220.0;
    let left = 80.0;
    let top_rouge = 80.0;
    let top_latency = 380.0;
    let bar_width = 70.0;
    let gap = 30.0;
    let group_gap = 80.0;

    let rouge_max = candle
        .finetuned
        .avg_rouge_l
        .max(burn.finetuned.avg_rouge_l)
        .max(candle.base.avg_rouge_l)
        .max(burn.base.avg_rouge_l)
        .max(0.1);
    let latency_max = candle
        .base
        .avg_latency_ms
        .max(candle.finetuned.avg_latency_ms)
        .max(burn.base.avg_latency_ms)
        .max(burn.finetuned.avg_latency_ms)
        .max(1.0);
    let train_max = candle.training_wall_ms.max(burn.training_wall_ms).max(1) as f64;

    let candle_x = left;
    let burn_x = candle_x + (bar_width * 2.0) + gap + group_gap;
    let train_chart_height = 240.0;
    let train_top = 170.0;

    let bar =
        |value: f64, max: f64, y_base: f64| chart_height * (value / max).clamp(0.0, 1.0) + y_base;
    let train_bar = |value: f64| train_chart_height * (value / train_max).clamp(0.0, 1.0);

    let candle_base_rouge_h = bar(candle.base.avg_rouge_l, rouge_max, 0.0);
    let candle_ft_rouge_h = bar(candle.finetuned.avg_rouge_l, rouge_max, 0.0);
    let burn_base_rouge_h = bar(burn.base.avg_rouge_l, rouge_max, 0.0);
    let burn_ft_rouge_h = bar(burn.finetuned.avg_rouge_l, rouge_max, 0.0);

    let candle_base_latency_h = bar(candle.base.avg_latency_ms, latency_max, 0.0);
    let candle_ft_latency_h = bar(candle.finetuned.avg_latency_ms, latency_max, 0.0);
    let burn_base_latency_h = bar(burn.base.avg_latency_ms, latency_max, 0.0);
    let burn_ft_latency_h = bar(burn.finetuned.avg_latency_ms, latency_max, 0.0);

    let candle_train_h = train_bar(candle.training_wall_ms as f64);
    let burn_train_h = train_bar(burn.training_wall_ms as f64);

    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\
<style>\
text{{font-family:Menlo,Monaco,monospace;fill:#15202b}}\
.title{{font-size:28px;font-weight:700}}\
.subtitle{{font-size:16px;fill:#425466}}\
.axis{{stroke:#ccd6dd;stroke-width:1}}\
.label{{font-size:13px}}\
.small{{font-size:12px;fill:#5b6875}}\
</style>\
<rect width=\"100%\" height=\"100%\" fill=\"#f7fafc\"/>\
<text x=\"80\" y=\"45\" class=\"title\">Candle vs Burn true pre/post benchmark</text>\
<text x=\"80\" y=\"68\" class=\"subtitle\">Top: ROUGE-L before/after. Bottom-left: avg latency before/after. Right: train wall time.</text>\
<line x1=\"{left}\" y1=\"{top_rouge}\" x2=\"{left}\" y2=\"{rouge_bottom}\" class=\"axis\"/>\
<line x1=\"{left}\" y1=\"{rouge_bottom}\" x2=\"{rouge_right}\" y2=\"{rouge_bottom}\" class=\"axis\"/>\
<text x=\"{left}\" y=\"{rouge_title_y}\" class=\"label\">ROUGE-L</text>\
<rect x=\"{candle_x}\" y=\"{candle_base_rouge_y}\" width=\"{bar_width}\" height=\"{candle_base_rouge_h}\" fill=\"#2b6cb0\" rx=\"6\"/>\
<rect x=\"{candle_ft_x}\" y=\"{candle_ft_rouge_y}\" width=\"{bar_width}\" height=\"{candle_ft_rouge_h}\" fill=\"#2f855a\" rx=\"6\"/>\
<rect x=\"{burn_x}\" y=\"{burn_base_rouge_y}\" width=\"{bar_width}\" height=\"{burn_base_rouge_h}\" fill=\"#2b6cb0\" rx=\"6\"/>\
<rect x=\"{burn_ft_x}\" y=\"{burn_ft_rouge_y}\" width=\"{bar_width}\" height=\"{burn_ft_rouge_h}\" fill=\"#2f855a\" rx=\"6\"/>\
<text x=\"{candle_x}\" y=\"{candle_group_y}\" class=\"small\">candle</text>\
<text x=\"{burn_x}\" y=\"{burn_group_y}\" class=\"small\">burn</text>\
<text x=\"{candle_x}\" y=\"{rouge_bottom_plus}\" class=\"small\">base</text>\
<text x=\"{candle_ft_x}\" y=\"{rouge_bottom_plus}\" class=\"small\">ft</text>\
<text x=\"{burn_x}\" y=\"{rouge_bottom_plus}\" class=\"small\">base</text>\
<text x=\"{burn_ft_x}\" y=\"{rouge_bottom_plus}\" class=\"small\">ft</text>\
<text x=\"{candle_x}\" y=\"{candle_base_rouge_value_y}\" class=\"small\">{candle_base_rouge:.3}</text>\
<text x=\"{candle_ft_x}\" y=\"{candle_ft_rouge_value_y}\" class=\"small\">{candle_ft_rouge:.3}</text>\
<text x=\"{burn_x}\" y=\"{burn_base_rouge_value_y}\" class=\"small\">{burn_base_rouge:.3}</text>\
<text x=\"{burn_ft_x}\" y=\"{burn_ft_rouge_value_y}\" class=\"small\">{burn_ft_rouge:.3}</text>\
<line x1=\"{left}\" y1=\"{top_latency}\" x2=\"{left}\" y2=\"{latency_bottom}\" class=\"axis\"/>\
<line x1=\"{left}\" y1=\"{latency_bottom}\" x2=\"{rouge_right}\" y2=\"{latency_bottom}\" class=\"axis\"/>\
<text x=\"{left}\" y=\"{latency_title_y}\" class=\"label\">Avg latency (ms)</text>\
<rect x=\"{candle_x}\" y=\"{candle_base_latency_y}\" width=\"{bar_width}\" height=\"{candle_base_latency_h}\" fill=\"#805ad5\" rx=\"6\"/>\
<rect x=\"{candle_ft_x}\" y=\"{candle_ft_latency_y}\" width=\"{bar_width}\" height=\"{candle_ft_latency_h}\" fill=\"#dd6b20\" rx=\"6\"/>\
<rect x=\"{burn_x}\" y=\"{burn_base_latency_y}\" width=\"{bar_width}\" height=\"{burn_base_latency_h}\" fill=\"#805ad5\" rx=\"6\"/>\
<rect x=\"{burn_ft_x}\" y=\"{burn_ft_latency_y}\" width=\"{bar_width}\" height=\"{burn_ft_latency_h}\" fill=\"#dd6b20\" rx=\"6\"/>\
<text x=\"{candle_x}\" y=\"{latency_bottom_plus}\" class=\"small\">base</text>\
<text x=\"{candle_ft_x}\" y=\"{latency_bottom_plus}\" class=\"small\">ft</text>\
<text x=\"{burn_x}\" y=\"{latency_bottom_plus}\" class=\"small\">base</text>\
<text x=\"{burn_ft_x}\" y=\"{latency_bottom_plus}\" class=\"small\">ft</text>\
<text x=\"{candle_x}\" y=\"{candle_base_latency_value_y}\" class=\"small\">{candle_base_latency:.0}</text>\
<text x=\"{candle_ft_x}\" y=\"{candle_ft_latency_value_y}\" class=\"small\">{candle_ft_latency:.0}</text>\
<text x=\"{burn_x}\" y=\"{burn_base_latency_value_y}\" class=\"small\">{burn_base_latency:.0}</text>\
<text x=\"{burn_ft_x}\" y=\"{burn_ft_latency_value_y}\" class=\"small\">{burn_ft_latency:.0}</text>\
<line x1=\"720\" y1=\"{train_top}\" x2=\"720\" y2=\"{train_bottom}\" class=\"axis\"/>\
<line x1=\"720\" y1=\"{train_bottom}\" x2=\"1080\" y2=\"{train_bottom}\" class=\"axis\"/>\
<text x=\"720\" y=\"140\" class=\"label\">Train wall time (s)</text>\
<rect x=\"760\" y=\"{candle_train_y}\" width=\"90\" height=\"{candle_train_h}\" fill=\"#c05621\" rx=\"6\"/>\
<rect x=\"900\" y=\"{burn_train_y}\" width=\"90\" height=\"{burn_train_h}\" fill=\"#3182ce\" rx=\"6\"/>\
<text x=\"760\" y=\"{train_bottom_plus}\" class=\"small\">candle</text>\
<text x=\"900\" y=\"{train_bottom_plus}\" class=\"small\">burn</text>\
<text x=\"760\" y=\"{candle_train_value_y}\" class=\"small\">{candle_train:.2}s</text>\
<text x=\"900\" y=\"{burn_train_value_y}\" class=\"small\">{burn_train:.2}s</text>\
</svg>",
        width = width,
        height = height,
        left = left,
        top_rouge = top_rouge,
        rouge_bottom = top_rouge + chart_height,
        rouge_right = left + chart_width,
        rouge_title_y = top_rouge - 18.0,
        candle_x = candle_x,
        candle_ft_x = candle_x + bar_width + gap,
        burn_x = burn_x,
        burn_ft_x = burn_x + bar_width + gap,
        candle_group_y = top_rouge + chart_height + 48.0,
        burn_group_y = top_rouge + chart_height + 48.0,
        rouge_bottom_plus = top_rouge + chart_height + 24.0,
        candle_base_rouge_y = top_rouge + chart_height - candle_base_rouge_h,
        candle_ft_rouge_y = top_rouge + chart_height - candle_ft_rouge_h,
        burn_base_rouge_y = top_rouge + chart_height - burn_base_rouge_h,
        burn_ft_rouge_y = top_rouge + chart_height - burn_ft_rouge_h,
        candle_base_rouge_h = candle_base_rouge_h,
        candle_ft_rouge_h = candle_ft_rouge_h,
        burn_base_rouge_h = burn_base_rouge_h,
        burn_ft_rouge_h = burn_ft_rouge_h,
        candle_base_rouge = candle.base.avg_rouge_l,
        candle_ft_rouge = candle.finetuned.avg_rouge_l,
        burn_base_rouge = burn.base.avg_rouge_l,
        burn_ft_rouge = burn.finetuned.avg_rouge_l,
        candle_base_rouge_value_y = top_rouge + chart_height - candle_base_rouge_h - 8.0,
        candle_ft_rouge_value_y = top_rouge + chart_height - candle_ft_rouge_h - 8.0,
        burn_base_rouge_value_y = top_rouge + chart_height - burn_base_rouge_h - 8.0,
        burn_ft_rouge_value_y = top_rouge + chart_height - burn_ft_rouge_h - 8.0,
        top_latency = top_latency,
        latency_bottom = top_latency + chart_height,
        latency_title_y = top_latency - 18.0,
        latency_bottom_plus = top_latency + chart_height + 24.0,
        candle_base_latency_y = top_latency + chart_height - candle_base_latency_h,
        candle_ft_latency_y = top_latency + chart_height - candle_ft_latency_h,
        burn_base_latency_y = top_latency + chart_height - burn_base_latency_h,
        burn_ft_latency_y = top_latency + chart_height - burn_ft_latency_h,
        candle_base_latency_h = candle_base_latency_h,
        candle_ft_latency_h = candle_ft_latency_h,
        burn_base_latency_h = burn_base_latency_h,
        burn_ft_latency_h = burn_ft_latency_h,
        candle_base_latency = candle.base.avg_latency_ms,
        candle_ft_latency = candle.finetuned.avg_latency_ms,
        burn_base_latency = burn.base.avg_latency_ms,
        burn_ft_latency = burn.finetuned.avg_latency_ms,
        candle_base_latency_value_y = top_latency + chart_height - candle_base_latency_h - 8.0,
        candle_ft_latency_value_y = top_latency + chart_height - candle_ft_latency_h - 8.0,
        burn_base_latency_value_y = top_latency + chart_height - burn_base_latency_h - 8.0,
        burn_ft_latency_value_y = top_latency + chart_height - burn_ft_latency_h - 8.0,
        train_top = train_top,
        train_bottom = train_top + train_chart_height,
        candle_train_y = train_top + train_chart_height - candle_train_h,
        burn_train_y = train_top + train_chart_height - burn_train_h,
        candle_train_h = candle_train_h,
        burn_train_h = burn_train_h,
        candle_train = candle.training_wall_ms as f64 / 1000.0,
        burn_train = burn.training_wall_ms as f64 / 1000.0,
        train_bottom_plus = train_top + train_chart_height + 24.0,
        candle_train_value_y = train_top + train_chart_height - candle_train_h - 8.0,
        burn_train_value_y = train_top + train_chart_height - burn_train_h - 8.0,
    )
}
