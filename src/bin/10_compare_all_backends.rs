use clap::Parser;
use mistral_fintune::AppResult;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "10_compare_all_backends",
    about = "Compare true pre/post benchmark reports from any backend set"
)]
struct Cli {
    #[arg(long, required = true)]
    report: Vec<PathBuf>,

    #[arg(long, default_value = "artifacts/all-backend-comparison")]
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
struct AllBackendComparisonReport {
    backends: Vec<BackendReport>,
    best_rouge_backend: String,
    fastest_train_backend: String,
    fastest_finetuned_inference_backend: String,
    recommendation: String,
    visualization: String,
}

fn main() -> AppResult<()> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.out_dir)?;

    let mut reports = Vec::new();
    for report_path in &cli.report {
        let report: BackendReport = serde_json::from_str(&fs::read_to_string(report_path)?)?;
        reports.push(report);
    }
    reports.sort_by(|left, right| {
        right
            .rouge_l_delta
            .partial_cmp(&left.rouge_l_delta)
            .unwrap_or(Ordering::Equal)
    });

    let best_rouge_backend = reports
        .iter()
        .max_by(|left, right| {
            left.rouge_l_delta
                .partial_cmp(&right.rouge_l_delta)
                .unwrap_or(Ordering::Equal)
        })
        .map(|report| report.backend.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let fastest_train_backend = reports
        .iter()
        .min_by_key(|report| report.training_wall_ms)
        .map(|report| report.backend.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let fastest_finetuned_inference_backend = reports
        .iter()
        .min_by(|left, right| {
            left.finetuned
                .avg_latency_ms
                .partial_cmp(&right.finetuned.avg_latency_ms)
                .unwrap_or(Ordering::Equal)
        })
        .map(|report| report.backend.clone())
        .unwrap_or_else(|| "unknown".to_string());

    let recommendation = format!(
        "{} wins on held-out ROUGE-L delta, {} trains fastest, and {} has the lowest fine-tuned latency.",
        best_rouge_backend, fastest_train_backend, fastest_finetuned_inference_backend
    );

    let svg_path = cli.out_dir.join("all_backend_comparison.svg");
    fs::write(&svg_path, render_svg(&reports, &recommendation))?;

    let report = AllBackendComparisonReport {
        backends: reports.clone(),
        best_rouge_backend,
        fastest_train_backend,
        fastest_finetuned_inference_backend,
        recommendation,
        visualization: svg_path.to_string_lossy().to_string(),
    };

    let report_path = cli.out_dir.join("all_backend_comparison_report.json");
    let leaderboard_path = cli.out_dir.join("all_backend_comparison_leaderboard.md");
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    fs::write(
        &leaderboard_path,
        render_markdown(&report.backends, &report.recommendation),
    )?;

    println!("comparison report    : {}", report_path.to_string_lossy());
    println!(
        "leaderboard          : {}",
        leaderboard_path.to_string_lossy()
    );
    println!("visualization        : {}", svg_path.to_string_lossy());
    for backend in &report.backends {
        println!(
            "loaded report        : {} (ROUGE-L delta {:.4})",
            backend.backend, backend.rouge_l_delta
        );
    }

    Ok(())
}

fn render_markdown(backends: &[BackendReport], recommendation: &str) -> String {
    let mut out = String::from("# All Backend Comparison\n\n");
    out.push_str("## Summary\n");
    out.push_str(&format!("- recommendation: {recommendation}\n\n"));
    out.push_str("## Quality\n\n");
    out.push_str("| Backend | Base EM | FT EM | Base ROUGE-L | FT ROUGE-L | ROUGE-L Delta |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for backend in backends {
        out.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            backend.backend,
            backend.base.exact_match_rate,
            backend.finetuned.exact_match_rate,
            backend.base.avg_rouge_l,
            backend.finetuned.avg_rouge_l,
            backend.rouge_l_delta,
        ));
    }

    out.push_str("\n## Runtime\n\n");
    out.push_str(
        "| Backend | Train Wall (s) | Eval Wall (s) | Base Latency (ms) | FT Latency (ms) |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|\n");
    for backend in backends {
        out.push_str(&format!(
            "| {} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
            backend.backend,
            backend.training_wall_ms as f64 / 1000.0,
            backend.eval_wall_ms as f64 / 1000.0,
            backend.base.avg_latency_ms,
            backend.finetuned.avg_latency_ms,
        ));
    }

    out.push_str("\n## Response Shape\n\n");
    out.push_str("| Backend | Base Avg Len | FT Avg Len | Len Delta |\n");
    out.push_str("|---|---:|---:|---:|\n");
    for backend in backends {
        out.push_str(&format!(
            "| {} | {:.2} | {:.2} | {:.2} |\n",
            backend.backend,
            backend.base.avg_response_len,
            backend.finetuned.avg_response_len,
            backend.response_len_delta,
        ));
    }

    out
}

fn render_svg(backends: &[BackendReport], recommendation: &str) -> String {
    let width = 1500.0;
    let row_height = 48.0;
    let header_height = 130.0;
    let footer_height = 40.0;
    let height = header_height + (backends.len() as f64 * row_height) + footer_height;

    let rouge_x = 360.0;
    let train_x = 770.0;
    let latency_x = 1180.0;
    let bar_max_width = 220.0;

    let rouge_max = backends
        .iter()
        .map(|report| report.rouge_l_delta.max(0.0))
        .fold(0.001_f64, f64::max);
    let train_max = backends
        .iter()
        .map(|report| report.training_wall_ms as f64 / 1000.0)
        .fold(0.001_f64, f64::max);
    let ft_latency_max = backends
        .iter()
        .map(|report| report.finetuned.avg_latency_ms)
        .fold(0.001_f64, f64::max);

    let mut body = String::new();
    for (index, backend) in backends.iter().enumerate() {
        let y = header_height + (index as f64 * row_height);
        let rouge_width = (backend.rouge_l_delta.max(0.0) / rouge_max) * bar_max_width;
        let train_width = ((backend.training_wall_ms as f64 / 1000.0) / train_max) * bar_max_width;
        let latency_width = (backend.finetuned.avg_latency_ms / ft_latency_max) * bar_max_width;

        body.push_str(&format!(
            "<text x=\"40\" y=\"{name_y}\" class=\"label\">{}</text>\
<rect x=\"{rouge_x}\" y=\"{bar_y}\" width=\"{rouge_width}\" height=\"18\" rx=\"4\" fill=\"#0f766e\"/>\
<text x=\"{rouge_text_x}\" y=\"{name_y}\" class=\"small\">{:.4} -> {:.4} (delta {:.4})</text>\
<rect x=\"{train_x}\" y=\"{bar_y}\" width=\"{train_width}\" height=\"18\" rx=\"4\" fill=\"#1d4ed8\"/>\
<text x=\"{train_text_x}\" y=\"{name_y}\" class=\"small\">{:.2}s</text>\
<rect x=\"{latency_x}\" y=\"{bar_y}\" width=\"{latency_width}\" height=\"18\" rx=\"4\" fill=\"#b45309\"/>\
<text x=\"{latency_text_x}\" y=\"{name_y}\" class=\"small\">{:.2} ms (base {:.2})</text>",
            backend.backend,
            backend.base.avg_rouge_l,
            backend.finetuned.avg_rouge_l,
            backend.rouge_l_delta,
            backend.training_wall_ms as f64 / 1000.0,
            backend.finetuned.avg_latency_ms,
            backend.base.avg_latency_ms,
            name_y = y + 18.0,
            bar_y = y + 24.0,
            rouge_x = rouge_x,
            train_x = train_x,
            latency_x = latency_x,
            rouge_width = rouge_width,
            train_width = train_width,
            latency_width = latency_width,
            rouge_text_x = rouge_x + bar_max_width + 16.0,
            train_text_x = train_x + bar_max_width + 16.0,
            latency_text_x = latency_x + bar_max_width + 16.0,
        ));
    }

    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\
<style>\
text{{font-family:Menlo,Monaco,monospace;fill:#15202b}}\
.title{{font-size:28px;font-weight:700}}\
.subtitle{{font-size:15px;fill:#425466}}\
.label{{font-size:15px;font-weight:600}}\
.small{{font-size:12px;fill:#425466}}\
.axis{{stroke:#d0d7de;stroke-width:1}}\
</style>\
<rect width=\"100%\" height=\"100%\" fill=\"#f8fafc\"/>\
<text x=\"40\" y=\"42\" class=\"title\">All backend true pre/post comparison</text>\
<text x=\"40\" y=\"68\" class=\"subtitle\">Each row shows before vs after fine-tuning for one backend.</text>\
<text x=\"40\" y=\"92\" class=\"subtitle\">{recommendation}</text>\
<text x=\"360\" y=\"118\" class=\"label\">ROUGE-L delta</text>\
<text x=\"770\" y=\"118\" class=\"label\">Train wall time</text>\
<text x=\"1180\" y=\"118\" class=\"label\">Fine-tuned latency</text>\
<line x1=\"340\" y1=\"124\" x2=\"1450\" y2=\"124\" class=\"axis\"/>\
{body}\
</svg>",
        width = width,
        height = height,
        recommendation = xml_escape(recommendation),
        body = body,
    )
}

fn xml_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
