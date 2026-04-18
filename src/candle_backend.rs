#[cfg(target_os = "macos")]
extern crate accelerate_src;

use crate::{AppError, AppResult, TrainSample, eval_row_metrics, split_for_eval};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    Activation, AdamW, Embedding, LayerNorm, Linear, Module, Optimizer, VarBuilder, VarMap,
    embedding, layer_norm, linear, loss,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;
use std::time::Instant;

const NEG_INFINITY: f32 = -1.0e9;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteTokenizer {
    stoi: HashMap<String, u32>,
    itos: Vec<String>,
    pad_token_id: u32,
    eos_token_id: u32,
}

impl ByteTokenizer {
    pub fn from_texts(texts: &[String]) -> AppResult<Self> {
        let mut charset = BTreeSet::new();
        for text in texts {
            for token in text.split_whitespace() {
                charset.insert(token.to_string());
            }
        }

        if charset.is_empty() {
            return Err(AppError::Validation(
                "cannot build tokenizer from empty corpus".to_string(),
            ));
        }

        let itos = charset.into_iter().collect::<Vec<_>>();
        let stoi = itos
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, token)| (token, index as u32))
            .collect::<HashMap<_, _>>();
        let pad_token_id = itos.len() as u32;
        let eos_token_id = pad_token_id + 1;

        Ok(Self {
            stoi,
            itos,
            pad_token_id,
            eos_token_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len() + 2
    }

    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn encode_prompt(&self, value: &str) -> AppResult<Vec<u32>> {
        value
            .split_whitespace()
            .map(|token| {
                self.stoi.get(token).copied().ok_or_else(|| {
                    AppError::Validation(format!("tokenizer missing token {token:?}"))
                })
            })
            .collect()
    }

    pub fn encode_with_eos(&self, value: &str) -> AppResult<Vec<u32>> {
        let mut out = self.encode_prompt(value)?;
        out.push(self.eos_token_id());
        Ok(out)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .copied()
            .take_while(|id| *id != self.eos_token_id() && *id != self.pad_token_id())
            .filter_map(|id| self.itos.get(id as usize).cloned())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinyLmConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub mlp_hidden: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleRunManifest {
    pub model: TinyLmConfig,
    pub tokenizer: ByteTokenizer,
    pub train_limit: usize,
    pub eval_limit: usize,
    pub steps: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub base_checkpoint: String,
    pub finetuned_checkpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossRecord {
    pub step: usize,
    pub loss: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainSummary {
    pub final_loss: f64,
    pub min_loss: f64,
    pub loss_history: Vec<LossRecord>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrainHyperparameters {
    pub steps: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub log_every: usize,
}

#[derive(Debug, Clone)]
pub struct EvalExample {
    pub prompt: String,
    pub target: String,
}

pub struct TinyCausalLm {
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    lm_head: Linear,
    config: TinyLmConfig,
}

struct TransformerBlock {
    norm_attn: LayerNorm,
    norm_mlp: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    fc_in: Linear,
    fc_out: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl TransformerBlock {
    fn new(vb: VarBuilder<'_>, config: &TinyLmConfig) -> AppResult<Self> {
        let head_dim = config.d_model / config.n_heads;
        let norm_cfg = candle_nn::LayerNormConfig::default();
        Ok(Self {
            norm_attn: layer_norm(config.d_model, norm_cfg, vb.pp("ln_attn"))?,
            norm_mlp: layer_norm(config.d_model, norm_cfg, vb.pp("ln_mlp"))?,
            q_proj: linear(config.d_model, config.d_model, vb.pp("q_proj"))?,
            k_proj: linear(config.d_model, config.d_model, vb.pp("k_proj"))?,
            v_proj: linear(config.d_model, config.d_model, vb.pp("v_proj"))?,
            out_proj: linear(config.d_model, config.d_model, vb.pp("out_proj"))?,
            fc_in: linear(config.d_model, config.mlp_hidden, vb.pp("fc_in"))?,
            fc_out: linear(config.mlp_hidden, config.d_model, vb.pp("fc_out"))?,
            n_heads: config.n_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> AppResult<Tensor> {
        let attn_input = self.norm_attn.forward(xs)?;
        let attn_out = self.self_attention(&attn_input)?;
        let resid = (xs + attn_out)?;
        let mlp_input = self.norm_mlp.forward(&resid)?;
        let mlp_hidden = self.fc_in.forward(&mlp_input)?;
        let mlp_hidden = Activation::Gelu.forward(&mlp_hidden)?;
        let mlp_out = self.fc_out.forward(&mlp_hidden)?;
        Ok((resid + mlp_out)?)
    }

    fn self_attention(&self, xs: &Tensor) -> AppResult<Tensor> {
        let (batch, seq_len, width) = xs.dims3()?;
        let q = self.project_heads(&self.q_proj.forward(xs)?, batch, seq_len)?;
        let k = self.project_heads(&self.k_proj.forward(xs)?, batch, seq_len)?;
        let v = self.project_heads(&self.v_proj.forward(xs)?, batch, seq_len)?;
        let k = k.transpose(2, 3)?.contiguous()?;

        let scores = q.matmul(&k)?;
        let scores = scores.affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let mask = causal_mask(seq_len, xs.device())?;
        let scores = scores.broadcast_add(&mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = weights.matmul(&v)?;
        let attended = attended.transpose(1, 2)?.reshape((batch, seq_len, width))?;
        Ok(self.out_proj.forward(&attended)?)
    }

    fn project_heads(&self, xs: &Tensor, batch: usize, seq_len: usize) -> AppResult<Tensor> {
        Ok(xs
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?)
    }
}

impl TinyCausalLm {
    pub fn new(vb: VarBuilder<'_>, config: TinyLmConfig) -> AppResult<Self> {
        let norm_cfg = candle_nn::LayerNormConfig::default();
        let token_embedding =
            embedding(config.vocab_size, config.d_model, vb.pp("token_embedding"))?;
        let position_embedding = embedding(
            config.max_seq_len,
            config.d_model,
            vb.pp("position_embedding"),
        )?;
        let mut blocks = Vec::with_capacity(config.n_layers);
        for layer in 0..config.n_layers {
            blocks.push(TransformerBlock::new(
                vb.pp(format!("blocks.{layer}")),
                &config,
            )?);
        }
        let final_norm = layer_norm(config.d_model, norm_cfg, vb.pp("final_norm"))?;
        let lm_head = linear(config.d_model, config.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            lm_head,
            config,
        })
    }

    pub fn forward(&self, token_ids: &Tensor) -> AppResult<Tensor> {
        let (_, seq_len) = token_ids.dims2()?;
        if seq_len > self.config.max_seq_len {
            return Err(AppError::Validation(format!(
                "sequence length {seq_len} exceeds max_seq_len {}",
                self.config.max_seq_len
            )));
        }

        let token_embeddings = self.token_embedding.forward(token_ids)?;
        let positions = (0..seq_len as u32).collect::<Vec<_>>();
        let position_ids = Tensor::from_slice(&positions, (1, seq_len), token_ids.device())?;
        let position_embeddings = self.position_embedding.forward(&position_ids)?;

        let mut hidden = token_embeddings.broadcast_add(&position_embeddings)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        let hidden = self.final_norm.forward(&hidden)?;
        Ok(self.lm_head.forward(&hidden)?)
    }

    pub fn generate(
        &self,
        tokenizer: &ByteTokenizer,
        prompt: &str,
        max_new_tokens: usize,
    ) -> AppResult<String> {
        let mut token_ids = tokenizer.encode_prompt(prompt)?;
        let prompt_len = token_ids.len();

        for _ in 0..max_new_tokens {
            let start = token_ids
                .len()
                .saturating_sub(self.config.max_seq_len.max(1));
            let window = &token_ids[start..];
            let input = Tensor::from_slice(window, (1, window.len()), &Device::Cpu)?;
            let logits = self.forward(&input)?;
            let (_, seq_len, _) = logits.dims3()?;
            let last = logits.i((0, seq_len - 1))?;
            let next_id = argmax_token(&last)?;
            if next_id == tokenizer.eos_token_id() {
                break;
            }
            token_ids.push(next_id);
        }

        Ok(tokenizer.decode(&token_ids[prompt_len..]))
    }
}

pub fn cpu_device() -> Device {
    Device::Cpu
}

pub fn build_model(config: &TinyLmConfig, device: &Device) -> AppResult<(VarMap, TinyCausalLm)> {
    let var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
    let model = TinyCausalLm::new(vb, config.clone())?;
    // Ensure there is no stale state from a previous builder clone.
    var_map.data();
    Ok((var_map, model))
}

pub fn save_checkpoint(var_map: &VarMap, checkpoint_path: &Path) -> AppResult<()> {
    if let Some(parent) = checkpoint_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    var_map.save(checkpoint_path)?;
    Ok(())
}

pub fn load_model(
    config: &TinyLmConfig,
    checkpoint_path: &Path,
    device: &Device,
) -> AppResult<(VarMap, TinyCausalLm)> {
    let mut var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
    let model = TinyCausalLm::new(vb, config.clone())?;
    var_map.load(checkpoint_path)?;
    Ok((var_map, model))
}

pub fn training_texts_from_rows(rows: &[TrainSample], limit: usize) -> AppResult<Vec<String>> {
    rows.iter()
        .take(limit)
        .map(|row| {
            let split = split_for_eval(row)?;
            Ok(format!("{}{}", split.prompt_text(), split.target.as_str()))
        })
        .collect()
}

pub fn eval_examples_from_rows(rows: &[TrainSample], limit: usize) -> AppResult<Vec<EvalExample>> {
    rows.iter()
        .take(limit)
        .map(|row| {
            let split = split_for_eval(row)?;
            Ok(EvalExample {
                prompt: split.prompt_text(),
                target: split.target.as_str().to_string(),
            })
        })
        .collect()
}

pub fn train_model(
    model: &TinyCausalLm,
    var_map: &VarMap,
    tokenizer: &ByteTokenizer,
    texts: &[String],
    hyperparameters: TrainHyperparameters,
) -> AppResult<TrainSummary> {
    if texts.is_empty() {
        return Err(AppError::Validation(
            "no training texts available".to_string(),
        ));
    }

    let mut optimizer = AdamW::new_lr(var_map.all_vars(), hyperparameters.learning_rate)?;
    let mut rng = StdRng::seed_from_u64(hyperparameters.seed);
    let mut history = Vec::new();
    let mut min_loss = f64::INFINITY;
    let mut final_loss = 0.0;

    for step in 1..=hyperparameters.steps {
        let sample_index = rng.gen_range(0..texts.len());
        let (input, target) = sample_window(
            &texts[sample_index],
            tokenizer,
            model.config.max_seq_len,
            &mut rng,
        )?;
        let logits = model.forward(&input)?;
        let (_, seq_len, vocab) = logits.dims3()?;
        let logits = logits.reshape((seq_len, vocab))?;
        let loss = loss::cross_entropy(&logits, &target)?;
        optimizer.backward_step(&loss)?;
        final_loss = loss.to_scalar::<f32>()? as f64;
        min_loss = min_loss.min(final_loss);

        if step == 1
            || step == hyperparameters.steps
            || step % hyperparameters.log_every.max(1) == 0
        {
            history.push(LossRecord {
                step,
                loss: final_loss,
            });
        }
    }

    Ok(TrainSummary {
        final_loss,
        min_loss,
        loss_history: history,
    })
}

pub fn evaluate_pair(
    base_model: &TinyCausalLm,
    finetuned_model: &TinyCausalLm,
    tokenizer: &ByteTokenizer,
    eval_examples: &[EvalExample],
    max_new_tokens: usize,
) -> AppResult<(AggregateMetrics, AggregateMetrics, Vec<RowMetrics>)> {
    if eval_examples.is_empty() {
        return Err(AppError::Validation(
            "no evaluation rows available".to_string(),
        ));
    }

    let mut rows = Vec::with_capacity(eval_examples.len());
    let mut base = AggregateMetrics::default();
    let mut finetuned = AggregateMetrics::default();

    for (index, example) in eval_examples.iter().enumerate() {
        let started = Instant::now();
        let base_output = base_model.generate(tokenizer, &example.prompt, max_new_tokens)?;
        let base_latency = started.elapsed().as_millis();
        let base_metrics = eval_row_metrics(&base_output, &example.target, base_latency);

        let started = Instant::now();
        let finetuned_output =
            finetuned_model.generate(tokenizer, &example.prompt, max_new_tokens)?;
        let finetuned_latency = started.elapsed().as_millis();
        let finetuned_metrics =
            eval_row_metrics(&finetuned_output, &example.target, finetuned_latency);

        base.record(base_metrics);
        finetuned.record(finetuned_metrics);
        rows.push(RowMetrics {
            index: index + 1,
            prompt: example.prompt.clone(),
            target: example.target.clone(),
            base_output,
            finetuned_output,
            base_exact_match: base_metrics.exact_match,
            finetuned_exact_match: finetuned_metrics.exact_match,
            base_rouge_l: base_metrics.rouge_l,
            finetuned_rouge_l: finetuned_metrics.rouge_l,
            base_response_len: base_metrics.response_len,
            finetuned_response_len: finetuned_metrics.response_len,
            base_latency_ms: base_metrics.latency_ms,
            finetuned_latency_ms: finetuned_metrics.latency_ms,
        });
    }

    base.finalize(eval_examples.len());
    finetuned.finalize(eval_examples.len());
    Ok((base, finetuned, rows))
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregateMetrics {
    pub exact_match_rate: f64,
    pub avg_rouge_l: f64,
    pub avg_response_len: f64,
    pub avg_latency_ms: f64,
    pub count: usize,
    exact_matches: usize,
    rouge_sum: f64,
    response_len_sum: usize,
    latency_sum: u128,
}

impl AggregateMetrics {
    fn record(&mut self, row: crate::EvalMetrics) {
        if row.exact_match {
            self.exact_matches += 1;
        }
        self.rouge_sum += row.rouge_l;
        self.response_len_sum += row.response_len;
        self.latency_sum += row.latency_ms;
    }

    fn finalize(&mut self, count: usize) {
        let denom = count as f64;
        self.count = count;
        self.exact_match_rate = self.exact_matches as f64 / denom;
        self.avg_rouge_l = self.rouge_sum / denom;
        self.avg_response_len = self.response_len_sum as f64 / denom;
        self.avg_latency_ms = self.latency_sum as f64 / denom;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowMetrics {
    pub index: usize,
    pub prompt: String,
    pub target: String,
    pub base_output: String,
    pub finetuned_output: String,
    pub base_exact_match: bool,
    pub finetuned_exact_match: bool,
    pub base_rouge_l: f64,
    pub finetuned_rouge_l: f64,
    pub base_response_len: usize,
    pub finetuned_response_len: usize,
    pub base_latency_ms: u128,
    pub finetuned_latency_ms: u128,
}

pub fn write_manifest(path: &Path, manifest: &CandleRunManifest) -> AppResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(manifest)?)?;
    Ok(())
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> AppResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

pub fn sample_window(
    text: &str,
    tokenizer: &ByteTokenizer,
    max_seq_len: usize,
    rng: &mut StdRng,
) -> AppResult<(Tensor, Tensor)> {
    let encoded = tokenizer.encode_with_eos(text)?;
    if encoded.len() < 2 {
        return Err(AppError::Validation(
            "training sample is too short after tokenization".to_string(),
        ));
    }

    let window = max_seq_len.min(encoded.len() - 1);
    let max_start = encoded.len() - (window + 1);
    let start = if max_start == 0 {
        0
    } else {
        rng.gen_range(0..=max_start)
    };
    let slice = &encoded[start..start + window + 1];
    let input = Tensor::from_slice(&slice[..window], (1, window), &Device::Cpu)?;
    let target = Tensor::from_slice(&slice[1..], window, &Device::Cpu)?;
    Ok((input, target))
}

pub fn render_leaderboard_markdown(
    report_title: &str,
    dataset_path: &str,
    train_summary: &TrainSummary,
    base: &AggregateMetrics,
    finetuned: &AggregateMetrics,
) -> String {
    format!(
        "# {report_title}\n\n\
## Run\n\
- dataset: `{dataset_path}`\n\
- loss min: `{:.6}`\n\
- loss final: `{:.6}`\n\n\
## Aggregate Metrics\n\n\
| Checkpoint | Exact Match | ROUGE-L | Avg Response Len | Avg Latency (ms) |\n\
|---|---:|---:|---:|---:|\n\
| Base | {:.3} | {:.3} | {:.2} | {:.2} |\n\
| Fine-tuned | {:.3} | {:.3} | {:.2} | {:.2} |\n\n\
## Deltas\n\
- Exact Match delta: {:.3}\n\
- ROUGE-L delta: {:.3}\n\
- Response length delta: {:.2}\n\
- Latency delta (ms): {:.2}\n",
        train_summary.min_loss,
        train_summary.final_loss,
        base.exact_match_rate,
        base.avg_rouge_l,
        base.avg_response_len,
        base.avg_latency_ms,
        finetuned.exact_match_rate,
        finetuned.avg_rouge_l,
        finetuned.avg_response_len,
        finetuned.avg_latency_ms,
        finetuned.exact_match_rate - base.exact_match_rate,
        finetuned.avg_rouge_l - base.avg_rouge_l,
        finetuned.avg_response_len - base.avg_response_len,
        finetuned.avg_latency_ms - base.avg_latency_ms,
    )
}

fn argmax_token(logits: &Tensor) -> AppResult<u32> {
    let values = logits.to_vec1::<f32>()?;
    let (index, _) = values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| AppError::Validation("logits vector is empty".to_string()))?;
    Ok(index as u32)
}

fn causal_mask(seq_len: usize, device: &Device) -> AppResult<Tensor> {
    let mut values = vec![0f32; seq_len * seq_len];
    for row in 0..seq_len {
        for col in (row + 1)..seq_len {
            values[row * seq_len + col] = NEG_INFINITY;
        }
    }
    Ok(Tensor::from_slice(
        &values,
        (1, 1, seq_len, seq_len),
        device,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_round_trip_preserves_text() -> AppResult<()> {
        let text = "Rust only";
        let tokenizer = ByteTokenizer::from_texts(&[text.to_string()])?;
        let encoded = tokenizer.encode_with_eos(text)?;
        assert_eq!(tokenizer.decode(&encoded), text);
        Ok(())
    }

    #[test]
    fn leaderboard_contains_expected_sections() {
        let markdown = render_leaderboard_markdown(
            "Report",
            "data/valid_mistral.jsonl",
            &TrainSummary {
                final_loss: 0.5,
                min_loss: 0.25,
                loss_history: vec![],
            },
            &AggregateMetrics {
                exact_match_rate: 0.1,
                avg_rouge_l: 0.2,
                avg_response_len: 10.0,
                avg_latency_ms: 12.0,
                count: 2,
                ..AggregateMetrics::default()
            },
            &AggregateMetrics {
                exact_match_rate: 0.2,
                avg_rouge_l: 0.3,
                avg_response_len: 11.0,
                avg_latency_ms: 13.0,
                count: 2,
                ..AggregateMetrics::default()
            },
        );
        assert!(markdown.contains("# Report"));
        assert!(markdown.contains("Fine-tuned"));
        assert!(markdown.contains("ROUGE-L delta"));
    }
}
