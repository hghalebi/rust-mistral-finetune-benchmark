//! Pure-Rust Burn backend for true pre/post checkpoint benchmarking.

use crate::{AppError, AppResult, EvalMetrics, TrainSample, eval_row_metrics, split_for_eval};
use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        PositionalEncoding, PositionalEncodingConfig,
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLoss,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub type InferBackend = NdArray<f32>;
pub type TrainBackend = Autodiff<InferBackend>;
pub type BurnDevice = NdArrayDevice;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTokenizer {
    stoi: HashMap<String, i64>,
    itos: Vec<String>,
    pad_token_id: i64,
    eos_token_id: i64,
}

impl WordTokenizer {
    pub fn from_texts(texts: &[String]) -> AppResult<Self> {
        let mut vocab = BTreeSet::new();
        for text in texts {
            for token in text.split_whitespace() {
                vocab.insert(token.to_string());
            }
        }

        if vocab.is_empty() {
            return Err(AppError::Validation(
                "cannot build tokenizer from empty corpus".to_string(),
            ));
        }

        let itos = vocab.into_iter().collect::<Vec<_>>();
        let stoi = itos
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, token)| (token, index as i64))
            .collect::<HashMap<_, _>>();
        let pad_token_id = itos.len() as i64;
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

    pub fn pad_token_id(&self) -> i64 {
        self.pad_token_id
    }

    pub fn eos_token_id(&self) -> i64 {
        self.eos_token_id
    }

    pub fn encode_prompt(&self, value: &str) -> AppResult<Vec<i64>> {
        value
            .split_whitespace()
            .map(|token| {
                self.stoi.get(token).copied().ok_or_else(|| {
                    AppError::Validation(format!("tokenizer missing token {token:?}"))
                })
            })
            .collect()
    }

    pub fn encode_with_eos(&self, value: &str) -> AppResult<Vec<i64>> {
        let mut out = self.encode_prompt(value)?;
        out.push(self.eos_token_id());
        Ok(out)
    }

    pub fn decode(&self, ids: &[i64]) -> String {
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
pub struct BurnRunManifest {
    pub model: TinyLmConfig,
    pub tokenizer: WordTokenizer,
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

#[derive(Module, Debug)]
pub struct TinyBurnCausalLm<B: Backend> {
    token_embedding: Embedding<B>,
    positional_encoding: PositionalEncoding<B>,
    encoder: TransformerEncoder<B>,
    final_norm: LayerNorm<B>,
    lm_head: Linear<B>,
}

#[derive(Debug, Clone)]
pub struct TinyBurnCausalLmConfig {
    config: TinyLmConfig,
}

impl TinyBurnCausalLmConfig {
    pub fn new(config: TinyLmConfig) -> Self {
        Self { config }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> TinyBurnCausalLm<B> {
        let token_embedding =
            EmbeddingConfig::new(self.config.vocab_size, self.config.d_model).init(device);
        let positional_encoding = PositionalEncodingConfig::new(self.config.d_model)
            .with_max_sequence_size(self.config.max_seq_len)
            .init(device);
        let encoder = TransformerEncoderConfig::new(
            self.config.d_model,
            self.config.mlp_hidden,
            self.config.n_heads,
            self.config.n_layers,
        )
        .with_dropout(0.0)
        .with_norm_first(true)
        .init(device);
        let final_norm = LayerNormConfig::new(self.config.d_model).init(device);
        let lm_head = LinearConfig::new(self.config.d_model, self.config.vocab_size)
            .with_bias(false)
            .init(device);

        TinyBurnCausalLm {
            token_embedding,
            positional_encoding,
            encoder,
            final_norm,
            lm_head,
        }
    }
}

impl<B> TinyBurnCausalLm<B>
where
    B: Backend,
{
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = token_ids.dims();
        let device = token_ids.device();
        let hidden = self.token_embedding.forward(token_ids);
        let hidden = self.positional_encoding.forward(hidden);
        let mask = generate_autoregressive_mask::<B>(batch_size, seq_len, &device);
        let hidden = self
            .encoder
            .forward(TransformerEncoderInput::new(hidden).mask_attn(mask));
        let hidden = self.final_norm.forward(hidden);
        self.lm_head.forward(hidden)
    }

    pub fn generate(
        &self,
        tokenizer: &WordTokenizer,
        prompt: &str,
        max_new_tokens: usize,
        max_seq_len: usize,
        device: &B::Device,
    ) -> AppResult<String> {
        let mut token_ids = tokenizer.encode_prompt(prompt)?;
        let prompt_len = token_ids.len();

        for _ in 0..max_new_tokens {
            let start = token_ids.len().saturating_sub(max_seq_len.max(1));
            let window = &token_ids[start..];
            let input = int_tensor_2d::<B>(window, [1, window.len()], device);
            let logits = self.forward(input);
            let [_, seq_len, vocab_size] = logits.dims();
            let last_logits = logits
                .slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size])
                .reshape([1, vocab_size]);
            let next_id = argmax_token(last_logits)?;
            if next_id == tokenizer.eos_token_id() {
                break;
            }
            token_ids.push(next_id);
        }

        Ok(tokenizer.decode(&token_ids[prompt_len..]))
    }
}

pub fn cpu_device() -> BurnDevice {
    NdArrayDevice::Cpu
}

pub fn seed_backend(seed: u64, device: &BurnDevice) {
    InferBackend::seed(device, seed);
    TrainBackend::seed(device, seed);
}

pub fn build_model(config: &TinyLmConfig, device: &BurnDevice) -> TinyBurnCausalLm<TrainBackend> {
    TinyBurnCausalLmConfig::new(config.clone()).init::<TrainBackend>(device)
}

pub fn save_checkpoint(
    model: TinyBurnCausalLm<InferBackend>,
    checkpoint_stem: &Path,
) -> AppResult<PathBuf> {
    if let Some(parent) = checkpoint_stem.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.save_file(checkpoint_stem.to_path_buf(), &recorder)?;
    resolve_checkpoint_path(checkpoint_stem)
}

pub fn load_model(
    config: &TinyLmConfig,
    checkpoint_stem: &Path,
    device: &BurnDevice,
) -> AppResult<TinyBurnCausalLm<InferBackend>> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = TinyBurnCausalLmConfig::new(config.clone()).init::<InferBackend>(device);
    Ok(model.load_file(checkpoint_stem.to_path_buf(), &recorder, device)?)
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
    mut model: TinyBurnCausalLm<TrainBackend>,
    tokenizer: &WordTokenizer,
    texts: &[String],
    hyperparameters: TrainHyperparameters,
    device: &BurnDevice,
) -> AppResult<(TinyBurnCausalLm<TrainBackend>, TrainSummary)> {
    if texts.is_empty() {
        return Err(AppError::Validation(
            "no training texts available".to_string(),
        ));
    }

    let criterion = CrossEntropyLoss::new(None, device);
    let mut optimizer = AdamWConfig::new().init::<TrainBackend, TinyBurnCausalLm<TrainBackend>>();
    let mut rng = StdRng::seed_from_u64(hyperparameters.seed);
    let mut history = Vec::new();
    let mut min_loss = f64::INFINITY;
    let mut final_loss = 0.0;

    for step in 1..=hyperparameters.steps {
        let sample_index = rng.gen_range(0..texts.len());
        let (input_ids, target_ids) = sample_window(
            &texts[sample_index],
            tokenizer,
            max_seq_len(&model),
            &mut rng,
        )?;
        let input = int_tensor_2d::<TrainBackend>(&input_ids, [1, input_ids.len()], device);
        let targets = int_tensor_1d::<TrainBackend>(&target_ids, [target_ids.len()], device);
        let logits = model.forward(input);
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let loss = criterion.forward(logits.reshape([batch_size * seq_len, vocab_size]), targets);
        final_loss = scalar_f32(loss.clone())? as f64;
        min_loss = min_loss.min(final_loss);

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(hyperparameters.learning_rate, model, grads);

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

    Ok((
        model,
        TrainSummary {
            final_loss,
            min_loss,
            loss_history: history,
        },
    ))
}

pub fn evaluate_pair(
    base_model: &TinyBurnCausalLm<InferBackend>,
    finetuned_model: &TinyBurnCausalLm<InferBackend>,
    tokenizer: &WordTokenizer,
    eval_examples: &[EvalExample],
    max_new_tokens: usize,
    config: &TinyLmConfig,
    device: &BurnDevice,
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
        let base_output = base_model.generate(
            tokenizer,
            &example.prompt,
            max_new_tokens,
            config.max_seq_len,
            device,
        )?;
        let base_latency = started.elapsed().as_millis();
        let base_metrics = eval_row_metrics(&base_output, &example.target, base_latency);

        let started = Instant::now();
        let finetuned_output = finetuned_model.generate(
            tokenizer,
            &example.prompt,
            max_new_tokens,
            config.max_seq_len,
            device,
        )?;
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
    fn record(&mut self, row: EvalMetrics) {
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

pub fn write_manifest(path: &Path, manifest: &BurnRunManifest) -> AppResult<()> {
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

fn resolve_checkpoint_path(checkpoint_stem: &Path) -> AppResult<PathBuf> {
    let candidates = [
        checkpoint_stem.to_path_buf(),
        checkpoint_stem.with_extension("mpk"),
        checkpoint_stem.with_extension("mpk.gz"),
    ];

    candidates
        .into_iter()
        .find(|candidate| candidate.exists())
        .ok_or_else(|| {
            AppError::Validation(format!(
                "could not resolve recorded checkpoint path for {}",
                checkpoint_stem.display()
            ))
        })
}

fn max_seq_len<B: Backend>(model: &TinyBurnCausalLm<B>) -> usize {
    model.positional_encoding.max_sequence_size
}

fn int_tensor_2d<B: Backend>(
    values: &[i64],
    shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    Tensor::from_data(TensorData::new(values.to_vec(), shape), device)
}

fn int_tensor_1d<B: Backend>(
    values: &[i64],
    shape: [usize; 1],
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    Tensor::from_data(TensorData::new(values.to_vec(), shape), device)
}

fn scalar_f32<B: Backend>(tensor: Tensor<B, 1>) -> AppResult<f32> {
    tensor
        .into_data()
        .into_vec::<f32>()?
        .into_iter()
        .next()
        .ok_or_else(|| AppError::Validation("scalar tensor was empty".to_string()))
}

fn argmax_token<B: Backend>(tensor: Tensor<B, 2>) -> AppResult<i64> {
    tensor
        .argmax(1)
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()?
        .into_iter()
        .next()
        .ok_or_else(|| AppError::Validation("argmax tensor was empty".to_string()))
}

fn sample_window(
    text: &str,
    tokenizer: &WordTokenizer,
    max_seq_len: usize,
    rng: &mut StdRng,
) -> AppResult<(Vec<i64>, Vec<i64>)> {
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
    Ok((slice[..window].to_vec(), slice[1..].to_vec()))
}
