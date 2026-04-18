//! Shared types and helpers for preparing chat-style JSONL data for supervised fine-tuning.

use serde::{Deserialize, Serialize};
use serde_json::Value;
pub mod backend_runtime;
#[cfg(feature = "burn")]
pub mod burn_backend;
#[cfg(feature = "candle")]
pub mod candle_backend;

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use thiserror::Error;

/// Project-wide result type.
pub type AppResult<T> = std::result::Result<T, AppError>;

/// Typed error surface for the prep pipeline.
#[derive(Debug, Error)]
pub enum AppError {
    /// I/O failed while reading or writing files.
    #[error("input/output failed: {0}")]
    Io(#[from] std::io::Error),

    /// JSON parsing or serialization failed.
    #[error("json failed: {0}")]
    Json(#[from] serde_json::Error),

    /// Hugging Face Hub API error.
    #[error("hugging face hub failed: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),

    /// Domain or dataset validation failed.
    #[error("validation failed: {0}")]
    Validation(String),

    /// Fine-tuning backend is unavailable in the current binary.
    #[error("backend '{backend}' unavailable: {reason}")]
    BackendUnavailable { backend: String, reason: String },

    /// YAML serialization/deserialization failed.
    #[error("yaml failed: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// External Rust command failed while training or generating.
    #[error("command failed: {command}")]
    CommandFailure {
        command: String,
        status: i32,
        stdout: String,
        stderr: String,
    },

    /// Candle backend failure.
    #[cfg(feature = "candle")]
    #[error("candle failed: {0}")]
    Candle(#[from] candle_core::Error),

    /// Burn checkpoint save/load failure.
    #[cfg(feature = "burn")]
    #[error("burn recorder failed: {0}")]
    BurnRecorder(#[from] burn::record::RecorderError),

    /// Burn tensor data conversion failure.
    #[cfg(feature = "burn")]
    #[error("burn tensor data failed: {0}")]
    BurnData(#[from] burn::tensor::DataError),
}

/// One evaluation row derived from a chat sample.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalSample {
    /// Messages used as the prompt/input side.
    pub prompt_messages: Messages,
    /// The canonical expected assistant message.
    pub target: MessageContent,
}

impl EvalSample {
    /// Builds a canonical model input prompt from the messages.
    pub fn prompt_text(&self) -> String {
        let mut out = String::new();

        for message in self.prompt_messages.as_slice() {
            let role = match message.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            out.push_str(&format!("[{role}]\n{}\n", message.content.as_str()));
        }

        out.push_str("[assistant]\n");
        out
    }
}

/// Row-level evaluation metrics for one prompt.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EvalMetrics {
    /// Exact-match comparison after normalization.
    pub exact_match: bool,
    /// ROUGE-L F1 score.
    pub rouge_l: f64,
    /// Response length in whitespace tokens.
    pub response_len: usize,
    /// End-to-end generation time in milliseconds.
    pub latency_ms: u128,
}

/// Repository identifier in a strong type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DatasetRepo(String);

impl DatasetRepo {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Dataset file path in a strong type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DatasetFile(String);

impl DatasetFile {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Generic filesystem path wrapper used for boundary inputs/outputs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LocalPath(String);

impl LocalPath {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Maximum characters allowed in a full sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaxTotalChars(usize);

impl MaxTotalChars {
    pub fn new(value: usize) -> Self {
        Self(value)
    }

    pub fn get(self) -> usize {
        self.0
    }
}

/// Validation ratio helper with runtime checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidationRatio(f64);

impl ValidationRatio {
    pub fn new(value: f64) -> AppResult<Self> {
        if !(0.0..1.0).contains(&value) {
            return Err(AppError::Validation(
                "validation ratio must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f64 {
        self.0
    }
}

/// Seed value for deterministic shuffling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Seed(u64);

impl Seed {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

/// A raw line from an input `.jsonl` file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RawJsonLine(String);

impl RawJsonLine {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Chat message content wrapped as a typed value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MessageContent(String);

impl MessageContent {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn normalized(value: &str) -> Self {
        let normalized = value
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();

        Self(normalized)
    }

    pub fn is_blank(&self) -> bool {
        self.0.trim().is_empty()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Message roles expected by Mistral-style chat datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System instructions.
    System,
    /// User prompt.
    User,
    /// Assistant completion.
    Assistant,
}

impl TryFrom<&str> for Role {
    type Error = AppError;

    fn try_from(value: &str) -> AppResult<Self> {
        match value.trim() {
            "system" => Ok(Self::System),
            "user" => Ok(Self::User),
            "assistant" => Ok(Self::Assistant),
            other => Err(AppError::Validation(format!("unsupported role: {other}"))),
        }
    }
}

/// One chat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: MessageContent,
}

impl ChatMessage {
    pub fn new(role: Role, content: MessageContent) -> Self {
        Self { role, content }
    }
}

/// Ordered list of messages composing one training sample.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Messages(Vec<ChatMessage>);

impl Messages {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self(messages)
    }

    pub fn as_slice(&self) -> &[ChatMessage] {
        &self.0
    }

    pub fn into_inner(self) -> Vec<ChatMessage> {
        self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn has_user(&self) -> bool {
        self.0.iter().any(|m| m.role == Role::User)
    }

    pub fn has_assistant(&self) -> bool {
        self.0.iter().any(|m| m.role == Role::Assistant)
    }

    pub fn ends_with_assistant(&self) -> bool {
        self.0
            .last()
            .map(|m| m.role == Role::Assistant)
            .unwrap_or(false)
    }

    pub fn total_content_len(&self) -> usize {
        self.0.iter().map(|m| m.content.len()).sum()
    }

    pub fn with_stable_system_message(self) -> Self {
        let stable = MessageContent::new("You are a helpful AI assistant.");
        let mut inner = self.0;

        if inner.is_empty() {
            return Self(inner);
        }

        if inner[0].role == Role::System {
            inner[0].content = stable;
            return Self(inner);
        }

        let mut out = Vec::with_capacity(inner.len() + 1);
        out.push(ChatMessage::new(Role::System, stable));
        out.extend(inner);
        Self(out)
    }
}

/// Kept intact from raw data; used to retain metadata if needed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Metadata(Value);

impl Metadata {
    pub fn new(value: Value) -> Self {
        Self(value)
    }
}

/// Clean sample with raw metadata preserved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CleanSample {
    pub messages: Messages,
    pub metadata: Option<Metadata>,
}

impl CleanSample {
    pub fn new(messages: Messages, metadata: Option<Metadata>) -> Self {
        Self { messages, metadata }
    }
}

/// Final training sample shape expected by downstream Mistral tooling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainSample {
    pub messages: Messages,
}

impl TrainSample {
    pub fn new(messages: Messages) -> Self {
        Self { messages }
    }
}

/// Read all lines from a `.jsonl` file.
pub fn read_jsonl_lines(path: &LocalPath) -> AppResult<Vec<RawJsonLine>> {
    let file = File::open(path.as_str())?;
    let reader = BufReader::new(file);

    let mut lines = Vec::new();
    for line in reader.lines() {
        lines.push(RawJsonLine::new(line?));
    }
    Ok(lines)
}

/// Write a sequence of serializable rows to `.jsonl`.
pub fn write_jsonl_rows<T: Serialize>(path: &LocalPath, rows: &[T]) -> AppResult<()> {
    let file = File::create(path.as_str())?;
    let mut writer = BufWriter::new(file);

    for row in rows {
        serde_json::to_writer(&mut writer, row)?;
        writer.write_all(b"\n")?;
    }

    writer.flush()?;
    Ok(())
}

/// Pretty-print JSON for a human-readable sample.
pub fn pretty_json<T: Serialize>(value: &T) -> AppResult<String> {
    Ok(serde_json::to_string_pretty(value)?)
}

/// Convert one raw JSON value into the internal clean sample shape.
pub fn to_clean_sample(value: &Value) -> AppResult<CleanSample> {
    let metadata = value.get("metadata").cloned().map(Metadata::new);

    let raw_messages = value
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| AppError::Validation("row is missing the messages array".to_string()))?;

    let mut messages = Vec::new();

    for raw_message in raw_messages {
        let raw_role = raw_message
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or_else(|| AppError::Validation("message is missing role".to_string()))?;

        let raw_content = raw_message
            .get("content")
            .and_then(|c| c.as_str())
            .ok_or_else(|| AppError::Validation("message is missing content".to_string()))?;

        let role = Role::try_from(raw_role)?;
        let content = MessageContent::normalized(raw_content);

        if content.is_blank() {
            return Err(AppError::Validation(
                "message content is blank after normalization".to_string(),
            ));
        }

        messages.push(ChatMessage::new(role, content));
    }

    let messages = Messages::new(messages).with_stable_system_message();

    if messages.is_empty() {
        return Err(AppError::Validation("messages list is empty".to_string()));
    }

    if !messages.has_user() {
        return Err(AppError::Validation("row has no user message".to_string()));
    }

    if !messages.has_assistant() {
        return Err(AppError::Validation(
            "row has no assistant message".to_string(),
        ));
    }

    if !messages.ends_with_assistant() {
        return Err(AppError::Validation(
            "row does not end with an assistant message".to_string(),
        ));
    }

    Ok(CleanSample::new(messages, metadata))
}

/// Validate all rows that are expected to match the training schema.
pub fn validate_train_sample(sample: &TrainSample) -> AppResult<()> {
    if sample.messages.is_empty() {
        return Err(AppError::Validation("messages list is empty".to_string()));
    }

    if !sample.messages.has_user() {
        return Err(AppError::Validation("row has no user message".to_string()));
    }

    if !sample.messages.has_assistant() {
        return Err(AppError::Validation(
            "row has no assistant message".to_string(),
        ));
    }

    if !sample.messages.ends_with_assistant() {
        return Err(AppError::Validation(
            "row does not end with an assistant message".to_string(),
        ));
    }

    for message in sample.messages.as_slice() {
        if message.content.is_blank() {
            return Err(AppError::Validation("row has blank content".to_string()));
        }
    }

    Ok(())
}

/// Load all training rows from JSONL and validate them before training.
pub fn load_and_validate_train_rows(path: &LocalPath) -> AppResult<Vec<TrainSample>> {
    let lines = read_jsonl_lines(path)?;
    let mut rows = Vec::with_capacity(lines.len());

    for (index, raw_line) in lines.iter().enumerate() {
        let sample: TrainSample = serde_json::from_str(raw_line.as_str()).map_err(|err| {
            AppError::Validation(format!("line {} parse failed: {err}", index + 1))
        })?;

        validate_train_sample(&sample).map_err(|err| {
            AppError::Validation(format!("line {} validation failed: {err}", index + 1))
        })?;

        rows.push(sample);
    }

    Ok(rows)
}

/// Split one training sample into inference input and expected target.
/// The target is the final assistant message and the prompt is every prior message.
pub fn split_for_eval(sample: &TrainSample) -> AppResult<EvalSample> {
    let messages = sample.messages.as_slice();

    let target_index = messages
        .iter()
        .rposition(|m| m.role == Role::Assistant)
        .ok_or_else(|| AppError::Validation("sample has no assistant message".to_string()))?;

    if target_index == 0 {
        return Err(AppError::Validation(
            "sample starts with assistant message".to_string(),
        ));
    }

    if !messages
        .iter()
        .take(target_index)
        .any(|m| m.role == Role::User)
    {
        return Err(AppError::Validation(
            "sample has no user message before target assistant message".to_string(),
        ));
    }

    let target = messages[target_index].content.clone();
    let prompt_messages = Messages::new(messages[..target_index].to_vec());

    if prompt_messages.is_empty() {
        return Err(AppError::Validation("sample prompt is empty".to_string()));
    }

    Ok(EvalSample {
        prompt_messages,
        target,
    })
}

/// Normalize response text for exact-match checks.
pub fn normalize_for_exact_match(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_lowercase()
}

/// Length of a response in whitespace tokens.
pub fn token_length(value: &str) -> usize {
    value.split_whitespace().count()
}

/// Exact-match metric over normalized strings.
pub fn exact_match(actual: &str, expected: &str) -> bool {
    normalize_for_exact_match(actual) == normalize_for_exact_match(expected)
}

fn lcs_length(a: &[&str], b: &[&str]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    let mut matrix = vec![vec![0usize; b.len() + 1]; a.len() + 1];

    for i in 1..=a.len() {
        for j in 1..=b.len() {
            if a[i - 1] == b[j - 1] {
                matrix[i][j] = matrix[i - 1][j - 1] + 1;
            } else {
                matrix[i][j] = matrix[i - 1][j].max(matrix[i][j - 1]);
            }
        }
    }

    matrix[a.len()][b.len()]
}

/// ROUGE-L F1 for two pieces of text.
pub fn rouge_l_f1(actual: &str, expected: &str) -> f64 {
    let actual_tokens: Vec<&str> = actual.split_whitespace().collect();
    let expected_tokens: Vec<&str> = expected.split_whitespace().collect();

    let lcs = lcs_length(&actual_tokens, &expected_tokens);

    if actual_tokens.is_empty() || expected_tokens.is_empty() || lcs == 0 {
        return 0.0;
    }

    let precision = lcs as f64 / actual_tokens.len() as f64;
    let recall = lcs as f64 / expected_tokens.len() as f64;

    if (precision + recall) <= f64::EPSILON {
        0.0
    } else {
        (2.0 * precision * recall) / (precision + recall)
    }
}

/// Compute a row-level metric bundle.
pub fn eval_row_metrics(output: &str, target: &str, latency_ms: u128) -> EvalMetrics {
    EvalMetrics {
        exact_match: exact_match(output, target),
        rouge_l: rouge_l_f1(output, target),
        response_len: token_length(output),
        latency_ms,
    }
}

/// Aggregate lightweight dataset stats used by Rust-side fine-tuning planners.
pub fn summarize_train_rows(rows: &[TrainSample]) -> (usize, usize, usize) {
    let total_rows = rows.len();
    let total_messages = rows
        .iter()
        .map(|row| row.messages.as_slice().len())
        .sum::<usize>();
    let approx_tokens = rows
        .iter()
        .flat_map(|row| row.messages.as_slice().iter())
        .map(|message| message.content.as_str().split_whitespace().count())
        .sum();

    (total_rows, total_messages, approx_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_known_roles() -> AppResult<()> {
        assert_eq!(Role::try_from("system")?, Role::System);
        assert_eq!(Role::try_from("user")?, Role::User);
        assert_eq!(Role::try_from("assistant")?, Role::Assistant);
        assert!(Role::try_from("bad-role").is_err());
        Ok(())
    }

    #[test]
    fn normalizes_message_content() {
        let value = MessageContent::normalized(" line one \n  \nline two  \n");
        assert_eq!(value.as_str(), "line one\n\nline two");
        assert!(!value.is_blank());
    }

    #[test]
    fn stabilizes_system_message() {
        let messages = Messages::new(vec![
            ChatMessage::new(Role::User, MessageContent::new("hi")),
            ChatMessage::new(Role::Assistant, MessageContent::new("hello")),
        ]);
        let stabilized = messages.with_stable_system_message();
        assert_eq!(stabilized.as_slice().len(), 3);
        assert_eq!(stabilized.as_slice()[0].role, Role::System);
        assert_eq!(
            stabilized.as_slice()[0].content.as_str(),
            "You are a helpful AI assistant."
        );
    }

    #[test]
    fn parses_clean_sample_from_json() -> AppResult<()> {
        let raw = json!({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ],
            "metadata": {"source":"test"}
        });
        let sample = to_clean_sample(&raw)?;
        assert_eq!(sample.messages.as_slice().len(), 3);
        assert!(sample.metadata.is_some());
        assert_eq!(sample.messages.as_slice()[0].role, Role::System);
        assert_eq!(sample.messages.as_slice()[2].role, Role::Assistant);
        Ok(())
    }

    #[test]
    fn validates_train_sample() -> AppResult<()> {
        let sample = TrainSample::new(Messages::new(vec![
            ChatMessage::new(Role::System, MessageContent::new("system")),
            ChatMessage::new(Role::User, MessageContent::new("question")),
            ChatMessage::new(Role::Assistant, MessageContent::new("answer")),
        ]));
        validate_train_sample(&sample)?;
        Ok(())
    }
}
