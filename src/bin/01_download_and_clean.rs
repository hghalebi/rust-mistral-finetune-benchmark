use hf_hub::{Repo, api::sync::ApiBuilder};
use mistral_fintune::{
    AppError, AppResult, DatasetFile, DatasetRepo, LocalPath, MaxTotalChars, pretty_json,
    read_jsonl_lines, to_clean_sample, write_jsonl_rows,
};
use std::collections::HashSet;
use std::env;
use std::fs;

fn main() -> AppResult<()> {
    let data_dir = LocalPath::new("data");
    let raw_path = LocalPath::new("data/raw.jsonl");
    let cleaned_path = LocalPath::new("data/cleaned_chat.jsonl");

    let (dataset_repo, dataset_file) =
        parse_dataset_ref(&env::var("HF_DATASET_REF").unwrap_or_else(|_| {
            "hf://datasets/Roman1111111/claude-opus-4.6-10000x/opus46_final.jsonl".to_string()
        }))?;
    let max_total_chars = MaxTotalChars::new(12_000);

    fs::create_dir_all(data_dir.as_str())?;

    let token = std::env::var("HF_TOKEN").ok();
    let api = ApiBuilder::new().with_token(token).build()?;
    let repo = api.repo(Repo::dataset(dataset_repo.as_str().to_owned()));
    let downloaded_path = repo.download(dataset_file.as_str())?;

    fs::copy(&downloaded_path, raw_path.as_str())?;
    println!("saved raw file to {}", raw_path.as_str());

    let raw_lines = read_jsonl_lines(&raw_path)?;

    let mut cleaned_rows = Vec::new();
    let mut seen = HashSet::new();
    let mut dropped_bad_json = 0usize;
    let mut dropped_invalid_row = 0usize;
    let mut dropped_too_long = 0usize;
    let mut dropped_duplicate = 0usize;

    for (index, raw_line) in raw_lines.iter().enumerate() {
        if index < 5 {
            println!("\n===== RAW SAMPLE {} =====", index + 1);
            println!("{}", raw_line.as_str());
        }

        let parsed = match serde_json::from_str::<serde_json::Value>(raw_line.as_str()) {
            Ok(value) => value,
            Err(_) => {
                dropped_bad_json += 1;
                continue;
            }
        };

        let cleaned = match to_clean_sample(&parsed) {
            Ok(sample) => sample,
            Err(AppError::Validation(_)) => {
                dropped_invalid_row += 1;
                continue;
            }
            Err(err) => return Err(err),
        };

        if cleaned.messages.total_content_len() > max_total_chars.get() {
            dropped_too_long += 1;
            continue;
        }

        let fingerprint = serde_json::to_string(&cleaned.messages)?;
        if !seen.insert(fingerprint) {
            dropped_duplicate += 1;
            continue;
        }

        cleaned_rows.push(cleaned);
    }

    write_jsonl_rows(&cleaned_path, &cleaned_rows)?;

    println!("\n===== SUMMARY =====");
    println!("raw rows           : {}", raw_lines.len());
    println!("kept rows          : {}", cleaned_rows.len());
    println!("dropped bad json   : {}", dropped_bad_json);
    println!("dropped invalid    : {}", dropped_invalid_row);
    println!("dropped too long   : {}", dropped_too_long);
    println!("dropped duplicate  : {}", dropped_duplicate);
    println!("cleaned output     : {}", cleaned_path.as_str());

    println!("\n===== FIRST 3 CLEANED ROWS =====");
    for row in cleaned_rows.iter().take(3) {
        println!("{}", pretty_json(row)?);
    }

    Ok(())
}

fn parse_dataset_ref(value: &str) -> AppResult<(DatasetRepo, DatasetFile)> {
    let normalized = value.trim();

    let without_prefix = normalized
        .strip_prefix("hf://datasets/")
        .unwrap_or(normalized)
        .trim_start_matches('/');

    let (repo, file) = without_prefix.rsplit_once('/').ok_or_else(|| {
        AppError::Validation("dataset ref must be like <repo>/<file>".to_string())
    })?;

    if repo.is_empty() || file.is_empty() {
        return Err(AppError::Validation(
            "dataset ref must include both repository and filename".to_string(),
        ));
    }

    Ok((
        DatasetRepo::new(repo.to_string()),
        DatasetFile::new(file.to_string()),
    ))
}
