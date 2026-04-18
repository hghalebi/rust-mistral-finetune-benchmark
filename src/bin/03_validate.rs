use mistral_fintune::{
    AppError, AppResult, LocalPath, TrainSample, read_jsonl_lines, validate_train_sample,
};

fn main() -> AppResult<()> {
    let train_path = LocalPath::new("data/train_mistral.jsonl");
    let valid_path = LocalPath::new("data/valid_mistral.jsonl");

    validate_file(&train_path)?;
    validate_file(&valid_path)?;

    println!("validation passed");
    Ok(())
}

fn validate_file(path: &LocalPath) -> AppResult<()> {
    let lines = read_jsonl_lines(path)?;

    for (index, line) in lines.iter().enumerate() {
        let row: TrainSample = serde_json::from_str(line.as_str())?;

        validate_train_sample(&row).map_err(|error| match error {
            AppError::Validation(message) => {
                AppError::Validation(format!("{} line {} {}", path.as_str(), index + 1, message))
            }
            other => other,
        })?;
    }

    Ok(())
}
