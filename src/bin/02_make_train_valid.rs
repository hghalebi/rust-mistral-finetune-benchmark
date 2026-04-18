use mistral_fintune::{
    AppError, AppResult, CleanSample, LocalPath, Seed, TrainSample, ValidationRatio, pretty_json,
    read_jsonl_lines, write_jsonl_rows,
};
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};

fn main() -> AppResult<()> {
    let input_path = LocalPath::new("data/cleaned_chat.jsonl");
    let train_path = LocalPath::new("data/train_mistral.jsonl");
    let valid_path = LocalPath::new("data/valid_mistral.jsonl");

    let validation_ratio = ValidationRatio::new(0.05)?;
    let seed = Seed::new(42);

    let lines = read_jsonl_lines(&input_path)?;

    let mut rows = Vec::new();
    for line in lines {
        let cleaned: CleanSample = serde_json::from_str(line.as_str())?;
        rows.push(TrainSample::new(cleaned.messages));
    }

    if rows.len() < 2 {
        return Err(AppError::Validation(
            "need at least 2 rows to create train and validation files".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed.get());
    rows.shuffle(&mut rng);

    let valid_count = ((rows.len() as f64) * validation_ratio.get()).round() as usize;
    let valid_count = valid_count.max(1).min(rows.len() - 1);

    let valid_rows = rows[..valid_count].to_vec();
    let train_rows = rows[valid_count..].to_vec();

    write_jsonl_rows(&train_path, &train_rows)?;
    write_jsonl_rows(&valid_path, &valid_rows)?;

    println!(
        "loaded cleaned rows : {}",
        train_rows.len() + valid_rows.len()
    );
    println!("train rows          : {}", train_rows.len());
    println!("valid rows          : {}", valid_rows.len());

    println!("\n===== FIRST 2 TRAIN ROWS =====");
    for row in train_rows.iter().take(2) {
        println!("{}", pretty_json(row)?);
    }

    println!("\n===== FIRST 2 VALID ROWS =====");
    for row in valid_rows.iter().take(2) {
        println!("{}", pretty_json(row)?);
    }

    Ok(())
}
