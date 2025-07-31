use std::fs;
use std::path::Path;
use std::process;

pub fn setup(data_path: &str) {
    // check if data exists, throw otherwise
    if !Path::new(data_path).exists() {
        eprintln!("Error: 'data' directory does not exist.");
        process::exit(1);
    }

    if !Path::new("finetuning_data").exists() {
        fs::create_dir_all("finetuning_data").expect("Failed to create finetuning_data directory");
    }

    if !Path::new("tokenized_finetuning_data").exists() {
        fs::create_dir_all("tokenized_finetuning_data").expect("Failed to create tokenized_finetuning_data directory");
    }

    if !Path::new("finetuned").exists() {
        fs::create_dir_all("finetuned").expect("Failed to create finetuned directory");
    }

    // check if checkpoints exists
    if !Path::new("checkpoints").exists() {
        fs::create_dir_all("checkpoints").expect("Failed to create checkpoints directory");
    }

    // check if tokenized exists
    if !Path::new("tokenized").exists() {
        fs::create_dir_all("tokenized").expect("Failed to create tokenized directory");
    }
}