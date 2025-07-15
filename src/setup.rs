use std::fs;
use std::path::Path;
use std::process;

pub fn setup() {
    // check if data exists, throw otherwise
    if !Path::new("data").exists() {
        eprintln!("Error: 'data' directory does not exist.");
        process::exit(1);
    }

    // check if checkpoints exists
    if !Path::new("checkpoints").exists() {
        fs::create_dir_all("checkpoints").expect("Failed to create checkpoints directory");
    }
}