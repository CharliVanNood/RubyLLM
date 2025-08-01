use std::fs::File;
use std::io::Write;
use std::env;
use dotenv::dotenv;

mod data;
mod tokenizer;
mod setup;

pub const SEQUENCE_LENGTH: usize = 128;

pub const TRAIN_FULL: bool = true;
// set this to false if you only want short responses
// it will not merge sentences for continuation and only make it train on the individual sentences

fn main() {
    println!("\nNyoLLM Data Transformer\n");

    let mut train = true;
    for arg in std::env::args() {
        if arg == "tune" {
            train = false;
        } else if arg == "train" {
            train = true;
        }
    }

    dotenv().ok();
    let mut training_data_directory = env::var("TRAIN_DATA").expect("TRAIN_DATA not set in .env");
    setup::setup(&training_data_directory);

    if train {
        _ = tokenizer::tokenize(&training_data_directory);
    }

    let tokenizer = match tokenizers::Tokenizer::from_file("checkpoints/tokenizer.json") {
        Ok(tokenizer_out) => {
            println!("Created the tokenizer with vocab size {}\n", tokenizer_out.get_vocab_size(false));
            tokenizer_out
        }
        Err(e) => {
            eprintln!("Error during tokenization: {}", e);
            return;
        }
    };

    if !train {
        training_data_directory = "finetuning_data".to_string();
    }

    let encoding = tokenizer.encode("Nyo is amazing![CLS]", true).unwrap();
    println!("Test sentence: {:?} Result: {:?}", encoding.get_tokens(), encoding.get_ids());

    // Get Enter Token
    let encoding = tokenizer.encode("[SEP][CLS][IGN][STA]", true).unwrap();
    println!("Special Tokens: {:?} Result: {:?}", encoding.get_tokens(), encoding.get_ids());
    let seperation_token = encoding.get_ids()[0];
    let classification_token = encoding.get_ids()[1];
    let ignore_token = encoding.get_ids()[2];
    let start_token = encoding.get_ids()[3];

    let text = data::load_data(&training_data_directory);
    let text_reduced = data::reduce_spaces(&text);
    let text_sanitized = data::sanitize(&text_reduced);
    println!("The text has been split into {} words\nTokenizing Dataset\n", text_sanitized.split(" ").count());
    let text_tokenized = tokenizer.encode(text_sanitized, true).unwrap().get_ids().to_vec();

    println!("Creating sequences");
    let mut sequences = Vec::new();
    let mut results = Vec::new();
    if TRAIN_FULL {
        for sequence in text_tokenized.windows(SEQUENCE_LENGTH + 1) {
            let input = sequence[..SEQUENCE_LENGTH].to_vec();
            let target = sequence[SEQUENCE_LENGTH];

            sequences.push(input);
            results.push(target);
        }
    }
    let base_sequences_len = sequences.len();
    println!("Amount of sequences: {}\nAmount of results: {}\n", sequences.len(), results.len());

    println!("Creating partial sequences");
    let mut temp_sequence = [0; SEQUENCE_LENGTH];
    let mut previous_token = 0;
    let mut seperators = 0;
    let mut ignore = false;
    for token in text_tokenized {
        for i in 0..SEQUENCE_LENGTH-1 {
            temp_sequence[i] = temp_sequence[i + 1];
        }
        temp_sequence[SEQUENCE_LENGTH - 1] = previous_token;

        if temp_sequence != [0; SEQUENCE_LENGTH] && !ignore {
            sequences.push(temp_sequence.to_vec());
            results.push(token);
        }

        if token == ignore_token { ignore = true; }
        if token == start_token { ignore = false; }
        if token == seperation_token { ignore = false; }

        if (previous_token == seperation_token && temp_sequence[0] != 0) || previous_token == classification_token {
            temp_sequence = [0; SEQUENCE_LENGTH];
            seperators += 1;
            previous_token = 0;
        } else {
            previous_token = token;
        }
    }
    for sequence in 0..5 {
        println!("{:?}", sequences[sequence + base_sequences_len]);
    }
    println!("Amount of sequences: {}\nAmount of results: {}\nAmount of seperations: {}\n\nWriting to file", sequences.len(), results.len(), seperators);
    let resulting_array = (sequences, results, tokenizer.get_vocab_size(false));

    if train {
        let json = serde_json::to_string(&resulting_array).unwrap();
        let mut file = File::create("tokenized/data.json").unwrap();
        file.write_all(json.as_bytes()).unwrap();
        println!("Data has been written to tokenized/data.json\n\nStarting NyoLLM\n");
    } else {
        let json = serde_json::to_string(&resulting_array).unwrap();
        let mut file = File::create("tokenized_finetuning_data/data.json").unwrap();
        file.write_all(json.as_bytes()).unwrap();
        println!("Data has been written to tokenized_finetuning_data/data.json\n\nStarting NyoLLM\n");
    }
}