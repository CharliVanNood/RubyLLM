use std::fs::File;
use std::io::Write;

mod data;
mod tokenizer;
mod setup;

pub const SEQUENCE_LENGTH: usize = 64;

fn main() {
    println!("\nRUBY AI\n");

    setup::setup();

    let tokenizer_result = tokenizer::tokenize();
    let tokenizer = match tokenizer_result {
        Ok(tokenizer_out) => {
            println!("Created the tokenizer with vocab size {}\n", tokenizer_out.get_vocab_size(false));
            tokenizer_out
        }
        Err(e) => {
            eprintln!("Error during tokenization: {}", e);
            return;
        }
    };

    let encoding = tokenizer.encode("Nyo is amazing!", true).unwrap();
    println!("Test sentence: {:?} Result: {:?}", encoding.get_tokens(), encoding.get_ids());

    let text = data::load_data();
    let text_reduced = data::reduce_spaces(&text);
    let text_sanitized = data::sanitize(&text_reduced);
    println!("The text has been split into {} words\nTokenizing Dataset\n", text_sanitized.split(" ").count());
    let text_tokenized = tokenizer.encode(text_sanitized, true).unwrap().get_ids().to_vec();

    println!("Creating sequences");
    let mut sequences = Vec::new();
    let mut results = Vec::new();
    for sequence in text_tokenized.windows(SEQUENCE_LENGTH + 1) {
        let input = sequence[..SEQUENCE_LENGTH].to_vec();
        let target = sequence[SEQUENCE_LENGTH];

        sequences.push(input);
        results.push(target);
    }
    println!("Amount of sequences: {}\nAmount of results: {}\n\nWriting to file", sequences.len(), results.len());
    let resulting_array = (sequences, results);

    let json = serde_json::to_string(&resulting_array).unwrap();
    let mut file = File::create("tokenized/data.json").unwrap();
    file.write_all(json.as_bytes()).unwrap();
}
