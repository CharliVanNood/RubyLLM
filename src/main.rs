mod data;
mod tokenizer;
mod setup;

fn main() {
    println!("\nRUBY AI\n");

    setup::setup();

    let text = data::load_data();
    let text_reduced = data::reduce_spaces(&text);
    let sentences = data::split_sentences(&text_reduced);
    println!("The text has been split into {} sentences\n", sentences.len());

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

    let encoding = tokenizer.encode("Rust is amazing!", true).unwrap();
    println!("Test sentence: {:?} Result: {:?}", encoding.get_tokens(), encoding.get_ids());
}
