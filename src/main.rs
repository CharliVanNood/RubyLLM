mod data;
mod tokenizer;

fn main() {
    println!("\nRUBY AI\n");

    let text = data::load_data();
    let text_reduced = data::reduce_spaces(&text);
    let sentences = data::split_sentences(&text_reduced);
    println!("The text has been split into {} sentences", sentences.len());

    let _ = tokenizer::tokenize();
}
