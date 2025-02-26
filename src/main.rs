mod data;

fn main() {
    println!("\nRUBY AI\n");

    let text = data::load_data();
    let text_reduced = data::reduce_spaces(&text);
    
    println!("{}", text_reduced);
}
