use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::{Result, TokenizerBuilder, AddedToken};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

pub fn tokenize() -> Result<()> {
    println!("Creating a tokenizer");
    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(48000)
        .min_frequency(0)
        .show_progress(true)
        .special_tokens(
            vec![
                AddedToken::from(String::from("[UNK]"), true),
                AddedToken::from(String::from("[PAD]"), true),
                AddedToken::from(String::from("[CLS]"), true),
                AddedToken::from(String::from("[SEP]"), true),
                AddedToken::from(String::from("[MASK]"), true)
            ]).build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;
    
    tokenizer.train_from_files(&mut trainer, vec!["data/data1.txt".to_string(), "data/data2.txt".to_string()]).unwrap();

    tokenizer.save("checkpoints/tokenizer.json", false).unwrap();

    let encoding = tokenizer.encode("Rust is amazing!", true).unwrap();
    println!("Test sentence: {:?}", encoding.get_tokens());

    Ok(())
}