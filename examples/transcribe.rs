#[path = "shared/utils.rs"]
mod utils;

use canary_rs::Canary;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading Canary model...");

    let config = utils::execution_config_from_env();

    let model_dir =
        std::env::var("CANARY_MODEL_DIR").unwrap_or_else(|_| "canary-180m-flash".to_string());
    let model = Canary::from_pretrained(&model_dir, Some(config))?;
    let mut session = model.session();

    println!("Model loaded successfully!");

    // Transcribe the test audio file
    // The audio is in English, transcribe to English
    println!("Transcribing src/audio.wav (English)...");
    let result = session.transcribe_file("audio.wav", "en", "en")?;

    println!("\n=== Transcription Result ===");
    println!("Text: {}", result.text);

    println!("\n=== Token-level Timestamps ===");
    for token in result.tokens.iter().take(20) {
        // Show first 20 tokens
        println!(
            "[{:.3}s - {:.3}s] {} (prob: {:.3})",
            token.start, token.end, token.text, token.prob
        );
    }

    if result.tokens.len() > 20 {
        println!("... and {} more tokens", result.tokens.len() - 20);
    }

    Ok(())
}
