use canary_rs::{Canary, ExecutionConfig, ExecutionProvider};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading Canary model...");

    // Load the model with CPU execution provider
    let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cpu);

    let model = Canary::from_pretrained("canary-1b-v2", Some(config))?;
    let mut session = model.session();

    println!("Model loaded successfully!");

    // Transcribe the test audio file
    // The audio is in English, transcribe to English
    println!("Transcribing src/loading.raw (English)...");
    let result = session.transcribe_file("audio.wav", "en", "en")?;

    println!("\n=== Transcription Result ===");
    println!("Text: {}", result.text);

    println!("\n=== Token-level Timestamps ===");
    for token in result.tokens.iter().take(20) {
        // Show first 20 tokens
        println!("[{:.3}s - {:.3}s] {}", token.start, token.end, token.text);
    }

    if result.tokens.len() > 20 {
        println!("... and {} more tokens", result.tokens.len() - 20);
    }

    Ok(())
}
