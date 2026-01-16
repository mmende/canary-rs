use canary_rs::{Canary, Result};

#[test]
fn test_transcribe_loading_audio() -> Result<()> {
    let model_path = "canary-1b-v2";
    let model = Canary::from_pretrained(model_path, None)?;
    let mut session = model.session();

    // Test with the loading.raw file
    let audio_path = "audio.wav";
    let result = session.transcribe_file(audio_path, "en", "en")?;

    println!("Transcription result: {}", result.text);
    println!("Tokens: {:#?}", result.tokens);

    // The audio should say something like "Multitask loaded successfully"
    // We'll just check it's not empty for now
    assert!(!result.text.is_empty(), "Transcription should not be empty");

    Ok(())
}

#[test]
#[ignore]
fn test_model_loads() -> Result<()> {
    let model_path = "canary-1b-v2";
    let _model = Canary::from_pretrained(model_path, None)?;
    println!("Model loaded successfully!");
    Ok(())
}
