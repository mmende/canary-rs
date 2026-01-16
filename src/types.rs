use thiserror::Error;

#[derive(Debug, Error)]
pub enum CanaryError {
    #[error("ONNX Runtime error: {0}")]
    OrtError(#[from] ort::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Audio error: {0}")]
    AudioError(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),
}

pub type Result<T> = std::result::Result<T, CanaryError>;

/// Token with timestamp information
#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub start: f32,
    pub end: f32,
    /// Token probability (0..1).
    pub prob: f32,
}

/// Transcription result
#[derive(Debug, Clone)]
pub struct CanaryResult {
    pub text: String,
    pub tokens: Vec<Token>,
}
