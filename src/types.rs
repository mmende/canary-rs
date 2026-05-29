use thiserror::Error;

#[derive(Debug, Error)]
pub enum CanaryError {
    #[error("ONNX Runtime error: {0}")]
    OrtError(String),

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

impl<R> From<ort::Error<R>> for CanaryError {
    fn from(error: ort::Error<R>) -> Self {
        Self::OrtError(error.to_string())
    }
}

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

/// Hook into the autoregressive decoding loop to modify logits before argmax.
///
/// Implement this to apply grammar constraints, language steering, or any other
/// token-level masking. Called once per decoding step, before the argmax selection.
pub trait LogitsProcessor: Send {
    /// Modify `logits` in place (e.g. set disallowed token positions to `f32::NEG_INFINITY`).
    /// `tokens_so_far` contains every token ID generated so far in this sequence.
    fn process(&mut self, logits: &mut [f32], tokens_so_far: &[usize]);
}

/// Extension of [`LogitsProcessor`] that can be cloned for beam-search decoding.
///
/// Each active beam in a beam search needs its own independent copy of the
/// processor state.  Implement this trait (in addition to `LogitsProcessor`) to
/// enable `transcribe_file_with_beam_processor`.
pub trait BeamLogitsProcessor: LogitsProcessor {
    /// Return a heap-allocated clone of this processor in its current internal state.
    fn clone_beam_state(&self) -> Box<dyn BeamLogitsProcessor>;
}
