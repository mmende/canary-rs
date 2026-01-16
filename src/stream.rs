use crate::audio;
use crate::model::Canary;
use crate::session::CanarySession;
use crate::types::{CanaryError, CanaryResult, Result};
use std::collections::VecDeque;

/// Configuration for chunked streaming transcription.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Duration of each decoding window in seconds.
    pub window_duration: f32,
    /// Step size between windows in seconds.
    pub step_duration: f32,
    /// Emit partial windows by zero-padding to the full window size.
    pub emit_partial: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            window_duration: 10.0,
            step_duration: 2.0,
            emit_partial: false,
        }
    }
}

impl StreamConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the decoding window size in seconds.
    ///
    /// Larger windows provide more context (often better accuracy) but increase latency and
    /// per-decode compute cost.
    pub fn with_window_duration(mut self, seconds: f32) -> Self {
        self.window_duration = seconds;
        self
    }

    /// Set how far the window advances between decodes in seconds.
    ///
    /// Smaller steps yield more frequent updates but also more overlap and compute.
    pub fn with_step_duration(mut self, seconds: f32) -> Self {
        self.step_duration = seconds;
        self
    }

    /// Enable emitting partial windows by zero-padding to the full window size.
    ///
    /// When false, decoding starts only after a full window of audio is available.
    pub fn with_emit_partial(mut self, emit_partial: bool) -> Self {
        self.emit_partial = emit_partial;
        self
    }
}

/// Result of a streaming window decode.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Full transcription result for the window.
    pub result: CanaryResult,
    /// Text that is new relative to the previous window.
    pub delta_text: String,
    /// Window start time in seconds.
    pub window_start: f32,
    /// Window end time in seconds.
    pub window_end: f32,
}

/// Windowed streaming transcription helper.
pub struct CanaryStream {
    session: CanarySession,
    source_lang: String,
    target_lang: String,
    window_samples: usize,
    step_samples: usize,
    model_sample_rate: usize,
    total_samples: usize,
    pending_samples: usize,
    emit_partial: bool,
    buffer: VecDeque<f32>,
    last_text: String,
}

impl CanaryStream {
    pub(crate) fn new(
        model: Canary,
        source_lang: String,
        target_lang: String,
        config: StreamConfig,
    ) -> Result<Self> {
        let model_sample_rate = model.sample_rate;
        let window_samples = seconds_to_samples(config.window_duration, model_sample_rate)?;
        let step_samples = seconds_to_samples(config.step_duration, model_sample_rate)?;
        if step_samples > window_samples {
            return Err(CanaryError::ModelError(
                "Stream step must be <= window duration".into(),
            ));
        }

        Ok(Self {
            session: model.session(),
            source_lang,
            target_lang,
            window_samples,
            step_samples,
            model_sample_rate,
            total_samples: 0,
            pending_samples: 0,
            emit_partial: config.emit_partial,
            buffer: VecDeque::new(),
            last_text: String::new(),
        })
    }

    /// Push new audio samples and return any available window results.
    pub fn push_samples(
        &mut self,
        audio: &[f32],
        sample_rate: usize,
        channels: usize,
    ) -> Result<Vec<StreamChunk>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let mono_audio = if channels > 1 {
            audio::to_mono(audio, channels)
        } else {
            audio.to_vec()
        };
        let resampled_audio = if sample_rate != self.model_sample_rate {
            audio::resample(&mono_audio, sample_rate, self.model_sample_rate)?
        } else {
            mono_audio
        };

        self.append_samples(&resampled_audio)
    }

    /// Flush any remaining audio by decoding the last window (zero-padded if needed).
    pub fn flush(&mut self) -> Result<Vec<StreamChunk>> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        let mut window: Vec<f32> = self.buffer.iter().copied().collect();
        let window_end = self.total_samples;
        let window_start = window_end.saturating_sub(window.len());
        if window.len() < self.window_samples {
            if !self.emit_partial {
                self.reset();
                return Ok(Vec::new());
            }
            window.resize(self.window_samples, 0.0);
        } else if window.len() > self.window_samples {
            window = window[window.len() - self.window_samples..].to_vec();
        }

        let chunk = self.decode_window(&window, window_start, window_end)?;
        self.reset();
        Ok(vec![chunk])
    }

    /// Clear buffered audio and cached state.
    pub fn reset(&mut self) {
        self.session.reset();
        self.buffer.clear();
        self.total_samples = 0;
        self.pending_samples = 0;
        self.last_text.clear();
    }

    fn append_samples(&mut self, audio: &[f32]) -> Result<Vec<StreamChunk>> {
        let mut chunks = Vec::new();
        for &sample in audio {
            self.buffer.push_back(sample);
            if self.buffer.len() > self.window_samples {
                self.buffer.pop_front();
            }
            self.total_samples += 1;
            self.pending_samples += 1;

            if self.pending_samples < self.step_samples {
                continue;
            }

            if !self.emit_partial && self.buffer.len() < self.window_samples {
                self.pending_samples = 0;
                continue;
            }

            let mut window: Vec<f32> = self.buffer.iter().copied().collect();
            let window_end = self.total_samples;
            let window_start = window_end.saturating_sub(window.len());

            if window.len() < self.window_samples {
                window.resize(self.window_samples, 0.0);
            }

            chunks.push(self.decode_window(&window, window_start, window_end)?);
            self.pending_samples = self.pending_samples.saturating_sub(self.step_samples);
        }

        Ok(chunks)
    }

    fn decode_window(
        &mut self,
        window: &[f32],
        window_start_sample: usize,
        window_end_sample: usize,
    ) -> Result<StreamChunk> {
        let result = self.session.transcribe_samples(
            window,
            self.model_sample_rate,
            1,
            &self.source_lang,
            &self.target_lang,
        )?;
        let delta_text = delta_text(&self.last_text, &result.text);
        self.last_text = result.text.clone();
        Ok(StreamChunk {
            result,
            delta_text,
            window_start: window_start_sample as f32 / self.model_sample_rate as f32,
            window_end: window_end_sample as f32 / self.model_sample_rate as f32,
        })
    }
}

fn seconds_to_samples(seconds: f32, sample_rate: usize) -> Result<usize> {
    if !(seconds > 0.0) {
        return Err(CanaryError::ModelError(
            "Stream duration must be positive".into(),
        ));
    }
    Ok((seconds * sample_rate as f32).round().max(1.0) as usize)
}

fn delta_text(previous: &str, current: &str) -> String {
    if previous.is_empty() {
        return current.trim().to_string();
    }

    let mut prefix_len = 0;
    for (a, b) in previous.chars().zip(current.chars()) {
        if a == b {
            prefix_len += a.len_utf8();
        } else {
            break;
        }
    }

    current[prefix_len..].trim_start().to_string()
}
