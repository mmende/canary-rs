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
    /// Pad partial windows to the full window size when emitting partials.
    pub pad_partial: bool,
    /// Number of recent windows required to stabilize text.
    pub stability_window: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            window_duration: 10.0,
            step_duration: 2.0,
            emit_partial: false,
            pad_partial: true,
            stability_window: 0,
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

    /// Enable zero-padding for partial windows.
    ///
    /// When false, partial windows are decoded at their true length.
    pub fn with_pad_partial(mut self, pad_partial: bool) -> Self {
        self.pad_partial = pad_partial;
        self
    }

    /// Require this many recent windows to agree before emitting stabilized text.
    ///
    /// Values <= 1 disable stabilization.
    pub fn with_stability_window(mut self, stability_window: usize) -> Self {
        self.stability_window = stability_window;
        self
    }
}

/// Result of a streaming window decode.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Full transcription result for the window.
    pub result: CanaryResult,
    /// Text newly committed since the previous window.
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
    pad_partial: bool,
    stability_window: usize,
    buffer: VecDeque<f32>,
    last_text: String,
    recent_texts: VecDeque<String>,
    committed_text: String,
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
            pad_partial: config.pad_partial,
            stability_window: config.stability_window,
            buffer: VecDeque::new(),
            last_text: String::new(),
            recent_texts: VecDeque::new(),
            committed_text: String::new(),
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
            if self.pad_partial {
                window.resize(self.window_samples, 0.0);
            }
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
        self.recent_texts.clear();
        self.committed_text.clear();
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
                if self.pad_partial {
                    window.resize(self.window_samples, 0.0);
                }
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
        let window_text = result.text.trim();
        let delta_text = self.update_delta_text(window_text);
        Ok(StreamChunk {
            result,
            delta_text,
            window_start: window_start_sample as f32 / self.model_sample_rate as f32,
            window_end: window_end_sample as f32 / self.model_sample_rate as f32,
        })
    }

    fn update_delta_text(&mut self, window_text: &str) -> String {
        let window_text = window_text.trim();
        if window_text.is_empty() {
            self.last_text.clear();
            self.recent_texts.clear();
            return String::new();
        }
        let delta_text = if self.stability_window > 1 {
            self.recent_texts.push_back(window_text.to_string());
            while self.recent_texts.len() > self.stability_window {
                self.recent_texts.pop_front();
            }

            if self.recent_texts.len() < self.stability_window {
                String::new()
            } else if let Some(stable_text) = stable_prefix_text(&self.recent_texts) {
                append_stable_words(&mut self.committed_text, &stable_text)
            } else {
                String::new()
            }
        } else {
            delta_text_words(&self.last_text, window_text)
        };

        self.last_text = window_text.to_string();
        delta_text
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

fn delta_text_words(previous: &str, current: &str) -> String {
    let previous = previous.trim();
    let current = current.trim();
    if previous.is_empty() {
        return current.to_string();
    }
    if current.is_empty() {
        return String::new();
    }

    let prev_words: Vec<&str> = previous.split_whitespace().collect();
    let curr_words: Vec<&str> = current.split_whitespace().collect();
    if prev_words.is_empty() {
        return current.to_string();
    }
    if curr_words.is_empty() {
        return String::new();
    }

    let overlap = suffix_prefix_overlap(&prev_words, &curr_words);
    curr_words[overlap..].join(" ")
}

fn stable_prefix_text(texts: &VecDeque<String>) -> Option<String> {
    if texts.is_empty() {
        return None;
    }

    let mut word_lists: Vec<Vec<&str>> = Vec::with_capacity(texts.len());
    for text in texts {
        word_lists.push(text.split_whitespace().collect());
    }

    let min_len = word_lists
        .iter()
        .map(|words| words.len())
        .min()
        .unwrap_or(0);
    if min_len == 0 {
        return None;
    }

    let mut stable_words: Vec<&str> = Vec::new();
    'outer: for idx in 0..min_len {
        let word = word_lists[0][idx];
        for list in &word_lists[1..] {
            if list[idx] != word {
                break 'outer;
            }
        }
        stable_words.push(word);
    }

    if stable_words.is_empty() {
        None
    } else {
        Some(stable_words.join(" "))
    }
}

fn append_stable_words(committed: &mut String, stable_text: &str) -> String {
    let stable_words: Vec<&str> = stable_text.split_whitespace().collect();
    if stable_words.is_empty() {
        return String::new();
    }

    let committed_words: Vec<&str> = committed.split_whitespace().collect();
    if is_prefix_words(&committed_words, &stable_words) {
        return String::new();
    }

    let overlap = suffix_prefix_overlap(&committed_words, &stable_words);
    let new_words = &stable_words[overlap..];
    if new_words.is_empty() {
        return String::new();
    }

    if !committed.is_empty() {
        committed.push(' ');
    }
    let new_text = new_words.join(" ");
    committed.push_str(&new_text);
    new_text
}

fn suffix_prefix_overlap(a: &[&str], b: &[&str]) -> usize {
    let max_len = a.len().min(b.len());
    for len in (1..=max_len).rev() {
        if a[a.len() - len..] == b[..len] {
            return len;
        }
    }
    0
}

fn is_prefix_words(haystack: &[&str], needle: &[&str]) -> bool {
    if needle.len() > haystack.len() {
        return false;
    }
    haystack[..needle.len()] == needle[..]
}
