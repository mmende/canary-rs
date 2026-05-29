use crate::audio;
use crate::model::Canary;
use crate::types::{
    BeamLogitsProcessor, CanaryError, CanaryResult, LogitsProcessor, Result, Token,
};
use ndarray::Array3;
use ort::session::{OutputSelector, RunOptions, Session, SessionOutputs};
use ort::value::{DynTensor, DynValue, Shape, Tensor, TensorElementType, ValueType};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;
use std::sync::MutexGuard;

/// Per-session state for running the model.
pub struct CanarySession {
    model: Canary,
    encoder_mask: Option<Vec<i64>>,
    encoder_mask_shape: Option<Vec<usize>>,
}

impl CanarySession {
    pub(crate) fn new(model: Canary) -> Self {
        Self {
            model,
            encoder_mask: None,
            encoder_mask_shape: None,
        }
    }

    fn lock_encoder(&self) -> Result<MutexGuard<'_, Session>> {
        self.model
            .encoder
            .lock()
            .map_err(|_| CanaryError::InferenceError("Encoder session mutex poisoned".into()))
    }

    fn lock_decoder(&self) -> Result<MutexGuard<'_, Session>> {
        self.model
            .decoder
            .lock()
            .map_err(|_| CanaryError::InferenceError("Decoder session mutex poisoned".into()))
    }

    /// Clear cached encoder outputs for a new utterance.
    pub fn reset(&mut self) {
        self.encoder_mask = None;
        self.encoder_mask_shape = None;
    }

    /// Transcribe an audio file
    ///
    /// # Arguments
    /// * `path` - Path to the audio file
    /// * `source_lang` - Source language code (e.g., "en", "de", "fr")
    /// * `target_lang` - Target language code for translation, or same as source for transcription
    pub fn transcribe_file<P: AsRef<Path>>(
        &mut self,
        path: P,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<CanaryResult> {
        let (audio, sample_rate, channels) = audio::load_audio_file(path)?;
        self.transcribe_samples(&audio, sample_rate, channels, source_lang, target_lang)
    }

    /// Transcribe in-memory audio samples
    ///
    /// # Arguments
    /// * `audio` - Audio samples as f32 slice
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    /// * `source_lang` - Source language code (e.g., "en", "de", "fr")
    /// * `target_lang` - Target language code for translation, or same as source for transcription
    pub fn transcribe_samples(
        &mut self,
        audio: &[f32],
        sample_rate: usize,
        channels: usize,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<CanaryResult> {
        self.reset();

        // Convert to mono if needed
        let mono_audio = if channels > 1 {
            audio::to_mono(audio, channels)
        } else {
            audio.to_vec()
        };

        // Resample if needed
        let resampled_audio = if sample_rate != self.model.sample_rate {
            audio::resample(&mono_audio, sample_rate, self.model.sample_rate)?
        } else {
            mono_audio
        };

        // Extract features
        let features = audio::extract_features(&resampled_audio, self.model.sample_rate)?;

        // Run encoder
        let encoded = self.run_encoder(&features)?;

        // Run decoder
        let token_results = self.run_decoder(&encoded, source_lang, target_lang, None)?;

        // Decode tokens to text
        let result = self.decode_tokens(&token_results)?;

        Ok(result)
    }

    /// Transcribe in-memory audio samples with a [`LogitsProcessor`] applied before each argmax.
    pub fn transcribe_samples_with_processor<P: LogitsProcessor>(
        &mut self,
        audio: &[f32],
        sample_rate: usize,
        channels: usize,
        source_lang: &str,
        target_lang: &str,
        processor: &mut P,
    ) -> Result<CanaryResult> {
        self.reset();
        let mono_audio = if channels > 1 {
            audio::to_mono(audio, channels)
        } else {
            audio.to_vec()
        };
        let resampled_audio = if sample_rate != self.model.sample_rate {
            audio::resample(&mono_audio, sample_rate, self.model.sample_rate)?
        } else {
            mono_audio
        };
        let features = audio::extract_features(&resampled_audio, self.model.sample_rate)?;
        let encoded = self.run_encoder(&features)?;
        let token_results =
            self.run_decoder(&encoded, source_lang, target_lang, Some(processor))?;
        self.decode_tokens(&token_results)
    }

    /// Transcribe an audio file with a [`LogitsProcessor`] applied before each argmax.
    pub fn transcribe_file_with_processor<P: LogitsProcessor, Q: AsRef<Path>>(
        &mut self,
        path: Q,
        source_lang: &str,
        target_lang: &str,
        processor: &mut P,
    ) -> Result<CanaryResult> {
        let (audio, sample_rate, channels) = audio::load_audio_file(path)?;
        self.transcribe_samples_with_processor(
            &audio,
            sample_rate,
            channels,
            source_lang,
            target_lang,
            processor,
        )
    }

    /// Transcribe an audio file using beam search with a [`BeamLogitsProcessor`].
    ///
    /// `beam_width` controls how many hypotheses are tracked simultaneously.
    /// Width 1 degenerates to greedy search.  Values of 3–5 give the best
    /// accuracy-vs-latency trade-off for grammar-constrained decoding.
    pub fn transcribe_file_with_beam_processor<P: BeamLogitsProcessor, Q: AsRef<Path>>(
        &mut self,
        path: Q,
        source_lang: &str,
        target_lang: &str,
        processor: &P,
        beam_width: usize,
    ) -> Result<CanaryResult> {
        let (audio, sample_rate, channels) = audio::load_audio_file(path)?;
        self.transcribe_samples_with_beam_processor(
            &audio,
            sample_rate,
            channels,
            source_lang,
            target_lang,
            processor,
            beam_width,
        )
    }

    /// Transcribe in-memory audio samples using beam search with a [`BeamLogitsProcessor`].
    pub fn transcribe_samples_with_beam_processor<P: BeamLogitsProcessor>(
        &mut self,
        audio: &[f32],
        sample_rate: usize,
        channels: usize,
        source_lang: &str,
        target_lang: &str,
        processor: &P,
        beam_width: usize,
    ) -> Result<CanaryResult> {
        self.reset();
        let mono = if channels > 1 {
            audio::to_mono(audio, channels)
        } else {
            audio.to_vec()
        };
        let resampled = if sample_rate != self.model.sample_rate {
            audio::resample(&mono, sample_rate, self.model.sample_rate)?
        } else {
            mono
        };
        let features = audio::extract_features(&resampled, self.model.sample_rate)?;
        let encoded = self.run_encoder(&features)?;
        let tokens =
            self.run_decoder_beam(&encoded, source_lang, target_lang, processor, beam_width)?;
        self.decode_tokens(&tokens)
    }

    fn run_encoder(&mut self, features: &Array3<f32>) -> Result<Array3<f32>> {
        let shape = features.shape();

        // Convert to format expected by Tensor::from_array
        let features_vec: Vec<f32> = features.iter().copied().collect();
        let features_shape = [shape[0], shape[1], shape[2]];
        let audio_tensor = Tensor::from_array((features_shape, features_vec))?;

        let length_vec = vec![shape[2] as i64];
        let length_tensor = Tensor::from_array(([1], length_vec))?;

        let (encoded_dims, encoded_data, mask_dims, mask_data) = {
            let mut encoder = self.lock_encoder()?;
            let outputs = encoder.run(ort::inputs![
                "audio_signal" => audio_tensor,
                "length" => length_tensor,
            ])?;
            let encoded_dynvalue = &outputs["encoder_embeddings"];
            let encoder_mask_dynvalue = &outputs["encoder_mask"];
            let (encoded_shape, encoded_data) = encoded_dynvalue.try_extract_tensor::<f32>()?;
            let (mask_shape, mask_data) = encoder_mask_dynvalue.try_extract_tensor::<i64>()?;
            let encoded_dims: Vec<usize> =
                encoded_shape.as_ref().iter().map(|&d| d as usize).collect();
            let mask_dims: Vec<usize> = mask_shape.as_ref().iter().map(|&d| d as usize).collect();
            (
                encoded_dims,
                encoded_data.to_vec(),
                mask_dims,
                mask_data.to_vec(),
            )
        };

        if encoded_dims.len() != 3 {
            return Err(CanaryError::InferenceError(format!(
                "Expected 3D tensor, got {}D",
                encoded_dims.len()
            )));
        }

        let encoded_array = Array3::from_shape_vec(
            (encoded_dims[0], encoded_dims[1], encoded_dims[2]),
            encoded_data,
        )
        .map_err(|e| CanaryError::InferenceError(format!("Shape error: {}", e)))?;

        self.encoder_mask = Some(mask_data);
        self.encoder_mask_shape = Some(mask_dims);

        Ok(encoded_array)
    }

    fn run_decoder(
        &mut self,
        encoded: &Array3<f32>,
        source_lang: &str,
        target_lang: &str,
        mut processor: Option<&mut dyn LogitsProcessor>,
    ) -> Result<Vec<(usize, f32)>> {
        let max_length = 512;
        let session_cfg = &self.model.config.session;
        let preallocate_outputs = session_cfg.preallocate_outputs;
        let preallocate_logits = session_cfg.preallocate_logits;
        let vocab_len = self.model.vocab.len();

        // Get special token IDs
        let bos_id = self
            .model
            .token_to_id
            .get("<|startoftranscript|>")
            .or_else(|| self.model.token_to_id.get("<s>"))
            .copied()
            .unwrap_or(0);

        let eos_id = self
            .model
            .token_to_id
            .get("<|endoftext|>")
            .or_else(|| self.model.token_to_id.get("</s>"))
            .copied()
            .unwrap_or(1);

        // Format language tokens for Canary
        // Format: <|startoftranscript|> <|source_lang|> <|target_lang|> <|pnc|>
        let source_lang_token = format!("<|{}|>", source_lang);
        let target_lang_token = format!("<|{}|>", target_lang);

        fn parse_prompt_override(
            token_to_id: &HashMap<String, usize>,
            vocab_len: usize,
            spec: &str,
        ) -> Vec<usize> {
            let mut out = Vec::new();
            for raw in spec.split(|c| c == ',' || c == ' ' || c == '\n' || c == '\t') {
                let token = raw.trim();
                if token.is_empty() {
                    continue;
                }
                if token.chars().all(|c| c.is_ascii_digit()) {
                    if let Ok(id) = token.parse::<usize>() {
                        if id < vocab_len {
                            out.push(id);
                        } else {
                            log::warn!("Prompt id {} out of range", id);
                        }
                    }
                } else if let Some(&id) = token_to_id.get(token) {
                    out.push(id);
                } else {
                    log::warn!("Prompt token '{}' not found in vocabulary", token);
                }
            }
            out
        }

        let prompt_override = session_cfg.prompt_override.as_deref().and_then(|spec| {
            if spec.trim().is_empty() {
                None
            } else {
                Some(spec)
            }
        });
        let default_prompt_tokens = {
            let mut tokens = vec![bos_id];

            if let Some(&id) = self.model.token_to_id.get(&source_lang_token) {
                tokens.push(id);
            } else {
                log::warn!(
                    "Source language token '{}' not found in vocabulary",
                    source_lang_token
                );
            }

            if let Some(&id) = self.model.token_to_id.get(&target_lang_token) {
                tokens.push(id);
            } else {
                log::warn!(
                    "Target language token '{}' not found in vocabulary",
                    target_lang_token
                );
            }

            let use_pnc = session_cfg.use_pnc;
            let pnc_token = if use_pnc { "<|pnc|>" } else { "<|nopnc|>" };
            if let Some(&id) = self.model.token_to_id.get(pnc_token) {
                tokens.push(id);
            }

            if let Some(use_itn) = session_cfg.use_itn {
                let itn_token = if use_itn { "<|itn|>" } else { "<|noitn|>" };
                if let Some(&id) = self.model.token_to_id.get(itn_token) {
                    tokens.push(id);
                }
            }

            let use_ts = session_cfg.use_timestamps;
            let ts_token = if use_ts {
                "<|timestamp|>"
            } else {
                "<|notimestamp|>"
            };
            if let Some(&id) = self.model.token_to_id.get(ts_token) {
                tokens.push(id);
            }

            let use_diarize = session_cfg.use_diarize;
            let diarize_token = if use_diarize {
                "<|diarize|>"
            } else {
                "<|nodiarize|>"
            };
            if let Some(&id) = self.model.token_to_id.get(diarize_token) {
                tokens.push(id);
            }

            tokens
        };
        let prompt_tokens = if let Some(spec) = prompt_override {
            let tokens = parse_prompt_override(self.model.token_to_id.as_ref(), vocab_len, spec);
            if tokens.is_empty() {
                log::warn!("Prompt override produced no valid tokens; using default prompt.");
                default_prompt_tokens.clone()
            } else {
                tokens
            }
        } else {
            default_prompt_tokens
        };
        if prompt_tokens.is_empty() {
            return Err(CanaryError::InferenceError(
                "Prompt tokens are empty".into(),
            ));
        }

        // Prepare encoder tensors once
        let encoded_shape = encoded.shape();
        let encoder_hidden = *encoded_shape.get(2).ok_or_else(|| {
            CanaryError::InferenceError("Encoded tensor missing hidden dimension".into())
        })?;
        let encoded_vec: Vec<f32> = encoded.iter().copied().collect();
        let encoded_tensor = Tensor::<f32>::from_array((
            [encoded_shape[0], encoded_shape[1], encoded_shape[2]],
            encoded_vec,
        ))?;

        // Use cached encoder_mask from encoder output if available
        let mask_tensor = if let (Some(mask_data), Some(mask_shape)) =
            (&self.encoder_mask, &self.encoder_mask_shape)
        {
            Tensor::from_array((mask_shape.as_slice(), mask_data.clone()))?
        } else {
            // Fallback: create a mask of all ones
            let mask_vec: Vec<i64> = vec![1; encoded_shape[0] * 1 * encoded_shape[2]];
            Tensor::from_array(([encoded_shape[0], 1, encoded_shape[2]], mask_vec))?
        };

        // Autoregressive decoding.
        //
        // WORKAROUND for zero-length tensor limitation:
        // Create a zero-length cache for the first call, then use the returned
        // cache for incremental decoding (or run logits-only mode for debugging).
        fn infer_decoder_dims_from_output(decoder: &Session) -> Option<(usize, usize)> {
            let output = decoder
                .outputs()
                .iter()
                .find(|out| out.name() == "decoder_hidden_states")?;
            let ValueType::Tensor { shape, .. } = output.dtype() else {
                return None;
            };
            if shape.len() != 4 {
                return None;
            }
            let num_layers = shape[0];
            let hidden = shape[3];
            if num_layers > 0 && hidden > 0 {
                Some((num_layers as usize, hidden as usize))
            } else {
                None
            }
        }

        fn extract_last_logits(outputs: &SessionOutputs<'_>) -> Result<Vec<f32>> {
            let logits_dynvalue = &outputs["logits"];
            let (logits_shape, logits_data) = logits_dynvalue.try_extract_tensor::<f32>()?;
            let dims = logits_shape.as_ref();

            if dims.len() != 3 {
                return Err(CanaryError::InferenceError(format!(
                    "Expected logits [B, T, V], got {:?}",
                    dims
                )));
            }

            let t = dims[1] as usize;
            let v = dims[2] as usize;
            if t == 0 {
                return Err(CanaryError::InferenceError(
                    "Logits sequence length is zero".into(),
                ));
            }

            let last_logits_start = (t - 1) * v;
            Ok(logits_data[last_logits_start..last_logits_start + v].to_vec())
        }

        fn extract_decoder_mems(outputs: &mut SessionOutputs<'_>) -> Result<DynValue> {
            outputs.remove("decoder_hidden_states").ok_or_else(|| {
                CanaryError::InferenceError("Missing decoder_hidden_states output".into())
            })
        }

        fn empty_mems_tensor(
            allocator: &ort::memory::Allocator,
            num_layers: usize,
            hidden: usize,
        ) -> Result<DynTensor> {
            let empty_mems_shape = Shape::new([num_layers as i64, 1, 0, hidden as i64]);
            Ok(DynTensor::new(
                allocator,
                TensorElementType::Float32,
                empty_mems_shape,
            )?)
        }

        fn cmp_logits(a: &f32, b: &f32) -> Ordering {
            let a = if a.is_nan() { f32::NEG_INFINITY } else { *a };
            let b = if b.is_nan() { f32::NEG_INFINITY } else { *b };
            a.total_cmp(&b)
        }

        let mut generated: Vec<(usize, f32)> = Vec::new();

        let mut cache_len: usize = 0;
        let mut decoder = self.lock_decoder()?;
        let (num_layers, hidden) = {
            let mut num_layers = session_cfg.decoder_num_layers;
            let mut hidden = session_cfg.decoder_hidden_size;

            if num_layers.is_none() || hidden.is_none() {
                if let Some((out_layers, out_hidden)) = infer_decoder_dims_from_output(&decoder) {
                    num_layers.get_or_insert(out_layers);
                    hidden.get_or_insert(out_hidden);
                }
            }

            if hidden.is_none() {
                hidden = Some(encoder_hidden);
            }

            let num_layers = match num_layers {
                Some(value) => value,
                None => {
                    log::warn!(
                        "Decoder layer count is unknown; defaulting to 10. Set SessionConfig::decoder_num_layers if incorrect."
                    );
                    10
                }
            };
            let hidden = hidden.unwrap_or(encoder_hidden);
            (num_layers, hidden)
        };

        // Step 1: First call with empty cache and full prompt
        log::debug!("Running decoder with empty cache.");
        let prompt_ids_i64: Vec<i64> = prompt_tokens.iter().map(|&x| x as i64).collect();
        let prompt_tokens_tensor =
            Tensor::<i64>::from_array(([1, prompt_ids_i64.len()], prompt_ids_i64))?;

        let (mut last_logits, mut decoder_mems) = {
            let input_len = prompt_tokens.len();
            let mems_len = cache_len;
            let (last_logits, decoder_mems) = if preallocate_outputs {
                let mems_shape = Shape::new([
                    num_layers as i64,
                    1,
                    (mems_len + input_len) as i64,
                    hidden as i64,
                ]);
                let mems_prealloc =
                    DynTensor::new(decoder.allocator(), TensorElementType::Float32, mems_shape)?;
                let mut output_selector =
                    OutputSelector::default().preallocate("decoder_hidden_states", mems_prealloc);
                if preallocate_logits {
                    let logits_shape = Shape::new([1, input_len as i64, vocab_len as i64]);
                    let logits_prealloc = DynTensor::new(
                        decoder.allocator(),
                        TensorElementType::Float32,
                        logits_shape,
                    )?;
                    output_selector = output_selector.preallocate("logits", logits_prealloc);
                }
                let run_options = RunOptions::new()?.with_outputs(output_selector);
                let empty_mems = empty_mems_tensor(decoder.allocator(), num_layers, hidden)?;
                let mut outputs = decoder.run_with_options(
                    ort::inputs![
                        "input_ids" => prompt_tokens_tensor,
                        "encoder_embeddings" => &encoded_tensor,
                        "encoder_mask" => &mask_tensor,
                        "decoder_mems" => empty_mems,
                    ],
                    &run_options,
                )?;
                let last_logits = extract_last_logits(&outputs)?;
                let decoder_mems = extract_decoder_mems(&mut outputs)?;
                (last_logits, decoder_mems)
            } else {
                let empty_mems = empty_mems_tensor(decoder.allocator(), num_layers, hidden)?;
                let mut outputs = decoder.run(ort::inputs![
                    "input_ids" => prompt_tokens_tensor,
                    "encoder_embeddings" => &encoded_tensor,
                    "encoder_mask" => &mask_tensor,
                    "decoder_mems" => empty_mems,
                ])?;
                let last_logits = extract_last_logits(&outputs)?;
                let decoder_mems = extract_decoder_mems(&mut outputs)?;
                (last_logits, decoder_mems)
            };

            cache_len += input_len;
            (last_logits, Some(decoder_mems))
        };

        // Step 3: Autoregressive generation (incremental, one token at a time)
        let mut tokens_so_far: Vec<usize> = Vec::new();
        for _step in 0..max_length {
            if let Some(ref mut p) = processor {
                p.process(&mut last_logits, &tokens_so_far);
            }

            let (max_idx, max_val) = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| cmp_logits(a, b))
                .unwrap_or((eos_id, &f32::NEG_INFINITY));

            let next_token = max_idx;

            if next_token == eos_id {
                break;
            }

            let prob = {
                let max_logit = *max_val;
                let sum_exp: f32 = last_logits.iter().map(|&x| (x - max_logit).exp()).sum();
                1.0 / sum_exp
            };

            generated.push((next_token, prob));
            tokens_so_far.push(next_token);

            let input_ids_i64 = vec![next_token as i64];
            let tokens_tensor = Tensor::<i64>::from_array(([1, 1], input_ids_i64))?;

            let mems_to_use = decoder_mems.take().ok_or_else(|| {
                CanaryError::InferenceError("Missing decoder cache during generation".into())
            })?;

            let input_len = 1;
            let mems_len = cache_len;
            let (new_logits, new_mems) = if preallocate_outputs {
                let mems_shape = Shape::new([
                    num_layers as i64,
                    1,
                    (mems_len + input_len) as i64,
                    hidden as i64,
                ]);
                let mems_prealloc =
                    DynTensor::new(decoder.allocator(), TensorElementType::Float32, mems_shape)?;
                let mut output_selector =
                    OutputSelector::default().preallocate("decoder_hidden_states", mems_prealloc);
                if preallocate_logits {
                    let logits_shape = Shape::new([1, input_len as i64, vocab_len as i64]);
                    let logits_prealloc = DynTensor::new(
                        decoder.allocator(),
                        TensorElementType::Float32,
                        logits_shape,
                    )?;
                    output_selector = output_selector.preallocate("logits", logits_prealloc);
                }
                let run_options = RunOptions::new()?.with_outputs(output_selector);
                let mut outputs = decoder.run_with_options(
                    ort::inputs![
                        "input_ids" => tokens_tensor,
                        "encoder_embeddings" => &encoded_tensor,
                        "encoder_mask" => &mask_tensor,
                        "decoder_mems" => mems_to_use,
                    ],
                    &run_options,
                )?;
                let last_logits = extract_last_logits(&outputs)?;
                let decoder_mems = extract_decoder_mems(&mut outputs)?;
                (last_logits, decoder_mems)
            } else {
                let mut outputs = decoder.run(ort::inputs![
                    "input_ids" => tokens_tensor,
                    "encoder_embeddings" => &encoded_tensor,
                    "encoder_mask" => &mask_tensor,
                    "decoder_mems" => mems_to_use,
                ])?;
                let last_logits = extract_last_logits(&outputs)?;
                let decoder_mems = extract_decoder_mems(&mut outputs)?;
                (last_logits, decoder_mems)
            };

            last_logits = new_logits;
            decoder_mems = Some(new_mems);
            cache_len += 1;
        }

        Ok(generated)
    }

    fn run_decoder_beam(
        &mut self,
        encoded: &Array3<f32>,
        source_lang: &str,
        target_lang: &str,
        initial_processor: &dyn BeamLogitsProcessor,
        beam_width: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let beam_width = beam_width.max(1);
        let max_length = 512_usize;
        let session_cfg = &self.model.config.session;
        let _vocab_len = self.model.vocab.len();

        let bos_id = self
            .model
            .token_to_id
            .get("<|startoftranscript|>")
            .or_else(|| self.model.token_to_id.get("<s>"))
            .copied()
            .unwrap_or(0);
        let eos_id = self
            .model
            .token_to_id
            .get("<|endoftext|>")
            .or_else(|| self.model.token_to_id.get("</s>"))
            .copied()
            .unwrap_or(1);

        // Build prompt tokens (same logic as run_decoder).
        let source_lang_token = format!("<|{}|>", source_lang);
        let target_lang_token = format!("<|{}|>", target_lang);
        let mut prompt_tokens = vec![bos_id];
        if let Some(&id) = self.model.token_to_id.get(&source_lang_token) {
            prompt_tokens.push(id);
        } else {
            log::warn!(
                "Source language token '{}' not found in vocabulary",
                source_lang_token
            );
        }
        if let Some(&id) = self.model.token_to_id.get(&target_lang_token) {
            prompt_tokens.push(id);
        } else {
            log::warn!(
                "Target language token '{}' not found in vocabulary",
                target_lang_token
            );
        }
        let pnc_token = if session_cfg.use_pnc {
            "<|pnc|>"
        } else {
            "<|nopnc|>"
        };
        if let Some(&id) = self.model.token_to_id.get(pnc_token) {
            prompt_tokens.push(id);
        }
        if let Some(use_itn) = session_cfg.use_itn {
            let itn_token = if use_itn { "<|itn|>" } else { "<|noitn|>" };
            if let Some(&id) = self.model.token_to_id.get(itn_token) {
                prompt_tokens.push(id);
            }
        }
        let ts_token = if session_cfg.use_timestamps {
            "<|timestamp|>"
        } else {
            "<|notimestamp|>"
        };
        if let Some(&id) = self.model.token_to_id.get(ts_token) {
            prompt_tokens.push(id);
        }
        let diarize_token = if session_cfg.use_diarize {
            "<|diarize|>"
        } else {
            "<|nodiarize|>"
        };
        if let Some(&id) = self.model.token_to_id.get(diarize_token) {
            prompt_tokens.push(id);
        }
        if prompt_tokens.is_empty() {
            return Err(CanaryError::InferenceError(
                "Prompt tokens are empty".into(),
            ));
        }

        // Encoder tensors (shared by all beams).
        let encoded_shape = encoded.shape();
        let encoder_hidden = *encoded_shape.get(2).ok_or_else(|| {
            CanaryError::InferenceError("Encoded tensor missing hidden dimension".into())
        })?;
        let encoded_vec: Vec<f32> = encoded.iter().copied().collect();
        let encoded_tensor = Tensor::<f32>::from_array((
            [encoded_shape[0], encoded_shape[1], encoded_shape[2]],
            encoded_vec,
        ))?;
        let mask_tensor = if let (Some(mask_data), Some(mask_shape)) =
            (&self.encoder_mask, &self.encoder_mask_shape)
        {
            Tensor::<i64>::from_array((mask_shape.as_slice(), mask_data.clone()))?
        } else {
            let mask_vec: Vec<i64> = vec![1; encoded_shape[0] * 1 * encoded_shape[2]];
            Tensor::<i64>::from_array(([encoded_shape[0], 1, encoded_shape[2]], mask_vec))?
        };

        // Decoder dims.
        let mut decoder = self.lock_decoder()?;
        let (num_layers, hidden) = {
            let mut num_layers = session_cfg.decoder_num_layers;
            let mut hidden_opt = session_cfg.decoder_hidden_size;
            if num_layers.is_none() || hidden_opt.is_none() {
                if let Some(out) = decoder
                    .outputs()
                    .iter()
                    .find(|o| o.name() == "decoder_hidden_states")
                {
                    if let ValueType::Tensor { shape, .. } = out.dtype() {
                        if shape.len() == 4 && shape[0] > 0 && shape[3] > 0 {
                            num_layers.get_or_insert(shape[0] as usize);
                            hidden_opt.get_or_insert(shape[3] as usize);
                        }
                    }
                }
            }
            if hidden_opt.is_none() {
                hidden_opt = Some(encoder_hidden);
            }
            (
                num_layers.unwrap_or(10),
                hidden_opt.unwrap_or(encoder_hidden),
            )
        };

        // Helpers (plain fns - cannot capture locals).
        fn last_logits(outputs: &SessionOutputs<'_>) -> Result<Vec<f32>> {
            let (shape, data) = outputs["logits"].try_extract_tensor::<f32>()?;
            let dims = shape.as_ref();
            if dims.len() != 3 {
                return Err(CanaryError::InferenceError("Expected 3D logits".into()));
            }
            let t = dims[1] as usize;
            let v = dims[2] as usize;
            if t == 0 {
                return Err(CanaryError::InferenceError("Empty logits".into()));
            }
            Ok(data[(t - 1) * v..t * v].to_vec())
        }

        fn pull_cache(outputs: &mut SessionOutputs<'_>) -> Result<(Vec<f32>, [usize; 4])> {
            let val = outputs.remove("decoder_hidden_states").ok_or_else(|| {
                CanaryError::InferenceError("Missing decoder_hidden_states".into())
            })?;
            let (shape, data) = val.try_extract_tensor::<f32>()?;
            let s = shape.as_ref();
            if s.len() != 4 {
                return Err(CanaryError::InferenceError(format!(
                    "Cache expected 4D, got {}D",
                    s.len()
                )));
            }
            Ok((
                data.to_vec(),
                [s[0] as usize, s[1] as usize, s[2] as usize, s[3] as usize],
            ))
        }

        // Prompt step - shared initial state for all beams.
        let prompt_ids: Vec<i64> = prompt_tokens.iter().map(|&x| x as i64).collect();
        let prompt_tensor = Tensor::<i64>::from_array(([1, prompt_ids.len()], prompt_ids))?;
        let empty_mems = {
            let shape = Shape::new([num_layers as i64, 1, 0, hidden as i64]);
            DynTensor::new(decoder.allocator(), TensorElementType::Float32, shape)?
        };
        let (init_logits, init_cache_data, init_cache_shape) = {
            let mut outputs = decoder.run(ort::inputs![
                "input_ids" => prompt_tensor,
                "encoder_embeddings" => &encoded_tensor,
                "encoder_mask" => &mask_tensor,
                "decoder_mems" => empty_mems,
            ])?;
            let logits = last_logits(&outputs)?;
            let (cache_data, cache_shape) = pull_cache(&mut outputs)?;
            (logits, cache_data, cache_shape)
        };

        // Per-beam state.
        struct Beam {
            tokens: Vec<(usize, f32)>,
            tokens_so_far: Vec<usize>,
            log_prob: f64,
            cache_data: Vec<f32>,
            cache_shape: [usize; 4],
            logits: Vec<f32>,
            processor: Box<dyn BeamLogitsProcessor>,
        }

        let mut beams: Vec<Beam> = (0..beam_width)
            .map(|_| Beam {
                tokens: Vec::new(),
                tokens_so_far: Vec::new(),
                log_prob: 0.0,
                cache_data: init_cache_data.clone(),
                cache_shape: init_cache_shape,
                logits: init_logits.clone(),
                processor: initial_processor.clone_beam_state(),
            })
            .collect();

        let mut finished: Vec<(Vec<(usize, f32)>, f64)> = Vec::new();

        for _step in 0..max_length {
            // Apply each beam's processor to its logits, collect scored candidates.
            let mut candidates: Vec<(usize, usize, f64)> = Vec::new(); // (beam_idx, token, score)

            for (bi, beam) in beams.iter_mut().enumerate() {
                let mut masked = beam.logits.clone();
                beam.processor.process(&mut masked, &beam.tokens_so_far);

                let max_l = masked
                    .iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);
                if max_l.is_infinite() {
                    continue;
                }
                let sum_exp: f64 = masked
                    .iter()
                    .map(|&x| {
                        if x.is_finite() {
                            ((x - max_l) as f64).exp()
                        } else {
                            0.0
                        }
                    })
                    .sum();
                let log_denom = sum_exp.ln();

                for (token, &logit) in masked.iter().enumerate() {
                    if !logit.is_finite() {
                        continue;
                    }
                    let log_p = (logit - max_l) as f64 - log_denom;
                    candidates.push((bi, token, beam.log_prob + log_p));
                }
            }

            if candidates.is_empty() {
                break;
            }

            // Keep the top `beam_width` candidates globally.
            candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
            candidates.truncate(beam_width);

            // Expand each surviving candidate into a new beam.
            let mut new_beams: Vec<Beam> = Vec::new();
            for (parent_idx, token, score) in candidates {
                let parent = &beams[parent_idx];

                if token == eos_id {
                    finished.push((parent.tokens.clone(), score));
                    continue;
                }

                // Clone processor state (already advanced for tokens_so_far.last()).
                let new_proc = parent.processor.clone_beam_state();

                // Token probability from the parent's raw logits.
                let raw = &parent.logits;
                let max_raw = raw
                    .iter()
                    .copied()
                    .filter(|x| x.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);
                let sum_raw: f32 = raw
                    .iter()
                    .map(|&x| {
                        if x.is_finite() {
                            (x - max_raw).exp()
                        } else {
                            0.0
                        }
                    })
                    .sum();
                let prob = if token < raw.len() && raw[token].is_finite() {
                    (raw[token] - max_raw).exp() / sum_raw
                } else {
                    0.0
                };

                // Run one decoder step.
                let token_tensor = Tensor::<i64>::from_array(([1, 1], vec![token as i64]))?;
                let cache_tensor =
                    Tensor::<f32>::from_array((parent.cache_shape, parent.cache_data.clone()))?;
                let mut step_out = decoder.run(ort::inputs![
                    "input_ids" => token_tensor,
                    "encoder_embeddings" => &encoded_tensor,
                    "encoder_mask" => &mask_tensor,
                    "decoder_mems" => cache_tensor,
                ])?;
                let new_logits = last_logits(&step_out)?;
                let (new_cache_data, new_cache_shape) = pull_cache(&mut step_out)?;

                let mut new_tokens = parent.tokens.clone();
                new_tokens.push((token, prob));
                let mut new_tokens_so_far = parent.tokens_so_far.clone();
                new_tokens_so_far.push(token);

                new_beams.push(Beam {
                    tokens: new_tokens,
                    tokens_so_far: new_tokens_so_far,
                    log_prob: score,
                    cache_data: new_cache_data,
                    cache_shape: new_cache_shape,
                    logits: new_logits,
                    processor: new_proc,
                });
            }

            if new_beams.is_empty() {
                break;
            }
            beams = new_beams;
        }

        // Collect any active beams that reached max_length without EOS.
        for beam in beams {
            finished.push((beam.tokens, beam.log_prob));
        }

        if finished.is_empty() {
            return Ok(Vec::new());
        }

        // Length-normalised ranking (Google NMT penalty, α = 0.7).
        finished.sort_by(|a, b| {
            let lp = |n: usize| ((5.0_f64 + n as f64) / 6.0).powf(0.7);
            let sa = a.1 / lp(a.0.len());
            let sb = b.1 / lp(b.0.len());
            sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
        });

        Ok(finished.remove(0).0)
    }

    fn decode_tokens(&self, token_results: &[(usize, f32)]) -> Result<CanaryResult> {
        let mut text = String::new();
        let mut result_tokens = Vec::new();
        let spm_space = "\u{2581}";
        let use_sentencepiece = self.model.vocab.iter().any(|tok| tok.contains(spm_space));

        for (i, &(token_id, prob)) in token_results.iter().enumerate() {
            if token_id >= self.model.vocab.len() {
                continue;
            }

            let token_text = &self.model.vocab[token_id];

            // Skip special tokens
            if token_text.starts_with("<|")
                || token_text.starts_with("</")
                || token_text == "<unk>"
                || token_text == "<pad>"
            {
                continue;
            }

            // Simple detokenization (SentencePiece preferred, fallback to BPE "##")
            if token_text.is_empty() {
                continue;
            }

            let (cleaned, is_new_word) = if use_sentencepiece {
                let cleaned = token_text.replace(spm_space, " ");
                let is_new_word = cleaned.starts_with(' ');
                (cleaned, is_new_word)
            } else {
                let cleaned = token_text.trim_start_matches("##").to_string();
                let is_new_word = !token_text.starts_with("##");
                (cleaned, is_new_word)
            };

            if cleaned.is_empty() {
                continue;
            }

            if use_sentencepiece {
                text.push_str(&cleaned);
            } else {
                if !text.is_empty() && is_new_word {
                    text.push(' ');
                }
                text.push_str(&cleaned);
            }

            // Create token with dummy timestamps
            result_tokens.push(Token {
                text: cleaned.trim_start().to_string(),
                start: i as f32 * 0.02,
                end: (i + 1) as f32 * 0.02,
                prob,
            });
        }

        Ok(CanaryResult {
            text: text.trim().to_string(),
            tokens: result_tokens,
        })
    }
}
