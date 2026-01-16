use canary_rs::{Canary, ExecutionConfig, ExecutionProvider, StreamConfig};
use cpal::Sample;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

// Live streaming example:
// - Captures microphone audio via cpal.
// - Runs a rolling window decode for low-latency updates.
// - Tracks a simple noise floor to avoid decoding silence.
// - On detected utterance end, re-decodes the full utterance for a final result.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use en -> en or the first and second command line arguments as source and target language codes
    let source_lang = match std::env::args().nth(1) {
        Some(lang) => lang,
        None => "en".to_string(),
    };
    let target_lang = match std::env::args().nth(2) {
        Some(lang) => lang,
        None => source_lang.clone(),
    };
    println!(
        "Source language: {}, Target language: {}",
        source_lang, target_lang
    );

    println!("Loading Canary model...");

    let config = ExecutionConfig::new().with_execution_provider(ExecutionProvider::Cpu);
    let model = Canary::from_pretrained("canary-180m-flash", Some(config))?;
    let stream_cfg = StreamConfig::new()
        .with_window_duration(8.0)
        .with_step_duration(0.5)
        .with_emit_partial(true);
    let mut stream_state =
        model.stream(source_lang.clone(), target_lang.clone(), stream_cfg.clone())?;
    let mut full_session = model.session();

    println!("Model loaded successfully!");

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;
    let supported_config = device.default_input_config()?;
    let sample_rate = supported_config.sample_rate() as usize;
    let channels = supported_config.channels() as usize;
    let cpal_config: cpal::StreamConfig = supported_config.clone().into();

    // Feed chunks at roughly the streaming step size to avoid multi-second output bursts.
    let chunk_seconds = stream_cfg.step_duration.max(0.05);
    let chunk_samples = ((sample_rate as f32 * chunk_seconds).round() as usize).max(1) * channels;
    let min_silence_threshold = 0.0008_f32;
    let max_silence_threshold = 0.02_f32;
    let silence_hold_seconds = 0.8_f32;
    let silence_hold_samples =
        ((sample_rate as f32 * silence_hold_seconds).round() as usize).max(1) * channels;
    let min_utterance_seconds = 0.3_f32;
    let min_utterance_samples =
        ((sample_rate as f32 * min_utterance_seconds).round() as usize).max(1) * channels;

    let buffer = Arc::new(Mutex::new(Vec::<f32>::with_capacity(chunk_samples * 2)));
    let (tx, rx) = mpsc::channel::<Vec<f32>>();

    let err_fn = |err| eprintln!("stream error: {}", err);

    let buffer_clone = Arc::clone(&buffer);
    let tx_clone = tx.clone();
    let stream = match supported_config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &cpal_config,
            move |data: &[f32], _| {
                push_input_data(data, chunk_samples, &buffer_clone, &tx_clone);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &cpal_config,
            move |data: &[i16], _| {
                push_input_data(data, chunk_samples, &buffer_clone, &tx_clone);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &cpal_config,
            move |data: &[u16], _| {
                push_input_data(data, chunk_samples, &buffer_clone, &tx_clone);
            },
            err_fn,
            None,
        )?,
        sample_format => {
            return Err(format!("Unsupported sample format: {:?}", sample_format).into());
        }
    };

    println!("Listening... press Enter to stop.");
    stream.play()?;

    let stop = Arc::new(AtomicBool::new(false));
    let stop_clone = Arc::clone(&stop);
    std::thread::spawn(move || {
        let mut line = String::new();
        let _ = std::io::stdin().read_line(&mut line);
        stop_clone.store(true, Ordering::SeqCst);
    });

    let mut last_len = 0usize;
    let mut final_line = String::new();
    let mut silence_samples = 0usize;
    let mut noise_floor = 0.0_f32;
    let mut recent_texts: VecDeque<String> = VecDeque::new();
    let mut utterance_audio: Vec<f32> = Vec::new();
    let mut in_utterance = false;
    while !stop.load(Ordering::SeqCst) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => {
                let chunk_rms = rms(&chunk);
                if noise_floor == 0.0 {
                    noise_floor = chunk_rms;
                } else if chunk_rms < noise_floor * 1.5 {
                    noise_floor = noise_floor * 0.95 + chunk_rms * 0.05;
                }
                let silence_threshold =
                    (noise_floor * 3.0).clamp(min_silence_threshold, max_silence_threshold);

                if chunk_rms < silence_threshold {
                    if in_utterance {
                        silence_samples = silence_samples.saturating_add(chunk.len());
                        utterance_audio.extend_from_slice(&chunk);
                        if silence_samples >= silence_hold_samples {
                            if utterance_audio.len() >= min_utterance_samples {
                                let final_result = full_session.transcribe_samples(
                                    &utterance_audio,
                                    sample_rate,
                                    channels,
                                    &source_lang,
                                    &target_lang,
                                )?;
                                let final_text = final_result.text.trim();
                                if !final_text.is_empty() {
                                    if last_len > 0 {
                                        print!("\r{:<width$}", "", width = last_len);
                                    }
                                    print!("\r{}", final_text);
                                    println!();
                                    let _ = std::io::stdout().flush();
                                    last_len = 0;
                                }
                            } else if last_len > 0 {
                                println!();
                                last_len = 0;
                            }

                            stream_state.reset();
                            final_line.clear();
                            recent_texts.clear();
                            utterance_audio.clear();
                            silence_samples = 0;
                            in_utterance = false;
                        }
                    }
                    continue;
                }
                silence_samples = 0;
                in_utterance = true;
                utterance_audio.extend_from_slice(&chunk);

                let results = stream_state.push_samples(&chunk, sample_rate, channels)?;
                for result in results {
                    let text = result.result.text.trim();
                    if !text.is_empty() {
                        recent_texts.push_back(text.to_string());
                        if recent_texts.len() > 3 {
                            recent_texts.pop_front();
                        }

                        let mut committed_updated = false;
                        if recent_texts.len() == 3 {
                            if let Some(stable_text) = stable_prefix_text(&recent_texts) {
                                committed_updated =
                                    append_stable_words(&mut final_line, &stable_text);
                            }
                        }

                        let current = strip_committed_prefix(&final_line, text);
                        let mut display_text = if final_line.is_empty() {
                            current.clone()
                        } else if current.is_empty() {
                            final_line.clone()
                        } else {
                            format!("{} {}", final_line, current)
                        };

                        let mut line_break = false;
                        if committed_updated && ends_with_sentence_punct(&final_line) {
                            display_text = final_line.clone();
                            line_break = true;
                        }

                        if !display_text.is_empty() && has_alnum(&display_text) {
                            last_len = last_len.max(display_text.len());
                            print!("\r{:<width$}", display_text, width = last_len);
                            let _ = std::io::stdout().flush();
                        }

                        if line_break {
                            println!();
                            last_len = 0;
                            final_line.clear();
                            recent_texts.clear();
                        }
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    Ok(())
}

fn push_input_data<T: Sample>(
    input: &[T],
    chunk_samples: usize,
    buffer: &Arc<Mutex<Vec<f32>>>,
    sender: &mpsc::Sender<Vec<f32>>,
) where
    f32: cpal::FromSample<T>,
{
    let mut buffer = match buffer.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    buffer.extend(input.iter().map(|sample| f32::from_sample(*sample)));

    while buffer.len() >= chunk_samples {
        let chunk: Vec<f32> = buffer.drain(..chunk_samples).collect();
        let _ = sender.send(chunk);
    }
}

fn ends_with_sentence_punct(text: &str) -> bool {
    text.chars()
        .rev()
        .find(|ch| !ch.is_whitespace())
        .map_or(false, |ch| matches!(ch, '.' | '!' | '?'))
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|v| v * v).sum();
    (sum / samples.len() as f32).sqrt()
}

fn has_alnum(text: &str) -> bool {
    text.chars().any(|ch| ch.is_alphanumeric())
}

fn stable_prefix_text(texts: &VecDeque<String>) -> Option<String> {
    let mut word_lists: Vec<Vec<&str>> = Vec::with_capacity(texts.len());
    for text in texts {
        word_lists.push(text.split_whitespace().collect());
    }
    if word_lists.is_empty() {
        return None;
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

fn append_stable_words(final_line: &mut String, stable_text: &str) -> bool {
    let stable_words: Vec<&str> = stable_text.split_whitespace().collect();
    if stable_words.is_empty() {
        return false;
    }

    let final_words: Vec<&str> = final_line.split_whitespace().collect();
    if is_prefix_words(&final_words, &stable_words) {
        return false;
    }

    let overlap = suffix_prefix_overlap(&final_words, &stable_words);
    let new_words = &stable_words[overlap..];
    if new_words.is_empty() {
        return false;
    }

    if !final_line.is_empty() {
        final_line.push(' ');
    }
    final_line.push_str(&new_words.join(" "));
    true
}

fn strip_committed_prefix(final_line: &str, text: &str) -> String {
    let text_words: Vec<&str> = text.split_whitespace().collect();
    if text_words.is_empty() {
        return String::new();
    }
    let final_words: Vec<&str> = final_line.split_whitespace().collect();
    let overlap = suffix_prefix_overlap(&final_words, &text_words);
    text_words[overlap..].join(" ")
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
