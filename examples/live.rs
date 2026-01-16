use canary_rs::{Canary, ExecutionConfig, ExecutionProvider, StreamConfig};
use cpal::Sample;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
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
        .with_emit_partial(true)
        .with_pad_partial(false)
        .with_stability_window(3);
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
                    let delta = result.delta_text.trim();
                    if delta.is_empty() {
                        continue;
                    }

                    append_with_space(&mut final_line, delta);

                    let mut line_break = false;
                    if ends_with_sentence_punct(&final_line) {
                        line_break = true;
                    }

                    if !final_line.is_empty() && has_alnum(&final_line) {
                        last_len = last_len.max(final_line.len());
                        print!("\r{:<width$}", final_line, width = last_len);
                        let _ = std::io::stdout().flush();
                    }

                    if line_break {
                        println!();
                        last_len = 0;
                        final_line.clear();
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

fn append_with_space(line: &mut String, chunk: &str) {
    if chunk.is_empty() {
        return;
    }
    let needs_space = !line.is_empty() && !starts_with_punct(chunk);
    if needs_space {
        line.push(' ');
    }
    line.push_str(chunk);
}

fn starts_with_punct(text: &str) -> bool {
    text.chars()
        .next()
        .map_or(false, |ch| matches!(ch, '.' | ',' | '!' | '?' | ';' | ':'))
}
