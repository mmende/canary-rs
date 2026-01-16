use canary_rs::{Canary, ExecutionConfig, ExecutionProvider};
use cpal::Sample;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

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
    let model = Canary::from_pretrained("canary-1b-v2", Some(config))?;
    let mut session = model.session();

    println!("Model loaded successfully!");

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;
    let supported_config = device.default_input_config()?;
    let sample_rate = supported_config.sample_rate() as usize;
    let channels = supported_config.channels() as usize;
    let stream_config: cpal::StreamConfig = supported_config.clone().into();

    let chunk_seconds = 3.0_f32;
    let chunk_samples = (sample_rate as f32 * chunk_seconds) as usize * channels;

    let buffer = Arc::new(Mutex::new(Vec::<f32>::with_capacity(chunk_samples * 2)));
    let (tx, rx) = mpsc::channel::<Vec<f32>>();

    let err_fn = |err| eprintln!("stream error: {}", err);

    let buffer_clone = Arc::clone(&buffer);
    let tx_clone = tx.clone();
    let stream = match supported_config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                push_input_data(data, chunk_samples, &buffer_clone, &tx_clone);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                push_input_data(data, chunk_samples, &buffer_clone, &tx_clone);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &stream_config,
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

    while !stop.load(Ordering::SeqCst) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => {
                let result = session.transcribe_samples(
                    &chunk,
                    sample_rate,
                    channels,
                    &source_lang,
                    &target_lang,
                )?;
                let text = result.text.trim();
                if !text.is_empty() {
                    println!("{}", text);
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
