# canary-rs

A Rust implementation for NVIDIA's Canary multilingual ASR/AST model using ONNX Runtime.

## Usage

Download **Canary-1b-v2** from [HuggingFace](https://huggingface.co/istupakov/canary-1b-v2-onnx/tree/main):

- `encoder-model.onnx`
- `encoder-model.onnx.data`
- `decoder-model.onnx`
- `vocab.txt`

or for int8 quantization:

- `encoder-model.int8.onnx`
- `decoder-model.int8.onnx`
- `vocab.txt`

and place them in a directory, e.g., `canary-1b-v2`.

Alternative checkpoints/models:
- [canary-180m-flash](https://huggingface.co/istupakov/canary-180m-flash-onnx/tree/main).

```rust
use canary_rs::{Canary, StreamConfig};

let model = Canary::from_pretrained("canary-1b-v2", None)?;
let mut session = model.session();
let result = session.transcribe_file("audio.wav", "en", "en")?;
println!("Transcription: {}", result.text);

// Or transcribe in-memory audio
// let result = session.transcribe_samples(&audio_samples, sample_rate, channels, "en", "en")?;

for token in result.tokens {
    println!("Token: {} ({} - {})", token.text, token.start, token.end);
}

// Windowed streaming helper for live audio.
let stream_cfg = StreamConfig::new().with_window_duration(10.0).with_step_duration(2.0);
let mut stream = model.stream("en", "en", stream_cfg)?;
// stream.push_samples(&audio_chunk, sample_rate, channels)?;
```

## Features

- `ort-defaults` (default): Enable ONNX Runtime default features.
- Execution providers: `cuda`, `tensorrt`, `coreml`, `directml`, `rocm`, `openvino`, `webgpu`, `nnapi`.
- Dynamic loading: `load-dynamic`, `preload-dylibs` (see `ort` docs).

## Logging

This crate uses the `log` crate for warnings and diagnostic messages. Configure a logger in
your binary (for example, `env_logger` or `tracing-subscriber`) to see output.

## License

MIT License. See [LICENSE.md](LICENSE.md) for details.

## Acknowledgments

- NVIDIA for the Canary models
- [ort](https://github.com/pykeio/ort) crate maintainers
- [parakeet-rs](https://github.com/altunenes/parakeet-rs) for API inspiration
- [ONNX exports by istupakov](https://huggingface.co/istupakov)