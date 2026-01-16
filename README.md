# canary-rs

A Rust implementation for NVIDIA's Canary multilingual ASR/AST model using ONNX Runtime.

## Usage

Download [Canary-1b-v2](https://huggingface.co/istupakov/canary-1b-v2-onnx/tree/main) or [Canary-180m-flash](https://huggingface.co/istupakov/canary-180m-flash-onnx/tree/main) model files from HuggingFace:

- `encoder-model.onnx`
- `encoder-model.onnx.data` (Canary-1b-v2 only)
- `decoder-model.onnx`
- `vocab.txt`

or for int8 quantization:

- `encoder-model.int8.onnx`
- `decoder-model.int8.onnx`
- `vocab.txt`

and place them in a directory, e.g., `canary-1b-v2`.

```rust
use canary_rs::{Canary, StreamConfig};

let model = Canary::from_pretrained("canary-1b-v2", None)?;
let mut session = model.session();
let result = session.transcribe_file("audio.wav", "en", "en")?;
println!("Transcription: {}", result.text);

// Or transcribe in-memory audio
// let result = session.transcribe_samples(&audio_samples, sample_rate, channels, "en", "en")?;

for token in result.tokens {
    // Note: Timestamps are dummy values for now.
    println!("Token: {} ({} - {}) (prob: {:.3})", token.text, token.start, token.end, token.prob);
}

// Windowed streaming helper for live audio.
let stream_cfg = StreamConfig::new().with_window_duration(10.0).with_step_duration(2.0);
let mut stream = model.stream("en", "en", stream_cfg)?;
// stream.push_samples(&audio_chunk, sample_rate, channels)?;
```

## Features

- `ort-defaults` (default): Enable ONNX Runtime default features.
- Execution providers (🚧 Mostly untested): `cuda`, `tensorrt`, `coreml`, `directml`, `rocm`, `openvino`, `webgpu`, `nnapi`.
- Dynamic loading: `load-dynamic`, `preload-dylibs` (see `ort` docs).

## Logging

This crate uses the `log` crate for warnings and diagnostic messages. Configure a logger in
your binary (for example, `env_logger` or `tracing-subscriber`) to see output.

## Notes

Timestamps aren't working right now and are just dummy values because Canary doesn't emit timestamp tokens from the decoder. In the original NeMo implementation, timestamps are generated in a separate post-decode step using forced alignment with an auxiliary CTC model.

## License

MIT License. See [LICENSE.md](LICENSE.md) for details.

## Acknowledgments

- NVIDIA for the Canary models
- [ort](https://github.com/pykeio/ort) crate maintainers
- [parakeet-rs](https://github.com/altunenes/parakeet-rs) for API inspiration
- [ONNX exports by istupakov](https://huggingface.co/istupakov)