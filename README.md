# canary-rs

A Rust implementation for NVIDIA's Canary-1b-v2 multilingual ASR/AST model using ONNX Runtime.

## Usage

Download from [HuggingFace](https://huggingface.co/istupakov/canary-1b-v2-onnx/tree/main):

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
- [canary-1b-flash](https://huggingface.co/istupakov/canary-1b-flash-onnx/tree/main).

```rust
use canary_rs::Canary;

let model = Canary::from_pretrained("canary-1b-v2", None)?;
let mut session = model.session();
let result = session.transcribe_file("audio.wav", "en", "en")?;
println!("Transcription: {}", result.text);

// Or transcribe in-memory audio
// let result = session.transcribe_samples(&audio_samples, sample_rate, channels, "en", "en")?;

for token in result.tokens {
    println!("Token: {} ({} - {})", token.text, token.start, token.end);
}
```

If you are using a different Canary checkpoint and the decoder dimensions cannot be inferred,
set them explicitly via `SessionConfig::with_decoder_dims`.

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

- NVIDIA for Canary-1b-v2
- `ort` crate maintainers
- `parakeet-rs` for API inspiration
