use canary_rs::{CoreMLComputeUnits, ExecutionConfig, ExecutionProvider, SessionConfig};

/// Builds an `ExecutionConfig` from environment variables.
///
/// - `CANARY_EXECUTION_PROVIDER`: `cpu` (default), `coreml`, `cuda`, `tensorrt`,
///   `directml`, `rocm`, `openvino`, `webgpu`, `nnapi`
/// - `CANARY_COREML_COMPUTE_UNITS`: `all`, `cpu_and_ne` (default), `cpu_and_gpu`, `cpu_only`
/// - `CANARY_COREML_LOW_PRECISION`: `1` or `true`
/// - `CANARY_CUDA_DEVICE_ID`: integer device index (default: `0`)
/// - `CANARY_ITN`: `1`/`true` = force `<|itn|>`, `0`/`false` = force `<|noitn|>`, unset = omit token
pub fn execution_config_from_env() -> ExecutionConfig {
    let provider = std::env::var("CANARY_EXECUTION_PROVIDER")
        .unwrap_or_default()
        .to_lowercase();

    let provider = match provider.trim() {
        "coreml" => ExecutionProvider::CoreML,
        "cuda" => ExecutionProvider::Cuda,
        "tensorrt" => ExecutionProvider::TensorRT,
        "directml" => ExecutionProvider::DirectML,
        "rocm" => ExecutionProvider::ROCm,
        "openvino" => ExecutionProvider::OpenVINO,
        "webgpu" => ExecutionProvider::WebGPU,
        "nnapi" => ExecutionProvider::NNAPI,
        _ => ExecutionProvider::Cpu,
    };

    eprintln!("Execution provider: {:?}", provider);

    let mut config = ExecutionConfig::new().with_execution_provider(provider.clone());

    if matches!(provider, ExecutionProvider::CoreML) {
        let compute_units = std::env::var("CANARY_COREML_COMPUTE_UNITS")
            .unwrap_or_default()
            .to_lowercase();
        let compute_units = match compute_units.trim() {
            "all" => CoreMLComputeUnits::All,
            "cpu_and_gpu" => CoreMLComputeUnits::CPUAndGPU,
            "cpu_only" => CoreMLComputeUnits::CPUOnly,
            _ => CoreMLComputeUnits::CPUAndNeuralEngine,
        };
        config = config.with_coreml_compute_units(compute_units);

        let low_precision = std::env::var("CANARY_COREML_LOW_PRECISION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        config = config.with_coreml_low_precision(low_precision);
    }

    if matches!(
        provider,
        ExecutionProvider::Cuda | ExecutionProvider::TensorRT
    ) {
        let device_id = std::env::var("CANARY_CUDA_DEVICE_ID")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(0);
        config = config.with_cuda_device_id(device_id);
    }

    let itn = std::env::var("CANARY_ITN").ok().and_then(|v| {
        let v = v.trim().to_lowercase();
        match v.as_str() {
            "1" | "true" => Some(true),
            "0" | "false" => Some(false),
            _ => None,
        }
    });
    config = config.with_session_config(SessionConfig::new().with_itn(itn));

    config
}
