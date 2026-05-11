use crate::model::{ExecutionConfig, ExecutionProvider};
use crate::types::{CanaryError, Result};
use ort::ep::ExecutionProviderDispatch;
use ort::session::builder::SessionBuilder;

pub fn apply_execution_providers(
    mut builder: SessionBuilder,
    config: &ExecutionConfig,
) -> Result<SessionBuilder> {
    let providers = execution_providers(config)?;
    builder = builder.with_execution_providers(providers)?;
    Ok(builder)
}

fn execution_providers(config: &ExecutionConfig) -> Result<Vec<ExecutionProviderDispatch>> {
    let mut providers = Vec::new();

    match config.execution_provider {
        ExecutionProvider::Cpu => {
            providers.push(cpu_provider());
        }
        ExecutionProvider::Cuda => {
            providers.push(cuda_provider(config.cuda_device_id)?);
            providers.push(cpu_provider());
        }
        ExecutionProvider::TensorRT => {
            providers.push(tensorrt_provider(config.cuda_device_id)?);
            providers.push(cpu_provider());
        }
        ExecutionProvider::CoreML => {
            providers.push(coreml_provider(
                config.coreml_cache_dir.as_deref(),
                config.coreml_compute_units,
                config.coreml_low_precision,
                config.coreml_model_format,
                config.coreml_static_input_shapes,
                config.coreml_enable_subgraphs,
                config.coreml_specialization_strategy,
            )?);
            providers.push(cpu_provider());
        }
        ExecutionProvider::DirectML => {
            providers.push(directml_provider()?);
            providers.push(cpu_provider());
        }
        ExecutionProvider::ROCm => {
            providers.push(rocm_provider()?);
            providers.push(cpu_provider());
        }
        ExecutionProvider::OpenVINO => {
            providers.push(openvino_provider()?);
            providers.push(cpu_provider());
        }
        ExecutionProvider::WebGPU => {
            providers.push(webgpu_provider()?);
            providers.push(cpu_provider());
        }
        ExecutionProvider::NNAPI => {
            providers.push(nnapi_provider()?);
            providers.push(cpu_provider());
        }
    }

    Ok(providers)
}

fn cpu_provider() -> ExecutionProviderDispatch {
    ort::ep::CPU::default().build()
}

fn cuda_provider(device_id: i32) -> Result<ExecutionProviderDispatch> {
    #[cfg(feature = "cuda")]
    {
        Ok(ort::ep::CUDA::default().with_device_id(device_id).build())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device_id;
        Err(CanaryError::ModelError(
            "CUDA execution provider not enabled; build with feature \"cuda\"".into(),
        ))
    }
}

fn tensorrt_provider(device_id: i32) -> Result<ExecutionProviderDispatch> {
    #[cfg(feature = "tensorrt")]
    {
        Ok(ort::ep::TensorRT::default().with_device_id(device_id).build())
    }
    #[cfg(not(feature = "tensorrt"))]
    {
        let _ = device_id;
        Err(CanaryError::ModelError(
            "TensorRT execution provider not enabled; build with feature \"tensorrt\"".into(),
        ))
    }
}

fn coreml_provider(
    cache_dir: Option<&str>,
    compute_units: Option<ort::ep::coreml::ComputeUnits>,
    low_precision: bool,
    model_format: Option<ort::ep::coreml::ModelFormat>,
    static_input_shapes: bool,
    enable_subgraphs: bool,
    specialization_strategy: Option<ort::ep::coreml::SpecializationStrategy>,
) -> Result<ExecutionProviderDispatch> {
    #[cfg(feature = "coreml")]
    {
        let mut provider = ort::ep::CoreML::default()
            .with_compute_units(
                compute_units.unwrap_or(ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine),
            )
            .with_low_precision_accumulation_on_gpu(low_precision);
        if let Some(model_format) = model_format {
            provider = provider.with_model_format(model_format);
        }
        if static_input_shapes {
            provider = provider.with_static_input_shapes(true);
        }
        if enable_subgraphs {
            provider = provider.with_subgraphs(true);
        }
        if let Some(strategy) = specialization_strategy {
            provider = provider.with_specialization_strategy(strategy);
        }
        if let Some(cache_dir) = cache_dir {
            if !cache_dir.trim().is_empty() {
                provider = provider.with_model_cache_dir(cache_dir.to_string());
            }
        }
        Ok(provider.build())
    }
    #[cfg(not(feature = "coreml"))]
    {
        let _ = cache_dir;
        let _ = compute_units;
        let _ = low_precision;
        let _ = model_format;
        let _ = static_input_shapes;
        let _ = enable_subgraphs;
        let _ = specialization_strategy;
        Err(CanaryError::ModelError(
            "CoreML execution provider not enabled; build with feature \"coreml\"".into(),
        ))
    }
}

fn directml_provider() -> Result<ExecutionProviderDispatch> {
    #[cfg(feature = "directml")]
    {
        Ok(ort::ep::DirectML::default().build())
    }
    #[cfg(not(feature = "directml"))]
    {
        Err(CanaryError::ModelError(
            "DirectML execution provider not enabled; build with feature \"directml\"".into(),
        ))
    }
}

fn rocm_provider() -> Result<ExecutionProviderDispatch> {
    #[cfg(feature = "rocm")]
    {
        Ok(ort::ep::ROCm::default().build())
    }
    #[cfg(not(feature = "rocm"))]
    {
        Err(CanaryError::ModelError(
            "ROCm execution provider not enabled; build with feature \"rocm\"".into(),
        ))
    }
}

fn openvino_provider() -> Result<ExecutionProviderDispatch> {
    #[cfg(feature = "openvino")]
    {
        Ok(ort::ep::OpenVINO::default().build())
    }
    #[cfg(not(feature = "openvino"))]
    {
        Err(CanaryError::ModelError(
            "OpenVINO execution provider not enabled; build with feature \"openvino\"".into(),
        ))
    }
}

fn webgpu_provider() -> Result<ExecutionProviderDispatch> {
    #[cfg(feature = "webgpu")]
    {
        Ok(ort::ep::WebGPU::default().build())
    }
    #[cfg(not(feature = "webgpu"))]
    {
        Err(CanaryError::ModelError(
            "WebGPU execution provider not enabled; build with feature \"webgpu\"".into(),
        ))
    }
}

fn nnapi_provider() -> Result<ExecutionProviderDispatch> {
    #[cfg(feature = "nnapi")]
    {
        Ok(ort::ep::NNAPI::default().build())
    }
    #[cfg(not(feature = "nnapi"))]
    {
        Err(CanaryError::ModelError(
            "NNAPI execution provider not enabled; build with feature \"nnapi\"".into(),
        ))
    }
}
