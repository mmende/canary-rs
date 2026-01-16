use crate::execution;
use crate::session::CanarySession;
use crate::types::{CanaryError, CanaryResult, Result};
use ort::logging::LogLevel;
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Execution provider configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Cpu,
    Cuda,
    TensorRT,
    CoreML,
    DirectML,
    ROCm,
    OpenVINO,
    WebGPU,
    NNAPI,
}

/// Configuration for session-level decoding and debug options.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Preallocate decoder outputs to reduce per-step allocations.
    pub preallocate_outputs: bool,
    /// Preallocate decoder logits when `preallocate_outputs` is enabled.
    pub preallocate_logits: bool,
    /// Override the default prompt tokens by specifying token ids or token strings.
    pub prompt_override: Option<String>,
    /// Override decoder layer count if it cannot be inferred from the model outputs.
    pub decoder_num_layers: Option<usize>,
    /// Override decoder hidden size if it cannot be inferred from the model outputs.
    pub decoder_hidden_size: Option<usize>,
    /// Enable punctuation and capitalization tokens in the prompt.
    pub use_pnc: bool,
    /// Enable inverse text normalization tokens in the prompt.
    pub use_itn: bool,
    /// Enable timestamp tokens in the prompt.
    pub use_timestamps: bool,
    /// Enable diarization tokens in the prompt.
    pub use_diarize: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            preallocate_outputs: false,
            preallocate_logits: false,
            prompt_override: None,
            decoder_num_layers: None,
            decoder_hidden_size: None,
            use_pnc: true,
            use_itn: false,
            use_timestamps: false,
            use_diarize: false,
        }
    }
}

impl SessionConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_decoder_dims(mut self, num_layers: usize, hidden_size: usize) -> Self {
        self.decoder_num_layers = Some(num_layers);
        self.decoder_hidden_size = Some(hidden_size);
        self
    }
}

/// Configuration for model execution
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Execution provider to use for the model sessions.
    pub execution_provider: ExecutionProvider,
    /// Number of inter-op threads for ORT.
    pub inter_threads: usize,
    /// Number of intra-op threads for ORT.
    pub intra_threads: usize,
    /// Use ORT default session options without applying this config.
    pub use_ort_default_options: bool,
    /// Disable ORT graph optimizations.
    pub disable_ort_optimization: bool,
    /// Skip configuring ORT thread pools.
    pub skip_ort_threads: bool,
    /// Disable ORT prepacking of weights.
    pub disable_ort_prepack: bool,
    /// Enable verbose ORT logging.
    pub ort_verbose: bool,
    /// Optional CoreML model cache directory.
    pub coreml_cache_dir: Option<String>,
    /// Decoder/session runtime options.
    pub session: SessionConfig,
}

impl ExecutionConfig {
    pub fn new() -> Self {
        Self {
            execution_provider: ExecutionProvider::Cpu,
            inter_threads: 1,
            intra_threads: 4,
            use_ort_default_options: false,
            disable_ort_optimization: false,
            skip_ort_threads: false,
            disable_ort_prepack: false,
            ort_verbose: false,
            coreml_cache_dir: None,
            session: SessionConfig::default(),
        }
    }

    pub fn with_execution_provider(mut self, provider: ExecutionProvider) -> Self {
        self.execution_provider = provider;
        self
    }

    pub fn with_threads(mut self, inter: usize, intra: usize) -> Self {
        self.inter_threads = inter;
        self.intra_threads = intra;
        self
    }

    pub fn with_session_config(mut self, session: SessionConfig) -> Self {
        self.session = session;
        self
    }

    pub fn with_coreml_cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.coreml_cache_dir = Some(dir.into());
        self
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Main Canary model interface
#[derive(Clone)]
pub struct Canary {
    pub(crate) encoder: Arc<Mutex<Session>>,
    pub(crate) decoder: Arc<Mutex<Session>>,
    pub(crate) vocab: Arc<Vec<String>>,
    pub(crate) token_to_id: Arc<HashMap<String, usize>>,
    pub(crate) config: ExecutionConfig,
    pub(crate) sample_rate: usize,
}

impl Canary {
    /// Load model from a directory containing the ONNX files
    pub fn from_pretrained<P: AsRef<Path>>(
        model_path: P,
        config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let config = config.unwrap_or_default();

        // Expand a leading "~" (only "~" and "~/" forms are supported).
        let model_path = if let Ok(stripped) = model_path.strip_prefix("~") {
            let home = std::env::var("HOME")
                .map_err(|_| CanaryError::ModelError("Cannot expand ~ without HOME".into()))?;
            if stripped.as_os_str().is_empty() {
                PathBuf::from(home)
            } else if stripped.is_absolute() {
                return Err(CanaryError::ModelError(
                    "Unsupported path form after '~' expansion".into(),
                ));
            } else {
                PathBuf::from(home).join(stripped)
            }
        } else {
            model_path.to_path_buf()
        };

        let encoder_path = Self::select_model_file(
            &model_path,
            &["encoder-model.onnx", "encoder-model.int8.onnx"],
        )?;
        let decoder_path = Self::select_model_file(
            &model_path,
            &["decoder-model.onnx", "decoder-model.int8.onnx"],
        )?;

        if !encoder_path.exists() {
            return Err(CanaryError::ModelError(format!(
                "Encoder model not found at {:?}",
                encoder_path
            )));
        }

        if !decoder_path.exists() {
            return Err(CanaryError::ModelError(format!(
                "Decoder model not found at {:?}",
                decoder_path
            )));
        }

        // Load vocabulary
        let vocab_path = model_path.join("vocab.txt");
        let vocab = Self::load_vocab(&vocab_path)?;
        let token_to_id: HashMap<String, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        // Create ONNX sessions
        let encoder = Arc::new(Mutex::new(Self::create_session(&encoder_path, &config)?));
        let decoder_config = match config.execution_provider {
            ExecutionProvider::CoreML => ExecutionConfig {
                execution_provider: ExecutionProvider::Cpu,
                ..config.clone()
            },
            _ => config.clone(),
        };
        let decoder = Arc::new(Mutex::new(Self::create_session(&decoder_path, &decoder_config)?));

        Ok(Self {
            encoder,
            decoder,
            vocab: Arc::new(vocab),
            token_to_id: Arc::new(token_to_id),
            config,
            sample_rate: 16000,
        })
    }

    /// Create a new session with isolated per-run state.
    pub fn session(&self) -> CanarySession {
        CanarySession::new(self.clone())
    }

    /// Transcribe an audio file using a fresh session.
    pub fn transcribe_file<P: AsRef<Path>>(
        &self,
        path: P,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<CanaryResult> {
        let mut session = self.session();
        session.transcribe_file(path, source_lang, target_lang)
    }

    /// Transcribe in-memory audio samples using a fresh session.
    pub fn transcribe_samples(
        &self,
        audio: &[f32],
        sample_rate: usize,
        channels: usize,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<CanaryResult> {
        let mut session = self.session();
        session.transcribe_samples(audio, sample_rate, channels, source_lang, target_lang)
    }

    fn load_vocab(path: &Path) -> Result<Vec<String>> {
        let content = std::fs::read_to_string(path)?;

        // Parse vocab.txt format: "token_text token_id"
        // We need to extract just the token_text and maintain order by token_id
        let mut tokens_with_ids: Vec<(String, usize)> = Vec::new();

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            // Split by last space to separate token from ID
            if let Some(last_space_idx) = line.rfind(' ') {
                let token_text = &line[..last_space_idx];
                let token_id_str = &line[last_space_idx + 1..];

                if let Ok(token_id) = token_id_str.parse::<usize>() {
                    tokens_with_ids.push((token_text.to_string(), token_id));
                }
            }
        }

        if tokens_with_ids.is_empty() {
            return Err(CanaryError::ModelError("Vocabulary is empty".into()));
        }

        // Sort by ID and extract just the tokens
        tokens_with_ids.sort_by_key(|(_, id)| *id);
        let vocab: Vec<String> = tokens_with_ids
            .into_iter()
            .map(|(token, _)| token)
            .collect();

        Ok(vocab)
    }

    fn select_model_file(model_dir: &Path, candidates: &[&str]) -> Result<PathBuf> {
        for name in candidates {
            let candidate = model_dir.join(name);
            if candidate.exists() {
                return Ok(candidate);
            }
        }

        Err(CanaryError::ModelError(format!(
            "Model file not found; tried: {}",
            candidates.join(", ")
        )))
    }

    fn create_session(path: &Path, config: &ExecutionConfig) -> Result<Session> {
        let mut builder = Session::builder()?;

        if !config.use_ort_default_options {
            let opt_level = if config.disable_ort_optimization {
                GraphOptimizationLevel::Disable
            } else {
                GraphOptimizationLevel::Level1
            };

            builder = builder
                .with_optimization_level(opt_level)?
                .with_memory_pattern(false)?;

            if !config.skip_ort_threads {
                builder = builder
                    .with_intra_threads(config.intra_threads)?
                    .with_inter_threads(config.inter_threads)?;
            }

            if config.disable_ort_prepack {
                builder = builder.with_prepacking(false)?;
            }
        }

        if config.ort_verbose {
            builder = builder.with_logger(Arc::new(
                |level: LogLevel, category: &str, id: &str, location: &str, message: &str| {
                    let log_level = match level {
                        LogLevel::Verbose => log::Level::Debug,
                        LogLevel::Info => log::Level::Info,
                        LogLevel::Warning => log::Level::Warn,
                        LogLevel::Error | LogLevel::Fatal => log::Level::Error,
                    };
                    log::log!(
                        log_level,
                        "[ort::{:?}] {} {} {}: {}",
                        level,
                        category,
                        id,
                        location,
                        message
                    );
                },
            ))?;
            builder = builder
                .with_log_level(LogLevel::Verbose)?
                .with_log_verbosity(4)?;
        }

        builder = execution::apply_execution_providers(builder, config)?;
        builder = Self::attach_external_initializers(builder, path, config)?;

        let session = builder.commit_from_file(path)?;

        // Note: In ort v2.0.0-rc.11, execution providers are configured differently
        // For now, we'll use the default provider which is CPU
        // CUDA and CoreML would need to be configured through environment variables
        // or different API calls

        Ok(session)
    }

    fn attach_external_initializers(
        mut builder: ort::session::builder::SessionBuilder,
        model_path: &Path,
        config: &ExecutionConfig,
    ) -> Result<ort::session::builder::SessionBuilder> {
        let use_external_data = match config.execution_provider {
            ExecutionProvider::CoreML => true,
            _ => false,
        };

        if !use_external_data {
            return Ok(builder);
        }

        let data_path = model_path.with_extension("onnx.data");
        if !data_path.exists() {
            return Ok(builder);
        }

        let file_name = data_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| CanaryError::ModelError("Invalid external data file name".into()))?;
        let data = std::fs::read(&data_path)?;

        builder = builder.with_external_initializer_file_in_memory(file_name, data.into())?;
        Ok(builder)
    }
}
