mod audio;
mod execution;
mod model;
mod session;
mod types;

pub use model::{Canary, ExecutionConfig, ExecutionProvider, SessionConfig};
pub use session::CanarySession;
pub use types::{CanaryError, CanaryResult, Result, Token};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_config() {
        let config = ExecutionConfig::new()
            .with_execution_provider(ExecutionProvider::Cpu)
            .with_threads(2, 2);

        assert_eq!(config.execution_provider, ExecutionProvider::Cpu);
        assert_eq!(config.inter_threads, 2);
        assert_eq!(config.intra_threads, 2);
    }
}
