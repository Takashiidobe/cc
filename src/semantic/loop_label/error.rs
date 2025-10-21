use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoopLabelerError {
    #[error("Break not in loop")]
    BreakNotInLoop,
    #[error("Continue not in loop")]
    ContinueNotInLoop,
}
