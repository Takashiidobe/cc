use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum LoopLabelerError {
    #[error("Break not in loop")]
    BreakNotInLoop,
    #[error("Continue not in loop")]
    ContinueNotInLoop,
}
