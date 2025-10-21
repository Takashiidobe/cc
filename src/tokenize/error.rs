use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("unsupported escape `{0}` at byte {1}")]
    UnsupportedEscape(char, usize),

    #[error("invalid character `{0}`")]
    InvalidCharacter(char),

    #[error("invalid string literal: {0}")]
    InvalidStringLiteral(String),

    #[error("unterminated string literal: {0}")]
    UnterminatedStringLiteral(String),

    #[error("truncated escape at byte {0} in {1}")]
    TruncatedEscapeSequence(usize, String),

    #[error("invalid character literal: {0}")]
    InvalidCharLiteral(String),

    #[error("unterminated character literal: {0}")]
    UnterminatedCharLiteral(String),

    #[error("empty character literal: {0}")]
    EmptyCharLiteral(String),

    #[error("too many chars in char literal: first={0}, extra={1}")]
    TooManyCharsInCharLiteral(char, String),

    #[error("char does not fit in i8: {0}")]
    CharOutOfRange(char),
}
