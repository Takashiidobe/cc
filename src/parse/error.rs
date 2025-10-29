use thiserror::Error;

use crate::{parse::ExprKind, tokenize::TokenKind};

#[derive(Error, Debug, Clone)]
pub(crate) enum ParserError {
    #[error("unexpected end of input{0}")]
    UnexpectedEof(&'static str),

    #[error("expected token {0:?}, found {1:?}")]
    ExpectedToken(TokenKind, Option<TokenKind>),

    #[error("not at end of program (next token: {0:?})")]
    NotAtEnd(Option<TokenKind>),

    #[error("storage class specifiers are not allowed here")]
    StorageNotAllowedHere,

    #[error("multiple storage class specifiers in declaration")]
    MultipleStorageClasses,

    #[error("unsupported type specifier {0:?}")]
    UnsupportedTypeSpecifier(TokenKind),

    #[error("duplicate type specifier '{0}'")]
    DuplicateTypeSpecifier(&'static str),

    #[error("conflicting type specifiers '{0}' and '{1}'")]
    ConflictingTypeSpecifiers(&'static str, &'static str),

    #[error("'void' cannot be combined with other type specifiers")]
    VoidCannotCombine,

    #[error("'double' cannot be combined with other type specifiers")]
    DoubleCannotCombine,

    #[error("declaration missing type specifier")]
    MissingTypeSpecifier,

    #[error("expected declaration specifiers")]
    ExpectedDeclSpecifiers,

    #[error("array size must be non-negative")]
    NegativeArraySize,

    #[error("array size does not fit in usize ({0})")]
    ArraySizeTooLarge(i128),

    #[error("expected constant array size, found {0:?}")]
    ExpectedConstArraySize(TokenKind),

    #[error("'void' parameter must be the only parameter")]
    VoidOnlyParameter,

    #[error("variable declared with void type")]
    VariableWithVoidType,

    #[error("function declarations are not allowed in block scope")]
    FunctionDeclInBlockScope,

    #[error("variable declared with function type")]
    VariableWithFunctionType,

    #[error("invalid function call target: {0:?}")]
    InvalidFunctionCallTarget(ExprKind),

    #[error("unsupported compound assignment token: {0:?}")]
    UnsupportedCompoundAssign(TokenKind),

    #[error("unsupported cast target: void")]
    UnsupportedCastTargetVoid,

    #[error("expected identifier or '(' in declarator, found {0:?}")]
    ExpectedIdentOrParen(TokenKind),

    #[error("Expected primary expression, found {0:?}")]
    ExpectedPrimary(TokenKind),

    #[error("expected identifier")]
    ExpectedIdentifier,
}

pub(crate) type PResult<T> = Result<T, ParserError>;
