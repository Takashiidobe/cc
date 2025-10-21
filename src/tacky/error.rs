use thiserror::Error;

use crate::parse::Type;

pub type IResult<T> = Result<T, IRError>;

#[derive(Error, Debug, Clone)]
pub enum IRError {
    #[error("unexpected state: {0}")]
    State(&'static str),
    #[error("loop stack underflow")]
    LoopUnderflow,
    #[error("loop id mismatch (expected {0}, got {1})")]
    LoopIdMismatch(usize, usize),
    #[error("no loop context for id {0}")]
    NoLoopContext(usize),
    #[error("missing loop id in {0}")]
    MissingLoopId(&'static str),
    #[error("return outside function context")]
    ReturnOutsideFunction,
    #[error("unsupported array initializer for static variable '{0}'")]
    UnsupportedArrayInit(String),
    #[error("division by zero in static initializer")]
    DivByZeroInStaticInit,
    #[error("non-constant expression in static initializer")]
    NonConstInitializer,
    #[error("unknown temporary {0}")]
    UnknownTemporary(String),
    #[error("unknown global {0}")]
    UnknownGlobal(String),
    #[error("unsupported conversion from {0:?} to {1:?}")]
    UnsupportedConversion(Type, Type),
    #[error("value of type {0:?} has no bit width")]
    NoBitWidth(Type),
    #[error("type {0:?} has no integer rank")]
    NoRank(Type),
    #[error("unsupported operand types {0:?} and {1:?}")]
    UnsupportedOperands(Type, Type),
    #[error("mismatched constant for double conversion")]
    BadDoubleConstant,
    #[error("cannot convert constant to {0:?}")]
    BadConstTarget(Type),
    #[error("cannot convert floating constant to {0}")]
    BadFloatConstTarget(&'static str),
    #[error("string literal \"{0}\" does not fit in array of size {1}")]
    StringTooLarge(String, usize),
    #[error("static initializer: non-constant double initializer")]
    NonConstDoubleInit,
    #[error("assignment target must be assignable")]
    BadAssignTarget,
    #[error("unsupported lvalue for address-of")]
    BadLValueAddressOf,
    #[error("call to undeclared function {0}")]
    CallUndeclared(String),
    #[error("function '{0}' called with wrong number of arguments (expected {1}, got {2})")]
    WrongArity(String, usize, usize),
    #[error("pointer addition is not supported")]
    PointerAddUnsupported,
    #[error("integer minus pointer is not supported")]
    IntMinusPtrUnsupported,
    #[error("invalid pointer element size {0}")]
    BadPointerElemSize(i64),
    #[error("expected pointer type, found {0:?}")]
    ExpectedPointer(Type),
    #[error("array size exceeds i64")]
    ArraySizeI64Overflow,
    #[error("{0}")]
    Generic(&'static str),
}
