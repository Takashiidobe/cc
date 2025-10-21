use thiserror::Error;

use crate::parse::Type;

pub(crate) type IResult<T> = Result<T, IRError>;

#[derive(Error, Debug, Clone)]
pub(crate) enum IRError {
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
    #[error("expected pointer type, found {0:?}")]
    ExpectedPointer(Type),
    #[error("{0}")]
    Generic(&'static str),
}
