use std::io;

use thiserror::Error;

use crate::{
    parse::{Const, Type},
    tacky::{BinaryOp, UnaryOp, Value},
};

#[derive(Error, Debug)]
pub(crate) enum CodegenError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("invalid float register index {0}")]
    InvalidFloatRegIndex(usize),
    #[error("unsupported xmm register index {0}")]
    UnsupportedXmmIndex(usize),
    #[error("general-purpose register requested for unsupported type {0:?}")]
    GprRequestedFor(Type),

    #[error("compound static initializers are not supported yet")]
    CompoundStaticInitializerUnsupported,
    #[error("unsupported scalar static initializer {0:?} <- {1:?}")]
    UnsupportedScalarStaticInitializer(Type, Const),
    #[error("byte initializer only supported for array types (found {0:?})")]
    ByteInitializerOnlyForArray(Type),
    #[error("label initializer requires pointer type (found {0:?})")]
    LabelInitializerRequiresPointer(Type),
    #[error("null-terminated initializer missing trailing NUL byte")]
    MissingTrailingNullInAsciz,

    #[error("unsupported function return type {0:?}")]
    UnsupportedFunctionReturnType(Type),
    #[error("array return type not yet supported in codegen")]
    ArrayReturnTypeUnsupported,
    #[error("missing type information for value '{0}'")]
    MissingTypeInfoForValue(String),
    #[error("missing type information for parameter '{0}'")]
    MissingTypeInfoForParam(String),

    #[error("attempted to access undefined stack slot '{0}'")]
    UndefinedStackSlot(String),
    #[error("address destination cannot be a constant")]
    AddressDestCannotBeConstant,
    #[error("cannot take address of a constant")]
    CannotTakeAddressOfConstant,

    #[error("copy destination cannot be a constant")]
    CopyDestCannotBeConstant,
    #[error("cannot store into a constant")]
    CannotStoreIntoConstant,
    #[error("cannot store double via general-purpose register")]
    StoreDoubleViaGpr,
    #[error("attempted to load double into general-purpose register")]
    LoadDoubleIntoGpr,
    #[error("unsupported load into xmm for value {0:?}")]
    UnsupportedLoadIntoXmm(Value),

    #[error("invalid unary op {0:?} for type {1:?}")]
    InvalidUnaryOpForType(UnaryOp, Type),
    #[error("invalid binary op {0:?} for types {1:?} and {2:?}")]
    InvalidBinaryOpForTypes(BinaryOp, Type, Type),

    #[error("division or remainder not supported for type {0:?}")]
    DivisionUnsupportedForType(Type),

    #[error("AddPtr destination cannot be a constant")]
    AddPtrDestCannotBeConstant,
    #[error("AddPtr destination must be a pointer type (found {0:?})")]
    AddPtrDestMustBePointer(Type),

    #[error("unsupported conversion {0:?} -> {1:?}")]
    UnsupportedConversion(Type, Type),
    #[error("unsupported sign extension {0:?} -> {1:?}")]
    UnsupportedSignExtend(Type, Type),

    #[error("CopyToOffset not supported for type {0:?}")]
    CopyToOffsetUnsupported(Type),

    #[error("unknown value type")]
    UnknownValueType,

    #[error("mov instruction requested for unsupported type {0:?}")]
    MovUnsupported(Type),
}

pub(crate) type Result<T = ()> = std::result::Result<T, CodegenError>;
