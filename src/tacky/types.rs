use std::collections::BTreeMap;

use crate::parse::{Const, Type};

#[derive(Debug, Clone)]
pub(crate) struct Program {
    pub(crate) items: Vec<TopLevel>,
    pub(crate) global_types: BTreeMap<String, Type>,
}

#[derive(Debug, Clone)]
pub(crate) enum TopLevel {
    Function(Function),
    StaticVariable(StaticVariable),
    StaticConstant(StaticConstant),
}

#[derive(Debug, Clone)]
pub(crate) struct Function {
    pub(crate) name: String,
    pub(crate) global: bool,
    pub(crate) params: Vec<String>,
    pub(crate) return_type: Type,
    pub(crate) instructions: Vec<Instruction>,
    pub(crate) value_types: BTreeMap<String, Type>,
}

#[derive(Debug, Clone)]
pub(crate) struct StaticVariable {
    pub(crate) name: String,
    pub(crate) global: bool,
    pub(crate) ty: Type,
    pub(crate) init: Vec<StaticInit>,
}

#[derive(Debug, Clone)]
pub(crate) struct StaticConstant {
    pub(crate) name: String,
    pub(crate) ty: Type,
    pub(crate) init: StaticInit,
}

#[derive(Debug, Clone)]
pub(crate) enum StaticInit {
    Scalar {
        offset: i64,
        value: Const,
    },
    Bytes {
        offset: i64,
        value: Vec<u8>,
        null_terminated: bool,
    },
    Label {
        offset: i64,
        symbol: String,
    },
}

#[derive(Debug, Clone)]
pub(crate) enum Value {
    Constant(Const),
    Var(String),
    Global(String),
}

#[derive(Debug, Clone)]
pub(crate) enum Instruction {
    Return(Value),
    Unary {
        op: UnaryOp,
        src: Value,
        dst: Value,
    },
    Binary {
        op: BinaryOp,
        src1: Value,
        src2: Value,
        dst: Value,
    },
    Copy {
        src: Value,
        dst: Value,
    },
    FunCall {
        name: String,
        args: Vec<Value>,
        dst: Value,
    },
    Jump(String),
    JumpIfZero {
        condition: Value,
        target: String,
    },
    JumpIfNotZero {
        condition: Value,
        target: String,
    },
    Label(String),
    SignExtend {
        src: Value,
        dst: Value,
    },
    ZeroExtend {
        src: Value,
        dst: Value,
    },
    Truncate {
        src: Value,
        dst: Value,
    },
    Convert {
        src: Value,
        dst: Value,
        from: Type,
        to: Type,
    },
    GetAddress {
        src: Value,
        dst: Value,
    },
    Load {
        src_ptr: Value,
        dst: Value,
    },
    Store {
        src: Value,
        dst_ptr: Value,
    },
    AddPtr {
        ptr: Value,
        index: Value,
        scale: i64,
        dst: Value,
    },
    CopyToOffset {
        src: Value,
        dst: String,
        offset: i64,
    },
    CopyFromOffset {
        src: String,
        offset: i64,
        dst: Value,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOp {
    Negate,
    Complement,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
    BitAnd,
    BitOr,
    BitXor,
    LeftShift,
    RightShift,
}

pub(crate) struct LoopContext {
    pub(crate) id: usize,
    pub(crate) break_label: String,
    pub(crate) continue_label: String,
}
