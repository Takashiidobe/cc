use std::collections::BTreeMap;

use crate::parse::{Const, Type};

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<TopLevel>,
    pub global_types: BTreeMap<String, Type>,
}

#[derive(Debug, Clone)]
pub enum TopLevel {
    Function(Function),
    StaticVariable(StaticVariable),
    StaticConstant(StaticConstant),
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub global: bool,
    pub params: Vec<String>,
    pub return_type: Type,
    pub instructions: Vec<Instruction>,
    pub value_types: BTreeMap<String, Type>,
}

#[derive(Debug, Clone)]
pub struct StaticVariable {
    pub name: String,
    pub global: bool,
    pub ty: Type,
    pub init: Vec<StaticInit>,
}

#[derive(Debug, Clone)]
pub struct StaticConstant {
    pub name: String,
    pub ty: Type,
    pub init: StaticInit,
}

#[derive(Debug, Clone)]
pub enum StaticInit {
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
pub enum Value {
    Constant(Const),
    Var(String),
    Global(String),
}

#[derive(Debug, Clone)]
pub enum Instruction {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Complement,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
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

pub struct LoopContext {
    pub id: usize,
    pub break_label: String,
    pub continue_label: String,
}
