use std::fmt;

use crate::{
    codegen::CodegenError,
    parse::{Type, Width},
};

#[derive(PartialEq, Debug, Clone, Copy)]
pub(crate) enum Reg {
    AX,
    CX,
    DX,
    DI,
    SI,
    R8,
    R9,
    R10,
    R11,
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub(crate) enum FloatReg {
    XMM0,
    XMM1,
    XMM2,
    XMM3,
    XMM4,
    XMM5,
    XMM6,
    XMM7,
}

impl FloatReg {
    pub(crate) const COUNT: usize = 8;
}

impl From<usize> for FloatReg {
    fn from(value: usize) -> Self {
        match value {
            0 => FloatReg::XMM0,
            1 => FloatReg::XMM1,
            2 => FloatReg::XMM2,
            3 => FloatReg::XMM3,
            4 => FloatReg::XMM4,
            5 => FloatReg::XMM5,
            6 => FloatReg::XMM6,
            7 => FloatReg::XMM7,
            _ => FloatReg::XMM7,
        }
    }
}

impl fmt::Display for FloatReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            FloatReg::XMM0 => "%xmm0",
            FloatReg::XMM1 => "%xmm1",
            FloatReg::XMM2 => "%xmm2",
            FloatReg::XMM3 => "%xmm3",
            FloatReg::XMM4 => "%xmm4",
            FloatReg::XMM5 => "%xmm5",
            FloatReg::XMM6 => "%xmm6",
            FloatReg::XMM7 => "%xmm7",
        })
    }
}

pub(crate) const ARGUMENT_REGISTERS: [Reg; 6] =
    [Reg::DI, Reg::SI, Reg::DX, Reg::CX, Reg::R8, Reg::R9];

impl Reg {
    pub(crate) fn reg_name_for_type(
        &self,
        ty: &Type,
    ) -> anyhow::Result<&'static str, CodegenError> {
        match ty {
            Type::Char | Type::SChar | Type::UChar => Ok(self.reg_name8()),
            Type::Short | Type::UShort => Ok(self.reg_name16()),
            Type::Int | Type::UInt => Ok(self.reg_name32()),
            Type::Long | Type::ULong | Type::Void | Type::Pointer(_) => Ok(self.reg_name64()),
            Type::Double
            | Type::FunType(_, _)
            | Type::Array(_, _)
            | Type::IncompleteArray(_)
            | Type::Struct(_) => Err(CodegenError::GprRequestedFor(ty.clone())),
        }
    }

    pub(crate) fn reg_name64(&self) -> &'static str {
        match self {
            Reg::AX => "%rax",
            Reg::CX => "%rcx",
            Reg::DX => "%rdx",
            Reg::DI => "%rdi",
            Reg::SI => "%rsi",
            Reg::R8 => "%r8",
            Reg::R9 => "%r9",
            Reg::R10 => "%r10",
            Reg::R11 => "%r11",
        }
    }

    pub(crate) fn reg_name32(&self) -> &'static str {
        match self {
            Reg::AX => "%eax",
            Reg::CX => "%ecx",
            Reg::DX => "%edx",
            Reg::DI => "%edi",
            Reg::SI => "%esi",
            Reg::R8 => "%r8d",
            Reg::R9 => "%r9d",
            Reg::R10 => "%r10d",
            Reg::R11 => "%r11d",
        }
    }

    pub(crate) fn reg_name16(&self) -> &'static str {
        match self {
            Reg::AX => "%ax",
            Reg::CX => "%cx",
            Reg::DX => "%dx",
            Reg::DI => "%di",
            Reg::SI => "%si",
            Reg::R8 => "%r8w",
            Reg::R9 => "%r9w",
            Reg::R10 => "%r10w",
            Reg::R11 => "%r11w",
        }
    }

    pub(crate) fn reg_name8(&self) -> &'static str {
        match self {
            Reg::AX => "%al",
            Reg::CX => "%cl",
            Reg::DX => "%dl",
            Reg::DI => "%dil",
            Reg::SI => "%sil",
            Reg::R8 => "%r8b",
            Reg::R9 => "%r9b",
            Reg::R10 => "%r10b",
            Reg::R11 => "%r11b",
        }
    }
}

pub(crate) fn width_suffix(w: Width) -> char {
    match w {
        Width::W8 => 'b',
        Width::W16 => 'w',
        Width::W32 => 'l',
        Width::W64 => 'q',
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Ext {
    Unknown,
    Zero,
    Sign,
}
