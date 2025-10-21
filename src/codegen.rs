use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryFrom;
use std::fmt::{self, Write as FmtWrite};
use std::io::Write;

use crate::parse::{Const, Type};
use crate::tacky::{
    BinaryOp, Function, Instruction, Program as TackyProgram, StaticConstant, StaticInit,
    StaticVariable, TopLevel, UnaryOp, Value,
};

#[derive(thiserror::Error, Debug)]
pub enum CodegenError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

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

    #[error("type has no size: {0:?}")]
    TypeHasNoSize(Type),
    #[error("type has no alignment: {0:?}")]
    TypeHasNoAlignment(Type),
    #[error("array size exceeds i64: {0}")]
    ArraySizeTooLarge(u128),

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

type Result<T = ()> = std::result::Result<T, CodegenError>;

impl Type {
    fn type_size(&self) -> Result<i64> {
        let sz = match self {
            Type::Void => 0,
            Type::Char | Type::SChar | Type::UChar => 1,
            Type::Int | Type::UInt => 4,
            Type::Long | Type::ULong | Type::Double | Type::Pointer(_) => 8,
            Type::FunType(_, _) | Type::IncompleteArray(_) => {
                return Err(CodegenError::TypeHasNoSize(self.clone()));
            }
            Type::Array(inner, len) => {
                let len_i64 = i64::try_from(*len)
                    .map_err(|_| CodegenError::ArraySizeTooLarge((*len) as u128))?;
                len_i64 * inner.type_size()?
            }
        };
        Ok(sz)
    }

    fn type_align(&self) -> Result<i64> {
        let a = match self {
            Type::Char | Type::SChar | Type::UChar | Type::Void => 1,
            Type::Int | Type::UInt => 4,
            Type::Long | Type::ULong | Type::Double | Type::Pointer(_) => 8,
            Type::FunType(_, _) | Type::IncompleteArray(_) => {
                return Err(CodegenError::TypeHasNoAlignment(self.clone()));
            }
            Type::Array(inner, _) => inner.type_align()?,
        };
        Ok(a)
    }
}

pub struct Codegen<W: Write> {
    pub buf: W,
    stack_map: BTreeMap<String, i64>,
    frame_size: i64,
    global_types: BTreeMap<String, Type>,
    value_types: BTreeMap<String, Type>,
}

#[derive(Debug, Clone, Copy)]
enum Reg {
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

#[derive(Debug, Clone, Copy)]
enum FloatReg {
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
    const COUNT: usize = 8;
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

const ARGUMENT_REGISTERS: [Reg; 6] = [Reg::DI, Reg::SI, Reg::DX, Reg::CX, Reg::R8, Reg::R9];

impl<W: Write> Codegen<W> {
    pub fn new(buf: W) -> Self {
        Self {
            buf,
            stack_map: BTreeMap::new(),
            frame_size: 0,
            global_types: BTreeMap::new(),
            value_types: BTreeMap::new(),
        }
    }

    pub fn lower(&mut self, program: &TackyProgram) -> Result<()> {
        self.global_types = program.global_types.clone();
        for item in &program.items {
            match item {
                TopLevel::StaticVariable(var) => self.emit_static_variable(var)?,
                TopLevel::StaticConstant(constant) => self.emit_static_constant(constant)?,
                TopLevel::Function(function) => self.emit_function(function)?,
            }
        }
        Ok(())
    }

    fn emit_static_variable(&mut self, var: &StaticVariable) -> Result<()> {
        if var.init.len() > 1 {
            return Err(CodegenError::CompoundStaticInitializerUnsupported);
        }

        let init_entry = var.init.first();

        let is_zero_init = match init_entry {
            None => true,
            Some(StaticInit::Scalar { value, .. }) => Self::const_is_zero(value),
            Some(StaticInit::Bytes { value, .. }) => value.iter().all(|&b| b == 0),
            Some(StaticInit::Label { .. }) => false,
        };

        if is_zero_init {
            writeln!(self.buf, ".bss")?;
        } else {
            writeln!(self.buf, ".data")?;
        }
        if var.global {
            writeln!(self.buf, ".globl {}", var.name)?;
        }
        let align = var.ty.type_align()?;
        writeln!(self.buf, ".align {}", align)?;
        writeln!(self.buf, "{}:", var.name)?;
        let size = var.ty.type_size()?;
        match init_entry {
            None => writeln!(self.buf, "  .zero {}", size)?,
            Some(init) => self.emit_static_init_for_type(&var.ty, init)?,
        }
        Ok(())
    }

    fn emit_scalar_static(&mut self, ty: &Type, value: &Const) -> Result<()> {
        match (ty, value) {
            (Type::Int, Const::Int(v)) => {
                writeln!(self.buf, "  .long {}", v)?;
            }
            (Type::UInt, Const::UInt(v)) => {
                writeln!(self.buf, "  .long {}", v)?;
            }
            (Type::Long, Const::Long(v)) | (Type::Pointer(_), Const::Long(v)) => {
                writeln!(self.buf, "  .quad {}", v)?;
            }
            (Type::ULong, Const::ULong(v)) | (Type::Pointer(_), Const::ULong(v)) => {
                writeln!(self.buf, "  .quad {}", v)?;
            }
            (Type::Double, Const::Double(v)) => {
                writeln!(self.buf, "  .quad {}", v.to_bits())?;
            }
            (Type::Char | Type::SChar, Const::Char(v)) => {
                writeln!(self.buf, "  .byte {}", v)?;
            }
            (Type::UChar, Const::UChar(v)) => {
                writeln!(self.buf, "  .byte {}", v)?;
            }
            _ => {
                return Err(CodegenError::UnsupportedScalarStaticInitializer(
                    ty.clone(),
                    value.clone(),
                ));
            }
        }
        Ok(())
    }

    fn emit_zero(&mut self, offset: i64) -> Result<()> {
        if offset != 0 {
            writeln!(self.buf, "  .zero {}", offset)?;
        }
        Ok(())
    }

    fn emit_static_init_for_type(&mut self, ty: &Type, init: &StaticInit) -> Result<()> {
        match init {
            StaticInit::Scalar { offset, value } => {
                self.emit_zero(*offset)?;
                self.emit_scalar_static(ty, value)
            }
            StaticInit::Bytes {
                offset,
                value,
                null_terminated,
            } => {
                self.emit_zero(*offset)?;
                if !matches!(ty, Type::Array(_, _)) {
                    return Err(CodegenError::ByteInitializerOnlyForArray(ty.clone()));
                }
                self.emit_bytes_directive(value, *null_terminated)
            }
            StaticInit::Label { offset, symbol } => {
                self.emit_zero(*offset)?;
                if !matches!(ty, Type::Pointer(_)) {
                    return Err(CodegenError::LabelInitializerRequiresPointer(ty.clone()));
                }
                writeln!(self.buf, "  .quad {}", symbol)?;
                Ok(())
            }
        }
    }

    fn const_is_zero(value: &Const) -> bool {
        match value {
            Const::Char(v) => *v == 0,
            Const::Int(v) => *v == 0,
            Const::Long(v) => *v == 0,
            Const::UChar(v) => *v == 0,
            Const::UInt(v) => *v == 0,
            Const::ULong(v) => *v == 0,
            Const::Double(v) => *v == 0.0,
        }
    }

    fn emit_bytes_directive(&mut self, bytes: &[u8], null_terminated: bool) -> Result<()> {
        if null_terminated && bytes.is_empty() {
            writeln!(self.buf, "  .asciz \"\"")?;
            return Ok(());
        }
        if !null_terminated && bytes.is_empty() {
            writeln!(self.buf, "  .ascii \"\"")?;
            return Ok(());
        }

        if null_terminated && bytes.last() != Some(&0) {
            return Err(CodegenError::MissingTrailingNullInAsciz);
        }

        let slice = if null_terminated {
            &bytes[..bytes.len().saturating_sub(1)]
        } else {
            bytes
        };

        let escaped = Self::escape_bytes(slice);
        let directive = if null_terminated { ".asciz" } else { ".ascii" };
        writeln!(self.buf, "  {} \"{}\"", directive, escaped)?;
        Ok(())
    }

    fn escape_bytes(bytes: &[u8]) -> String {
        let mut escaped = String::new();
        for &b in bytes {
            match b {
                b'\n' => escaped.push_str("\\n"),
                b'\r' => escaped.push_str("\\r"),
                b'\t' => escaped.push_str("\\t"),
                b'\0' => escaped.push_str("\\0"),
                b'\\' => escaped.push_str("\\\\"),
                b'"' => escaped.push_str("\\\""),
                b'\x07' => escaped.push_str("\\a"),
                b'\x08' => escaped.push_str("\\b"),
                b'\x0b' => escaped.push_str("\\v"),
                b'\x0c' => escaped.push_str("\\f"),
                0x20..=0x7e => escaped.push(char::from(b)),
                _ => {
                    let _ = write!(escaped, "\\x{:02x}", b);
                }
            }
        }
        escaped
    }

    fn emit_static_constant(&mut self, constant: &StaticConstant) -> Result<()> {
        writeln!(self.buf, ".data")?;
        let align = constant.ty.type_align()?;
        writeln!(self.buf, ".align {}", align)?;
        writeln!(self.buf, "{}:", constant.name)?;
        self.emit_static_init_for_type(&constant.ty, &constant.init)
    }

    fn emit_function(&mut self, function: &Function) -> Result<()> {
        self.stack_map.clear();
        self.frame_size = 0;
        self.value_types = function.value_types.clone();

        self.collect_stack_slots(function)?;

        self.emit_prologue(function)?;
        if self.frame_size > 0 {
            writeln!(self.buf, "  sub ${}, %rsp", self.frame_size)?;
        }
        self.move_params_to_stack(function)?;

        for instr in &function.instructions {
            self.emit_instruction(instr)?;
        }
        let ty = &function.return_type;
        match ty {
            Type::Char | Type::SChar | Type::UChar => {
                return Err(CodegenError::UnsupportedFunctionReturnType(ty.clone()));
            }
            Type::Int | Type::UInt => writeln!(
                self.buf,
                "  {} $0, {}",
                self.mov_instr(ty)?,
                Reg::AX.reg_name32()
            )?,
            Type::Long | Type::ULong | Type::Pointer(_) => writeln!(
                self.buf,
                "  {} $0, {}",
                self.mov_instr(ty)?,
                Reg::AX.reg_name64()
            )?,
            Type::Double => writeln!(self.buf, "  xorpd {}, {}", FloatReg::XMM0, FloatReg::XMM0)?,
            Type::Void => {}
            Type::FunType(_, _) => {
                return Err(CodegenError::UnsupportedFunctionReturnType(ty.clone()));
            }
            Type::Array(_, _) | Type::IncompleteArray(_) => {
                return Err(CodegenError::ArrayReturnTypeUnsupported);
            }
        }
        let result = self.emit_epilogue();
        self.value_types.clear();
        result
    }

    fn collect_stack_slots(&mut self, function: &Function) -> Result<()> {
        let mut vars = BTreeSet::new();
        for param in &function.params {
            vars.insert(param.clone());
        }

        for instr in &function.instructions {
            match instr {
                Instruction::Return(val) => Self::collect_value(&mut vars, val),
                Instruction::Unary { src, dst, .. } => {
                    Self::collect_value(&mut vars, src);
                    Self::collect_value(&mut vars, dst);
                }
                Instruction::Binary {
                    src1, src2, dst, ..
                } => {
                    Self::collect_value(&mut vars, src1);
                    Self::collect_value(&mut vars, src2);
                    Self::collect_value(&mut vars, dst);
                }
                Instruction::Copy { src, dst } => {
                    Self::collect_value(&mut vars, src);
                    Self::collect_value(&mut vars, dst);
                }
                Instruction::GetAddress { src, dst } => {
                    Self::collect_value(&mut vars, src);
                    Self::collect_value(&mut vars, dst);
                }
                Instruction::Load { src_ptr, dst } => {
                    Self::collect_value(&mut vars, src_ptr);
                    Self::collect_value(&mut vars, dst);
                }
                Instruction::Store { src, dst_ptr } => {
                    Self::collect_value(&mut vars, src);
                    Self::collect_value(&mut vars, dst_ptr);
                }
                Instruction::AddPtr {
                    ptr, index, dst, ..
                } => {
                    Self::collect_value(&mut vars, ptr);
                    Self::collect_value(&mut vars, index);
                    Self::collect_value(&mut vars, dst);
                }
                Instruction::CopyToOffset { src, dst, .. } => {
                    Self::collect_value(&mut vars, src);
                    if self.value_types.contains_key(dst) {
                        vars.insert(dst.clone());
                    }
                }
                Instruction::FunCall { args, dst, .. } => {
                    for arg in args {
                        Self::collect_value(&mut vars, arg);
                    }
                    Self::collect_value(&mut vars, dst);
                }
                Instruction::SignExtend { src, dst }
                | Instruction::ZeroExtend { src, dst }
                | Instruction::Truncate { src, dst }
                | Instruction::Convert { src, dst, .. } => {
                    Self::collect_value(&mut vars, src);
                    Self::collect_value(&mut vars, dst);
                }
                Instruction::Jump(_) | Instruction::Label(_) => {}
                Instruction::JumpIfZero { condition, .. }
                | Instruction::JumpIfNotZero { condition, .. } => {
                    Self::collect_value(&mut vars, condition);
                }
            }
        }

        let mut offset = 0;
        for name in vars {
            let ty = self.value_types.get(&name).cloned().ok_or_else(|| {
                CodegenError::MissingTypeInfoForValue(format!(
                    "{name}: Values: {:?}",
                    self.value_types
                ))
            })?;
            let size = ty.type_size()?;
            let align = ty.type_align()?;
            if offset % align != 0 {
                offset += align - (offset % align);
            }
            offset += size;
            self.stack_map.insert(name, -offset);
        }
        let mut frame_size = offset;
        if frame_size % 16 != 0 {
            frame_size += 16 - (frame_size % 16);
        }
        self.frame_size = frame_size;
        Ok(())
    }

    fn collect_value(vars: &mut BTreeSet<String>, value: &Value) {
        if let Value::Var(name) = value {
            vars.insert(name.clone());
        }
    }

    fn move_params_to_stack(&mut self, function: &Function) -> Result<()> {
        let mut int_reg_idx = 0usize;
        let mut float_reg_idx = 0usize;
        let mut stack_arg_idx = 0i64;

        for param in &function.params {
            let ty = self
                .value_types
                .get(param)
                .cloned()
                .ok_or_else(|| CodegenError::MissingTypeInfoForParam(param.clone()))?;
            let dest = self.stack_operand(param)?;

            if ty == Type::Double {
                if float_reg_idx < FloatReg::COUNT {
                    let src = FloatReg::from(float_reg_idx);
                    float_reg_idx += 1;
                    writeln!(self.buf, "  movsd {}, {}", src, dest)?;
                } else {
                    let offset = 16 + stack_arg_idx * 8;
                    stack_arg_idx += 1;
                    writeln!(self.buf, "  movsd {}(%rbp), %xmm0", offset)?;
                    writeln!(self.buf, "  movsd %xmm0, {}", dest)?;
                }
            } else if int_reg_idx < ARGUMENT_REGISTERS.len() {
                let reg = ARGUMENT_REGISTERS[int_reg_idx];
                int_reg_idx += 1;
                let reg_name = reg.reg_name_for_type(&ty)?;
                writeln!(
                    self.buf,
                    "  {} {}, {}",
                    self.mov_instr(&ty)?,
                    reg_name,
                    dest
                )?;
            } else {
                let offset = 16 + stack_arg_idx * 8;
                stack_arg_idx += 1;
                let temp_reg = Reg::R10.reg_name_for_type(&ty)?;
                writeln!(
                    self.buf,
                    "  {} {}(%rbp), {}",
                    self.mov_instr(&ty)?,
                    offset,
                    temp_reg
                )?;
                writeln!(
                    self.buf,
                    "  {} {}, {}",
                    self.mov_instr(&ty)?,
                    temp_reg,
                    dest
                )?;
            }
        }
        Ok(())
    }

    fn emit_instruction(&mut self, instr: &Instruction) -> Result<()> {
        match instr {
            Instruction::Return(value) => {
                let ty = self.value_type(value)?;
                match ty {
                    Type::Double => {
                        self.load_value_into_xmm(value, 0)?;
                        self.emit_epilogue()
                    }
                    Type::Void => self.emit_epilogue(),
                    _ => {
                        self.load_value_into_reg(value, Reg::AX)?;
                        self.emit_epilogue()
                    }
                }
            }
            Instruction::Copy { src, dst } => self.emit_copy(src, dst),
            Instruction::GetAddress { src, dst } => self.emit_get_address(src, dst),
            Instruction::Load { src_ptr, dst } => self.emit_load(src_ptr, dst),
            Instruction::Store { src, dst_ptr } => self.emit_store(src, dst_ptr),
            Instruction::FunCall { name, args, dst } => self.emit_fun_call(name, args, dst),
            Instruction::Unary { op, src, dst } => self.emit_unary(*op, src, dst),
            Instruction::Binary {
                op,
                src1,
                src2,
                dst,
            } => self.emit_binary(*op, src1, src2, dst),
            Instruction::AddPtr {
                ptr,
                index,
                scale,
                dst,
            } => self.emit_add_ptr(ptr, index, *scale, dst),
            Instruction::CopyToOffset { src, dst, offset } => {
                self.emit_copy_to_offset(src, dst, *offset)
            }
            Instruction::SignExtend { src, dst } => self.emit_sign_extend(src, dst),
            Instruction::ZeroExtend { src, dst } => self.emit_zero_extend(src, dst),
            Instruction::Truncate { src, dst } => self.emit_truncate(src, dst),
            Instruction::Convert { src, dst, from, to } => self.emit_convert(src, dst, from, to),
            Instruction::Jump(label) => {
                writeln!(self.buf, "  jmp {}", label)?;
                Ok(())
            }
            Instruction::JumpIfZero { condition, target } => {
                self.load_value_into_reg(condition, Reg::AX)?;
                let ty = self.value_type(condition)?;
                let reg_name = Reg::AX.reg_name_for_type(&ty)?;
                writeln!(self.buf, "  cmp $0, {}", reg_name)?;
                writeln!(self.buf, "  je {}", target)?;
                Ok(())
            }
            Instruction::JumpIfNotZero { condition, target } => {
                self.load_value_into_reg(condition, Reg::AX)?;
                let ty = self.value_type(condition)?;
                let reg_name = Reg::AX.reg_name_for_type(&ty)?;
                writeln!(self.buf, "  cmp $0, {}", reg_name)?;
                writeln!(self.buf, "  jne {}", target)?;
                Ok(())
            }
            Instruction::Label(name) => {
                writeln!(self.buf, "{}:", name)?;
                Ok(())
            }
        }
    }

    fn emit_copy(&mut self, src: &Value, dst: &Value) -> Result<()> {
        if matches!((src, dst), (Value::Var(a), Value::Var(b)) if a == b)
            || matches!((src, dst), (Value::Global(a), Value::Global(b)) if a == b)
        {
            return Ok(());
        }

        if matches!(dst, Value::Constant(_)) {
            return Err(CodegenError::CopyDestCannotBeConstant);
        }

        let ty = self.value_type(dst)?;
        if ty == Type::Double {
            self.load_value_into_xmm(src, 0)?;
            return self.store_xmm_into_value(0, dst);
        }

        self.load_value_into_reg(src, Reg::R11)?;
        self.store_reg_into_value(Reg::R11, dst)
    }

    fn emit_get_address(&mut self, src: &Value, dst: &Value) -> Result<()> {
        if matches!(dst, Value::Constant(_)) {
            return Err(CodegenError::AddressDestCannotBeConstant);
        }

        let reg = Reg::R11;
        match src {
            Value::Var(name) => {
                let operand = self.stack_operand(name)?;
                writeln!(self.buf, "  leaq {}, {}", operand, reg.reg_name64())?;
            }
            Value::Global(name) => {
                writeln!(self.buf, "  leaq {}(%rip), {}", name, reg.reg_name64())?;
            }
            Value::Constant(_) => return Err(CodegenError::CannotTakeAddressOfConstant),
        }

        self.store_reg_into_value(reg, dst)
    }

    fn emit_load(&mut self, src_ptr: &Value, dst: &Value) -> Result<()> {
        let dst_ty = self.value_type(dst)?;
        self.load_value_into_reg(src_ptr, Reg::R11)?;
        match dst_ty {
            Type::Double => {
                writeln!(
                    self.buf,
                    "  movsd ({}), {}",
                    Reg::R11.reg_name64(),
                    Self::xmm_name(0)?
                )?;
                self.store_xmm_into_value(0, dst)
            }
            _ => {
                let reg = Reg::R10;
                writeln!(
                    self.buf,
                    "  {} ({}), {}",
                    self.mov_instr(&dst_ty)?,
                    Reg::R11.reg_name64(),
                    reg.reg_name_for_type(&dst_ty)?
                )?;
                self.store_reg_into_value(reg, dst)
            }
        }
    }

    fn emit_store(&mut self, src: &Value, dst_ptr: &Value) -> Result<()> {
        let src_ty = self.value_type(src)?;
        self.load_value_into_reg(dst_ptr, Reg::R11)?;
        match src_ty {
            Type::Double => {
                self.load_value_into_xmm(src, 0)?;
                writeln!(
                    self.buf,
                    "  {} {}, ({})",
                    self.mov_instr(&src_ty)?,
                    Self::xmm_name(0)?,
                    Reg::R11.reg_name64()
                )?;
                Ok(())
            }
            _ => {
                self.load_value_into_reg(src, Reg::R10)?;
                writeln!(
                    self.buf,
                    "  {} {}, ({})",
                    self.mov_instr(&src_ty)?,
                    Reg::R10.reg_name_for_type(&src_ty)?,
                    Reg::R11.reg_name64()
                )?;
                Ok(())
            }
        }
    }

    fn emit_unary(&mut self, op: UnaryOp, src: &Value, dst: &Value) -> Result<()> {
        let ty = self.value_type(dst)?;
        if ty == Type::Double {
            match op {
                UnaryOp::Negate => {
                    self.load_value_into_xmm(src, 0)?;
                    writeln!(
                        self.buf,
                        "  xorpd {}, {}",
                        Self::xmm_name(1)?,
                        Self::xmm_name(1)?
                    )?;
                    writeln!(
                        self.buf,
                        "  subsd {}, {}",
                        Self::xmm_name(0)?,
                        Self::xmm_name(1)?
                    )?;
                    return self.store_xmm_into_value(1, dst);
                }
                UnaryOp::Complement | UnaryOp::Not => {
                    return Err(CodegenError::InvalidUnaryOpForType(op, ty));
                }
            }
        }

        self.load_value_into_reg(src, Reg::AX)?;

        match op {
            UnaryOp::Negate => writeln!(self.buf, "  neg {}", Reg::AX.reg_name_for_type(&ty)?)?,
            UnaryOp::Complement => writeln!(self.buf, "  not {}", Reg::AX.reg_name_for_type(&ty)?)?,
            UnaryOp::Not => {
                let reg_name = Reg::AX.reg_name_for_type(&ty)?;
                writeln!(self.buf, "  cmp $0, {}", reg_name)?;
                writeln!(self.buf, "  sete {}", Reg::AX.reg_name8())?;
                writeln!(
                    self.buf,
                    "  movzbq {}, {}",
                    Reg::AX.reg_name8(),
                    Reg::AX.reg_name64()
                )?;
            }
        }

        self.store_reg_into_value(Reg::AX, dst)
    }

    fn emit_binary(&mut self, op: BinaryOp, src1: &Value, src2: &Value, dst: &Value) -> Result<()> {
        let result_ty = self.value_type(dst)?;
        let lhs_ty = self.value_type(src1)?;
        let rhs_ty = self.value_type(src2)?;

        if result_ty == Type::Double {
            self.load_value_into_xmm(src1, 0)?;
            self.load_value_into_xmm(src2, 1)?;
            let instr = match op {
                BinaryOp::Add => "addsd",
                BinaryOp::Subtract => "subsd",
                BinaryOp::Multiply => "mulsd",
                BinaryOp::Divide => "divsd",
                _ => return Err(CodegenError::InvalidBinaryOpForTypes(op, lhs_ty, rhs_ty)),
            };
            writeln!(
                self.buf,
                "  {} {}, {}",
                instr,
                Self::xmm_name(1)?,
                Self::xmm_name(0)?
            )?;
            return self.store_xmm_into_value(0, dst);
        }

        if lhs_ty == Type::Double || rhs_ty == Type::Double {
            match op {
                BinaryOp::Equal
                | BinaryOp::NotEqual
                | BinaryOp::LessThan
                | BinaryOp::LessOrEqual
                | BinaryOp::GreaterThan
                | BinaryOp::GreaterOrEqual => {
                    self.load_value_into_xmm(src1, 0)?;
                    self.load_value_into_xmm(src2, 1)?;
                    writeln!(
                        self.buf,
                        "  ucomisd {}, {}",
                        Self::xmm_name(1)?,
                        Self::xmm_name(0)?
                    )?;

                    let set_instr = match op {
                        BinaryOp::Equal => "sete",
                        BinaryOp::NotEqual => "setne",
                        BinaryOp::LessThan => "setb",
                        BinaryOp::LessOrEqual => "setbe",
                        BinaryOp::GreaterThan => "seta",
                        BinaryOp::GreaterOrEqual => "setae",
                        _ => unreachable!(),
                    };

                    writeln!(self.buf, "  {} {}", set_instr, Reg::AX.reg_name8())?;
                    writeln!(
                        self.buf,
                        "  movzbq {}, {}",
                        Reg::AX.reg_name8(),
                        Reg::AX.reg_name64()
                    )?;
                    return self.store_reg_into_value(Reg::AX, dst);
                }
                _ => return Err(CodegenError::InvalidBinaryOpForTypes(op, lhs_ty, rhs_ty)),
            }
        }

        match op {
            BinaryOp::Divide | BinaryOp::Remainder => {
                let ty = result_ty;
                self.load_value_into_reg(src1, Reg::AX)?;
                self.load_value_into_reg(src2, Reg::R10)?;
                match ty {
                    Type::Char | Type::SChar | Type::UChar => {
                        return Err(CodegenError::DivisionUnsupportedForType(ty));
                    }
                    Type::Int => {
                        writeln!(self.buf, "  cltd")?;
                        writeln!(
                            self.buf,
                            "  idiv {}",
                            Reg::R10.reg_name_for_type(&Type::Int)?
                        )?;
                    }
                    Type::Long => {
                        writeln!(self.buf, "  cqo")?;
                        writeln!(
                            self.buf,
                            "  idiv {}",
                            Reg::R10.reg_name_for_type(&Type::Long)?
                        )?;
                    }
                    Type::UInt => {
                        writeln!(
                            self.buf,
                            "  xor {}, {}",
                            Reg::DX.reg_name32(),
                            Reg::DX.reg_name32(),
                        )?;
                        writeln!(
                            self.buf,
                            "  div {}",
                            Reg::R10.reg_name_for_type(&Type::UInt)?
                        )?;
                    }
                    Type::ULong => {
                        writeln!(
                            self.buf,
                            "  xor {}, {}",
                            Reg::DX.reg_name64(),
                            Reg::DX.reg_name64(),
                        )?;
                        writeln!(
                            self.buf,
                            "  div {}",
                            Reg::R10.reg_name_for_type(&Type::ULong)?
                        )?;
                    }
                    Type::Void
                    | Type::Array(_, _)
                    | Type::IncompleteArray(_)
                    | Type::Double
                    | Type::Pointer(_)
                    | Type::FunType(_, _) => {
                        return Err(CodegenError::DivisionUnsupportedForType(ty));
                    }
                }

                match op {
                    BinaryOp::Divide => self.store_reg_into_value(Reg::AX, dst),
                    BinaryOp::Remainder => self.store_reg_into_value(Reg::DX, dst),
                    _ => unreachable!(),
                }
            }
            BinaryOp::Add
            | BinaryOp::Subtract
            | BinaryOp::Multiply
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor => {
                let ty = self.value_type(dst)?;
                self.load_value_into_reg(src1, Reg::AX)?;
                self.load_value_into_reg(src2, Reg::R10)?;

                let op_str = match op {
                    BinaryOp::Add => "add",
                    BinaryOp::Subtract => "sub",
                    BinaryOp::Multiply => "imul",
                    BinaryOp::BitAnd => "and",
                    BinaryOp::BitOr => "or",
                    BinaryOp::BitXor => "xor",
                    _ => unreachable!(),
                };

                writeln!(
                    self.buf,
                    "  {} {}, {}",
                    op_str,
                    Reg::R10.reg_name_for_type(&ty)?,
                    Reg::AX.reg_name_for_type(&ty)?
                )?;
                self.store_reg_into_value(Reg::AX, dst)
            }
            BinaryOp::LeftShift | BinaryOp::RightShift => {
                let ty = self.value_type(dst)?;
                self.load_value_into_reg(src1, Reg::AX)?;
                self.load_value_into_reg(src2, Reg::CX)?;

                let op_str = match op {
                    BinaryOp::LeftShift => "shl",
                    BinaryOp::RightShift => {
                        if ty.is_unsigned() {
                            "shr"
                        } else {
                            "sar"
                        }
                    }
                    _ => unreachable!(),
                };

                writeln!(
                    self.buf,
                    "  {} {}, {}",
                    op_str,
                    Reg::CX.reg_name8(),
                    Reg::AX.reg_name_for_type(&ty)?
                )?;

                self.store_reg_into_value(Reg::AX, dst)
            }
            BinaryOp::Equal
            | BinaryOp::NotEqual
            | BinaryOp::LessThan
            | BinaryOp::LessOrEqual
            | BinaryOp::GreaterThan
            | BinaryOp::GreaterOrEqual => {
                let ty = self.value_type(src1)?;
                self.load_value_into_reg(src1, Reg::AX)?;
                self.load_value_into_reg(src2, Reg::R10)?;
                writeln!(
                    self.buf,
                    "  cmp {}, {}",
                    Reg::R10.reg_name_for_type(&ty)?,
                    Reg::AX.reg_name_for_type(&ty)?
                )?;

                let is_unsigned = ty.is_unsigned();
                let set_instr = match op {
                    BinaryOp::Equal => "sete",
                    BinaryOp::NotEqual => "setne",
                    BinaryOp::LessThan => {
                        if is_unsigned {
                            "setb"
                        } else {
                            "setl"
                        }
                    }
                    BinaryOp::LessOrEqual => {
                        if is_unsigned {
                            "setbe"
                        } else {
                            "setle"
                        }
                    }
                    BinaryOp::GreaterThan => {
                        if is_unsigned {
                            "seta"
                        } else {
                            "setg"
                        }
                    }
                    BinaryOp::GreaterOrEqual => {
                        if is_unsigned {
                            "setae"
                        } else {
                            "setge"
                        }
                    }
                    _ => unreachable!(),
                };

                writeln!(self.buf, "  {} {}", set_instr, Reg::AX.reg_name8())?;
                writeln!(
                    self.buf,
                    "  movzbq {}, {}",
                    Reg::AX.reg_name8(),
                    Reg::AX.reg_name64(),
                )?;

                self.store_reg_into_value(Reg::AX, dst)
            }
        }
    }

    fn emit_add_ptr(&mut self, ptr: &Value, index: &Value, scale: i64, dst: &Value) -> Result<()> {
        if matches!(dst, Value::Constant(_)) {
            return Err(CodegenError::AddPtrDestCannotBeConstant);
        }

        if !matches!(self.value_type(dst)?, Type::Pointer(_)) {
            return Err(CodegenError::AddPtrDestMustBePointer(self.value_type(dst)?));
        }

        self.load_value_into_reg(ptr, Reg::R11)?;
        self.load_value_into_reg(index, Reg::R10)?;

        let base = Reg::R11.reg_name64();
        let idx = Reg::R10.reg_name64();

        if matches!(scale, 1 | 2 | 4 | 8) {
            if scale == 1 {
                writeln!(self.buf, "  leaq ({},{}), {}", base, idx, base)?;
            } else {
                writeln!(self.buf, "  leaq ({},{},{}), {}", base, idx, scale, base)?;
            }
        } else {
            writeln!(self.buf, "  imul ${}, {}", scale, Reg::R10.reg_name64())?;
            writeln!(self.buf, "  add {}, {}", Reg::R10.reg_name64(), base)?;
        }

        self.store_reg_into_value(Reg::R11, dst)
    }

    fn emit_copy_to_offset(&mut self, src: &Value, dst: &str, offset: i64) -> Result<()> {
        let addr = if let Some(base) = self.stack_map.get(dst) {
            format!("{}(%rbp)", base + offset)
        } else {
            self.static_address(dst, offset)
        };
        let ty = self.value_type(src)?;

        match ty {
            Type::Double => {
                self.load_value_into_xmm(src, 0)?;
                writeln!(self.buf, "  movsd {}, {}", Self::xmm_name(0)?, addr)?;
            }
            Type::Char | Type::SChar | Type::UChar => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(self.buf, "  movb {}, {}", Reg::R11.reg_name8(), addr)?;
            }
            Type::Int | Type::UInt => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(self.buf, "  movl {}, {}", Reg::R11.reg_name32(), addr)?;
            }
            Type::Long | Type::ULong | Type::Pointer(_) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(self.buf, "  movq {}, {}", Reg::R11.reg_name64(), addr)?;
            }
            Type::Void | Type::FunType(_, _) | Type::Array(_, _) | Type::IncompleteArray(_) => {
                return Err(CodegenError::CopyToOffsetUnsupported(ty));
            }
        }

        Ok(())
    }

    fn static_address(&self, symbol: &str, offset: i64) -> String {
        if offset == 0 {
            format!("{}(%rip)", symbol)
        } else if offset > 0 {
            format!("{}+{}(%rip)", symbol, offset)
        } else {
            format!("{}{}(%rip)", symbol, offset)
        }
    }

    fn emit_sign_extend(&mut self, src: &Value, dst: &Value) -> Result<()> {
        let src_ty = self.value_type(src)?;
        let dst_ty = self.value_type(dst)?;
        match (&src_ty, &dst_ty) {
            (Type::Int, Type::Long) | (Type::Int, Type::ULong) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  movslq {}, {}",
                    Reg::R11.reg_name32(),
                    Reg::R11.reg_name64(),
                )?;
                self.store_reg_into_value(Reg::R11, dst)
            }
            (Type::Char, Type::Int) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  movzx {}, {}",
                    Reg::R11.reg_name8(),
                    Reg::R11.reg_name32(),
                )?;
                self.store_reg_into_value(Reg::R11, dst)
            }
            _ => Err(CodegenError::UnsupportedSignExtend(src_ty, dst_ty)),
        }
    }

    fn emit_zero_extend(&mut self, src: &Value, dst: &Value) -> Result<()> {
        self.load_value_into_reg(src, Reg::R11)?;
        self.store_reg_into_value(Reg::R11, dst)
    }

    fn emit_truncate(&mut self, src: &Value, dst: &Value) -> Result<()> {
        self.load_value_into_reg(src, Reg::R11)?;
        self.store_reg_into_value(Reg::R11, dst)
    }

    fn emit_convert(&mut self, src: &Value, dst: &Value, from: &Type, to: &Type) -> Result<()> {
        match (from, to) {
            (Type::Double, Type::Double) => {
                self.load_value_into_xmm(src, 0)?;
                self.store_xmm_into_value(0, dst)
            }
            (Type::Double, Type::Int) => {
                self.load_value_into_xmm(src, 0)?;
                writeln!(
                    self.buf,
                    "  cvttsd2si {}, {}",
                    Self::xmm_name(0)?,
                    Reg::R11.reg_name32(),
                )?;
                self.store_reg_into_value(Reg::R11, dst)
            }
            (Type::Double, Type::UInt) => {
                self.load_value_into_xmm(src, 0)?;
                writeln!(
                    self.buf,
                    "  cvttsd2siq {}, {}",
                    Self::xmm_name(0)?,
                    Reg::R11.reg_name64(),
                )?;
                self.store_reg_into_value(Reg::R11, dst)
            }
            (Type::Double, Type::Long) | (Type::Double, Type::ULong) => {
                self.load_value_into_xmm(src, 0)?;
                writeln!(
                    self.buf,
                    "  cvttsd2siq {}, {}",
                    Self::xmm_name(0)?,
                    Reg::R11.reg_name64(),
                )?;
                self.store_reg_into_value(Reg::R11, dst)
            }
            (Type::Char | Type::SChar, Type::Double) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  movsbl {}, {}",
                    Reg::R11.reg_name8(),
                    Reg::R11.reg_name32()
                )?;
                writeln!(
                    self.buf,
                    "  cvtsi2sd {}, {}",
                    Reg::R11.reg_name32(),
                    Self::xmm_name(0)?
                )?;
                self.store_xmm_into_value(0, dst)
            }
            (Type::UChar, Type::Double) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  movzbl {}, {}",
                    Reg::R11.reg_name8(),
                    Reg::R11.reg_name32()
                )?;
                writeln!(
                    self.buf,
                    "  cvtsi2sd {}, {}",
                    Reg::R11.reg_name32(),
                    Self::xmm_name(0)?
                )?;
                self.store_xmm_into_value(0, dst)
            }
            (Type::Int, Type::Double) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  cvtsi2sd {}, {}",
                    Reg::R11.reg_name32(),
                    Self::xmm_name(0)?
                )?;
                self.store_xmm_into_value(0, dst)
            }
            (Type::UInt, Type::Double) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  cvtsi2sdq {}, {}",
                    Reg::R11.reg_name64(),
                    Self::xmm_name(0)?
                )?;
                self.store_xmm_into_value(0, dst)
            }
            (Type::Long, Type::Double) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  cvtsi2sdq {}, {}",
                    Reg::R11.reg_name64(),
                    Self::xmm_name(0)?
                )?;
                self.store_xmm_into_value(0, dst)
            }
            (Type::ULong, Type::Double) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  mov {}, {}",
                    Reg::R11.reg_name64(),
                    Reg::AX.reg_name64(),
                )?;
                writeln!(self.buf, "  shr $1, {}", Reg::AX.reg_name64())?;
                writeln!(self.buf, "  and $1, {}", Reg::R11.reg_name64())?;
                writeln!(
                    self.buf,
                    "  cvtsi2sdq {}, {}",
                    Reg::AX.reg_name64(),
                    Self::xmm_name(0)?
                )?;
                writeln!(
                    self.buf,
                    "  addsd {}, {}",
                    Self::xmm_name(0)?,
                    Self::xmm_name(0)?
                )?;
                writeln!(
                    self.buf,
                    "  cvtsi2sdq {}, {}",
                    Reg::R11.reg_name64(),
                    Self::xmm_name(1)?
                )?;
                writeln!(
                    self.buf,
                    "  addsd {}, {}",
                    Self::xmm_name(1)?,
                    Self::xmm_name(0)?
                )?;
                self.store_xmm_into_value(0, dst)
            }
            _ => Err(CodegenError::UnsupportedConversion(
                from.clone(),
                to.clone(),
            )),
        }
    }

    fn emit_fun_call(&mut self, name: &str, args: &[Value], dst: &Value) -> Result<()> {
        let mut int_regs = Vec::new();
        let mut float_regs = Vec::new();
        let mut stack_args: Vec<(Value, Type)> = Vec::new();
        let mut int_reg_idx = 0;
        let mut float_reg_idx = 0;

        for arg in args {
            let ty = self.value_type(arg)?;
            if ty == Type::Double {
                if float_reg_idx < FloatReg::COUNT {
                    float_regs.push((float_reg_idx, arg.clone()));
                    float_reg_idx += 1;
                } else {
                    stack_args.push((arg.clone(), ty));
                }
            } else if int_reg_idx < ARGUMENT_REGISTERS.len() {
                let reg = ARGUMENT_REGISTERS[int_reg_idx];
                int_reg_idx += 1;
                int_regs.push((reg, arg.clone()));
            } else {
                stack_args.push((arg.clone(), ty));
            }
        }

        let mut stack_bytes: i64 = 0;
        if !stack_args.len().is_multiple_of(2) {
            writeln!(self.buf, "  sub $8, %rsp")?;
            writeln!(self.buf, "  movq $0, (%rsp)")?;
            stack_bytes += 8;
        }

        for (value, ty) in stack_args.iter().rev() {
            match ty {
                Type::Double => {
                    self.load_value_into_xmm(value, 0)?;
                    writeln!(self.buf, "  sub $8, %rsp")?;
                    writeln!(self.buf, "  movsd {}, (%rsp)", Self::xmm_name(0)?)?;
                }
                _ => {
                    self.load_value_into_reg(value, Reg::R11)?;
                    writeln!(self.buf, "  push {}", Reg::R11.reg_name64())?;
                }
            }
            stack_bytes += 8;
        }

        for (idx, value) in float_regs {
            if idx >= FloatReg::COUNT {
                return Err(CodegenError::InvalidFloatRegIndex(idx));
            }
            self.load_value_into_xmm(&value, idx)?;
        }

        for (reg, value) in int_regs {
            self.load_value_into_reg(&value, reg)?;
        }

        writeln!(self.buf, "  call {}", name)?;

        if stack_bytes > 0 {
            writeln!(self.buf, "  add ${}, %rsp", stack_bytes)?;
        }

        if let Some(ty) = self.value_type_optional(dst) {
            match ty {
                Type::Void => {}
                Type::Double => {
                    self.store_xmm_into_value(0, dst)?;
                }
                _ => {
                    self.store_reg_into_value(Reg::AX, dst)?;
                }
            }
        }
        Ok(())
    }

    fn emit_prologue(&mut self, function: &Function) -> Result<()> {
        writeln!(self.buf, ".text")?;
        if function.global {
            writeln!(self.buf, ".globl {}", function.name)?;
        }
        writeln!(self.buf, "{}:", function.name)?;
        writeln!(self.buf, "  push %rbp")?;
        writeln!(self.buf, "  mov %rsp, %rbp")?;
        Ok(())
    }

    fn emit_epilogue(&mut self) -> Result<()> {
        writeln!(self.buf, "  mov %rbp, %rsp")?;
        writeln!(self.buf, "  pop %rbp")?;
        writeln!(self.buf, "  ret")?;
        Ok(())
    }

    fn load_value_into_reg(&mut self, value: &Value, reg: Reg) -> Result<()> {
        let ty = self.value_type(value)?;
        if ty == Type::Double {
            return Err(CodegenError::LoadDoubleIntoGpr);
        }
        match value {
            Value::Constant(c @ Const::Char(_) | c @ Const::UChar(_)) => {
                let (imm, ty) = match c {
                    Const::Char(n) => (*n as u8, Type::Char),
                    Const::UChar(n) => (*n, Type::UChar),
                    _ => unreachable!(),
                };
                writeln!(self.buf, "  movb ${}, {}", imm, reg.reg_name_for_type(&ty)?)?;
            }
            Value::Constant(Const::Int(n)) => {
                writeln!(
                    self.buf,
                    "  movl ${}, {}",
                    n,
                    reg.reg_name_for_type(&Type::Int)?
                )?;
            }
            Value::Constant(Const::UInt(n)) => {
                writeln!(
                    self.buf,
                    "  movl ${}, {}",
                    n,
                    reg.reg_name_for_type(&Type::UInt)?
                )?;
            }
            Value::Constant(Const::Long(n)) => {
                writeln!(self.buf, "  movq ${}, {}", n, reg.reg_name64())?;
            }
            Value::Constant(Const::ULong(n)) => {
                writeln!(self.buf, "  movq ${}, {}", n, reg.reg_name64())?;
            }
            Value::Constant(Const::Double(_)) => return Err(CodegenError::LoadDoubleIntoGpr),
            Value::Var(name) => {
                let operand = self.stack_operand(name)?;
                writeln!(
                    self.buf,
                    "  {} {}, {}",
                    self.mov_instr(&ty)?,
                    operand,
                    reg.reg_name_for_type(&ty)?
                )?;
            }
            Value::Global(name) => {
                writeln!(
                    self.buf,
                    "  {} {}(%rip), {}",
                    self.mov_instr(&ty)?,
                    name,
                    reg.reg_name_for_type(&ty)?
                )?;
            }
        }
        Ok(())
    }

    fn store_reg_into_value(&mut self, reg: Reg, value: &Value) -> Result<()> {
        let ty = self.value_type(value)?;
        let reg_name = reg.reg_name_for_type(&ty)?;
        match value {
            Value::Var(name) => {
                let operand = self.stack_operand(name)?;
                writeln!(
                    self.buf,
                    "  {} {}, {}",
                    self.mov_instr(&ty)?,
                    reg_name,
                    operand
                )?;
            }
            Value::Global(name) => {
                writeln!(
                    self.buf,
                    "  {} {}, {}(%rip)",
                    self.mov_instr(&ty)?,
                    reg_name,
                    name
                )?;
            }
            Value::Constant(Const::Double(_)) => return Err(CodegenError::StoreDoubleViaGpr),
            Value::Constant(_) => return Err(CodegenError::CannotStoreIntoConstant),
        }
        Ok(())
    }

    fn load_value_into_xmm(&mut self, value: &Value, xmm: usize) -> Result<()> {
        let name = Self::xmm_name(xmm)?;
        match value {
            Value::Constant(Const::Double(v)) => self.emit_load_double_constant(*v, xmm),
            Value::Var(name_var) => {
                let operand = self.stack_operand(name_var)?;
                writeln!(self.buf, "  movsd {}, {}", operand, name)?;
                Ok(())
            }
            Value::Global(name_glob) => {
                writeln!(self.buf, "  movsd {}(%rip), {}", name_glob, name)?;
                Ok(())
            }
            other => Err(CodegenError::UnsupportedLoadIntoXmm(other.clone())),
        }
    }

    fn store_xmm_into_value(&mut self, xmm: usize, value: &Value) -> Result<()> {
        let name = Self::xmm_name(xmm)?;
        match value {
            Value::Var(var) => {
                let operand = self.stack_operand(var)?;
                writeln!(self.buf, "  movsd {}, {}", name, operand)?;
                Ok(())
            }
            Value::Global(glob) => {
                writeln!(self.buf, "  movsd {}, {}(%rip)", name, glob)?;
                Ok(())
            }
            Value::Constant(_) => Err(CodegenError::CannotStoreIntoConstant),
        }
    }

    fn emit_load_double_constant(&mut self, value: f64, xmm: usize) -> Result<()> {
        let bits = value.to_bits();
        writeln!(self.buf, "  sub $8, %rsp")?;
        writeln!(self.buf, "  movabs ${:#x}, %rax", bits)?;
        writeln!(self.buf, "  movq %rax, (%rsp)")?;
        writeln!(self.buf, "  movsd (%rsp), {}", Self::xmm_name(xmm)?)?;
        writeln!(self.buf, "  add $8, %rsp")?;
        Ok(())
    }

    fn xmm_name(index: usize) -> Result<&'static str> {
        match index {
            0 => Ok("%xmm0"),
            1 => Ok("%xmm1"),
            2 => Ok("%xmm2"),
            3 => Ok("%xmm3"),
            4 => Ok("%xmm4"),
            5 => Ok("%xmm5"),
            6 => Ok("%xmm6"),
            7 => Ok("%xmm7"),
            _ => Err(CodegenError::UnsupportedXmmIndex(index)),
        }
    }

    fn stack_operand(&self, name: &str) -> Result<String> {
        let offset = self
            .stack_map
            .get(name)
            .ok_or_else(|| CodegenError::UndefinedStackSlot(name.to_string()))?;
        Ok(format!("{}(%rbp)", offset))
    }

    fn mov_instr(&self, ty: &Type) -> Result<&'static str> {
        match ty {
            Type::Char | Type::SChar | Type::UChar => Ok("movb"),
            Type::Int | Type::UInt => Ok("movl"),
            Type::Long | Type::ULong | Type::Void | Type::Pointer(_) => Ok("movq"),
            Type::Double => Err(CodegenError::MovUnsupported(ty.clone())),
            Type::FunType(_, _) | Type::Array(_, _) | Type::IncompleteArray(_) => {
                Err(CodegenError::MovUnsupported(ty.clone()))
            }
        }
    }

    fn value_type(&self, value: &Value) -> Result<Type> {
        self.value_type_optional(value)
            .ok_or(CodegenError::UnknownValueType)
    }

    fn value_type_optional(&self, value: &Value) -> Option<Type> {
        match value {
            Value::Constant(Const::Char(_)) => Some(Type::Char),
            Value::Constant(Const::UChar(_)) => Some(Type::UChar),
            Value::Constant(Const::Int(_)) => Some(Type::Int),
            Value::Constant(Const::Long(_)) => Some(Type::Long),
            Value::Constant(Const::UInt(_)) => Some(Type::UInt),
            Value::Constant(Const::ULong(_)) => Some(Type::ULong),
            Value::Constant(Const::Double(_)) => Some(Type::Double),
            Value::Var(name) => self.value_types.get(name).cloned(),
            Value::Global(name) => self.global_types.get(name).cloned(),
        }
    }
}

impl Reg {
    fn reg_name_for_type(&self, ty: &Type) -> Result<&'static str> {
        match ty {
            Type::Char | Type::SChar | Type::UChar => Ok(self.reg_name8()),
            Type::Int | Type::UInt => Ok(self.reg_name32()),
            Type::Long | Type::ULong | Type::Void | Type::Pointer(_) => Ok(self.reg_name64()),
            Type::Double | Type::FunType(_, _) | Type::Array(_, _) | Type::IncompleteArray(_) => {
                Err(CodegenError::GprRequestedFor(ty.clone()))
            }
        }
    }

    fn reg_name64(&self) -> &'static str {
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

    fn reg_name32(&self) -> &'static str {
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

    #[allow(unused)]
    fn reg_name16(&self) -> &'static str {
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

    fn reg_name8(&self) -> &'static str {
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
