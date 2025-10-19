use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryFrom;
use std::io::{Result, Write};

use crate::parse::{Const, Type};
use crate::tacky::{
    BinaryOp, Function, Instruction, Program as TackyProgram, StaticVariable, TopLevel, UnaryOp,
    Value,
};

pub struct Codegen<W: Write> {
    pub buf: W,
    stack_map: BTreeMap<String, i64>,
    frame_size: i64,
    global_types: BTreeMap<String, Type>,
    value_types: BTreeMap<String, Type>,
}

#[derive(Clone, Copy)]
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

const ARGUMENT_REGISTERS: [Reg; 6] = [Reg::DI, Reg::SI, Reg::DX, Reg::CX, Reg::R8, Reg::R9];
const FLOAT_ARGUMENT_REGISTERS: [&str; 8] = [
    "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7",
];

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
                TopLevel::Function(function) => self.emit_function(function)?,
            }
        }
        Ok(())
    }

    fn emit_static_variable(&mut self, var: &StaticVariable) -> Result<()> {
        if !var.init.is_empty() && (var.init.len() != 1 || var.init[0].offset != 0) {
            panic!("compound static initializers are not supported yet");
        }

        let init_value_ref = var.init.first().map(|init| &init.value);
        let init_const = init_value_ref.cloned();

        let is_zero_init = match init_value_ref {
            None => true,
            Some(Const::Int(v)) => *v == 0,
            Some(Const::UInt(v)) => *v == 0,
            Some(Const::Long(v)) => *v == 0,
            Some(Const::ULong(v)) => *v == 0,
            Some(Const::Double(v)) => *v == 0.0,
        };

        if is_zero_init {
            writeln!(self.buf, ".bss")?;
        } else {
            writeln!(self.buf, ".data")?;
        }
        if var.global {
            writeln!(self.buf, ".globl {}", var.name)?;
        }
        let align = Self::type_align(&var.ty);
        writeln!(self.buf, ".align {}", align)?;
        writeln!(self.buf, "{}:", var.name)?;
        let size = Self::type_size(&var.ty);
        if is_zero_init {
            writeln!(self.buf, "  .zero {}", size)?
        } else {
            match (&var.ty, init_const.as_ref()) {
                (Type::Int, Some(Const::Int(value))) => {
                    writeln!(self.buf, "  .long {}", *value as i32)?;
                }
                (Type::UInt, Some(Const::UInt(value))) => {
                    let truncated = value & 0xffff_ffff;
                    writeln!(self.buf, "  .long {}", truncated)?;
                }
                (Type::Long, Some(Const::Long(value))) => {
                    writeln!(self.buf, "  .quad {}", value)?;
                }
                (Type::ULong, Some(Const::ULong(value))) => {
                    writeln!(self.buf, "  .quad {}", value)?;
                }
                (Type::Pointer(_), Some(Const::Long(value))) => {
                    writeln!(self.buf, "  .quad {}", value)?;
                }
                (Type::Pointer(_), Some(Const::ULong(value))) => {
                    writeln!(self.buf, "  .quad {}", value)?;
                }
                (Type::Double, Some(Const::Double(value))) => {
                    let bits = value.to_bits();
                    writeln!(self.buf, "  .quad {}", bits)?;
                }
                (Type::Int, None)
                | (Type::UInt, None)
                | (Type::Long, None)
                | (Type::ULong, None)
                | (Type::Pointer(_), None)
                | (Type::Double, None) => {
                    writeln!(self.buf, "  .zero {}", Self::type_size(&var.ty))?
                }
                _ => panic!(
                    "static initializer type mismatch: {:?} with {:?}",
                    var.ty, init_const
                ),
            }
        }
        Ok(())
    }

    fn emit_function(&mut self, function: &Function) -> Result<()> {
        self.stack_map.clear();
        self.frame_size = 0;
        self.value_types = function.value_types.clone();

        self.collect_stack_slots(function);

        self.emit_prologue(function)?;
        if self.frame_size > 0 {
            writeln!(self.buf, "  sub ${}, %rsp", self.frame_size)?;
        }
        self.move_params_to_stack(function)?;

        for instr in &function.instructions {
            self.emit_instruction(instr)?;
        }
        match function.return_type {
            Type::Int | Type::UInt => writeln!(self.buf, "  movl $0, %eax")?,
            Type::Long | Type::ULong | Type::Pointer(_) => writeln!(self.buf, "  movq $0, %rax")?,
            Type::Double => writeln!(self.buf, "  xorpd %xmm0, %xmm0")?,
            Type::Void => {}
            Type::FunType(_, _) => {
                panic!("unsupported function return type")
            }
            Type::Array(_, _) => panic!("array return type not yet supported in codegen"),
        }
        let result = self.emit_epilogue();
        self.value_types.clear();
        result
    }

    fn collect_stack_slots(&mut self, function: &Function) {
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
                Instruction::CopyToOffset { src, .. } => {
                    Self::collect_value(&mut vars, src);
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
            let ty = self.value_types.get(&name).cloned().unwrap_or_else(|| {
                panic!(
                    "missing type information for {name}: Values: {:?}",
                    self.value_types
                )
            });
            let size = Self::type_size(&ty);
            let align = Self::type_align(&ty);
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
                .unwrap_or_else(|| panic!("missing type information for parameter {param}"));
            let dest = self.stack_operand(param);

            if ty == Type::Double {
                if float_reg_idx < FLOAT_ARGUMENT_REGISTERS.len() {
                    let src = FLOAT_ARGUMENT_REGISTERS[float_reg_idx];
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
                let reg_name = Codegen::<W>::reg_name_for_type(reg, &ty);
                writeln!(self.buf, "  {} {}, {}", self.mov_instr(&ty), reg_name, dest)?;
            } else {
                let offset = 16 + stack_arg_idx * 8;
                stack_arg_idx += 1;
                let temp_reg = Codegen::<W>::reg_name_for_type(Reg::R10, &ty);
                writeln!(
                    self.buf,
                    "  {} {}(%rbp), {}",
                    self.mov_instr(&ty),
                    offset,
                    temp_reg
                )?;
                writeln!(self.buf, "  {} {}, {}", self.mov_instr(&ty), temp_reg, dest)?;
            }
        }
        Ok(())
    }

    fn emit_instruction(&mut self, instr: &Instruction) -> Result<()> {
        match instr {
            Instruction::Return(value) => {
                let ty = self.value_type(value);
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
            Instruction::Jump(label) => writeln!(self.buf, "  jmp {}", label),
            Instruction::JumpIfZero { condition, target } => {
                self.load_value_into_reg(condition, Reg::AX)?;
                let ty = self.value_type(condition);
                let reg_name = Codegen::<W>::reg_name_for_type(Reg::AX, &ty);
                writeln!(self.buf, "  cmp $0, {}", reg_name)?;
                writeln!(self.buf, "  je {}", target)
            }
            Instruction::JumpIfNotZero { condition, target } => {
                self.load_value_into_reg(condition, Reg::AX)?;
                let ty = self.value_type(condition);
                let reg_name = Codegen::<W>::reg_name_for_type(Reg::AX, &ty);
                writeln!(self.buf, "  cmp $0, {}", reg_name)?;
                writeln!(self.buf, "  jne {}", target)
            }
            Instruction::Label(name) => writeln!(self.buf, "{}:", name),
        }
    }

    fn emit_copy(&mut self, src: &Value, dst: &Value) -> Result<()> {
        if matches!((src, dst), (Value::Var(a), Value::Var(b)) if a == b)
            || matches!((src, dst), (Value::Global(a), Value::Global(b)) if a == b)
        {
            return Ok(());
        }

        if matches!(dst, Value::Constant(_)) {
            panic!("Copy destination cannot be a constant");
        }

        let ty = self.value_type(dst);
        if ty == Type::Double {
            self.load_value_into_xmm(src, 0)?;
            return self.store_xmm_into_value(0, dst);
        }

        self.load_value_into_reg(src, Reg::R11)?;
        self.store_reg_into_value(Reg::R11, dst)
    }

    fn emit_get_address(&mut self, src: &Value, dst: &Value) -> Result<()> {
        if matches!(dst, Value::Constant(_)) {
            panic!("address destination cannot be constant");
        }

        let reg = Reg::R11;
        match src {
            Value::Var(name) => {
                let operand = self.stack_operand(name);
                writeln!(
                    self.buf,
                    "  leaq {}, {}",
                    operand,
                    Codegen::<W>::reg_name(reg)
                )?;
            }
            Value::Global(name) => {
                writeln!(
                    self.buf,
                    "  leaq {}(%rip), {}",
                    name,
                    Codegen::<W>::reg_name(reg)
                )?;
            }
            Value::Constant(_) => panic!("cannot take address of constant"),
        }

        self.store_reg_into_value(reg, dst)
    }

    fn emit_load(&mut self, src_ptr: &Value, dst: &Value) -> Result<()> {
        let dst_ty = self.value_type(dst);
        self.load_value_into_reg(src_ptr, Reg::R11)?;
        match dst_ty {
            Type::Double => {
                writeln!(
                    self.buf,
                    "  movsd ({}), {}",
                    Self::reg_name(Reg::R11),
                    Self::xmm_name(0)
                )?;
                self.store_xmm_into_value(0, dst)
            }
            _ => {
                let reg = Reg::R10;
                writeln!(
                    self.buf,
                    "  {} ({}), {}",
                    self.mov_instr(&dst_ty),
                    Self::reg_name(Reg::R11),
                    Codegen::<W>::reg_name_for_type(reg, &dst_ty)
                )?;
                self.store_reg_into_value(reg, dst)
            }
        }
    }

    fn emit_store(&mut self, src: &Value, dst_ptr: &Value) -> Result<()> {
        let src_ty = self.value_type(src);
        self.load_value_into_reg(dst_ptr, Reg::R11)?;
        match src_ty {
            Type::Double => {
                self.load_value_into_xmm(src, 0)?;
                writeln!(
                    self.buf,
                    "  movsd {}, ({})",
                    Self::xmm_name(0),
                    Self::reg_name(Reg::R11)
                )
            }
            _ => {
                self.load_value_into_reg(src, Reg::R10)?;
                writeln!(
                    self.buf,
                    "  {} {}, ({})",
                    self.mov_instr(&src_ty),
                    Codegen::<W>::reg_name_for_type(Reg::R10, &src_ty),
                    Self::reg_name(Reg::R11)
                )
            }
        }
    }

    fn emit_unary(&mut self, op: UnaryOp, src: &Value, dst: &Value) -> Result<()> {
        let ty = self.value_type(dst);
        if ty == Type::Double {
            match op {
                UnaryOp::Negate => {
                    self.load_value_into_xmm(src, 0)?;
                    writeln!(
                        self.buf,
                        "  xorpd {}, {}",
                        Self::xmm_name(1),
                        Self::xmm_name(1)
                    )?;
                    writeln!(
                        self.buf,
                        "  subsd {}, {}",
                        Self::xmm_name(0),
                        Self::xmm_name(1)
                    )?;
                    return self.store_xmm_into_value(1, dst);
                }
                UnaryOp::Complement => panic!("bitwise complement not supported for doubles"),
                UnaryOp::Not => panic!("logical not not supported for doubles"),
            }
        }

        self.load_value_into_reg(src, Reg::AX)?;

        match op {
            UnaryOp::Negate => writeln!(
                self.buf,
                "  neg {}",
                Codegen::<W>::reg_name_for_type(Reg::AX, &ty)
            )?,
            UnaryOp::Complement => writeln!(
                self.buf,
                "  not {}",
                Codegen::<W>::reg_name_for_type(Reg::AX, &ty)
            )?,
            UnaryOp::Not => {
                let reg_name = Codegen::<W>::reg_name_for_type(Reg::AX, &ty);
                writeln!(self.buf, "  cmp $0, {}", reg_name)?;
                writeln!(self.buf, "  sete {}", Self::reg_name8(Reg::AX))?;
                writeln!(
                    self.buf,
                    "  movzbq {}, {}",
                    Self::reg_name8(Reg::AX),
                    Self::reg_name(Reg::AX)
                )?;
            }
        }

        self.store_reg_into_value(Reg::AX, dst)
    }

    fn emit_binary(&mut self, op: BinaryOp, src1: &Value, src2: &Value, dst: &Value) -> Result<()> {
        let result_ty = self.value_type(dst);
        let lhs_ty = self.value_type(src1);
        let rhs_ty = self.value_type(src2);

        if result_ty == Type::Double {
            self.load_value_into_xmm(src1, 0)?;
            self.load_value_into_xmm(src2, 1)?;
            let instr = match op {
                BinaryOp::Add => "addsd",
                BinaryOp::Subtract => "subsd",
                BinaryOp::Multiply => "mulsd",
                BinaryOp::Divide => "divsd",
                _ => panic!("unsupported floating binary op {:?}", op),
            };
            writeln!(
                self.buf,
                "  {} {}, {}",
                instr,
                Self::xmm_name(1),
                Self::xmm_name(0)
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
                        Self::xmm_name(1),
                        Self::xmm_name(0)
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

                    writeln!(self.buf, "  {} {}", set_instr, Self::reg_name8(Reg::AX))?;
                    writeln!(
                        self.buf,
                        "  movzbq {}, {}",
                        Self::reg_name8(Reg::AX),
                        Self::reg_name(Reg::AX)
                    )?;
                    return self.store_reg_into_value(Reg::AX, dst);
                }
                _ => panic!("unsupported floating-point binary op {:?}", op),
            }
        }

        match op {
            BinaryOp::Divide | BinaryOp::Remainder => {
                let ty = result_ty;
                self.load_value_into_reg(src1, Reg::AX)?;
                self.load_value_into_reg(src2, Reg::R10)?;
                match ty {
                    Type::Int => {
                        writeln!(self.buf, "  cltd")?;
                        writeln!(
                            self.buf,
                            "  idiv {}",
                            Codegen::<W>::reg_name_for_type(Reg::R10, &Type::Int)
                        )?;
                    }
                    Type::Long => {
                        writeln!(self.buf, "  cqo")?;
                        writeln!(
                            self.buf,
                            "  idiv {}",
                            Codegen::<W>::reg_name_for_type(Reg::R10, &Type::Long)
                        )?;
                    }
                    Type::UInt => {
                        writeln!(
                            self.buf,
                            "  xor {}, {}",
                            Self::reg_name32(Reg::DX),
                            Self::reg_name32(Reg::DX)
                        )?;
                        writeln!(
                            self.buf,
                            "  div {}",
                            Codegen::<W>::reg_name_for_type(Reg::R10, &Type::UInt)
                        )?;
                    }
                    Type::ULong => {
                        writeln!(
                            self.buf,
                            "  xor {}, {}",
                            Self::reg_name(Reg::DX),
                            Self::reg_name(Reg::DX)
                        )?;
                        writeln!(
                            self.buf,
                            "  div {}",
                            Codegen::<W>::reg_name_for_type(Reg::R10, &Type::ULong)
                        )?;
                    }
                    Type::Void => panic!("division on void type"),
                    Type::Array(_, _) => panic!("division on array type"),
                    Type::Double => panic!("integer division on double type"),
                    Type::Pointer(_) => panic!("division on pointer type"),
                    Type::FunType(_, _) => panic!("division on function type"),
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
                let ty = self.value_type(dst);
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
                    Codegen::<W>::reg_name_for_type(Reg::R10, &ty),
                    Codegen::<W>::reg_name_for_type(Reg::AX, &ty)
                )?;
                self.store_reg_into_value(Reg::AX, dst)
            }
            BinaryOp::LeftShift | BinaryOp::RightShift => {
                let ty = self.value_type(dst);
                self.load_value_into_reg(src1, Reg::AX)?;
                self.load_value_into_reg(src2, Reg::CX)?;

                let op_str = match op {
                    BinaryOp::LeftShift => "shl",
                    BinaryOp::RightShift => {
                        if Self::is_unsigned_type(&ty) {
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
                    Self::reg_name8(Reg::CX),
                    Codegen::<W>::reg_name_for_type(Reg::AX, &ty)
                )?;

                self.store_reg_into_value(Reg::AX, dst)
            }
            BinaryOp::Equal
            | BinaryOp::NotEqual
            | BinaryOp::LessThan
            | BinaryOp::LessOrEqual
            | BinaryOp::GreaterThan
            | BinaryOp::GreaterOrEqual => {
                let ty = self.value_type(src1);
                self.load_value_into_reg(src1, Reg::AX)?;
                self.load_value_into_reg(src2, Reg::R10)?;
                writeln!(
                    self.buf,
                    "  cmp {}, {}",
                    Codegen::<W>::reg_name_for_type(Reg::R10, &ty),
                    Codegen::<W>::reg_name_for_type(Reg::AX, &ty)
                )?;

                let is_unsigned = Self::is_unsigned_type(&ty);
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

                writeln!(self.buf, "  {} {}", set_instr, Self::reg_name8(Reg::AX))?;
                writeln!(
                    self.buf,
                    "  movzbq {}, {}",
                    Self::reg_name8(Reg::AX),
                    Self::reg_name(Reg::AX)
                )?;

                self.store_reg_into_value(Reg::AX, dst)
            }
        }
    }

    fn emit_add_ptr(&mut self, ptr: &Value, index: &Value, scale: i64, dst: &Value) -> Result<()> {
        if matches!(dst, Value::Constant(_)) {
            panic!("AddPtr destination cannot be a constant");
        }

        if !matches!(self.value_type(dst), Type::Pointer(_)) {
            panic!("AddPtr destination must have pointer type");
        }

        self.load_value_into_reg(ptr, Reg::R11)?;
        self.load_value_into_reg(index, Reg::R10)?;

        let base = Codegen::<W>::reg_name(Reg::R11);
        let idx = Codegen::<W>::reg_name(Reg::R10);

        if matches!(scale, 1 | 2 | 4 | 8) {
            if scale == 1 {
                writeln!(self.buf, "  leaq ({},{}), {}", base, idx, base)?;
            } else {
                writeln!(self.buf, "  leaq ({},{},{}), {}", base, idx, scale, base)?;
            }
        } else {
            writeln!(
                self.buf,
                "  imul ${}, {}",
                scale,
                Codegen::<W>::reg_name(Reg::R10)
            )?;
            writeln!(
                self.buf,
                "  add {}, {}",
                Codegen::<W>::reg_name(Reg::R10),
                base
            )?;
        }

        self.store_reg_into_value(Reg::R11, dst)
    }

    fn emit_copy_to_offset(&mut self, src: &Value, dst: &str, offset: i64) -> Result<()> {
        let addr = self.static_address(dst, offset);
        let ty = self.value_type(src);

        match ty {
            Type::Double => {
                self.load_value_into_xmm(src, 0)?;
                writeln!(self.buf, "  movsd {}, {}", Self::xmm_name(0), addr)?;
            }
            Type::Int | Type::UInt => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  movl {}, {}",
                    Codegen::<W>::reg_name32(Reg::R11),
                    addr
                )?;
            }
            Type::Long | Type::ULong | Type::Pointer(_) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  movq {}, {}",
                    Codegen::<W>::reg_name(Reg::R11),
                    addr
                )?;
            }
            Type::Void => panic!("cannot copy void value"),
            Type::FunType(_, _) => panic!("function type copy not supported"),
            Type::Array(_, _) => panic!("array copy via CopyToOffset not supported"),
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
        let src_ty = self.value_type(src);
        let dst_ty = self.value_type(dst);
        match (src_ty, dst_ty) {
            (Type::Int, Type::Long) | (Type::Int, Type::ULong) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  movsxd {}, {}",
                    Codegen::<W>::reg_name32(Reg::R11),
                    Codegen::<W>::reg_name(Reg::R11)
                )?;
                self.store_reg_into_value(Reg::R11, dst)
            }
            _ => panic!("unsupported sign extension"),
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
                    Self::xmm_name(0),
                    Codegen::<W>::reg_name32(Reg::R11)
                )?;
                self.store_reg_into_value(Reg::R11, dst)
            }
            (Type::Double, Type::UInt) => {
                self.load_value_into_xmm(src, 0)?;
                writeln!(
                    self.buf,
                    "  cvttsd2siq {}, {}",
                    Self::xmm_name(0),
                    Codegen::<W>::reg_name(Reg::R11)
                )?;
                self.store_reg_into_value(Reg::R11, dst)
            }
            (Type::Double, Type::Long) | (Type::Double, Type::ULong) => {
                self.load_value_into_xmm(src, 0)?;
                writeln!(
                    self.buf,
                    "  cvttsd2siq {}, {}",
                    Self::xmm_name(0),
                    Codegen::<W>::reg_name(Reg::R11)
                )?;
                self.store_reg_into_value(Reg::R11, dst)
            }
            (Type::Int, Type::Double) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  cvtsi2sd {}, {}",
                    Codegen::<W>::reg_name32(Reg::R11),
                    Self::xmm_name(0)
                )?;
                self.store_xmm_into_value(0, dst)
            }
            (Type::UInt, Type::Double) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  cvtsi2sdq {}, {}",
                    Codegen::<W>::reg_name(Reg::R11),
                    Self::xmm_name(0)
                )?;
                self.store_xmm_into_value(0, dst)
            }
            (Type::Long, Type::Double) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  cvtsi2sdq {}, {}",
                    Codegen::<W>::reg_name(Reg::R11),
                    Self::xmm_name(0)
                )?;
                self.store_xmm_into_value(0, dst)
            }
            (Type::ULong, Type::Double) => {
                self.load_value_into_reg(src, Reg::R11)?;
                writeln!(
                    self.buf,
                    "  mov {}, {}",
                    Codegen::<W>::reg_name(Reg::R11),
                    Codegen::<W>::reg_name(Reg::AX)
                )?;
                writeln!(self.buf, "  shr $1, {}", Codegen::<W>::reg_name(Reg::AX))?;
                writeln!(self.buf, "  and $1, {}", Codegen::<W>::reg_name(Reg::R11))?;
                writeln!(
                    self.buf,
                    "  cvtsi2sdq {}, {}",
                    Codegen::<W>::reg_name(Reg::AX),
                    Self::xmm_name(0)
                )?;
                writeln!(
                    self.buf,
                    "  addsd {}, {}",
                    Self::xmm_name(0),
                    Self::xmm_name(0)
                )?;
                writeln!(
                    self.buf,
                    "  cvtsi2sdq {}, {}",
                    Codegen::<W>::reg_name(Reg::R11),
                    Self::xmm_name(1)
                )?;
                writeln!(
                    self.buf,
                    "  addsd {}, {}",
                    Self::xmm_name(1),
                    Self::xmm_name(0)
                )?;
                self.store_xmm_into_value(0, dst)
            }
            _ => panic!("unsupported conversion {:?} -> {:?}", from, to),
        }
    }

    fn emit_fun_call(&mut self, name: &str, args: &[Value], dst: &Value) -> Result<()> {
        let mut int_regs = Vec::new();
        let mut float_regs = Vec::new();
        let mut stack_args: Vec<(Value, Type)> = Vec::new();
        let mut int_reg_idx = 0usize;
        let mut float_reg_idx = 0usize;

        for arg in args {
            let ty = self.value_type(arg);
            if ty == Type::Double {
                if float_reg_idx < FLOAT_ARGUMENT_REGISTERS.len() {
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
                    writeln!(self.buf, "  movsd {}, (%rsp)", Self::xmm_name(0))?;
                }
                _ => {
                    self.load_value_into_reg(value, Reg::R11)?;
                    writeln!(self.buf, "  push {}", Self::reg_name(Reg::R11))?;
                }
            }
            stack_bytes += 8;
        }

        for (idx, value) in float_regs {
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
        writeln!(self.buf, "  mov %rsp, %rbp")
    }

    fn emit_epilogue(&mut self) -> Result<()> {
        writeln!(self.buf, "  mov %rbp, %rsp")?;
        writeln!(self.buf, "  pop %rbp")?;
        writeln!(self.buf, "  ret")
    }

    fn load_value_into_reg(&mut self, value: &Value, reg: Reg) -> Result<()> {
        let ty = self.value_type(value);
        if ty == Type::Double {
            panic!("attempted to load double into general-purpose register");
        }
        match value {
            Value::Constant(Const::Int(n)) => {
                writeln!(
                    self.buf,
                    "  movl ${}, {}",
                    n,
                    Codegen::<W>::reg_name_for_type(reg, &Type::Int)
                )
            }
            Value::Constant(Const::Long(n)) => {
                writeln!(self.buf, "  movq ${}, {}", n, Codegen::<W>::reg_name(reg))
            }
            Value::Constant(Const::UInt(n)) => {
                writeln!(
                    self.buf,
                    "  movl ${}, {}",
                    n,
                    Codegen::<W>::reg_name_for_type(reg, &Type::UInt)
                )
            }
            Value::Constant(Const::ULong(n)) => {
                writeln!(self.buf, "  movq ${}, {}", n, Codegen::<W>::reg_name(reg))
            }
            Value::Constant(Const::Double(_)) => {
                panic!("attempted to load double into general-purpose register")
            }
            Value::Var(name) => {
                let operand = self.stack_operand(name);
                writeln!(
                    self.buf,
                    "  {} {}, {}",
                    self.mov_instr(&ty),
                    operand,
                    Codegen::<W>::reg_name_for_type(reg, &ty)
                )
            }
            Value::Global(name) => {
                writeln!(
                    self.buf,
                    "  {} {}(%rip), {}",
                    self.mov_instr(&ty),
                    name,
                    Codegen::<W>::reg_name_for_type(reg, &ty)
                )
            }
        }
    }

    fn store_reg_into_value(&mut self, reg: Reg, value: &Value) -> Result<()> {
        let ty = self.value_type(value);
        let reg_name = Codegen::<W>::reg_name_for_type(reg, &ty);
        match value {
            Value::Var(name) => {
                let operand = self.stack_operand(name);
                writeln!(
                    self.buf,
                    "  {} {}, {}",
                    self.mov_instr(&ty),
                    reg_name,
                    operand
                )
            }
            Value::Global(name) => {
                writeln!(
                    self.buf,
                    "  {} {}, {}(%rip)",
                    self.mov_instr(&ty),
                    reg_name,
                    name
                )
            }
            Value::Constant(Const::Double(_)) => {
                panic!("cannot store double via general-purpose register")
            }
            Value::Constant(_) => panic!("Cannot store into a constant"),
        }
    }

    fn load_value_into_xmm(&mut self, value: &Value, xmm: usize) -> Result<()> {
        match value {
            Value::Constant(Const::Double(v)) => self.emit_load_double_constant(*v, xmm),
            Value::Var(name) => {
                let operand = self.stack_operand(name);
                writeln!(self.buf, "  movsd {}, {}", operand, Self::xmm_name(xmm))
            }
            Value::Global(name) => {
                writeln!(self.buf, "  movsd {}(%rip), {}", name, Self::xmm_name(xmm))
            }
            other => panic!("unsupported load into xmm for value {:?}", other),
        }
    }

    fn store_xmm_into_value(&mut self, xmm: usize, value: &Value) -> Result<()> {
        match value {
            Value::Var(name) => {
                let operand = self.stack_operand(name);
                writeln!(self.buf, "  movsd {}, {}", Self::xmm_name(xmm), operand)
            }
            Value::Global(name) => {
                writeln!(self.buf, "  movsd {}, {}(%rip)", Self::xmm_name(xmm), name)
            }
            Value::Constant(_) => panic!("Cannot store into a constant"),
        }
    }

    fn emit_load_double_constant(&mut self, value: f64, xmm: usize) -> Result<()> {
        let bits = value.to_bits();
        writeln!(self.buf, "  sub $8, %rsp")?;
        writeln!(self.buf, "  movabs ${:#x}, %rax", bits)?;
        writeln!(self.buf, "  movq %rax, (%rsp)")?;
        writeln!(self.buf, "  movsd (%rsp), {}", Self::xmm_name(xmm))?;
        writeln!(self.buf, "  add $8, %rsp")
    }

    fn xmm_name(index: usize) -> &'static str {
        match index {
            0 => "%xmm0",
            1 => "%xmm1",
            2 => "%xmm2",
            3 => "%xmm3",
            4 => "%xmm4",
            5 => "%xmm5",
            6 => "%xmm6",
            7 => "%xmm7",
            _ => panic!("unsupported xmm register index {}", index),
        }
    }

    fn stack_operand(&self, name: &str) -> String {
        let offset = self.stack_map.get(name).unwrap_or_else(|| {
            panic!("Attempted to access undefined stack slot {}", name);
        });
        format!("{}(%rbp)", offset)
    }

    fn type_size(ty: &Type) -> i64 {
        match ty {
            Type::Int | Type::UInt => 4,
            Type::Long | Type::ULong | Type::Double => 8,
            Type::Void => 0,
            Type::Pointer(_) => 8,
            Type::FunType(_, _) => panic!("function type has no size"),
            Type::Array(inner, len) => {
                let len_i64 = i64::try_from(*len).expect("array size exceeds i64");
                len_i64 * Self::type_size(inner)
            }
        }
    }

    fn type_align(ty: &Type) -> i64 {
        match ty {
            Type::Long | Type::ULong => 8,
            Type::Int | Type::UInt => 4,
            Type::Double => 8,
            Type::Void => 1,
            Type::Pointer(_) => 8,
            Type::FunType(_, _) => panic!("function type has no alignment"),
            Type::Array(inner, _) => Self::type_align(inner),
        }
    }

    fn mov_instr(&self, ty: &Type) -> &'static str {
        match ty {
            Type::Int | Type::UInt => "movl",
            Type::Long | Type::ULong => "movq",
            Type::Void => "movq",
            Type::Pointer(_) => "movq",
            Type::Double => panic!("mov instruction requested for double"),
            Type::FunType(_, _) => panic!("function type move not supported"),
            Type::Array(_, _) => panic!("array type move not supported"),
        }
    }

    fn reg_name_for_type(reg: Reg, ty: &Type) -> &'static str {
        match ty {
            Type::Int | Type::UInt => Self::reg_name32(reg),
            Type::Long | Type::ULong | Type::Void | Type::Pointer(_) => Self::reg_name(reg),
            Type::Double => panic!("general-purpose register requested for double"),
            Type::FunType(_, _) => panic!("function type register request"),
            Type::Array(_, _) => panic!("array type register request"),
        }
    }

    fn is_unsigned_type(ty: &Type) -> bool {
        matches!(ty, Type::UInt | Type::ULong)
    }

    fn value_type(&self, value: &Value) -> Type {
        self.value_type_optional(value)
            .unwrap_or_else(|| panic!("unknown value"))
    }

    fn value_type_optional(&self, value: &Value) -> Option<Type> {
        match value {
            Value::Constant(Const::Int(_)) => Some(Type::Int),
            Value::Constant(Const::Long(_)) => Some(Type::Long),
            Value::Constant(Const::UInt(_)) => Some(Type::UInt),
            Value::Constant(Const::ULong(_)) => Some(Type::ULong),
            Value::Constant(Const::Double(_)) => Some(Type::Double),
            Value::Var(name) => self.value_types.get(name).cloned(),
            Value::Global(name) => self.global_types.get(name).cloned(),
        }
    }

    fn reg_name(reg: Reg) -> &'static str {
        match reg {
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

    fn reg_name32(reg: Reg) -> &'static str {
        match reg {
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

    fn reg_name8(reg: Reg) -> &'static str {
        match reg {
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
