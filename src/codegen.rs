use std::collections::{BTreeSet, BTreeMap};
use std::io::{Result, Write};

use crate::parse::{Const, Type};
use crate::tacky::{
    BinaryOp, Function, Instruction, Program as TackyProgram, StaticVariable, UnaryOp, Value,
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
        for static_var in &program.statics {
            self.emit_static_variable(static_var)?;
        }
        for function in &program.functions {
            self.emit_function(function)?;
        }
        Ok(())
    }

    fn emit_static_variable(&mut self, var: &StaticVariable) -> Result<()> {
        if var.init == 0 {
            writeln!(self.buf, ".bss")?;
        } else {
            writeln!(self.buf, ".data")?;
        }
        if var.global {
            writeln!(self.buf, ".globl {}", var.name)?;
        }
        let align = self.type_align(&var.ty);
        writeln!(self.buf, ".align {}", align)?;
        writeln!(self.buf, "{}:", var.name)?;
        let size = self.type_size(&var.ty);
        if var.init == 0 {
            writeln!(self.buf, "  .zero {}", size)?
        } else {
            match var.ty {
                Type::Int => writeln!(self.buf, "  .long {}", var.init as i32)?,
                Type::Long => writeln!(self.buf, "  .quad {}", var.init)?,
                Type::Void => panic!("static variable with void type"),
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
            Type::Int => writeln!(self.buf, "  movl $0, %eax")?,
            Type::Long => writeln!(self.buf, "  movq $0, %rax")?,
            Type::Void => {}
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
                Instruction::FunCall { args, dst, .. } => {
                    for arg in args {
                        Self::collect_value(&mut vars, arg);
                    }
                    Self::collect_value(&mut vars, dst);
                }
                Instruction::SignExtend { src, dst } | Instruction::Truncate { src, dst } => {
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

        let mut offset = 0i64;
        for name in vars {
            let ty = self.value_types.get(&name).cloned().unwrap_or_else(|| {
                panic!(
                    "missing type information for {name}: Values: {:?}",
                    self.value_types
                )
            });
            let size = self.type_size(&ty);
            let align = self.type_align(&ty);
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
        for (index, param) in function.params.iter().enumerate() {
            let ty = self
                .value_types
                .get(param)
                .cloned()
                .unwrap_or_else(|| panic!("missing type information for parameter {param}"));
            if index < ARGUMENT_REGISTERS.len() {
                let reg = ARGUMENT_REGISTERS[index];
                let dest = self.stack_operand(param);
                let reg_name = Codegen::<W>::reg_name_for_type(reg, &ty);
                writeln!(self.buf, "  {} {}, {}", self.mov_instr(&ty), reg_name, dest)?;
            } else {
                let stack_offset = 16 + ((index - ARGUMENT_REGISTERS.len()) as i64) * 8;
                let temp_reg = Codegen::<W>::reg_name_for_type(Reg::R10, &ty);
                writeln!(
                    self.buf,
                    "  {} {}(%rbp), {}",
                    self.mov_instr(&ty),
                    stack_offset,
                    temp_reg
                )?;
                let dest = self.stack_operand(param);
                writeln!(self.buf, "  {} {}, {}", self.mov_instr(&ty), temp_reg, dest)?;
            }
        }
        Ok(())
    }

    fn emit_instruction(&mut self, instr: &Instruction) -> Result<()> {
        match instr {
            Instruction::Return(value) => {
                self.load_value_into_reg(value, Reg::AX)?;
                self.emit_epilogue()
            }
            Instruction::Copy { src, dst } => self.emit_copy(src, dst),
            Instruction::FunCall { name, args, dst } => self.emit_fun_call(name, args, dst),
            Instruction::Unary { op, src, dst } => self.emit_unary(*op, src, dst),
            Instruction::Binary {
                op,
                src1,
                src2,
                dst,
            } => self.emit_binary(*op, src1, src2, dst),
            Instruction::SignExtend { src, dst } => self.emit_sign_extend(src, dst),
            Instruction::Truncate { src, dst } => self.emit_truncate(src, dst),
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

        self.load_value_into_reg(src, Reg::R11)?;
        self.store_reg_into_value(Reg::R11, dst)
    }

    fn emit_unary(&mut self, op: UnaryOp, src: &Value, dst: &Value) -> Result<()> {
        self.load_value_into_reg(src, Reg::AX)?;
        let ty = self.value_type(dst);

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
        match op {
            BinaryOp::Divide | BinaryOp::Remainder => {
                let ty = self.value_type(src1);
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
                    Type::Void => panic!("division on void type"),
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
                    BinaryOp::RightShift => "sar",
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

                let set_instr = match op {
                    BinaryOp::Equal => "sete",
                    BinaryOp::NotEqual => "setne",
                    BinaryOp::LessThan => "setl",
                    BinaryOp::LessOrEqual => "setle",
                    BinaryOp::GreaterThan => "setg",
                    BinaryOp::GreaterOrEqual => "setge",
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

    fn emit_sign_extend(&mut self, src: &Value, dst: &Value) -> Result<()> {
        let src_ty = self.value_type(src);
        let dst_ty = self.value_type(dst);
        match (src_ty, dst_ty) {
            (Type::Int, Type::Long) => {
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

    fn emit_truncate(&mut self, src: &Value, dst: &Value) -> Result<()> {
        self.load_value_into_reg(src, Reg::R11)?;
        self.store_reg_into_value(Reg::R11, dst)
    }

    fn emit_fun_call(&mut self, name: &str, args: &[Value], dst: &Value) -> Result<()> {
        let stack_args = if args.len() > ARGUMENT_REGISTERS.len() {
            &args[ARGUMENT_REGISTERS.len()..]
        } else {
            &[]
        };

        let mut stack_bytes: i64 = 0;
        if stack_args.len() % 2 != 0 {
            writeln!(self.buf, "  push $0")?;
            stack_bytes += 8;
        }

        for arg in stack_args.iter().rev() {
            self.load_value_into_reg(arg, Reg::R11)?;
            writeln!(self.buf, "  push {}", Self::reg_name(Reg::R11))?;
            stack_bytes += 8;
        }

        for (reg, arg) in ARGUMENT_REGISTERS.iter().zip(args.iter()) {
            self.load_value_into_reg(arg, *reg)?;
        }

        writeln!(self.buf, "  call {}", name)?;

        if stack_bytes > 0 {
            writeln!(self.buf, "  add ${}, %rsp", stack_bytes)?;
        }

        if let Some(ty) = self.value_type_optional(dst) {
            if ty != Type::Void {
                self.store_reg_into_value(Reg::AX, dst)?;
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
            Value::Constant(_) => panic!("Cannot store into a constant"),
        }
    }

    fn stack_operand(&self, name: &str) -> String {
        let offset = self.stack_map.get(name).unwrap_or_else(|| {
            panic!("Attempted to access undefined stack slot {}", name);
        });
        format!("{}(%rbp)", offset)
    }

    fn type_size(&self, ty: &Type) -> i64 {
        match ty {
            Type::Int => 4,
            Type::Long => 8,
            Type::Void => 0,
        }
    }

    fn type_align(&self, ty: &Type) -> i64 {
        match ty {
            Type::Long => 8,
            Type::Int => 4,
            Type::Void => 1,
        }
    }

    fn mov_instr(&self, ty: &Type) -> &'static str {
        match ty {
            Type::Int => "movl",
            Type::Long => "movq",
            Type::Void => "movq",
        }
    }

    fn reg_name_for_type(reg: Reg, ty: &Type) -> &'static str {
        match ty {
            Type::Int => Self::reg_name32(reg),
            Type::Long => Self::reg_name(reg),
            Type::Void => Self::reg_name(reg),
        }
    }

    fn value_type(&self, value: &Value) -> Type {
        self.value_type_optional(value)
            .unwrap_or_else(|| panic!("unknown value"))
    }

    fn value_type_optional(&self, value: &Value) -> Option<Type> {
        match value {
            Value::Constant(Const::Int(_)) => Some(Type::Int),
            Value::Constant(Const::Long(_)) => Some(Type::Long),
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
