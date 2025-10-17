use std::collections::{BTreeSet, HashMap};
use std::io::{Result, Write};

use crate::parse::Type;
use crate::tacky::{
    BinaryOp, Function, Instruction, Program as TackyProgram, StaticVariable, UnaryOp, Value,
};

pub struct Codegen<W: Write> {
    pub buf: W,
    stack_map: HashMap<String, i64>,
    frame_size: i64,
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
            stack_map: HashMap::new(),
            frame_size: 0,
        }
    }

    pub fn lower(&mut self, program: &TackyProgram) -> Result<()> {
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
        writeln!(self.buf, ".align 8")?;
        writeln!(self.buf, "{}:", var.name)?;
        if var.init == 0 {
            writeln!(self.buf, "  .zero 8")?
        } else {
            writeln!(self.buf, "  .quad {}", var.init)?;
        }
        Ok(())
    }

    fn emit_function(&mut self, function: &Function) -> Result<()> {
        self.stack_map.clear();
        self.frame_size = 0;

        self.collect_stack_slots(function);

        self.emit_prologue(function)?;
        if self.frame_size > 0 {
            writeln!(self.buf, "  sub ${}, %rsp", self.frame_size)?;
        }
        self.move_params_to_stack(function)?;

        for instr in &function.instructions {
            self.emit_instruction(instr)?;
        }
        if matches!(function.return_type, Type::Int) {
            writeln!(self.buf, "  mov $0, %rax")?;
        }
        self.emit_epilogue()
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
                Instruction::Jump(_) | Instruction::Label(_) => {}
                Instruction::JumpIfZero { condition, .. }
                | Instruction::JumpIfNotZero { condition, .. } => {
                    Self::collect_value(&mut vars, condition);
                }
            }
        }

        let mut offset = 0;
        for name in vars {
            offset += 8;
            self.stack_map.insert(name, -(offset as i64));
        }
        let mut frame_size = offset as i64;
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
            if index < ARGUMENT_REGISTERS.len() {
                let reg = ARGUMENT_REGISTERS[index];
                let dest = self.stack_operand(param);
                writeln!(self.buf, "  mov {}, {}", Self::reg_name(reg), dest)?;
            } else {
                let stack_offset = 16 + ((index - ARGUMENT_REGISTERS.len()) as i64) * 8;
                writeln!(
                    self.buf,
                    "  mov {}(%rbp), {}",
                    stack_offset,
                    Self::reg_name(Reg::R10)
                )?;
                let dest = self.stack_operand(param);
                writeln!(self.buf, "  mov {}, {}", Self::reg_name(Reg::R10), dest)?;
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
            Instruction::Jump(label) => writeln!(self.buf, "  jmp {}", label),
            Instruction::JumpIfZero { condition, target } => {
                self.load_value_into_reg(condition, Reg::AX)?;
                writeln!(self.buf, "  cmp $0, {}", Self::reg_name(Reg::AX))?;
                writeln!(self.buf, "  je {}", target)
            }
            Instruction::JumpIfNotZero { condition, target } => {
                self.load_value_into_reg(condition, Reg::AX)?;
                writeln!(self.buf, "  cmp $0, {}", Self::reg_name(Reg::AX))?;
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

        match op {
            UnaryOp::Negate => writeln!(self.buf, "  neg {}", Self::reg_name(Reg::AX))?,
            UnaryOp::Complement => writeln!(self.buf, "  not {}", Self::reg_name(Reg::AX))?,
            UnaryOp::Not => {
                writeln!(self.buf, "  cmp $0, {}", Self::reg_name(Reg::AX))?;
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
                self.load_value_into_reg(src1, Reg::AX)?;
                self.load_value_into_reg(src2, Reg::R10)?;
                writeln!(self.buf, "  cqo")?;
                writeln!(self.buf, "  idiv {}", Self::reg_name(Reg::R10))?;

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
                    Self::reg_name(Reg::R10),
                    Self::reg_name(Reg::AX)
                )?;

                self.store_reg_into_value(Reg::AX, dst)
            }
            BinaryOp::LeftShift | BinaryOp::RightShift => {
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
                    Self::reg_name(Reg::AX)
                )?;

                self.store_reg_into_value(Reg::AX, dst)
            }
            BinaryOp::Equal
            | BinaryOp::NotEqual
            | BinaryOp::LessThan
            | BinaryOp::LessOrEqual
            | BinaryOp::GreaterThan
            | BinaryOp::GreaterOrEqual => {
                self.load_value_into_reg(src1, Reg::AX)?;
                self.load_value_into_reg(src2, Reg::R10)?;
                writeln!(
                    self.buf,
                    "  cmp {}, {}",
                    Self::reg_name(Reg::R10),
                    Self::reg_name(Reg::AX)
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

        self.store_reg_into_value(Reg::AX, dst)
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
        match value {
            Value::Constant(n) => {
                writeln!(self.buf, "  mov ${}, {}", n, Self::reg_name(reg))
            }
            Value::Var(name) => {
                let operand = self.stack_operand(name);
                writeln!(self.buf, "  mov {}, {}", operand, Self::reg_name(reg))
            }
            Value::Global(name) => {
                writeln!(self.buf, "  mov {}(%rip), {}", name, Self::reg_name(reg))
            }
        }
    }

    fn store_reg_into_value(&mut self, reg: Reg, value: &Value) -> Result<()> {
        match value {
            Value::Var(name) => {
                let operand = self.stack_operand(name);
                writeln!(self.buf, "  mov {}, {}", Self::reg_name(reg), operand)
            }
            Value::Global(name) => {
                writeln!(self.buf, "  mov {}, {}(%rip)", Self::reg_name(reg), name)
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
