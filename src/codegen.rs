use std::collections::{BTreeSet, HashMap};
use std::io::{Result, Write};

use crate::tacky::{BinaryOp, Function, Instruction, Program as TackyProgram, UnaryOp, Value};

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
    R10,
    R11,
}

impl<W: Write> Codegen<W> {
    pub fn new(buf: W) -> Self {
        Self {
            buf,
            stack_map: HashMap::new(),
            frame_size: 0,
        }
    }

    pub fn lower(&mut self, program: &TackyProgram) -> Result<()> {
        for function in &program.functions {
            self.emit_function(function)?;
        }
        Ok(())
    }

    fn emit_function(&mut self, function: &Function) -> Result<()> {
        self.stack_map.clear();
        self.frame_size = 0;

        self.collect_stack_slots(function);

        self.emit_prologue(&function.name)?;
        if self.frame_size > 0 {
            writeln!(self.buf, "  sub ${}, %rsp", self.frame_size)?;
        }

        for instr in &function.instructions {
            self.emit_instruction(instr)?;
        }
        Ok(())
    }

    fn collect_stack_slots(&mut self, function: &Function) {
        let mut vars = BTreeSet::new();

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
        self.frame_size = offset as i64;
    }

    fn collect_value(vars: &mut BTreeSet<String>, value: &Value) {
        if let Value::Var(name) = value {
            vars.insert(name.clone());
        }
    }

    fn emit_instruction(&mut self, instr: &Instruction) -> Result<()> {
        match instr {
            Instruction::Return(value) => {
                self.load_value_into_reg(value, Reg::AX)?;
                self.emit_epilogue()
            }
            Instruction::Copy { src, dst } => self.emit_copy(src, dst),
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
        let dst_name = match dst {
            Value::Var(name) => name,
            Value::Constant(_) => panic!("Copy destination cannot be a constant"),
        };

        if matches!(src, Value::Var(name) if name == dst_name) {
            return Ok(());
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

    fn emit_prologue(&mut self, name: &str) -> Result<()> {
        writeln!(self.buf, ".globl {}", name)?;
        writeln!(self.buf, "{}:", name)?;
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
        }
    }

    fn store_reg_into_value(&mut self, reg: Reg, value: &Value) -> Result<()> {
        let name = match value {
            Value::Var(name) => name,
            Value::Constant(_) => panic!("Cannot store into a constant"),
        };

        let operand = self.stack_operand(name);
        writeln!(self.buf, "  mov {}, {}", Self::reg_name(reg), operand)
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
            Reg::R10 => "%r10",
            Reg::R11 => "%r11",
        }
    }

    fn reg_name8(reg: Reg) -> &'static str {
        match reg {
            Reg::AX => "%al",
            Reg::CX => "%cl",
            Reg::DX => "%dl",
            Reg::R10 => "%r10b",
            Reg::R11 => "%r11b",
        }
    }
}
