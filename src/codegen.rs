use std::collections::HashMap;
use std::io::{Result, Write};

use crate::tacky::{
    BinaryOp, Function, Instruction, Operand, Program as TackyProgram, Reg, UnaryOp,
};

pub struct Codegen<W: Write> {
    pub buf: W,
    stack_map: HashMap<String, i64>,
    next_offset: i64,
}

impl<W: Write> Codegen<W> {
    pub fn new(buf: W) -> Self {
        Self {
            buf,
            stack_map: HashMap::new(),
            next_offset: 0,
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
        self.next_offset = 0;

        self.emit_prologue(&function.name)?;
        for instr in &function.instructions {
            self.emit_instruction(instr)?;
        }
        Ok(())
    }

    fn emit_instruction(&mut self, instr: &Instruction) -> Result<()> {
        match instr {
            Instruction::AllocateStack(amount) => self.emit_allocate_stack(*amount),
            Instruction::Mov { src, dst } => self.emit_mov(src, dst),
            Instruction::Unary { op, operand } => self.emit_unary(*op, operand),
            Instruction::Binary { op, lhs, rhs, dst } => self.emit_binary(*op, lhs, rhs, dst),
            Instruction::Ret => self.emit_epilogue(),
            Instruction::Copy { src, dst } => todo!(),
            Instruction::JumpIfZero {
                condition,
                identifier,
            } => todo!(),
            Instruction::JumpIfNotZero {
                condition,
                identifier,
            } => todo!(),
            Instruction::Jump(_) => todo!(),
            Instruction::Label(_) => todo!(),
        }
    }

    fn emit_allocate_stack(&mut self, amount: i64) -> Result<()> {
        if amount == 0 {
            return Ok(());
        }
        writeln!(self.buf, "  sub ${}, %rsp", amount)
    }

    fn emit_mov(&mut self, src: &Operand, dst: &Operand) -> Result<()> {
        if Self::operands_equal(src, dst) {
            return Ok(());
        }

        let src_str = self.format_src(src);
        let dst_str = self.format_dst(dst);
        writeln!(self.buf, "  mov {}, {}", src_str, dst_str)
    }

    fn emit_unary(&mut self, op: UnaryOp, operand: &Operand) -> Result<()> {
        let reg = self.expect_reg(operand, "unary operand");
        let reg_name = Self::reg_name(reg);
        match op {
            UnaryOp::Neg => writeln!(self.buf, "  neg {}", reg_name),
            UnaryOp::Not => writeln!(self.buf, "  not {}", reg_name),
        }
    }

    fn emit_binary(
        &mut self,
        op: BinaryOp,
        lhs: &Operand,
        rhs: &Operand,
        dst: &Operand,
    ) -> Result<()> {
        let lhs_reg = self.expect_reg(lhs, "binary lhs");
        let rhs_reg = self.expect_reg(rhs, "binary rhs");
        let dst_reg = self.expect_reg(dst, "binary dst");

        if lhs_reg != dst_reg {
            // Ensure the destination starts with the lhs value.
            self.emit_mov(lhs, dst)?;
        }

        let rhs_name = Self::reg_name(rhs_reg);
        let dst_name = Self::reg_name(dst_reg);

        match op {
            BinaryOp::Add => writeln!(self.buf, "  add {}, {}", rhs_name, dst_name),
            BinaryOp::Sub => writeln!(self.buf, "  sub {}, {}", rhs_name, dst_name),
            BinaryOp::Mul => writeln!(self.buf, "  imul {}, {}", rhs_name, dst_name),
            BinaryOp::Rem | BinaryOp::Div => {
                if dst_reg != Reg::AX {
                    panic!("Division result must be placed in %rax");
                }
                writeln!(self.buf, "  cqo")?;
                writeln!(self.buf, "  idiv {}", rhs_name)
            }
            BinaryOp::LessThan => todo!(),
            BinaryOp::LessThanEqual => todo!(),
            BinaryOp::GreaterThan => todo!(),
            BinaryOp::GreaterThanEqual => todo!(),
            BinaryOp::Equal => todo!(),
            BinaryOp::NotEqual => todo!(),
            BinaryOp::And => todo!(),
            BinaryOp::Or => todo!(),
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

    fn operands_equal(lhs: &Operand, rhs: &Operand) -> bool {
        matches!(
            (lhs, rhs),
            (Operand::Reg(Reg::AX), Operand::Reg(Reg::AX))
                | (Operand::Reg(Reg::R10), Operand::Reg(Reg::R10))
        )
    }

    fn format_src(&self, operand: &Operand) -> String {
        match operand {
            Operand::Imm(value) => format!("${}", value),
            Operand::Reg(reg) => Self::reg_name(*reg).to_string(),
            Operand::Pseudo(name) => {
                let offset = self
                    .stack_map
                    .get(name)
                    .unwrap_or_else(|| panic!("Attempted to read from undefined pseudo {}", name));
                format!("{}(%rbp)", offset)
            }
            Operand::Stack(offset) => format!("{}(%rbp)", offset),
        }
    }

    fn format_dst(&mut self, operand: &Operand) -> String {
        match operand {
            Operand::Reg(reg) => Self::reg_name(*reg).to_string(),
            Operand::Pseudo(name) => {
                let offset = self.stack_slot(name);
                format!("{}(%rbp)", offset)
            }
            Operand::Stack(offset) => format!("{}(%rbp)", offset),
            Operand::Imm(_) => panic!("Cannot use an immediate as a destination"),
        }
    }

    fn stack_slot(&mut self, pseudo: &str) -> i64 {
        *self.stack_map.entry(pseudo.to_string()).or_insert_with(|| {
            self.next_offset -= 8;
            self.next_offset
        })
    }

    fn expect_reg(&self, operand: &Operand, context: &str) -> Reg {
        match operand {
            Operand::Reg(reg) => *reg,
            other => panic!("Expected register for {}, found {:?}", context, other),
        }
    }

    fn reg_name(reg: Reg) -> &'static str {
        match reg {
            Reg::AX => "%rax",
            Reg::DX => "%rdx",
            Reg::R10 => "%r10",
            Reg::R11 => "%r11",
        }
    }
}
