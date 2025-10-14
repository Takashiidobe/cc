use std::collections::HashMap;
use std::io::{Result, Write};

use crate::tacky::{BinaryOp, TackyInstr, TackyValue, UnaryOp};

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

    fn stack_slot(&mut self, var: &str) -> i64 {
        *self.stack_map.entry(var.to_string()).or_insert_with(|| {
            self.next_offset -= 8;
            self.next_offset
        })
    }

    fn emit_load(&mut self, val: &TackyValue) -> Result<()> {
        match val {
            TackyValue::Integer(i) => writeln!(self.buf, "  mov ${}, %rax", i),
            TackyValue::Var(v) => {
                let off = self.stack_slot(v);
                writeln!(self.buf, "  mov {}(%rbp), %rax", off)
            }
        }
    }

    fn emit_store(&mut self, var: &str) -> Result<()> {
        let off = self.stack_slot(var);
        writeln!(self.buf, "  mov %rax, {}(%rbp)", off)
    }

    pub fn emit_prologue(&mut self, name: &str) -> Result<()> {
        writeln!(self.buf, ".globl {}", name)?;
        writeln!(self.buf, "{}:", name)?;
        writeln!(self.buf, "  push %rbp")?;
        writeln!(self.buf, "  mov %rsp, %rbp")?;
        Ok(())
    }

    pub fn emit_epilogue(&mut self) -> Result<()> {
        writeln!(self.buf, "  mov %rbp, %rsp")?;
        writeln!(self.buf, "  pop %rbp")?;
        writeln!(self.buf, "  ret")?;
        Ok(())
    }

    pub fn lower(&mut self, instrs: &[TackyInstr]) -> Result<()> {
        for instr in instrs {
            match instr {
                TackyInstr::Label(name) => self.emit_prologue(name)?,

                TackyInstr::AssignVar(dst, src) => {
                    self.emit_load(src)?;
                    self.emit_store(dst)?;
                }

                TackyInstr::Unary { op, src, dst } => {
                    self.emit_load(src)?;
                    match op {
                        UnaryOp::Neg => writeln!(self.buf, "  neg %rax")?,
                        UnaryOp::BitNot => writeln!(self.buf, "  not %rax")?,
                    }
                    self.emit_store(dst)?;
                }

                TackyInstr::Binary { op, lhs, rhs, dst } => {
                    self.emit_load(lhs)?;
                    writeln!(self.buf, "  mov %rax, %r10")?;
                    self.emit_load(rhs)?;

                    match op {
                        BinaryOp::Add => writeln!(self.buf, "  add %r10, %rax")?,
                        BinaryOp::Sub => writeln!(self.buf, "  sub %r10, %rax")?,
                        BinaryOp::Mul => writeln!(self.buf, "  imul %r10, %rax")?,
                        BinaryOp::Div => {
                            writeln!(self.buf, "  mov %r10, %rbx")?;
                            writeln!(self.buf, "  cqo")?;
                            writeln!(self.buf, "  idiv %rbx")?;
                        }
                    }

                    self.emit_store(dst)?;
                }

                TackyInstr::Return(val) => {
                    self.emit_load(val)?;
                    self.emit_epilogue()?;
                }
            }
        }
        Ok(())
    }
}
