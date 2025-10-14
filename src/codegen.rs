use std::io::Write;

use anyhow::Result;

use crate::parse::{Expr, ExprKind, Program, Stmt, StmtKind};

pub struct Codegen<W: Write> {
    pub source: Vec<char>,
    pub program: Program,
    pub depth: i64,
    pub buf: W,
}

impl<W: Write> Codegen<W> {
    pub fn new(source: Vec<char>, program: Program, buf: W) -> Self {
        Self {
            source,
            program,
            depth: 0,
            buf,
        }
    }
    pub fn program(&mut self) -> Result<()> {
        self.stmt(&self.program.0.clone())?;
        Ok(())
    }

    fn push(&mut self) -> Result<()> {
        writeln!(self.buf, "  push %rax")?;
        self.depth += 1;
        Ok(())
    }

    fn pop(&mut self, arg: &str) -> Result<()> {
        writeln!(self.buf, "  pop {}", arg)?;
        self.depth -= 1;
        Ok(())
    }

    fn stmt(&mut self, stmt: &Stmt) -> Result<()> {
        match &stmt.kind {
            StmtKind::Expr(expr) => {
                self.expr(expr)?;
            }
            StmtKind::Return(expr) => {
                self.expr(expr)?;
                writeln!(self.buf, "  ret")?;
            }
            StmtKind::Block(stmts) => {
                for stmt in stmts {
                    self.stmt(stmt)?;
                }
            }
            StmtKind::FnDecl(name, stmts) => {
                writeln!(self.buf, "  .globl {name}")?;
                writeln!(self.buf, "{name}:")?;
                for stmt in stmts {
                    self.stmt(stmt)?;
                }
                writeln!(self.buf, "  ret")?;
            }
        }
        Ok(())
    }

    fn expr(&mut self, expr: &Expr) -> Result<()> {
        match expr.kind {
            ExprKind::Integer(val) => {
                writeln!(self.buf, "  mov ${}, %rax", val)?;
            }
            ExprKind::Add(ref lhs, ref rhs) => {
                self.expr(rhs)?;
                self.push()?;
                self.expr(lhs)?;
                self.pop("%rdi")?;
                writeln!(self.buf, "  add %rdi, %rax")?;
            }
            ExprKind::Sub(ref lhs, ref rhs) => {
                self.expr(rhs)?;
                self.push()?;
                self.expr(lhs)?;
                self.pop("%rdi")?;
                writeln!(self.buf, "  sub %rdi, %rax")?;
            }
        }
        Ok(())
    }
}
