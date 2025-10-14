use crate::parse::{Expr, ExprKind, Program, Stmt, StmtKind};

#[derive(Debug, Clone)]
pub enum TackyInstr {
    AssignVar(String, TackyValue),
    Unary {
        op: UnaryOp,
        src: TackyValue,
        dst: String,
    },
    Binary {
        op: BinaryOp,
        lhs: TackyValue,
        rhs: TackyValue,
        dst: String,
    },
    Return(TackyValue),
    Label(String),
}

#[derive(Debug, Clone)]
pub enum TackyValue {
    Integer(i64),
    Var(String),
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    BitNot,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Mul,
    Div,
    Add,
    Sub,
}

pub struct TackyGen {
    tmp_counter: usize,
    pub instructions: Vec<TackyInstr>,
    pub program: Program,
}

impl TackyGen {
    fn fresh_tmp(&mut self) -> String {
        let name = format!("tmp.{}", self.tmp_counter);
        self.tmp_counter += 1;
        name
    }

    pub fn new(program: Program) -> Self {
        Self {
            tmp_counter: 0,
            instructions: vec![],
            program,
        }
    }

    pub fn codegen(&mut self) -> Vec<TackyInstr> {
        self.gen_stmt(&self.program.0.clone());
        self.instructions.clone()
    }

    pub fn gen_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Return(expr) => {
                let value = self.gen_expr(expr);
                self.instructions.push(TackyInstr::Return(value));
            }
            StmtKind::Expr(expr) => {
                self.gen_expr(expr);
            }
            StmtKind::Block(stmts) => {
                for stmt in stmts {
                    self.gen_stmt(stmt);
                }
            }
            StmtKind::FnDecl(name, body) => {
                let label = name.to_string();
                self.instructions.push(TackyInstr::Label(label));

                for stmt in body {
                    self.gen_stmt(stmt);
                }
            }
        }
    }

    pub fn gen_expr(&mut self, expr: &Expr) -> TackyValue {
        match &expr.kind {
            ExprKind::Integer(n) => TackyValue::Integer(*n),
            ExprKind::Neg(rhs) => {
                let src = self.gen_expr(rhs);
                let dst = self.fresh_tmp();
                self.instructions.push(TackyInstr::Unary {
                    op: UnaryOp::Neg,
                    src,
                    dst: dst.clone(),
                });
                TackyValue::Var(dst)
            }
            ExprKind::BitNot(rhs) => {
                let src = self.gen_expr(rhs);
                let dst = self.fresh_tmp();
                self.instructions.push(TackyInstr::Unary {
                    op: UnaryOp::BitNot,
                    src,
                    dst: dst.clone(),
                });
                TackyValue::Var(dst)
            }
            kind @ ExprKind::Mul(lhs, rhs)
            | kind @ ExprKind::Div(lhs, rhs)
            | kind @ ExprKind::Add(lhs, rhs)
            | kind @ ExprKind::Sub(lhs, rhs) => {
                let lhs = self.gen_expr(lhs);
                let rhs = self.gen_expr(rhs);
                let dst = self.fresh_tmp();
                self.instructions.push(TackyInstr::Binary {
                    op: if matches!(kind, ExprKind::Div(_, _)) {
                        BinaryOp::Div
                    } else if matches!(kind, ExprKind::Mul(_, _)) {
                        BinaryOp::Mul
                    } else if matches!(kind, ExprKind::Sub(_, _)) {
                        BinaryOp::Sub
                    } else {
                        BinaryOp::Add
                    },
                    lhs,
                    rhs,
                    dst: dst.clone(),
                });
                TackyValue::Var(dst)
            }
        }
    }
}
