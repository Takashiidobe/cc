use crate::parse::{Expr, ExprKind, Program as AstProgram, Stmt, StmtKind};

#[derive(Debug, Clone)]
pub struct Program {
    pub functions: Vec<Function>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Mov { src: Operand, dst: Operand },
    Unary { op: UnaryOp, operand: Operand },
    Binary {
        op: BinaryOp,
        lhs: Operand,
        rhs: Operand,
        dst: Operand,
    },
    AllocateStack(i64),
    Ret,
}

#[derive(Debug, Clone)]
pub enum Operand {
    Imm(i64),
    Reg(Reg),
    Pseudo(String),
    Stack(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reg {
    AX,
    R10,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Not,
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
    program: AstProgram,
}

impl TackyGen {
    fn fresh_tmp(&mut self) -> String {
        let name = format!("tmp.{}", self.tmp_counter);
        self.tmp_counter += 1;
        name
    }

    pub fn new(program: AstProgram) -> Self {
        Self {
            tmp_counter: 0,
            program,
        }
    }

    pub fn codegen(mut self) -> Program {
        let function = self.gen_program();
        Program {
            functions: vec![function],
        }
    }

    fn gen_program(&mut self) -> Function {
        let root = self.program.0.clone();
        match root.kind {
            StmtKind::FnDecl(name, body) => self.gen_function(&name, &body),
            kind => panic!("Top-level statement must be a function declaration, found {kind:?}"),
        }
    }

    fn gen_function(&mut self, name: &str, body: &[Stmt]) -> Function {
        let mut instructions = Vec::new();
        let start_tmp = self.tmp_counter;

        for stmt in body {
            self.gen_stmt(stmt, &mut instructions);
        }

        let new_tmps = self.tmp_counter - start_tmp;
        if new_tmps > 0 {
            instructions.insert(
                0,
                Instruction::AllocateStack((new_tmps as i64) * 8),
            );
        }

        Function {
            name: name.to_string(),
            instructions,
        }
    }

    fn gen_stmt(&mut self, stmt: &Stmt, instructions: &mut Vec<Instruction>) {
        match &stmt.kind {
            StmtKind::Return(expr) => {
                let value = self.gen_expr(expr, instructions);
                self.move_to_reg(&value, Reg::AX, instructions);
                instructions.push(Instruction::Ret);
            }
            StmtKind::Expr(expr) => {
                let _ = self.gen_expr(expr, instructions);
            }
            StmtKind::Block(stmts) => {
                for stmt in stmts {
                    self.gen_stmt(stmt, instructions);
                }
            }
            StmtKind::FnDecl(_, _) => {
                panic!("Nested function declarations are not supported in this stage");
            }
        }
    }

    fn gen_expr(
        &mut self,
        expr: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> Operand {
        match &expr.kind {
            ExprKind::Integer(n) => {
                let tmp = self.fresh_tmp();
                instructions.push(Instruction::Mov {
                    src: Operand::Imm(*n),
                    dst: Operand::Pseudo(tmp.clone()),
                });
                Operand::Pseudo(tmp)
            }
            ExprKind::Neg(rhs) => self.gen_unary(UnaryOp::Neg, rhs, instructions),
            ExprKind::BitNot(rhs) => self.gen_unary(UnaryOp::Not, rhs, instructions),
            ExprKind::Add(lhs, rhs) => {
                self.gen_binary(BinaryOp::Add, lhs, rhs, instructions)
            }
            ExprKind::Sub(lhs, rhs) => {
                self.gen_binary(BinaryOp::Sub, lhs, rhs, instructions)
            }
            ExprKind::Mul(lhs, rhs) => {
                self.gen_binary(BinaryOp::Mul, lhs, rhs, instructions)
            }
            ExprKind::Div(lhs, rhs) => {
                self.gen_binary(BinaryOp::Div, lhs, rhs, instructions)
            }
        }
    }

    fn gen_unary(
        &mut self,
        op: UnaryOp,
        rhs: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> Operand {
        let operand = self.gen_expr(rhs, instructions);
        self.move_to_reg(&operand, Reg::AX, instructions);
        instructions.push(Instruction::Unary {
            op,
            operand: Operand::Reg(Reg::AX),
        });
        let tmp = self.fresh_tmp();
        instructions.push(Instruction::Mov {
            src: Operand::Reg(Reg::AX),
            dst: Operand::Pseudo(tmp.clone()),
        });
        Operand::Pseudo(tmp)
    }

    fn gen_binary(
        &mut self,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> Operand {
        let lhs_val = self.gen_expr(lhs, instructions);
        let rhs_val = self.gen_expr(rhs, instructions);

        self.move_to_reg(&lhs_val, Reg::AX, instructions);
        self.move_to_reg(&rhs_val, Reg::R10, instructions);

        instructions.push(Instruction::Binary {
            op,
            lhs: Operand::Reg(Reg::AX),
            rhs: Operand::Reg(Reg::R10),
            dst: Operand::Reg(Reg::AX),
        });

        let tmp = self.fresh_tmp();
        instructions.push(Instruction::Mov {
            src: Operand::Reg(Reg::AX),
            dst: Operand::Pseudo(tmp.clone()),
        });
        Operand::Pseudo(tmp)
    }

    fn move_to_reg(
        &mut self,
        operand: &Operand,
        reg: Reg,
        instructions: &mut Vec<Instruction>,
    ) {
        let dst = Operand::Reg(reg);
        if let Operand::Reg(existing) = operand {
            if *existing == reg {
                return;
            }
        }
        instructions.push(Instruction::Mov {
            src: operand.clone(),
            dst,
        });
    }
}
