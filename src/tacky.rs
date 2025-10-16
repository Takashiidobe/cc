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
pub enum Value {
    Constant(i64),
    Var(String),
}

#[derive(Debug, Clone)]
pub enum Instruction {
    Return(Value),
    Unary {
        op: UnaryOp,
        src: Value,
        dst: Value,
    },
    Binary {
        op: BinaryOp,
        src1: Value,
        src2: Value,
        dst: Value,
    },
    Copy {
        src: Value,
        dst: Value,
    },
    Jump(String),
    JumpIfZero {
        condition: Value,
        target: String,
    },
    JumpIfNotZero {
        condition: Value,
        target: String,
    },
    Label(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Complement,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
    BitAnd,
    BitOr,
    BitXor,
    LeftShift,
    RightShift,
}

pub struct TackyGen {
    tmp_counter: usize,
    label_counter: usize,
    program: AstProgram,
}

impl TackyGen {
    fn fresh_tmp(&mut self) -> String {
        let name = format!("tmp.{}", self.tmp_counter);
        self.tmp_counter += 1;
        name
    }

    fn fresh_label(&mut self, prefix: &str) -> String {
        let name = format!("{}.{}", prefix, self.label_counter);
        self.label_counter += 1;
        name
    }

    pub fn new(program: AstProgram) -> Self {
        Self {
            tmp_counter: 0,
            label_counter: 0,
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
        for stmt in body {
            self.gen_stmt(stmt, &mut instructions);
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
                instructions.push(Instruction::Return(value));
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

    fn gen_expr(&mut self, expr: &Expr, instructions: &mut Vec<Instruction>) -> Value {
        match &expr.kind {
            ExprKind::Integer(n) => Value::Constant(*n),
            ExprKind::Neg(rhs) => self.gen_unary_expr(UnaryOp::Negate, rhs, instructions),
            ExprKind::BitNot(rhs) => self.gen_unary_expr(UnaryOp::Complement, rhs, instructions),
            ExprKind::Not(rhs) => self.gen_unary_expr(UnaryOp::Not, rhs, instructions),
            ExprKind::Add(lhs, rhs) => self.gen_binary_expr(BinaryOp::Add, lhs, rhs, instructions),
            ExprKind::Sub(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::Subtract, lhs, rhs, instructions)
            }
            ExprKind::Mul(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::Multiply, lhs, rhs, instructions)
            }
            ExprKind::Div(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::Divide, lhs, rhs, instructions)
            }
            ExprKind::Rem(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::Remainder, lhs, rhs, instructions)
            }
            ExprKind::LessThan(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::LessThan, lhs, rhs, instructions)
            }
            ExprKind::LessThanEqual(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::LessOrEqual, lhs, rhs, instructions)
            }
            ExprKind::GreaterThan(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::GreaterThan, lhs, rhs, instructions)
            }
            ExprKind::GreaterThanEqual(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::GreaterOrEqual, lhs, rhs, instructions)
            }
            ExprKind::Equal(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::Equal, lhs, rhs, instructions)
            }
            ExprKind::NotEqual(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::NotEqual, lhs, rhs, instructions)
            }
            ExprKind::BitAnd(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::BitAnd, lhs, rhs, instructions)
            }
            ExprKind::BitOr(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::BitOr, lhs, rhs, instructions)
            }
            ExprKind::Xor(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::BitXor, lhs, rhs, instructions)
            }
            ExprKind::LeftShift(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::LeftShift, lhs, rhs, instructions)
            }
            ExprKind::RightShift(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::RightShift, lhs, rhs, instructions)
            }
            ExprKind::And(lhs, rhs) => self.gen_logical_and(lhs, rhs, instructions),
            ExprKind::Or(lhs, rhs) => self.gen_logical_or(lhs, rhs, instructions),
            ExprKind::Incr(rhs) => self.gen_inc_dec(BinaryOp::Add, rhs, instructions),
            ExprKind::Decr(rhs) => self.gen_inc_dec(BinaryOp::Subtract, rhs, instructions),
        }
    }

    fn gen_unary_expr(
        &mut self,
        op: UnaryOp,
        rhs: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let src = self.gen_expr(rhs, instructions);
        let tmp = self.fresh_tmp();
        let dst = Value::Var(tmp);
        instructions.push(Instruction::Unary {
            op,
            src,
            dst: dst.clone(),
        });
        dst
    }

    fn gen_binary_expr(
        &mut self,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let src1 = self.gen_expr(lhs, instructions);
        let src2 = self.gen_expr(rhs, instructions);
        let tmp = self.fresh_tmp();
        let dst = Value::Var(tmp);
        instructions.push(Instruction::Binary {
            op,
            src1,
            src2,
            dst: dst.clone(),
        });
        dst
    }

    fn gen_logical_and(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let result_tmp = self.fresh_tmp();
        let result = Value::Var(result_tmp.clone());
        let false_label = self.fresh_label("and.false");
        let end_label = self.fresh_label("and.end");

        instructions.push(Instruction::Copy {
            src: Value::Constant(0),
            dst: result.clone(),
        });

        let lhs_val = self.gen_expr(lhs, instructions);
        instructions.push(Instruction::JumpIfZero {
            condition: lhs_val,
            target: false_label.clone(),
        });

        let rhs_val = self.gen_expr(rhs, instructions);
        instructions.push(Instruction::JumpIfZero {
            condition: rhs_val,
            target: false_label.clone(),
        });

        instructions.push(Instruction::Copy {
            src: Value::Constant(1),
            dst: result.clone(),
        });
        instructions.push(Instruction::Jump(end_label.clone()));
        instructions.push(Instruction::Label(false_label));
        instructions.push(Instruction::Label(end_label));

        Value::Var(result_tmp)
    }

    fn gen_logical_or(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let result_tmp = self.fresh_tmp();
        let result = Value::Var(result_tmp.clone());
        let true_label = self.fresh_label("or.true");
        let end_label = self.fresh_label("or.end");

        instructions.push(Instruction::Copy {
            src: Value::Constant(0),
            dst: result.clone(),
        });

        let lhs_val = self.gen_expr(lhs, instructions);
        instructions.push(Instruction::JumpIfNotZero {
            condition: lhs_val,
            target: true_label.clone(),
        });

        let rhs_val = self.gen_expr(rhs, instructions);
        instructions.push(Instruction::JumpIfNotZero {
            condition: rhs_val,
            target: true_label.clone(),
        });

        instructions.push(Instruction::Jump(end_label.clone()));
        instructions.push(Instruction::Label(true_label));
        instructions.push(Instruction::Copy {
            src: Value::Constant(1),
            dst: result.clone(),
        });
        instructions.push(Instruction::Label(end_label));

        Value::Var(result_tmp)
    }

    fn gen_inc_dec(
        &mut self,
        op: BinaryOp,
        expr: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let src = self.gen_expr(expr, instructions);
        let tmp = self.fresh_tmp();
        let dst = Value::Var(tmp);
        instructions.push(Instruction::Binary {
            op,
            src1: src,
            src2: Value::Constant(1),
            dst: dst.clone(),
        });
        dst
    }
}
