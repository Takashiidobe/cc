use crate::parse::{Expr, ExprKind, ForInit, Program as AstProgram, Stmt, StmtKind};

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
    loop_stack: Vec<LoopContext>,
}

struct LoopContext {
    id: usize,
    break_label: String,
    continue_label: String,
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

    fn push_loop_context(&mut self, context: LoopContext) {
        self.loop_stack.push(context);
    }

    fn pop_loop_context(&mut self, expected_id: usize) {
        let context = self.loop_stack.pop().expect("loop stack underflow");
        debug_assert_eq!(context.id, expected_id, "loop id mismatch on pop");
    }

    fn loop_context(&self, id: usize) -> &LoopContext {
        self.loop_stack
            .iter()
            .rev()
            .find(|ctx| ctx.id == id)
            .unwrap_or_else(|| panic!("no loop context for id {id}"))
    }

    pub fn new(program: AstProgram) -> Self {
        Self {
            tmp_counter: 0,
            label_counter: 0,
            program,
            loop_stack: Vec::new(),
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
            StmtKind::Compound(stmts) => {
                for stmt in stmts {
                    self.gen_stmt(stmt, instructions);
                }
            }
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_val = self.gen_expr(condition, instructions);
                let end_label = self.fresh_label("if.end");
                if let Some(else_branch) = else_branch {
                    let else_label = self.fresh_label("if.else");
                    instructions.push(Instruction::JumpIfZero {
                        condition: cond_val,
                        target: else_label.clone(),
                    });
                    self.gen_stmt(then_branch, instructions);
                    instructions.push(Instruction::Jump(end_label.clone()));
                    instructions.push(Instruction::Label(else_label));
                    self.gen_stmt(else_branch, instructions);
                    instructions.push(Instruction::Label(end_label));
                } else {
                    instructions.push(Instruction::JumpIfZero {
                        condition: cond_val,
                        target: end_label.clone(),
                    });
                    self.gen_stmt(then_branch, instructions);
                    instructions.push(Instruction::Label(end_label));
                }
            }
            StmtKind::While {
                condition,
                body,
                loop_id,
            } => {
                let loop_id = loop_id.expect("while statement missing loop id");
                let cond_label = self.fresh_label("while.cond");
                let end_label = self.fresh_label("while.end");
                instructions.push(Instruction::Label(cond_label.clone()));
                let cond_val = self.gen_expr(condition, instructions);
                instructions.push(Instruction::JumpIfZero {
                    condition: cond_val,
                    target: end_label.clone(),
                });
                let context = LoopContext {
                    id: loop_id,
                    break_label: end_label.clone(),
                    continue_label: cond_label.clone(),
                };
                self.push_loop_context(context);
                self.gen_stmt(body, instructions);
                self.pop_loop_context(loop_id);
                instructions.push(Instruction::Jump(cond_label));
                instructions.push(Instruction::Label(end_label));
            }
            StmtKind::DoWhile {
                body,
                condition,
                loop_id,
            } => {
                let loop_id = loop_id.expect("do-while statement missing loop id");
                let body_label = self.fresh_label("do.body");
                let cond_label = self.fresh_label("do.cond");
                let end_label = self.fresh_label("do.end");
                instructions.push(Instruction::Label(body_label.clone()));
                let context = LoopContext {
                    id: loop_id,
                    break_label: end_label.clone(),
                    continue_label: cond_label.clone(),
                };
                self.push_loop_context(context);
                self.gen_stmt(body, instructions);
                instructions.push(Instruction::Label(cond_label.clone()));
                let cond_val = self.gen_expr(condition, instructions);
                instructions.push(Instruction::JumpIfNotZero {
                    condition: cond_val,
                    target: body_label,
                });
                self.pop_loop_context(loop_id);
                instructions.push(Instruction::Label(end_label));
            }
            StmtKind::For {
                init,
                condition,
                post,
                body,
                loop_id,
            } => {
                let loop_id = loop_id.expect("for statement missing loop id");
                match init {
                    ForInit::Declaration(decl) => {
                        self.gen_stmt(decl, instructions);
                    }
                    ForInit::Expr(Some(expr)) => {
                        let _ = self.gen_expr(expr, instructions);
                    }
                    ForInit::Expr(None) => {}
                }
                let cond_label = self.fresh_label("for.cond");
                let end_label = self.fresh_label("for.end");
                let post_label = post.as_ref().map(|_| self.fresh_label("for.post"));
                instructions.push(Instruction::Label(cond_label.clone()));
                if let Some(cond) = condition {
                    let cond_val = self.gen_expr(cond, instructions);
                    instructions.push(Instruction::JumpIfZero {
                        condition: cond_val,
                        target: end_label.clone(),
                    });
                }
                let continue_label = post_label.clone().unwrap_or_else(|| cond_label.clone());
                let context = LoopContext {
                    id: loop_id,
                    break_label: end_label.clone(),
                    continue_label: continue_label.clone(),
                };
                self.push_loop_context(context);
                self.gen_stmt(body, instructions);
                if let Some(post_expr) = post {
                    if let Some(label) = &post_label {
                        instructions.push(Instruction::Label(label.clone()));
                    }
                    let _ = self.gen_expr(post_expr, instructions);
                }
                self.pop_loop_context(loop_id);
                instructions.push(Instruction::Jump(cond_label));
                instructions.push(Instruction::Label(end_label));
            }
            StmtKind::Break { loop_id } => {
                let loop_id = loop_id.expect("break statement missing loop id");
                let context = self.loop_context(loop_id);
                instructions.push(Instruction::Jump(context.break_label.clone()));
            }
            StmtKind::Continue { loop_id } => {
                let loop_id = loop_id.expect("continue statement missing loop id");
                let context = self.loop_context(loop_id);
                instructions.push(Instruction::Jump(context.continue_label.clone()));
            }
            StmtKind::Declaration { name, init } => {
                if let Some(init_expr) = init {
                    let value = self.gen_expr(init_expr, instructions);
                    instructions.push(Instruction::Copy {
                        src: value,
                        dst: Value::Var(name.clone()),
                    });
                }
            }
            StmtKind::Null => {}
            StmtKind::FnDecl(_, _) => {
                panic!("Nested function declarations are not supported in this stage");
            }
        }
    }

    fn gen_expr(&mut self, expr: &Expr, instructions: &mut Vec<Instruction>) -> Value {
        match &expr.kind {
            ExprKind::Integer(n) => Value::Constant(*n),
            ExprKind::Var(name) => Value::Var(name.clone()),
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
            ExprKind::PreIncrement(expr) => {
                self.gen_inc_dec(expr, BinaryOp::Add, false, instructions)
            }
            ExprKind::PreDecrement(expr) => {
                self.gen_inc_dec(expr, BinaryOp::Subtract, false, instructions)
            }
            ExprKind::PostIncrement(expr) => {
                self.gen_inc_dec(expr, BinaryOp::Add, true, instructions)
            }
            ExprKind::PostDecrement(expr) => {
                self.gen_inc_dec(expr, BinaryOp::Subtract, true, instructions)
            }
            ExprKind::Conditional(cond, then_expr, else_expr) => {
                let result_tmp = self.fresh_tmp();
                let result = Value::Var(result_tmp.clone());
                let false_label = self.fresh_label("cond.false");
                let end_label = self.fresh_label("cond.end");

                let condition_value = self.gen_expr(cond, instructions);
                instructions.push(Instruction::JumpIfZero {
                    condition: condition_value,
                    target: false_label.clone(),
                });

                let then_value = self.gen_expr(then_expr, instructions);
                instructions.push(Instruction::Copy {
                    src: then_value,
                    dst: result.clone(),
                });
                instructions.push(Instruction::Jump(end_label.clone()));

                instructions.push(Instruction::Label(false_label));
                let else_value = self.gen_expr(else_expr, instructions);
                instructions.push(Instruction::Copy {
                    src: else_value,
                    dst: result.clone(),
                });
                instructions.push(Instruction::Label(end_label));

                Value::Var(result_tmp)
            }
            ExprKind::Assignment(lhs, rhs) => {
                let rhs_value = self.gen_expr(rhs, instructions);
                let lhs_expr = lhs.as_ref();
                let name = match &lhs_expr.kind {
                    ExprKind::Var(name) => name.clone(),
                    _ => panic!("Assignment target must be a variable"),
                };
                instructions.push(Instruction::Copy {
                    src: rhs_value,
                    dst: Value::Var(name.clone()),
                });
                Value::Var(name)
            }
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
        expr: &Expr,
        op: BinaryOp,
        is_post: bool,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let name = match &expr.kind {
            ExprKind::Var(name) => name.clone(),
            _ => panic!("Increment/decrement target must be a variable"),
        };

        let original_value = if is_post {
            let tmp_name = self.fresh_tmp();
            let tmp = Value::Var(tmp_name.clone());
            instructions.push(Instruction::Copy {
                src: Value::Var(name.clone()),
                dst: tmp.clone(),
            });
            Some(tmp)
        } else {
            None
        };

        let updated_tmp_name = self.fresh_tmp();
        let updated_tmp = Value::Var(updated_tmp_name.clone());
        instructions.push(Instruction::Binary {
            op,
            src1: Value::Var(name.clone()),
            src2: Value::Constant(1),
            dst: updated_tmp.clone(),
        });
        instructions.push(Instruction::Copy {
            src: updated_tmp.clone(),
            dst: Value::Var(name.clone()),
        });

        if let Some(original) = original_value {
            original
        } else {
            Value::Var(name)
        }
    }
}
