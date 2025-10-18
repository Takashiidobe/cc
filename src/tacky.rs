use std::collections::BTreeMap;

use crate::parse::{
    Const, DeclKind, Expr, ExprKind, ForInit, FunctionDecl, ParameterDecl, Program as AstProgram,
    Stmt, StmtKind, StorageClass, Type, VariableDecl,
};

#[derive(Debug, Clone)]
pub struct Program {
    pub functions: Vec<Function>,
    pub statics: Vec<StaticVariable>,
    pub global_types: BTreeMap<String, Type>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub global: bool,
    pub params: Vec<String>,
    pub return_type: Type,
    pub instructions: Vec<Instruction>,
    pub value_types: BTreeMap<String, Type>,
}

#[derive(Debug, Clone)]
pub struct StaticVariable {
    pub name: String,
    pub global: bool,
    pub ty: Type,
    pub init: i64,
}

#[derive(Debug, Clone)]
pub enum Value {
    Constant(Const),
    Var(String),
    Global(String),
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
    FunCall {
        name: String,
        args: Vec<Value>,
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
    SignExtend {
        src: Value,
        dst: Value,
    },
    Truncate {
        src: Value,
        dst: Value,
    },
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
    locals: BTreeMap<String, Type>,
    value_types: BTreeMap<String, Type>,
    global_types: BTreeMap<String, Type>,
    current_return_type: Option<Type>,
    function_signatures: BTreeMap<String, (Vec<Type>, Type)>,
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
        let mut global_types = BTreeMap::new();
        let mut function_signatures = BTreeMap::new();
        for decl in &program.0 {
            match &decl.kind {
                DeclKind::Variable(var) => {
                    global_types.insert(var.name.clone(), var.r#type.clone());
                }
                DeclKind::Function(func) => {
                    let param_types = func.params.iter().map(|p| p.r#type.clone()).collect();
                    function_signatures
                        .insert(func.name.clone(), (param_types, func.return_type.clone()));
                }
            }
        }

        Self {
            tmp_counter: 0,
            label_counter: 0,
            program,
            loop_stack: Vec::new(),
            locals: BTreeMap::new(),
            value_types: BTreeMap::new(),
            global_types,
            current_return_type: None,
            function_signatures,
        }
    }

    pub fn codegen(mut self) -> Program {
        let mut functions = Vec::new();
        let mut statics = Vec::new();

        let decls = std::mem::take(&mut self.program.0);
        for decl in decls {
            match decl.kind {
                DeclKind::Function(func) => {
                    if let Some(function) = self.gen_function_decl(func) {
                        functions.push(function);
                    }
                }
                DeclKind::Variable(var) => {
                    if let Some(static_var) = self.gen_static_variable(var) {
                        statics.push(static_var);
                    }
                }
            }
        }

        Program {
            functions,
            statics,
            global_types: self.global_types.clone(),
        }
    }

    fn gen_function_decl(&mut self, decl: FunctionDecl) -> Option<Function> {
        let FunctionDecl {
            name,
            params,
            body,
            storage_class,
            return_type,
        } = decl;

        let body = body?;
        let global = storage_class != Some(StorageClass::Static);
        Some(self.gen_function(name, global, params, body, return_type))
    }

    fn gen_function(
        &mut self,
        name: String,
        global: bool,
        params: Vec<ParameterDecl>,
        body: Vec<Stmt>,
        return_type: Type,
    ) -> Function {
        debug_assert!(
            self.loop_stack.is_empty(),
            "loop stack not empty at function entry"
        );

        let previous_return = self.current_return_type.replace(return_type.clone());

        self.locals.clear();
        self.value_types.clear();

        let mut param_names = Vec::new();
        for param in &params {
            self.register_local(&param.name, &param.r#type);
            param_names.push(param.name.clone());
        }

        let mut instructions = Vec::new();
        for stmt in &body {
            self.gen_stmt(stmt, &mut instructions);
        }
        debug_assert!(
            self.loop_stack.is_empty(),
            "loop stack not empty after generating function {}",
            name
        );
        self.locals.clear();
        let value_types = std::mem::take(&mut self.value_types);
        self.current_return_type = previous_return;

        Function {
            name,
            global,
            params: param_names,
            return_type,
            instructions,
            value_types,
        }
    }

    fn gen_static_variable(&mut self, decl: VariableDecl) -> Option<StaticVariable> {
        let VariableDecl {
            name,
            init,
            storage_class,
            r#type,
            is_definition,
        } = decl;

        if !is_definition {
            return None;
        }

        if matches!(storage_class, Some(StorageClass::Extern)) {
            return None;
        }

        let init_value = init.map(|expr| Self::eval_const_expr(&expr)).unwrap_or(0);
        let global = storage_class != Some(StorageClass::Static);

        Some(StaticVariable {
            name,
            global,
            ty: r#type,
            init: init_value,
        })
    }

    fn eval_const_expr(expr: &Expr) -> i64 {
        match &expr.kind {
            ExprKind::Constant(Const::Int(n)) => *n,
            ExprKind::Constant(Const::Long(n)) => *n,
            ExprKind::Neg(inner) => -Self::eval_const_expr(inner),
            ExprKind::BitNot(inner) => !Self::eval_const_expr(inner),
            ExprKind::Not(inner) => !Self::eval_const_expr(inner),
            ExprKind::Add(lhs, rhs) => Self::eval_const_expr(lhs) + Self::eval_const_expr(rhs),
            ExprKind::Sub(lhs, rhs) => Self::eval_const_expr(lhs) - Self::eval_const_expr(rhs),
            ExprKind::Mul(lhs, rhs) => Self::eval_const_expr(lhs) * Self::eval_const_expr(rhs),
            ExprKind::Div(lhs, rhs) => {
                let divisor = Self::eval_const_expr(rhs);
                if divisor == 0 {
                    panic!("division by zero in static initializer");
                }
                Self::eval_const_expr(lhs) / divisor
            }
            ExprKind::Rem(lhs, rhs) => {
                let divisor = Self::eval_const_expr(rhs);
                if divisor == 0 {
                    panic!("division by zero in static initializer");
                }
                Self::eval_const_expr(lhs) % divisor
            }
            ExprKind::Equal(lhs, rhs) => {
                (Self::eval_const_expr(lhs) == Self::eval_const_expr(rhs)) as i64
            }
            ExprKind::NotEqual(lhs, rhs) => {
                (Self::eval_const_expr(lhs) != Self::eval_const_expr(rhs)) as i64
            }
            ExprKind::LessThan(lhs, rhs) => {
                (Self::eval_const_expr(lhs) < Self::eval_const_expr(rhs)) as i64
            }
            ExprKind::LessThanEqual(lhs, rhs) => {
                (Self::eval_const_expr(lhs) <= Self::eval_const_expr(rhs)) as i64
            }
            ExprKind::GreaterThan(lhs, rhs) => {
                (Self::eval_const_expr(lhs) > Self::eval_const_expr(rhs)) as i64
            }
            ExprKind::GreaterThanEqual(lhs, rhs) => {
                (Self::eval_const_expr(lhs) >= Self::eval_const_expr(rhs)) as i64
            }
            ExprKind::BitAnd(lhs, rhs) => Self::eval_const_expr(lhs) & Self::eval_const_expr(rhs),
            ExprKind::BitOr(lhs, rhs) => Self::eval_const_expr(lhs) | Self::eval_const_expr(rhs),
            ExprKind::Xor(lhs, rhs) => Self::eval_const_expr(lhs) ^ Self::eval_const_expr(rhs),
            ExprKind::LeftShift(lhs, rhs) => {
                let shift = Self::eval_const_expr(rhs) as u32;
                Self::eval_const_expr(lhs) << shift
            }
            ExprKind::RightShift(lhs, rhs) => {
                let shift = Self::eval_const_expr(rhs) as u32;
                Self::eval_const_expr(lhs) >> shift
            }
            ExprKind::And(lhs, rhs) => {
                let left = Self::eval_const_expr(lhs);
                if left == 0 {
                    return 0;
                }
                (Self::eval_const_expr(rhs) != 0) as i64
            }
            ExprKind::Or(lhs, rhs) => {
                let left = Self::eval_const_expr(lhs);
                (left != 0 || Self::eval_const_expr(rhs) != 0) as i64
            }
            ExprKind::Conditional(cond, then_expr, else_expr) => {
                if Self::eval_const_expr(cond) != 0 {
                    Self::eval_const_expr(then_expr)
                } else {
                    Self::eval_const_expr(else_expr)
                }
            }
            ExprKind::Cast(_, inner) => Self::eval_const_expr(inner),
            _ => panic!("non-constant expression in static initializer"),
        }
    }

    fn register_local(&mut self, name: &str, ty: &Type) {
        self.locals.insert(name.to_string(), ty.clone());
        self.value_types.insert(name.to_string(), ty.clone());
    }

    fn record_temp(&mut self, name: &str, ty: &Type) {
        self.value_types.insert(name.to_string(), ty.clone());
    }

    fn convert_value(
        &mut self,
        value: Value,
        from_type: Type,
        to_type: Type,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        if from_type == to_type {
            return value;
        }

        match (value, from_type, to_type) {
            (Value::Constant(Const::Int(n)), Type::Int, Type::Long) => {
                Value::Constant(Const::Long(n))
            }
            (Value::Constant(Const::Long(n)), Type::Long, Type::Int) => {
                Value::Constant(Const::Int(n))
            }
            (val, Type::Int, Type::Long) => {
                let tmp = self.fresh_tmp();
                self.record_temp(&tmp, &Type::Long);
                let dst = Value::Var(tmp);
                instructions.push(Instruction::SignExtend {
                    src: val,
                    dst: dst.clone(),
                });
                dst
            }
            (val, Type::Long, Type::Int) => {
                let tmp = self.fresh_tmp();
                self.record_temp(&tmp, &Type::Int);
                let dst = Value::Var(tmp);
                instructions.push(Instruction::Truncate {
                    src: val,
                    dst: dst.clone(),
                });
                dst
            }
            (val, from, to) => {
                let _ = val;
                panic!("unsupported conversion from {:?} to {:?}", from, to)
            }
        }
    }

    fn type_of_value(&self, value: &Value) -> Type {
        match value {
            Value::Constant(Const::Int(_)) => Type::Int,
            Value::Constant(Const::Long(_)) => Type::Long,
            Value::Var(name) => self
                .value_types
                .get(name)
                .unwrap_or_else(|| panic!("unknown temporary {name}"))
                .clone(),
            Value::Global(name) => self
                .global_types
                .get(name)
                .unwrap_or_else(|| panic!("unknown global {name}"))
                .clone(),
        }
    }

    fn value_for_variable(&self, name: &str) -> Value {
        if self.is_local(name) {
            Value::Var(name.to_string())
        } else {
            Value::Global(name.to_string())
        }
    }

    fn is_local(&self, name: &str) -> bool {
        self.locals.contains_key(name)
    }

    fn gen_stmt(&mut self, stmt: &Stmt, instructions: &mut Vec<Instruction>) {
        match &stmt.kind {
            StmtKind::Return(expr) => {
                let value = self.gen_expr(expr, instructions);
                let target_type = self
                    .current_return_type
                    .clone()
                    .expect("return outside function context");
                let converted =
                    self.convert_value(value, expr.r#type.clone(), target_type, instructions);
                instructions.push(Instruction::Return(converted));
            }
            StmtKind::Expr(expr) => {
                self.gen_expr(expr, instructions);
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
            StmtKind::Declaration(decl) => {
                match decl.storage_class {
                    None => {
                        if decl.is_definition {
                            if let Some(init_expr) = &decl.init {
                                let value = self.gen_expr(init_expr, instructions);
                                let converted = self.convert_value(
                                    value,
                                    init_expr.r#type.clone(),
                                    decl.r#type.clone(),
                                    instructions,
                                );
                                instructions.push(Instruction::Copy {
                                    src: converted,
                                    dst: Value::Var(decl.name.clone()),
                                });
                            }
                            self.register_local(&decl.name, &decl.r#type);
                        }
                    }
                    Some(StorageClass::Extern) => {
                        // extern declarations do not produce code and remain global
                    }
                    Some(StorageClass::Static) => {
                        panic!("static local variables are not supported");
                    }
                }
            }
            StmtKind::Null => {}
        }
    }

    fn gen_expr(&mut self, expr: &Expr, instructions: &mut Vec<Instruction>) -> Value {
        let result_type = expr.r#type.clone();
        match &expr.kind {
            ExprKind::Constant(c) => Value::Constant(c.clone()),
            ExprKind::Var(name) => self.value_for_variable(name),
            ExprKind::FunctionCall(name, args) => {
                self.gen_function_call(name, args, &result_type, instructions)
            }
            ExprKind::Cast(target, inner) => {
                let value = self.gen_expr(inner, instructions);
                self.convert_value(value, inner.r#type.clone(), target.clone(), instructions)
            }
            ExprKind::Neg(rhs) => {
                self.gen_unary_expr(UnaryOp::Negate, rhs, &result_type, instructions)
            }
            ExprKind::BitNot(rhs) => {
                self.gen_unary_expr(UnaryOp::Complement, rhs, &result_type, instructions)
            }
            ExprKind::Not(rhs) => {
                self.gen_unary_expr(UnaryOp::Not, rhs, &result_type, instructions)
            }
            ExprKind::Add(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::Add, lhs, rhs, &result_type, instructions)
            }
            ExprKind::Sub(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::Subtract, lhs, rhs, &result_type, instructions)
            }
            ExprKind::Mul(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::Multiply, lhs, rhs, &result_type, instructions)
            }
            ExprKind::Div(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::Divide, lhs, rhs, &result_type, instructions)
            }
            ExprKind::Rem(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::Remainder, lhs, rhs, &result_type, instructions)
            }
            ExprKind::LessThan(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::LessThan, lhs, rhs, &result_type, instructions)
            }
            ExprKind::LessThanEqual(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::LessOrEqual, lhs, rhs, &result_type, instructions)
            }
            ExprKind::GreaterThan(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::GreaterThan, lhs, rhs, &result_type, instructions)
            }
            ExprKind::GreaterThanEqual(lhs, rhs) => self.gen_binary_expr(
                BinaryOp::GreaterOrEqual,
                lhs,
                rhs,
                &result_type,
                instructions,
            ),
            ExprKind::Equal(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::Equal, lhs, rhs, &result_type, instructions)
            }
            ExprKind::NotEqual(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::NotEqual, lhs, rhs, &result_type, instructions)
            }
            ExprKind::BitAnd(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::BitAnd, lhs, rhs, &result_type, instructions)
            }
            ExprKind::BitOr(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::BitOr, lhs, rhs, &result_type, instructions)
            }
            ExprKind::Xor(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::BitXor, lhs, rhs, &result_type, instructions)
            }
            ExprKind::LeftShift(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::LeftShift, lhs, rhs, &result_type, instructions)
            }
            ExprKind::RightShift(lhs, rhs) => {
                self.gen_binary_expr(BinaryOp::RightShift, lhs, rhs, &result_type, instructions)
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
                let cond_value = self.gen_expr(cond, instructions);
                let cond_bool =
                    self.convert_value(cond_value, cond.r#type.clone(), Type::Int, instructions);
                let false_label = self.fresh_label("cond.false");
                let end_label = self.fresh_label("cond.end");

                instructions.push(Instruction::JumpIfZero {
                    condition: cond_bool,
                    target: false_label.clone(),
                });

                let result_tmp = self.fresh_tmp();
                self.record_temp(&result_tmp, &result_type);
                let result = Value::Var(result_tmp.clone());

                let then_value = self.gen_expr(then_expr, instructions);
                let then_converted = self.convert_value(
                    then_value,
                    then_expr.r#type.clone(),
                    result_type.clone(),
                    instructions,
                );
                instructions.push(Instruction::Copy {
                    src: then_converted,
                    dst: result.clone(),
                });
                instructions.push(Instruction::Jump(end_label.clone()));

                instructions.push(Instruction::Label(false_label));
                let else_value = self.gen_expr(else_expr, instructions);
                let else_converted = self.convert_value(
                    else_value,
                    else_expr.r#type.clone(),
                    result_type.clone(),
                    instructions,
                );
                instructions.push(Instruction::Copy {
                    src: else_converted,
                    dst: result.clone(),
                });
                instructions.push(Instruction::Label(end_label));

                Value::Var(result_tmp)
            }
            ExprKind::Assignment(lhs, rhs) => {
                let lhs_expr = lhs.as_ref();
                let name = match &lhs_expr.kind {
                    ExprKind::Var(name) => name.clone(),
                    _ => panic!("assignment target must be a variable"),
                };
                let target = self.value_for_variable(&name);
                let target_type = self.type_of_value(&target);
                let rhs_value = self.gen_expr(rhs, instructions);
                let converted_rhs =
                    self.convert_value(rhs_value, rhs.r#type.clone(), target_type, instructions);
                instructions.push(Instruction::Copy {
                    src: converted_rhs.clone(),
                    dst: target.clone(),
                });
                target
            }
        }
    }

    fn gen_function_call(
        &mut self,
        name: &str,
        args: &[Expr],
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let signature = self
            .function_signatures
            .get(name)
            .cloned()
            .unwrap_or_else(|| panic!("call to undeclared function {name}"));
        let (param_types, return_type) = signature;

        if param_types.len() != args.len() {
            panic!(
                "function '{}' called with wrong number of arguments (expected {}, got {})",
                name,
                param_types.len(),
                args.len()
            );
        }

        let mut arg_values = Vec::new();
        for (arg, expected_type) in args.iter().zip(param_types.iter()) {
            let value = self.gen_expr(arg, instructions);
            let converted = self.convert_value(
                value,
                arg.r#type.clone(),
                expected_type.clone(),
                instructions,
            );
            arg_values.push(converted);
        }

        let tmp = self.fresh_tmp();
        self.record_temp(&tmp, &return_type);
        let dst = Value::Var(tmp);
        instructions.push(Instruction::FunCall {
            name: name.to_string(),
            args: arg_values,
            dst: dst.clone(),
        });

        if return_type == Type::Void {
            dst
        } else {
            self.convert_value(dst, return_type, result_type.clone(), instructions)
        }
    }

    fn gen_unary_expr(
        &mut self,
        op: UnaryOp,
        rhs: &Expr,
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let src_value = self.gen_expr(rhs, instructions);
        let src = self.convert_value(
            src_value,
            rhs.r#type.clone(),
            result_type.clone(),
            instructions,
        );
        let tmp = self.fresh_tmp();
        self.record_temp(&tmp, result_type);
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
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let lhs_value = self.gen_expr(lhs, instructions);
        let rhs_value = self.gen_expr(rhs, instructions);

        let (lhs_type, rhs_type) = match op {
            BinaryOp::LeftShift | BinaryOp::RightShift => (lhs.r#type.clone(), Type::Int),
            _ => match (lhs.r#type.clone(), rhs.r#type.clone()) {
                (Type::Long, _) | (_, Type::Long) => (Type::Long, Type::Long),
                _ => (Type::Int, Type::Int),
            },
        };

        let src1 = self.convert_value(
            lhs_value,
            lhs.r#type.clone(),
            lhs_type.clone(),
            instructions,
        );
        let src2 = self.convert_value(
            rhs_value,
            rhs.r#type.clone(),
            rhs_type.clone(),
            instructions,
        );

        let tmp = self.fresh_tmp();
        self.record_temp(&tmp, result_type);
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

        self.record_temp(&result_tmp, &Type::Int);

        instructions.push(Instruction::Copy {
            src: Value::Constant(Const::Int(0)),
            dst: result.clone(),
        });

        let lhs_val = self.gen_expr(lhs, instructions);
        let lhs_cond = self.convert_value(lhs_val, lhs.r#type.clone(), Type::Int, instructions);
        instructions.push(Instruction::JumpIfZero {
            condition: lhs_cond,
            target: false_label.clone(),
        });

        let rhs_val = self.gen_expr(rhs, instructions);
        let rhs_cond = self.convert_value(rhs_val, rhs.r#type.clone(), Type::Int, instructions);
        instructions.push(Instruction::JumpIfZero {
            condition: rhs_cond,
            target: false_label.clone(),
        });

        instructions.push(Instruction::Copy {
            src: Value::Constant(Const::Int(1)),
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

        self.record_temp(&result_tmp, &Type::Int);

        instructions.push(Instruction::Copy {
            src: Value::Constant(Const::Int(0)),
            dst: result.clone(),
        });

        let lhs_val = self.gen_expr(lhs, instructions);
        let lhs_cond = self.convert_value(lhs_val, lhs.r#type.clone(), Type::Int, instructions);
        instructions.push(Instruction::JumpIfNotZero {
            condition: lhs_cond,
            target: true_label.clone(),
        });

        let rhs_val = self.gen_expr(rhs, instructions);
        let rhs_cond = self.convert_value(rhs_val, rhs.r#type.clone(), Type::Int, instructions);
        instructions.push(Instruction::JumpIfNotZero {
            condition: rhs_cond,
            target: true_label.clone(),
        });

        instructions.push(Instruction::Jump(end_label.clone()));
        instructions.push(Instruction::Label(true_label));
        instructions.push(Instruction::Copy {
            src: Value::Constant(Const::Int(1)),
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
        let target = self.value_for_variable(&name);
        let target_type = self.type_of_value(&target);

        let original_value = if is_post {
            let tmp_name = self.fresh_tmp();
            self.record_temp(&tmp_name, &target_type);
            let tmp = Value::Var(tmp_name.clone());
            instructions.push(Instruction::Copy {
                src: target.clone(),
                dst: tmp.clone(),
            });
            Some(tmp)
        } else {
            None
        };

        let updated_tmp_name = self.fresh_tmp();
        self.record_temp(&updated_tmp_name, &target_type);
        let updated_tmp = Value::Var(updated_tmp_name.clone());
        let one = match target_type {
            Type::Long => Value::Constant(Const::Long(1)),
            _ => Value::Constant(Const::Int(1)),
        };
        instructions.push(Instruction::Binary {
            op,
            src1: target.clone(),
            src2: one,
            dst: updated_tmp.clone(),
        });
        instructions.push(Instruction::Copy {
            src: updated_tmp.clone(),
            dst: target.clone(),
        });

        if let Some(original) = original_value {
            original
        } else {
            target
        }
    }
}
