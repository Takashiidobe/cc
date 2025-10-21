mod error;
mod types;

use std::collections::BTreeMap;

pub use crate::{
    parse::{
        Const, DeclKind, Expr, ExprKind, ForInit, FunctionDecl, ParameterDecl,
        Program as AstProgram, Stmt, StmtKind, StorageClass, Type, VariableDecl,
    },
    tacky::{
        error::{IRError, IResult},
        types::{
            BinaryOp, Function, Instruction, LoopContext, Program, StaticConstant, StaticInit,
            StaticVariable, TopLevel, UnaryOp, Value,
        },
    },
};

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
    string_literals: Vec<StaticConstant>,
    string_counter: usize,
}

impl TackyGen {
    fn fresh_tmp(&mut self) -> String {
        let n = self.tmp_counter;
        self.tmp_counter += 1;
        format!("tmp.{n}")
    }
    fn fresh_label(&mut self, prefix: &str) -> String {
        let n = self.label_counter;
        self.label_counter += 1;
        format!("{prefix}.{n}")
    }
    fn push_loop_context(&mut self, context: LoopContext) {
        self.loop_stack.push(context);
    }
    fn pop_loop_context(&mut self, expected_id: usize) -> IResult<()> {
        let ctx = self.loop_stack.pop().ok_or(IRError::LoopUnderflow)?;
        if ctx.id != expected_id {
            return Err(IRError::LoopIdMismatch(expected_id, ctx.id));
        }
        Ok(())
    }
    fn loop_context(&self, id: usize) -> IResult<&LoopContext> {
        self.loop_stack
            .iter()
            .rev()
            .find(|c| c.id == id)
            .ok_or(IRError::NoLoopContext(id))
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
                    let ps = func.params.iter().map(|p| p.r#type.clone()).collect();
                    function_signatures.insert(func.name.clone(), (ps, func.return_type.clone()));
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
            string_literals: Vec::new(),
            string_counter: 0,
        }
    }

    pub fn codegen(mut self) -> IResult<Program> {
        let mut items = Vec::new();
        let decls = std::mem::take(&mut self.program.0);
        for decl in decls {
            match decl.kind {
                DeclKind::Function(func) => {
                    if let Some(function) = self.gen_function_decl(func)? {
                        items.push(TopLevel::Function(function));
                    }
                }
                DeclKind::Variable(var) => {
                    if let Some(static_var) = self.gen_static_variable(var)? {
                        items.push(TopLevel::StaticVariable(static_var));
                    }
                }
            }
        }
        for constant in self.string_literals.drain(..) {
            items.push(TopLevel::StaticConstant(constant));
        }
        Ok(Program {
            items,
            global_types: self.global_types.clone(),
        })
    }

    fn gen_function_decl(&mut self, decl: FunctionDecl) -> IResult<Option<Function>> {
        let FunctionDecl {
            name,
            params,
            body,
            storage_class,
            return_type,
        } = decl;
        let body = if let Some(b) = body {
            b
        } else {
            return Ok(None);
        };
        let global = storage_class != Some(StorageClass::Static);
        self.gen_function(name, global, params, body, return_type)
            .map(Some)
    }

    fn gen_function(
        &mut self,
        name: String,
        global: bool,
        params: Vec<ParameterDecl>,
        body: Vec<Stmt>,
        return_type: Type,
    ) -> IResult<Function> {
        if !self.loop_stack.is_empty() {
            return Err(IRError::State("loop stack not empty at fn entry"));
        }
        let prev_ret = self.current_return_type.replace(return_type.clone());
        self.locals.clear();
        self.value_types.clear();

        let mut param_names = Vec::new();
        for p in &params {
            self.register_local(&p.name, &p.r#type);
            param_names.push(p.name.clone());
        }

        let mut instructions = Vec::new();
        for s in &body {
            self.gen_stmt(s, &mut instructions)?;
        }

        if !self.loop_stack.is_empty() {
            return Err(IRError::State("loop stack not empty after function"));
        }
        self.locals.clear();
        let value_types = std::mem::take(&mut self.value_types);
        self.current_return_type = prev_ret;

        Ok(Function {
            name,
            global,
            params: param_names,
            return_type,
            instructions,
            value_types,
        })
    }

    fn gen_static_variable(&mut self, decl: VariableDecl) -> IResult<Option<StaticVariable>> {
        let VariableDecl {
            name,
            init,
            storage_class,
            r#type,
            is_definition,
        } = decl;
        if !is_definition {
            return Ok(None);
        }
        if matches!(storage_class, Some(StorageClass::Extern)) {
            return Ok(None);
        }

        let mut init_list = Vec::new();
        if let Some(expr) = init {
            match (&r#type, &expr.kind) {
                (Type::Array(elem, size), ExprKind::String(value)) if elem.as_ref().is_char() => {
                    let (bytes, null_terminated) = Self::char_array_bytes(value, *size)?;
                    init_list.push(StaticInit::Bytes {
                        offset: 0,
                        value: bytes,
                        null_terminated,
                    });
                }
                (Type::Array(_, _), _) => return Err(IRError::UnsupportedArrayInit(name.clone())),
                (Type::IncompleteArray(_), _) => {
                    return Err(IRError::UnsupportedArrayInit(name.clone()));
                }
                (_, ExprKind::String(value)) => {
                    let symbol = self.intern_string_literal(value);
                    init_list.push(StaticInit::Label { offset: 0, symbol });
                }
                (_, ExprKind::Constant(c)) => {
                    let const_value = Self::convert_constant(c.clone(), &expr.r#type, &r#type)?;
                    init_list.push(StaticInit::Scalar {
                        offset: 0,
                        value: const_value,
                    });
                }
                _ => {
                    if r#type == Type::Double {
                        return Err(IRError::NonConstDoubleInit);
                    }
                    let value = Self::eval_const_expr(&expr)?;
                    let const_value =
                        Self::convert_constant(Const::Long(value), &Type::Long, &r#type)?;
                    init_list.push(StaticInit::Scalar {
                        offset: 0,
                        value: const_value,
                    });
                }
            }
        }
        let global = storage_class != Some(StorageClass::Static);
        Ok(Some(StaticVariable {
            name,
            global,
            ty: r#type,
            init: init_list,
        }))
    }

    fn eval_const_expr(expr: &Expr) -> IResult<i64> {
        use ExprKind::*;
        Ok(match &expr.kind {
            ExprKind::Constant(Const::Char(n)) => *n as i64,
            ExprKind::Constant(Const::UChar(n)) => *n as i64,
            ExprKind::Constant(Const::Int(n)) => i64::from(*n),
            ExprKind::Constant(Const::Long(n)) => *n,
            ExprKind::Constant(Const::UInt(n)) => *n as i64,
            ExprKind::Constant(Const::ULong(n)) => *n as i64,
            Neg(inner) => -Self::eval_const_expr(inner)?,
            BitNot(inner) | Not(inner) => !Self::eval_const_expr(inner)?,
            Add(lhs, rhs) => Self::eval_const_expr(lhs)? + Self::eval_const_expr(rhs)?,
            Sub(lhs, rhs) => Self::eval_const_expr(lhs)? - Self::eval_const_expr(rhs)?,
            Mul(lhs, rhs) => Self::eval_const_expr(lhs)? * Self::eval_const_expr(rhs)?,
            Div(lhs, rhs) => {
                let d = Self::eval_const_expr(rhs)?;
                if d == 0 {
                    return Err(IRError::DivByZeroInStaticInit);
                }
                Self::eval_const_expr(lhs)? / d
            }
            Rem(lhs, rhs) => {
                let d = Self::eval_const_expr(rhs)?;
                if d == 0 {
                    return Err(IRError::DivByZeroInStaticInit);
                }
                Self::eval_const_expr(lhs)? % d
            }
            Equal(lhs, rhs) => (Self::eval_const_expr(lhs)? == Self::eval_const_expr(rhs)?) as i64,
            NotEqual(lhs, rhs) => {
                (Self::eval_const_expr(lhs)? != Self::eval_const_expr(rhs)?) as i64
            }
            LessThan(lhs, rhs) => {
                (Self::eval_const_expr(lhs)? < Self::eval_const_expr(rhs)?) as i64
            }
            LessThanEqual(lhs, rhs) => {
                (Self::eval_const_expr(lhs)? <= Self::eval_const_expr(rhs)?) as i64
            }
            GreaterThan(lhs, rhs) => {
                (Self::eval_const_expr(lhs)? > Self::eval_const_expr(rhs)?) as i64
            }
            GreaterThanEqual(lhs, rhs) => {
                (Self::eval_const_expr(lhs)? >= Self::eval_const_expr(rhs)?) as i64
            }
            BitAnd(lhs, rhs) => Self::eval_const_expr(lhs)? & Self::eval_const_expr(rhs)?,
            BitOr(lhs, rhs) => Self::eval_const_expr(lhs)? | Self::eval_const_expr(rhs)?,
            Xor(lhs, rhs) => Self::eval_const_expr(lhs)? ^ Self::eval_const_expr(rhs)?,
            LeftShift(lhs, rhs) => {
                Self::eval_const_expr(lhs)? << (Self::eval_const_expr(rhs)? as u32)
            }
            RightShift(lhs, rhs) => {
                Self::eval_const_expr(lhs)? >> (Self::eval_const_expr(rhs)? as u32)
            }
            And(lhs, rhs) => {
                let l = Self::eval_const_expr(lhs)?;
                if l == 0 {
                    0
                } else {
                    (Self::eval_const_expr(rhs)? != 0) as i64
                }
            }
            Or(lhs, rhs) => {
                let l = Self::eval_const_expr(lhs)?;
                (l != 0 || Self::eval_const_expr(rhs)? != 0) as i64
            }
            Conditional(c, t, e) => {
                if Self::eval_const_expr(c)? != 0 {
                    Self::eval_const_expr(t)?
                } else {
                    Self::eval_const_expr(e)?
                }
            }
            Cast(_, inner) => Self::eval_const_expr(inner)?,
            _ => return Err(IRError::NonConstInitializer),
        })
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

        if let Value::Constant(c) = value {
            let converted = match Self::convert_constant(c, &from_type, &to_type) {
                Ok(v) => v,
                Err(_) => return Value::Constant(Const::Int(0)), // unreachable in well-typed codegen; keep IR moving
            };
            return Value::Constant(converted);
        }

        let tmp = self.fresh_tmp();
        self.record_temp(&tmp, &to_type);
        let dst = Value::Var(tmp);

        if from_type.is_integer() && to_type.is_integer() {
            let from_bits = &from_type.bit_width();
            let to_bits = &to_type.bit_width();
            if from_bits == to_bits {
                instructions.push(Instruction::Copy {
                    src: value,
                    dst: dst.clone(),
                });
            } else if to_bits > from_bits {
                let instr = if from_type.is_unsigned() {
                    Instruction::ZeroExtend {
                        src: value,
                        dst: dst.clone(),
                    }
                } else {
                    Instruction::SignExtend {
                        src: value,
                        dst: dst.clone(),
                    }
                };
                instructions.push(instr);
            } else {
                instructions.push(Instruction::Truncate {
                    src: value,
                    dst: dst.clone(),
                });
            }
        } else if from_type.is_numeric() && to_type.is_numeric() {
            instructions.push(Instruction::Convert {
                src: value,
                dst: dst.clone(),
                from: from_type.clone(),
                to: to_type.clone(),
            });
        } else if (from_type.is_pointer() && to_type.is_pointer())
            || (from_type.is_pointer() && to_type.is_integer())
            || (from_type.is_integer() && to_type.is_pointer())
        {
            instructions.push(Instruction::Copy {
                src: value,
                dst: dst.clone(),
            });
        } else {
            // Keep IR consistent; emit a raw copy (front-end should prevent this)
            instructions.push(Instruction::Copy {
                src: value,
                dst: dst.clone(),
            });
        }
        dst
    }

    fn type_of_value(&self, value: &Value) -> IResult<Type> {
        Ok(match value {
            Value::Constant(Const::Char(_)) => Type::Int,
            Value::Constant(Const::UChar(_)) => Type::UInt,
            Value::Constant(Const::Short(_)) => Type::Short,
            Value::Constant(Const::UShort(_)) => Type::UShort,
            Value::Constant(Const::Int(_)) => Type::Int,
            Value::Constant(Const::Long(_)) => Type::Long,
            Value::Constant(Const::UInt(_)) => Type::UInt,
            Value::Constant(Const::ULong(_)) => Type::ULong,
            Value::Constant(Const::Double(_)) => Type::Double,
            Value::Var(name) => self
                .value_types
                .get(name)
                .cloned()
                .ok_or_else(|| IRError::UnknownTemporary(name.clone()))?,
            Value::Global(name) => self
                .global_types
                .get(name)
                .cloned()
                .ok_or_else(|| IRError::UnknownGlobal(name.clone()))?,
        })
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

    fn common_numeric_type(lhs: &Type, rhs: &Type) -> IResult<Type> {
        if lhs.is_floating() || rhs.is_floating() {
            return Ok(Type::Double);
        }
        if !lhs.is_integer() || !rhs.is_integer() {
            return Err(IRError::UnsupportedOperands(lhs.clone(), rhs.clone()));
        }
        let lr = lhs.type_rank();
        let rr = rhs.type_rank();
        Ok(if lr >= rr { lhs.clone() } else { rhs.clone() })
    }

    fn mask(bits: u32) -> u128 {
        if bits == 0 {
            0
        } else if bits >= 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        }
    }

    fn convert_constant(constant: Const, from_type: &Type, to_type: &Type) -> IResult<Const> {
        if from_type == to_type {
            return Ok(constant);
        }

        if matches!(to_type, Type::Double) {
            let value = match (constant, from_type) {
                (Const::Double(v), _) => v,
                (Const::Char(n), _) => n as f64,
                (Const::UChar(n), _) => n as f64,
                (Const::Short(n), _) => n as f64,
                (Const::UShort(n), _) => n as f64,
                (Const::Int(n), _) => f64::from(n),
                (Const::Long(n), _) => n as f64,
                (Const::UInt(n), _) => n as f64,
                (Const::ULong(n), _) => n as f64,
            };
            return Ok(Const::Double(value));
        }

        if matches!(from_type, Type::Double) {
            let value = match constant {
                Const::Double(v) => v,
                _ => return Err(IRError::BadDoubleConstant),
            };
            return Ok(match to_type {
                Type::Char | Type::SChar => Const::Char(value as i8),
                Type::UChar => Const::UChar(value as u8),
                Type::Short => Const::Short(value as i16),
                Type::UShort => Const::UShort(value as u16),
                Type::Int => Const::Int(value as i32),
                Type::Long => Const::Long(value as i64),
                Type::UInt => Const::UInt(value as u32),
                Type::ULong => Const::ULong(value as u64),
                Type::Double => unreachable!(),
                Type::Void => return Err(IRError::BadConstTarget(Type::Void)),
                Type::Pointer(_) => return Err(IRError::BadFloatConstTarget("pointer")),
                Type::FunType(_, _) => return Err(IRError::BadFloatConstTarget("function type")),
                Type::Array(_, _) => return Err(IRError::BadFloatConstTarget("array type")),
                Type::IncompleteArray(_) => {
                    return Err(IRError::BadFloatConstTarget("incomplete array type"));
                }
            });
        }

        let from_bits = from_type.bit_width();
        let to_bits = to_type.bit_width();
        let from_unsigned = from_type.is_unsigned();

        let mut raw = match constant {
            Const::Char(n) => n as u128,
            Const::UChar(n) => n as u128,
            Const::Short(n) => n as u128,
            Const::UShort(n) => n as u128,
            Const::Int(n) => n as u128,
            Const::UInt(n) => n as u128,
            Const::Long(n) => n as u128,
            Const::ULong(n) => n as u128,
            Const::Double(_) => unreachable!("handled above"),
        };

        raw &= Self::mask(from_bits as u32);
        if to_bits > from_bits && !from_unsigned {
            let sign_bit = 1u128 << (from_bits - 1);
            if raw & sign_bit != 0 {
                raw |= (!0u128) << from_bits;
            }
        }
        raw &= Self::mask(to_bits as u32);

        Ok(match to_type {
            Type::Char | Type::SChar => Const::Char(raw as i8),
            Type::UChar => Const::UChar(raw as u8),
            Type::Short => Const::Short(raw as i16),
            Type::UShort => Const::UShort(raw as u16),
            Type::Int => Const::Int(raw as i32),
            Type::UInt => Const::UInt(raw as u32),
            Type::Long => Const::Long(raw as i64),
            Type::ULong => Const::ULong(raw as u64),
            Type::Double => unreachable!(),
            Type::Void => return Err(IRError::BadConstTarget(Type::Void)),
            Type::Pointer(_) => {
                return Err(IRError::BadFloatConstTarget("pointer from integer const"));
            }
            Type::FunType(_, _) => {
                return Err(IRError::BadFloatConstTarget("function from integer const"));
            }
            Type::Array(_, _) => {
                return Err(IRError::BadFloatConstTarget("array from integer const"));
            }
            Type::IncompleteArray(_) => {
                return Err(IRError::BadFloatConstTarget(
                    "incomplete array from integer const",
                ));
            }
        })
    }

    fn char_array_bytes(value: &str, size: usize) -> IResult<(Vec<u8>, bool)> {
        let literal = value.as_bytes();
        if literal.len() > size {
            return Err(IRError::StringTooLarge(value.to_string(), size));
        }
        let mut bytes = value.as_bytes().to_vec();
        bytes.push(0);

        let mut data = Vec::with_capacity(size);
        for (idx, byte) in bytes.iter().enumerate() {
            if idx >= size {
                break;
            }
            data.push(*byte);
        }
        while data.len() < size {
            data.push(0);
        }
        let null_terminated = data.last().copied() == Some(0);
        Ok((data, null_terminated))
    }

    fn intern_string_literal(&mut self, value: &str) -> String {
        let name = format!(".LC{}", self.string_counter);
        self.string_counter += 1;
        let mut bytes = value.as_bytes().to_vec();
        bytes.push(0);
        let ty = Type::Array(Box::new(Type::Char), bytes.len());
        let init = StaticInit::Bytes {
            offset: 0,
            value: bytes,
            null_terminated: true,
        };
        self.string_literals.push(StaticConstant {
            name: name.clone(),
            ty: ty.clone(),
            init,
        });
        self.global_types.insert(name.clone(), ty);
        name
    }

    fn gen_stmt(&mut self, stmt: &Stmt, instructions: &mut Vec<Instruction>) -> IResult<()> {
        match &stmt.kind {
            StmtKind::Return(expr) => {
                let value = self.gen_expr(expr, instructions)?;
                let target_type = self
                    .current_return_type
                    .clone()
                    .ok_or(IRError::ReturnOutsideFunction)?;
                let converted =
                    self.convert_value(value, expr.r#type.clone(), target_type, instructions);
                instructions.push(Instruction::Return(converted));
            }
            StmtKind::Expr(expr) => {
                let _ = self.gen_expr(expr, instructions)?;
            }
            StmtKind::Compound(stmts) => {
                for s in stmts {
                    self.gen_stmt(s, instructions)?;
                }
            }
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_val = self.gen_expr(condition, instructions)?;
                let cond_bool =
                    self.convert_value(cond_val, condition.r#type.clone(), Type::Int, instructions);
                let end_label = self.fresh_label("if.end");
                if let Some(else_branch) = else_branch {
                    let else_label = self.fresh_label("if.else");
                    instructions.push(Instruction::JumpIfZero {
                        condition: cond_bool.clone(),
                        target: else_label.clone(),
                    });
                    self.gen_stmt(then_branch, instructions)?;
                    instructions.push(Instruction::Jump(end_label.clone()));
                    instructions.push(Instruction::Label(else_label));
                    self.gen_stmt(else_branch, instructions)?;
                    instructions.push(Instruction::Label(end_label));
                } else {
                    instructions.push(Instruction::JumpIfZero {
                        condition: cond_bool,
                        target: end_label.clone(),
                    });
                    self.gen_stmt(then_branch, instructions)?;
                    instructions.push(Instruction::Label(end_label));
                }
            }
            StmtKind::While {
                condition,
                body,
                loop_id,
            } => {
                let loop_id = loop_id.ok_or(IRError::MissingLoopId("while"))?;
                let cond_label = self.fresh_label("while.cond");
                let end_label = self.fresh_label("while.end");
                instructions.push(Instruction::Label(cond_label.clone()));
                let cond_val = self.gen_expr(condition, instructions)?;
                let cond_bool =
                    self.convert_value(cond_val, condition.r#type.clone(), Type::Int, instructions);
                instructions.push(Instruction::JumpIfZero {
                    condition: cond_bool,
                    target: end_label.clone(),
                });
                let context = LoopContext {
                    id: loop_id,
                    break_label: end_label.clone(),
                    continue_label: cond_label.clone(),
                };
                self.push_loop_context(context);
                self.gen_stmt(body, instructions)?;
                self.pop_loop_context(loop_id)?;
                instructions.push(Instruction::Jump(cond_label));
                instructions.push(Instruction::Label(end_label));
            }
            StmtKind::DoWhile {
                body,
                condition,
                loop_id,
            } => {
                let loop_id = loop_id.ok_or(IRError::MissingLoopId("do-while"))?;
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
                self.gen_stmt(body, instructions)?;
                instructions.push(Instruction::Label(cond_label.clone()));
                let cond_val = self.gen_expr(condition, instructions)?;
                let cond_bool =
                    self.convert_value(cond_val, condition.r#type.clone(), Type::Int, instructions);
                instructions.push(Instruction::JumpIfNotZero {
                    condition: cond_bool,
                    target: body_label,
                });
                self.pop_loop_context(loop_id)?;
                instructions.push(Instruction::Label(end_label));
            }
            StmtKind::For {
                init,
                condition,
                post,
                body,
                loop_id,
            } => {
                let loop_id = loop_id.ok_or(IRError::MissingLoopId("for"))?;
                match init {
                    ForInit::Declaration(decl) => self.gen_stmt(decl, instructions)?,
                    ForInit::Expr(Some(expr)) => {
                        let _ = self.gen_expr(expr, instructions)?;
                    }
                    ForInit::Expr(None) => {}
                }
                let cond_label = self.fresh_label("for.cond");
                let end_label = self.fresh_label("for.end");
                let post_label = post.as_ref().map(|_| self.fresh_label("for.post"));
                instructions.push(Instruction::Label(cond_label.clone()));
                if let Some(cond) = condition {
                    let cond_val = self.gen_expr(cond, instructions)?;
                    let cond_bool =
                        self.convert_value(cond_val, cond.r#type.clone(), Type::Int, instructions);
                    instructions.push(Instruction::JumpIfZero {
                        condition: cond_bool,
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
                self.gen_stmt(body, instructions)?;
                if let Some(post_expr) = post {
                    if let Some(label) = &post_label {
                        instructions.push(Instruction::Label(label.clone()));
                    }
                    let _ = self.gen_expr(post_expr, instructions)?;
                }
                self.pop_loop_context(loop_id)?;
                instructions.push(Instruction::Jump(cond_label));
                instructions.push(Instruction::Label(end_label));
            }
            StmtKind::Break { loop_id } => {
                let id = loop_id.ok_or(IRError::MissingLoopId("break"))?;
                let context = self.loop_context(id)?;
                instructions.push(Instruction::Jump(context.break_label.clone()));
            }
            StmtKind::Continue { loop_id } => {
                let id = loop_id.ok_or(IRError::MissingLoopId("continue"))?;
                let context = self.loop_context(id)?;
                instructions.push(Instruction::Jump(context.continue_label.clone()));
            }
            StmtKind::Declaration(decl) => {
                match decl.storage_class {
                    None => {
                        if decl.is_definition {
                            let mut handled_init = false;
                            if let (Type::Array(elem, size), Some(init_expr)) =
                                (&decl.r#type, &decl.init)
                                && elem.as_ref().is_char()
                            {
                                {
                                    if let ExprKind::String(value) = &init_expr.kind {
                                        let (bytes, _) = Self::char_array_bytes(value, *size)?;
                                        let elem_size = elem.as_ref().byte_size();
                                        for (idx, byte) in bytes.iter().enumerate() {
                                            let offset = (elem_size * idx) as i64;
                                            let const_value = match elem.as_ref() {
                                                Type::Char | Type::SChar => {
                                                    Const::Char(*byte as i8)
                                                }
                                                Type::UChar => Const::UChar(*byte),
                                                _ => unreachable!(
                                                    "element type is verified as char-compatible"
                                                ),
                                            };
                                            instructions.push(Instruction::CopyToOffset {
                                                src: Value::Constant(const_value),
                                                dst: decl.name.clone(),
                                                offset,
                                            });
                                        }
                                        handled_init = true;
                                    }
                                }
                            }

                            if !handled_init && let Some(init_expr) = &decl.init {
                                let value = self.gen_expr(init_expr, instructions)?;
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
                    Some(StorageClass::Extern) => { /* no code */ }
                    Some(StorageClass::Static) => {
                        return Err(IRError::Generic("static local variables are not supported"));
                    }
                }
            }
            StmtKind::Null => {}
        }
        Ok(())
    }

    fn gen_expr(&mut self, expr: &Expr, instructions: &mut Vec<Instruction>) -> IResult<Value> {
        let result_type = expr.r#type.clone();
        Ok(match &expr.kind {
            ExprKind::Constant(c) => Value::Constant(c.clone()),
            ExprKind::String(value) => {
                let symbol = self.intern_string_literal(value);
                let tmp = self.fresh_tmp();
                self.record_temp(&tmp, &result_type);
                let dst = Value::Var(tmp.clone());
                instructions.push(Instruction::GetAddress {
                    src: Value::Global(symbol),
                    dst: dst.clone(),
                });
                dst
            }
            ExprKind::Var(name) => {
                let v = self.value_for_variable(name);
                let storage_ty = self.type_of_value(&v)?;
                if matches!(storage_ty, Type::Array(_, _) | Type::IncompleteArray(_))
                    && matches!(result_type, Type::Pointer(_))
                {
                    let tmp = self.fresh_tmp();
                    self.record_temp(&tmp, &result_type);
                    let dst = Value::Var(tmp.clone());
                    instructions.push(Instruction::GetAddress {
                        src: v,
                        dst: dst.clone(),
                    });
                    dst
                } else {
                    v
                }
            }
            ExprKind::FunctionCall(name, args) => {
                self.gen_function_call(name, args, &result_type, instructions)?
            }
            ExprKind::Cast(target, inner) => {
                let v = self.gen_expr(inner, instructions)?;
                self.convert_value(v, inner.r#type.clone(), target.clone(), instructions)
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
            ExprKind::AddrOf(inner) => {
                let (ptr_value, ptr_type) = self.gen_lvalue_address(inner, instructions)?;
                self.convert_value(ptr_value, ptr_type, result_type.clone(), instructions)
            }
            ExprKind::Dereference(inner) => {
                let pointer_type = Type::Pointer(Box::new(result_type.clone()));
                let ptr_value = self.gen_expr(inner, instructions)?;
                let ptr =
                    self.convert_value(ptr_value, inner.r#type.clone(), pointer_type, instructions);
                let tmp = self.fresh_tmp();
                self.record_temp(&tmp, &result_type);
                let dst = Value::Var(tmp.clone());
                instructions.push(Instruction::Load {
                    src_ptr: ptr,
                    dst: dst.clone(),
                });
                dst
            }
            ExprKind::Add(lhs, rhs) => self.gen_add_expr(lhs, rhs, &result_type, instructions)?,
            ExprKind::Sub(lhs, rhs) => self.gen_sub_expr(lhs, rhs, &result_type, instructions)?,
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
                if lhs.r#type.is_pointer() || rhs.r#type.is_pointer() {
                    self.gen_pointer_equality(
                        BinaryOp::Equal,
                        lhs,
                        rhs,
                        &result_type,
                        instructions,
                    )?
                } else {
                    self.gen_binary_expr(BinaryOp::Equal, lhs, rhs, &result_type, instructions)
                }
            }
            ExprKind::NotEqual(lhs, rhs) => {
                if lhs.r#type.is_pointer() || rhs.r#type.is_pointer() {
                    self.gen_pointer_equality(
                        BinaryOp::NotEqual,
                        lhs,
                        rhs,
                        &result_type,
                        instructions,
                    )?
                } else {
                    self.gen_binary_expr(BinaryOp::NotEqual, lhs, rhs, &result_type, instructions)
                }
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
            ExprKind::And(lhs, rhs) => self.gen_logical_and(lhs, rhs, instructions)?,
            ExprKind::Or(lhs, rhs) => self.gen_logical_or(lhs, rhs, instructions)?,
            ExprKind::PreIncrement(e) => self.gen_inc_dec(e, BinaryOp::Add, true, instructions)?,
            ExprKind::PreDecrement(e) => {
                self.gen_inc_dec(e, BinaryOp::Subtract, true, instructions)?
            }
            ExprKind::PostIncrement(e) => {
                self.gen_inc_dec(e, BinaryOp::Add, false, instructions)?
            }
            ExprKind::PostDecrement(e) => {
                self.gen_inc_dec(e, BinaryOp::Subtract, false, instructions)?
            }
            ExprKind::Conditional(c, t, e) => {
                let cond_value = self.gen_expr(c, instructions)?;
                let cond_bool =
                    self.convert_value(cond_value, c.r#type.clone(), Type::Int, instructions);
                let false_label = self.fresh_label("cond.false");
                let end_label = self.fresh_label("cond.end");
                instructions.push(Instruction::JumpIfZero {
                    condition: cond_bool,
                    target: false_label.clone(),
                });

                let result_tmp = self.fresh_tmp();
                self.record_temp(&result_tmp, &result_type);
                let result = Value::Var(result_tmp.clone());

                let then_value = self.gen_expr(t, instructions)?;
                let then_conv = self.convert_value(
                    then_value,
                    t.r#type.clone(),
                    result_type.clone(),
                    instructions,
                );
                instructions.push(Instruction::Copy {
                    src: then_conv,
                    dst: result.clone(),
                });
                instructions.push(Instruction::Jump(end_label.clone()));

                instructions.push(Instruction::Label(false_label));
                let else_value = self.gen_expr(e, instructions)?;
                let else_conv = self.convert_value(
                    else_value,
                    e.r#type.clone(),
                    result_type.clone(),
                    instructions,
                );
                instructions.push(Instruction::Copy {
                    src: else_conv,
                    dst: result.clone(),
                });
                instructions.push(Instruction::Label(end_label));
                Value::Var(result_tmp)
            }
            ExprKind::Assignment(lhs, rhs) => match &lhs.kind {
                ExprKind::Var(name) => {
                    let target = self.value_for_variable(name);
                    let target_type = self.type_of_value(&target)?;
                    let rhs_value = self.gen_expr(rhs, instructions)?;
                    let converted_rhs = self.convert_value(
                        rhs_value,
                        rhs.r#type.clone(),
                        target_type.clone(),
                        instructions,
                    );
                    instructions.push(Instruction::Copy {
                        src: converted_rhs.clone(),
                        dst: target.clone(),
                    });
                    converted_rhs
                }
                ExprKind::Dereference(ptr_expr) => {
                    let pointer_type = ptr_expr.r#type.clone();
                    let expected_ptr = Type::Pointer(Box::new(lhs.r#type.clone()));
                    let ptr_value = self.gen_expr(ptr_expr, instructions)?;
                    let ptr =
                        self.convert_value(ptr_value, pointer_type, expected_ptr, instructions);
                    let rhs_value = self.gen_expr(rhs, instructions)?;
                    let converted_rhs = self.convert_value(
                        rhs_value,
                        rhs.r#type.clone(),
                        lhs.r#type.clone(),
                        instructions,
                    );
                    instructions.push(Instruction::Store {
                        src: converted_rhs.clone(),
                        dst_ptr: ptr,
                    });
                    converted_rhs
                }
                _ => return Err(IRError::BadAssignTarget),
            },
        })
    }

    fn gen_lvalue_address(
        &mut self,
        expr: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> IResult<(Value, Type)> {
        match &expr.kind {
            ExprKind::Var(name) => {
                let base = self.value_for_variable(name);
                let ptr_type = Type::Pointer(Box::new(expr.r#type.clone()));
                let tmp = self.fresh_tmp();
                self.record_temp(&tmp, &ptr_type);
                let dst = Value::Var(tmp);
                instructions.push(Instruction::GetAddress {
                    src: base,
                    dst: dst.clone(),
                });
                Ok((dst, ptr_type))
            }
            ExprKind::Dereference(inner) => {
                let value = self.gen_expr(inner, instructions)?;
                Ok((value, inner.r#type.clone()))
            }
            _ => Err(IRError::BadLValueAddressOf),
        }
    }

    fn gen_function_call(
        &mut self,
        name: &str,
        args: &[Expr],
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> IResult<Value> {
        let (param_types, return_type) = self
            .function_signatures
            .get(name)
            .cloned()
            .ok_or_else(|| IRError::CallUndeclared(name.to_string()))?;
        if param_types.len() != args.len() {
            return Err(IRError::WrongArity(
                name.to_string(),
                param_types.len(),
                args.len(),
            ));
        }
        let mut arg_values = Vec::new();
        for (arg, expected_type) in args.iter().zip(param_types.iter()) {
            let v = self.gen_expr(arg, instructions)?;
            let c = self.convert_value(v, arg.r#type.clone(), expected_type.clone(), instructions);
            arg_values.push(c);
        }
        let tmp = self.fresh_tmp();
        self.record_temp(&tmp, &return_type);
        let dst = Value::Var(tmp);
        instructions.push(Instruction::FunCall {
            name: name.to_string(),
            args: arg_values,
            dst: dst.clone(),
        });

        Ok(if return_type == Type::Void {
            dst
        } else {
            self.convert_value(dst, return_type, result_type.clone(), instructions)
        })
    }

    fn gen_unary_expr(
        &mut self,
        op: UnaryOp,
        rhs: &Expr,
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let src_value = self
            .gen_expr(rhs, instructions)
            .unwrap_or(Value::Constant(Const::Int(0)));
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
        let lhs_value = self
            .gen_expr(lhs, instructions)
            .unwrap_or(Value::Constant(Const::Int(0)));
        let rhs_value = self
            .gen_expr(rhs, instructions)
            .unwrap_or(Value::Constant(Const::Int(0)));

        let (lhs_target_type, rhs_target_type) = match op {
            BinaryOp::LeftShift | BinaryOp::RightShift => (lhs.r#type.clone(), Type::Int),
            _ => {
                let common =
                    Self::common_numeric_type(&lhs.r#type, &rhs.r#type).unwrap_or(Type::Int);
                (common.clone(), common)
            }
        };

        let src1 = self.convert_value(
            lhs_value,
            lhs.r#type.clone(),
            lhs_target_type.clone(),
            instructions,
        );
        let src2 = self.convert_value(
            rhs_value,
            rhs.r#type.clone(),
            rhs_target_type.clone(),
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

    fn gen_add_expr(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> IResult<Value> {
        let lp = lhs.r#type.is_pointer();
        let rp = rhs.r#type.is_pointer();
        Ok(match (lp, rp) {
            (true, false) => self.gen_pointer_add(lhs, rhs, result_type, false, instructions)?,
            (false, true) => self.gen_pointer_add(rhs, lhs, result_type, false, instructions)?,
            (true, true) => return Err(IRError::PointerAddUnsupported),
            (false, false) => {
                self.gen_binary_expr(BinaryOp::Add, lhs, rhs, result_type, instructions)
            }
        })
    }

    fn gen_sub_expr(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> IResult<Value> {
        let lp = lhs.r#type.is_pointer();
        let rp = rhs.r#type.is_pointer();
        Ok(match (lp, rp) {
            (true, false) => self.gen_pointer_add(lhs, rhs, result_type, true, instructions)?,
            (true, true) => self.gen_pointer_difference(lhs, rhs, result_type, instructions)?,
            (false, true) => return Err(IRError::IntMinusPtrUnsupported),
            (false, false) => {
                self.gen_binary_expr(BinaryOp::Subtract, lhs, rhs, result_type, instructions)
            }
        })
    }

    fn gen_pointer_add(
        &mut self,
        ptr_expr: &Expr,
        index_expr: &Expr,
        result_type: &Type,
        negate_index: bool,
        instructions: &mut Vec<Instruction>,
    ) -> IResult<Value> {
        let ptr_value = self.gen_expr(ptr_expr, instructions)?;
        let ptr = self.convert_value(
            ptr_value,
            ptr_expr.r#type.clone(),
            result_type.clone(),
            instructions,
        );

        let index_value = self.gen_expr(index_expr, instructions)?;
        let mut index = self.convert_value(
            index_value,
            index_expr.r#type.clone(),
            Type::Long,
            instructions,
        );

        if negate_index {
            let tmp_name = self.fresh_tmp();
            self.record_temp(&tmp_name, &Type::Long);
            let tmp = Value::Var(tmp_name.clone());
            instructions.push(Instruction::Unary {
                op: UnaryOp::Negate,
                src: index,
                dst: tmp.clone(),
            });
            index = tmp;
        }

        let base_type = self.pointer_base_type(&ptr_expr.r#type)?;
        let scale = base_type.byte_size();

        let dst_name = self.fresh_tmp();
        self.record_temp(&dst_name, result_type);
        let dst = Value::Var(dst_name.clone());
        instructions.push(Instruction::AddPtr {
            ptr,
            index,
            scale: scale as i64,
            dst: dst.clone(),
        });
        Ok(dst)
    }

    fn gen_pointer_difference(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> IResult<Value> {
        let lhs_value = self.gen_expr(lhs, instructions)?;
        let rhs_value = self.gen_expr(rhs, instructions)?;
        let lhs_long = self.convert_value(lhs_value, lhs.r#type.clone(), Type::Long, instructions);
        let rhs_long = self.convert_value(rhs_value, rhs.r#type.clone(), Type::Long, instructions);

        let diff_tmp_name = self.fresh_tmp();
        self.record_temp(&diff_tmp_name, &Type::Long);
        let diff_value = Value::Var(diff_tmp_name.clone());
        instructions.push(Instruction::Binary {
            op: BinaryOp::Subtract,
            src1: lhs_long,
            src2: rhs_long,
            dst: diff_value.clone(),
        });

        let base_type = self.pointer_base_type(&lhs.r#type)?;
        let scale = base_type.byte_size();

        let quotient = if scale != 1 {
            let scale_const = Value::Constant(Const::Long(scale as i64));
            let quot_tmp_name = self.fresh_tmp();
            self.record_temp(&quot_tmp_name, &Type::Long);
            let quot_value = Value::Var(quot_tmp_name.clone());
            instructions.push(Instruction::Binary {
                op: BinaryOp::Divide,
                src1: diff_value,
                src2: scale_const,
                dst: quot_value.clone(),
            });
            quot_value
        } else {
            diff_value
        };

        Ok(self.convert_value(quotient, Type::Long, result_type.clone(), instructions))
    }

    fn pointer_base_type<'a>(&self, pointer_type: &'a Type) -> IResult<&'a Type> {
        match pointer_type {
            Type::Pointer(inner) => Ok(inner.as_ref()),
            other => Err(IRError::ExpectedPointer(other.clone())),
        }
    }

    fn gen_pointer_equality(
        &mut self,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> IResult<Value> {
        let lhs_value = self.gen_expr(lhs, instructions)?;
        let rhs_value = self.gen_expr(rhs, instructions)?;
        let lhs_converted =
            self.convert_value(lhs_value, lhs.r#type.clone(), Type::ULong, instructions);
        let rhs_converted =
            self.convert_value(rhs_value, rhs.r#type.clone(), Type::ULong, instructions);
        let tmp = self.fresh_tmp();
        self.record_temp(&tmp, result_type);
        let dst = Value::Var(tmp);
        instructions.push(Instruction::Binary {
            op,
            src1: lhs_converted,
            src2: rhs_converted,
            dst: dst.clone(),
        });
        Ok(dst)
    }

    fn gen_logical_and(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> IResult<Value> {
        let result_tmp = self.fresh_tmp();
        let result = Value::Var(result_tmp.clone());
        let false_label = self.fresh_label("and.false");
        let end_label = self.fresh_label("and.end");
        self.record_temp(&result_tmp, &Type::Int);

        instructions.push(Instruction::Copy {
            src: Value::Constant(Const::Int(0)),
            dst: result.clone(),
        });
        let lhs_val = self.gen_expr(lhs, instructions)?;
        let lhs_cond = self.convert_value(lhs_val, lhs.r#type.clone(), Type::Int, instructions);
        instructions.push(Instruction::JumpIfZero {
            condition: lhs_cond,
            target: false_label.clone(),
        });
        let rhs_val = self.gen_expr(rhs, instructions)?;
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
        Ok(Value::Var(result_tmp))
    }

    fn gen_logical_or(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> IResult<Value> {
        let result_tmp = self.fresh_tmp();
        let result = Value::Var(result_tmp.clone());
        let true_label = self.fresh_label("or.true");
        let end_label = self.fresh_label("or.end");
        self.record_temp(&result_tmp, &Type::Int);

        instructions.push(Instruction::Copy {
            src: Value::Constant(Const::Int(0)),
            dst: result.clone(),
        });
        let lhs_val = self.gen_expr(lhs, instructions)?;
        let lhs_cond = self.convert_value(lhs_val, lhs.r#type.clone(), Type::Int, instructions);
        instructions.push(Instruction::JumpIfNotZero {
            condition: lhs_cond,
            target: true_label.clone(),
        });
        let rhs_val = self.gen_expr(rhs, instructions)?;
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
        Ok(Value::Var(result_tmp))
    }

    fn gen_inc_dec(
        &mut self,
        expr: &Expr,
        op: BinaryOp,
        is_post: bool,
        instructions: &mut Vec<Instruction>,
    ) -> IResult<Value> {
        let name = match &expr.kind {
            ExprKind::Var(n) => n.clone(),
            _ => {
                return Err(IRError::Generic(
                    "increment/decrement target must be a variable",
                ));
            }
        };
        let target = self.value_for_variable(&name);
        let target_type = self.type_of_value(&target)?;

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
        match &target_type {
            Type::Int => instructions.push(Instruction::Binary {
                op,
                src1: target.clone(),
                src2: Value::Constant(Const::Int(1)),
                dst: updated_tmp.clone(),
            }),
            Type::UInt => instructions.push(Instruction::Binary {
                op,
                src1: target.clone(),
                src2: Value::Constant(Const::UInt(1)),
                dst: updated_tmp.clone(),
            }),
            Type::Long => instructions.push(Instruction::Binary {
                op,
                src1: target.clone(),
                src2: Value::Constant(Const::Long(1)),
                dst: updated_tmp.clone(),
            }),
            Type::ULong => instructions.push(Instruction::Binary {
                op,
                src1: target.clone(),
                src2: Value::Constant(Const::ULong(1)),
                dst: updated_tmp.clone(),
            }),
            Type::Char | Type::SChar | Type::UChar => {
                return Err(IRError::Generic(
                    "increment/decrement for char types not yet supported",
                ));
            }
            Type::Short | Type::UShort => {
                return Err(IRError::Generic(
                    "increment/decrement for short types not yet supported",
                ));
            }
            Type::Double => instructions.push(Instruction::Binary {
                op,
                src1: target.clone(),
                src2: Value::Constant(Const::Double(1.0)),
                dst: updated_tmp.clone(),
            }),
            Type::Pointer(inner) => {
                let scale = inner.byte_size();
                let step = match op {
                    BinaryOp::Add => Const::Long(1),
                    BinaryOp::Subtract => Const::Long(-1),
                    _ => {
                        return Err(IRError::Generic(
                            "unsupported operation for pointer inc/dec",
                        ));
                    }
                };
                instructions.push(Instruction::AddPtr {
                    ptr: target.clone(),
                    index: Value::Constant(step),
                    scale: scale as i64,
                    dst: updated_tmp.clone(),
                });
            }
            Type::Void => return Err(IRError::Generic("cannot increment void")),
            Type::FunType(_, _) => {
                return Err(IRError::Generic("function type increment not supported"));
            }
            Type::Array(_, _) | Type::IncompleteArray(_) => {
                return Err(IRError::Generic("array increment not supported"));
            }
        }
        instructions.push(Instruction::Copy {
            src: updated_tmp.clone(),
            dst: target.clone(),
        });

        Ok(if let Some(orig) = original_value {
            orig
        } else {
            target
        })
    }
}
