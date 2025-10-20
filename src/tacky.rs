use std::collections::BTreeMap;
use std::convert::TryFrom;

use crate::parse::{
    Const, DeclKind, Expr, ExprKind, ForInit, FunctionDecl, ParameterDecl, Program as AstProgram,
    Stmt, StmtKind, StorageClass, Type, VariableDecl,
};

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<TopLevel>,
    pub global_types: BTreeMap<String, Type>,
}

#[derive(Debug, Clone)]
pub enum TopLevel {
    Function(Function),
    StaticVariable(StaticVariable),
    StaticConstant(StaticConstant),
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
    pub init: Vec<StaticInit>,
}

#[derive(Debug, Clone)]
pub struct StaticConstant {
    pub name: String,
    pub ty: Type,
    pub init: StaticInit,
}

#[derive(Debug, Clone)]
pub enum StaticInit {
    Scalar {
        offset: i64,
        value: Const,
    },
    Bytes {
        offset: i64,
        value: Vec<u8>,
        null_terminated: bool,
    },
    Label {
        offset: i64,
        symbol: String,
    },
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
    ZeroExtend {
        src: Value,
        dst: Value,
    },
    Truncate {
        src: Value,
        dst: Value,
    },
    Convert {
        src: Value,
        dst: Value,
        from: Type,
        to: Type,
    },
    GetAddress {
        src: Value,
        dst: Value,
    },
    Load {
        src_ptr: Value,
        dst: Value,
    },
    Store {
        src: Value,
        dst_ptr: Value,
    },
    AddPtr {
        ptr: Value,
        index: Value,
        scale: i64,
        dst: Value,
    },
    CopyToOffset {
        src: Value,
        dst: String,
        offset: i64,
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
    string_literals: Vec<StaticConstant>,
    string_counter: usize,
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
            string_literals: Vec::new(),
            string_counter: 0,
        }
    }

    pub fn codegen(mut self) -> Program {
        let mut items = Vec::new();

        let decls = std::mem::take(&mut self.program.0);
        for decl in decls {
            match decl.kind {
                DeclKind::Function(func) => {
                    if let Some(function) = self.gen_function_decl(func) {
                        items.push(TopLevel::Function(function));
                    }
                }
                DeclKind::Variable(var) => {
                    if let Some(static_var) = self.gen_static_variable(var) {
                        items.push(TopLevel::StaticVariable(static_var));
                    }
                }
            }
        }

        for constant in self.string_literals.drain(..) {
            items.push(TopLevel::StaticConstant(constant));
        }

        Program {
            items,
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

        let mut init_list = Vec::new();
        if let Some(expr) = init {
            match (&r#type, &expr.kind) {
                (Type::Array(elem, size), ExprKind::String(value))
                    if Self::is_char_type(elem.as_ref()) =>
                {
                    let (bytes, null_terminated) = Self::char_array_bytes(value, *size);
                    init_list.push(StaticInit::Bytes {
                        offset: 0,
                        value: bytes,
                        null_terminated,
                    });
                }
                (Type::Array(_, _), _) => {
                    panic!("unsupported array initializer for static variable '{name}'");
                }
                (_, ExprKind::String(value)) => {
                    let symbol = self.intern_string_literal(value);
                    init_list.push(StaticInit::Label { offset: 0, symbol });
                }
                (_, ExprKind::Constant(c)) => {
                    let const_value = Self::convert_constant(c.clone(), &expr.r#type, &r#type);
                    init_list.push(StaticInit::Scalar {
                        offset: 0,
                        value: const_value,
                    });
                }
                _ => {
                    if r#type == Type::Double {
                        panic!("non-constant double initializer");
                    }
                    let value = Self::eval_const_expr(&expr);
                    let const_value =
                        Self::convert_constant(Const::Long(value), &Type::Long, &r#type);
                    init_list.push(StaticInit::Scalar {
                        offset: 0,
                        value: const_value,
                    });
                }
            }
        }

        let global = storage_class != Some(StorageClass::Static);

        Some(StaticVariable {
            name,
            global,
            ty: r#type,
            init: init_list,
        })
    }

    fn eval_const_expr(expr: &Expr) -> i64 {
        match &expr.kind {
            ExprKind::Constant(Const::Char(n)) => *n as i64,
            ExprKind::Constant(Const::UChar(n)) => *n as i64,
            ExprKind::Constant(Const::Int(n)) => *n,
            ExprKind::Constant(Const::Long(n)) => *n,
            ExprKind::Constant(Const::UInt(n)) => (*n as u32) as i64,
            ExprKind::Constant(Const::ULong(n)) => *n as i64,
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

        if let Value::Constant(c) = value {
            let converted = Self::convert_constant(c, &from_type, &to_type);
            return Value::Constant(converted);
        }

        let tmp = self.fresh_tmp();
        self.record_temp(&tmp, &to_type);
        let dst = Value::Var(tmp);

        if Self::is_integer_type(&from_type) && Self::is_integer_type(&to_type) {
            let from_bits = Self::bit_width(&from_type);
            let to_bits = Self::bit_width(&to_type);

            if from_bits == to_bits {
                instructions.push(Instruction::Copy {
                    src: value,
                    dst: dst.clone(),
                });
            } else if to_bits > from_bits {
                let instr = if Self::is_unsigned(&from_type) {
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
        } else if Self::is_numeric_type(&from_type) && Self::is_numeric_type(&to_type) {
            instructions.push(Instruction::Convert {
                src: value,
                dst: dst.clone(),
                from: from_type.clone(),
                to: to_type.clone(),
            });
        } else if (Self::is_pointer_type(&from_type) && Self::is_pointer_type(&to_type))
            || (Self::is_pointer_type(&from_type) && Self::is_integer_type(&to_type))
            || (Self::is_integer_type(&from_type) && Self::is_pointer_type(&to_type))
        {
            instructions.push(Instruction::Copy {
                src: value,
                dst: dst.clone(),
            });
        } else {
            panic!(
                "unsupported conversion from {:?} to {:?}",
                from_type, to_type
            );
        }

        dst
    }

    fn type_of_value(&self, value: &Value) -> Type {
        match value {
            Value::Constant(Const::Char(_)) => Type::Int,
            Value::Constant(Const::UChar(_)) => Type::UInt,
            Value::Constant(Const::Int(_)) => Type::Int,
            Value::Constant(Const::Long(_)) => Type::Long,
            Value::Constant(Const::UInt(_)) => Type::UInt,
            Value::Constant(Const::ULong(_)) => Type::ULong,
            Value::Constant(Const::Double(_)) => Type::Double,
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

    fn bit_width(ty: &Type) -> u32 {
        match ty {
            Type::Char | Type::SChar | Type::UChar => 8,
            Type::Int | Type::UInt => 32,
            Type::Long | Type::ULong => 64,
            Type::Double => 64,
            Type::Void => panic!("void type has no bit width"),
            Type::Pointer(_) => 64,
            Type::FunType(_, _) => panic!("function type has no bit width"),
            Type::Array(_, _) => panic!("array type has no bit width"),
        }
    }

    fn is_unsigned(ty: &Type) -> bool {
        matches!(ty, Type::UChar | Type::UInt | Type::ULong)
    }

    fn is_integer_type(ty: &Type) -> bool {
        matches!(
            ty,
            Type::Char
                | Type::SChar
                | Type::UChar
                | Type::Int
                | Type::UInt
                | Type::Long
                | Type::ULong
        )
    }

    fn is_char_type(ty: &Type) -> bool {
        matches!(ty, Type::Char | Type::SChar | Type::UChar)
    }

    fn is_pointer_type(ty: &Type) -> bool {
        matches!(ty, Type::Pointer(_))
    }

    fn is_floating_type(ty: &Type) -> bool {
        matches!(ty, Type::Double)
    }

    fn is_numeric_type(ty: &Type) -> bool {
        Self::is_integer_type(ty) || Self::is_floating_type(ty)
    }

    fn type_rank(ty: &Type) -> usize {
        match ty {
            Type::Char | Type::SChar => 0,
            Type::UChar => 1,
            Type::Int => 2,
            Type::UInt => 3,
            Type::Long => 4,
            Type::ULong => 5,

            Type::Void => panic!("void type has no rank"),
            Type::Double => panic!("double type has no integer rank"),
            Type::Pointer(_) | Type::FunType(_, _) => panic!("pointer type has no rank"),
            Type::Array(_, _) => panic!("array type has no rank"),
        }
    }

    fn common_numeric_type(lhs: &Type, rhs: &Type) -> Type {
        if Self::is_floating_type(lhs) || Self::is_floating_type(rhs) {
            return Type::Double;
        }

        if !Self::is_integer_type(lhs) || !Self::is_integer_type(rhs) {
            panic!("unsupported operand types {:?} and {:?}", lhs, rhs);
        }

        if Self::type_rank(lhs) >= Self::type_rank(rhs) {
            lhs.clone()
        } else {
            rhs.clone()
        }
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

    fn convert_constant(constant: Const, from_type: &Type, to_type: &Type) -> Const {
        if from_type == to_type {
            return constant;
        }

        if matches!(to_type, Type::Double) {
            let value = match (constant, from_type) {
                (Const::Double(v), _) => v,
                (Const::Char(n), _) => n as f64,
                (Const::UChar(n), _) => n as f64,
                (Const::Int(n), _) => n as f64,
                (Const::Long(n), _) => n as f64,
                (Const::UInt(n), _) => n as f64,
                (Const::ULong(n), _) => n as f64,
            };
            return Const::Double(value);
        }

        if matches!(from_type, Type::Double) {
            let value = match constant {
                Const::Double(v) => v,
                _ => panic!("mismatched constant for double conversion"),
            };
            return match to_type {
                Type::Char | Type::SChar => Const::Char(value as i8),
                Type::UChar => Const::UChar(value as u8),
                Type::Int => Const::Int(value as i64),
                Type::Long => Const::Long(value as i64),
                Type::UInt => Const::UInt(value as u64),
                Type::ULong => Const::ULong(value as u64),
                Type::Double => unreachable!(),
                Type::Void => panic!("cannot convert constant to void"),
                Type::Pointer(_) => panic!("cannot convert floating constant to pointer"),
                Type::FunType(_, _) => panic!("cannot convert floating constant to function type"),
                Type::Array(_, _) => panic!("cannot convert floating constant to array type"),
            };
        }

        let from_bits = Self::bit_width(from_type);
        let to_bits = Self::bit_width(to_type);
        let from_unsigned = Self::is_unsigned(from_type);

        let mut raw = match constant {
            Const::Char(n) => (n as i32 as u32) as u128,
            Const::UChar(n) => (n as u32) as u128,
            Const::Int(n) => (n as i32 as u32) as u128,
            Const::UInt(n) => (n as u32) as u128,
            Const::Long(n) => (n as u64) as u128,
            Const::ULong(n) => n as u128,
            Const::Double(_) => unreachable!("double handled earlier"),
        };

        raw &= Self::mask(from_bits);

        if to_bits > from_bits && !from_unsigned {
            let sign_bit = 1u128 << (from_bits - 1);
            if raw & sign_bit != 0 {
                let extension_mask = (!0u128) << from_bits;
                raw |= extension_mask;
            }
        }

        raw &= Self::mask(to_bits);

        match to_type {
            Type::Char | Type::SChar => {
                let value = raw as i8;
                Const::Char(value)
            }
            Type::UChar => {
                let value = raw as u8;
                Const::UChar(value)
            }
            Type::Int => {
                let value = raw as i64;
                Const::Int(value)
            }
            Type::UInt => {
                let value = raw as u64;
                Const::UInt(value)
            }
            Type::Long => {
                let value = raw as i64;
                Const::Long(value)
            }
            Type::ULong => Const::ULong(raw as u64),
            Type::Double => unreachable!(),
            Type::Void => panic!("cannot convert constant to void"),
            Type::Pointer(_) => panic!("cannot convert integer constant to pointer"),
            Type::FunType(_, _) => panic!("cannot convert integer constant to function type"),
            Type::Array(_, _) => panic!("cannot convert integer constant to array type"),
        }
    }

    fn char_array_bytes(value: &str, size: usize) -> (Vec<u8>, bool) {
        let literal = value.as_bytes();
        if literal.len() > size {
            panic!(
                "string literal initializer \"{}\" does not fit in array of size {}",
                value, size
            );
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
        (data, null_terminated)
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
                let cond_bool =
                    self.convert_value(cond_val, condition.r#type.clone(), Type::Int, instructions);
                let end_label = self.fresh_label("if.end");
                if let Some(else_branch) = else_branch {
                    let else_label = self.fresh_label("if.else");
                    instructions.push(Instruction::JumpIfZero {
                        condition: cond_bool.clone(),
                        target: else_label.clone(),
                    });
                    self.gen_stmt(then_branch, instructions);
                    instructions.push(Instruction::Jump(end_label.clone()));
                    instructions.push(Instruction::Label(else_label));
                    self.gen_stmt(else_branch, instructions);
                    instructions.push(Instruction::Label(end_label));
                } else {
                    instructions.push(Instruction::JumpIfZero {
                        condition: cond_bool,
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
                let cond_bool =
                    self.convert_value(cond_val, condition.r#type.clone(), Type::Int, instructions);
                instructions.push(Instruction::JumpIfNotZero {
                    condition: cond_bool,
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
                let value = self.value_for_variable(name);
                let storage_ty = self.type_of_value(&value);
                if matches!(storage_ty, Type::Array(_, _))
                    && matches!(result_type, Type::Pointer(_))
                {
                    let tmp = self.fresh_tmp();
                    self.record_temp(&tmp, &result_type);
                    let dst = Value::Var(tmp.clone());
                    instructions.push(Instruction::GetAddress {
                        src: value,
                        dst: dst.clone(),
                    });
                    dst
                } else {
                    value
                }
            }
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
            ExprKind::AddrOf(inner) => {
                let (ptr_value, ptr_type) = self.gen_lvalue_address(inner, instructions);
                self.convert_value(ptr_value, ptr_type, result_type.clone(), instructions)
            }
            ExprKind::Dereference(inner) => {
                let pointer_type = Type::Pointer(Box::new(result_type.clone()));
                let ptr_value = self.gen_expr(inner, instructions);
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
            ExprKind::Add(lhs, rhs) => self.gen_add_expr(lhs, rhs, &result_type, instructions),
            ExprKind::Sub(lhs, rhs) => self.gen_sub_expr(lhs, rhs, &result_type, instructions),
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
                if Self::is_pointer_type(&lhs.r#type) || Self::is_pointer_type(&rhs.r#type) {
                    self.gen_pointer_equality(BinaryOp::Equal, lhs, rhs, &result_type, instructions)
                } else {
                    self.gen_binary_expr(BinaryOp::Equal, lhs, rhs, &result_type, instructions)
                }
            }
            ExprKind::NotEqual(lhs, rhs) => {
                if Self::is_pointer_type(&lhs.r#type) || Self::is_pointer_type(&rhs.r#type) {
                    self.gen_pointer_equality(
                        BinaryOp::NotEqual,
                        lhs,
                        rhs,
                        &result_type,
                        instructions,
                    )
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
            ExprKind::Assignment(lhs, rhs) => match &lhs.kind {
                ExprKind::Var(name) => {
                    let target = self.value_for_variable(name);
                    let target_type = self.type_of_value(&target);
                    let rhs_value = self.gen_expr(rhs, instructions);
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
                    let ptr_value = self.gen_expr(ptr_expr, instructions);
                    let ptr =
                        self.convert_value(ptr_value, pointer_type, expected_ptr, instructions);
                    let rhs_value = self.gen_expr(rhs, instructions);
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
                _ => panic!("assignment target must be assignable"),
            },
        }
    }

    fn gen_lvalue_address(
        &mut self,
        expr: &Expr,
        instructions: &mut Vec<Instruction>,
    ) -> (Value, Type) {
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
                (dst, ptr_type)
            }
            ExprKind::Dereference(inner) => {
                let value = self.gen_expr(inner, instructions);
                (value, inner.r#type.clone())
            }
            _ => panic!("unsupported lvalue for address-of"),
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

        let (lhs_target_type, rhs_target_type) = match op {
            BinaryOp::LeftShift | BinaryOp::RightShift => (lhs.r#type.clone(), Type::Int),
            _ => {
                let common = Self::common_numeric_type(&lhs.r#type, &rhs.r#type);
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
    ) -> Value {
        let lhs_is_pointer = Self::is_pointer_type(&lhs.r#type);
        let rhs_is_pointer = Self::is_pointer_type(&rhs.r#type);

        match (lhs_is_pointer, rhs_is_pointer) {
            (true, false) => self.gen_pointer_add(lhs, rhs, result_type, false, instructions),
            (false, true) => self.gen_pointer_add(rhs, lhs, result_type, false, instructions),
            (true, true) => panic!("pointer addition is not supported"),
            (false, false) => {
                self.gen_binary_expr(BinaryOp::Add, lhs, rhs, result_type, instructions)
            }
        }
    }

    fn gen_sub_expr(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let lhs_is_pointer = Self::is_pointer_type(&lhs.r#type);
        let rhs_is_pointer = Self::is_pointer_type(&rhs.r#type);

        match (lhs_is_pointer, rhs_is_pointer) {
            (true, false) => self.gen_pointer_add(lhs, rhs, result_type, true, instructions),
            (true, true) => self.gen_pointer_difference(lhs, rhs, result_type, instructions),
            (false, true) => panic!("integer minus pointer is not supported"),
            (false, false) => {
                self.gen_binary_expr(BinaryOp::Subtract, lhs, rhs, result_type, instructions)
            }
        }
    }

    fn gen_pointer_add(
        &mut self,
        ptr_expr: &Expr,
        index_expr: &Expr,
        result_type: &Type,
        negate_index: bool,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let ptr_value = self.gen_expr(ptr_expr, instructions);
        let ptr = self.convert_value(
            ptr_value,
            ptr_expr.r#type.clone(),
            result_type.clone(),
            instructions,
        );

        let index_value = self.gen_expr(index_expr, instructions);
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

        let base_type = self.pointer_base_type(&ptr_expr.r#type);
        let scale = Self::size_of_type(base_type);
        if scale <= 0 {
            panic!("invalid pointer element size {scale}");
        }

        let dst_name = self.fresh_tmp();
        self.record_temp(&dst_name, result_type);
        let dst = Value::Var(dst_name.clone());
        instructions.push(Instruction::AddPtr {
            ptr,
            index,
            scale,
            dst: dst.clone(),
        });

        dst
    }

    fn gen_pointer_difference(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let lhs_value = self.gen_expr(lhs, instructions);
        let rhs_value = self.gen_expr(rhs, instructions);

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

        let base_type = self.pointer_base_type(&lhs.r#type);
        let scale = Self::size_of_type(base_type);
        if scale <= 0 {
            panic!("invalid pointer element size {scale}");
        }

        let quotient = if scale != 1 {
            let scale_const = Value::Constant(Const::Long(scale));
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

        self.convert_value(quotient, Type::Long, result_type.clone(), instructions)
    }

    fn pointer_base_type<'a>(&self, pointer_type: &'a Type) -> &'a Type {
        match pointer_type {
            Type::Pointer(inner) => inner.as_ref(),
            other => panic!("expected pointer type, found {other:?}"),
        }
    }

    fn size_of_type(ty: &Type) -> i64 {
        match ty {
            Type::Char | Type::SChar | Type::UChar => 1,
            Type::Int | Type::UInt => 4,
            Type::Long | Type::ULong => 8,
            Type::Double => 8,
            Type::Pointer(_) => 8,
            Type::Array(inner, len) => {
                let len_i64 = i64::try_from(*len).expect("array size exceeds i64");
                len_i64 * Self::size_of_type(inner)
            }
            Type::Void => panic!("void type has no size"),
            Type::FunType(_, _) => panic!("function type has no size"),
        }
    }

    fn gen_pointer_equality(
        &mut self,
        op: BinaryOp,
        lhs: &Expr,
        rhs: &Expr,
        result_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> Value {
        let lhs_value = self.gen_expr(lhs, instructions);
        let rhs_value = self.gen_expr(rhs, instructions);

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
        match &target_type {
            Type::Int => {
                instructions.push(Instruction::Binary {
                    op,
                    src1: target.clone(),
                    src2: Value::Constant(Const::Int(1)),
                    dst: updated_tmp.clone(),
                });
            }
            Type::UInt => {
                instructions.push(Instruction::Binary {
                    op,
                    src1: target.clone(),
                    src2: Value::Constant(Const::UInt(1)),
                    dst: updated_tmp.clone(),
                });
            }
            Type::Long => {
                instructions.push(Instruction::Binary {
                    op,
                    src1: target.clone(),
                    src2: Value::Constant(Const::Long(1)),
                    dst: updated_tmp.clone(),
                });
            }
            Type::ULong => {
                instructions.push(Instruction::Binary {
                    op,
                    src1: target.clone(),
                    src2: Value::Constant(Const::ULong(1)),
                    dst: updated_tmp.clone(),
                });
            }
            Type::Char | Type::SChar | Type::UChar => {
                panic!("increment/decrement for char types not yet supported");
            }
            Type::Double => {
                instructions.push(Instruction::Binary {
                    op,
                    src1: target.clone(),
                    src2: Value::Constant(Const::Double(1.0)),
                    dst: updated_tmp.clone(),
                });
            }
            Type::Pointer(inner) => {
                let scale = Self::size_of_type(inner);
                if scale <= 0 {
                    panic!("invalid pointer element size {scale}");
                }
                let step = match op {
                    BinaryOp::Add => Const::Long(1),
                    BinaryOp::Subtract => Const::Long(-1),
                    _ => panic!("unsupported operation for pointer increment/decrement"),
                };
                instructions.push(Instruction::AddPtr {
                    ptr: target.clone(),
                    index: Value::Constant(step),
                    scale,
                    dst: updated_tmp.clone(),
                });
            }
            Type::Void => panic!("cannot increment void"),
            Type::FunType(_, _) => panic!("function type increment not supported"),
            Type::Array(_, _) => panic!("array increment not supported"),
        }
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
