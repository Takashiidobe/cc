use std::collections::BTreeMap;

use anyhow::Result;
use thiserror::Error;

use crate::{
    parse::{
        Const, Decl, DeclKind, Expr, ExprKind, ForInit, FunctionDecl, ParameterDecl, Program, Stmt,
        StmtKind, StorageClass, Type, VariableDecl,
    },
    tokenize::TokenKind,
};

#[derive(Error, Debug)]
pub enum SemanticError {
    #[error("Cannot cast to `{0}`")]
    CastTo(Type),
    #[error("{0}")]
    Unsupported(String),
    #[error("Variable `{0}` cannot be type `{1}`")]
    InvalidVarType(String, Type),
    #[error("Param `{0}` cannot be type `{1}`")]
    InvalidParamType(String, Type),
    #[error("Cannot dereference type: `{0}`")]
    DereferenceError(Type),
    #[error("String literal cannot initialize type {0}")]
    InvalidStringLiteral(Type),
    #[error("Function '{0}' called with invalid arguments (expected {1:?}, got {2:?})")]
    InvalidFunctionArguments(String, Vec<Type>, Vec<Type>),
    #[error("Undeclared identifier: {0}")]
    UndeclaredIdentifier(String),
    #[error("Invalid Binary Expression: {0:?} {1:?} {2:?}")]
    InvalidBinaryExpr(Expr, TokenKind, Expr),
}

#[derive(Clone)]
struct Symbol {
    unique: String,
    ty: Type,
}

#[derive(Clone)]
struct FunctionSignature {
    return_type: Type,
    param_types: Vec<Type>,
}

#[derive(Default)]
pub struct SemanticAnalyzer {
    counter: usize,
    scopes: Vec<BTreeMap<String, Symbol>>,
    functions: BTreeMap<String, FunctionSignature>,
    current_return_type: Option<Type>,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn analyze_program(mut self, program: Program) -> Result<Program, SemanticError> {
        self.enter_scope();

        for decl in &program.0 {
            match &decl.kind {
                DeclKind::Variable(var) => {
                    self.insert_symbol(
                        var.name.clone(),
                        Symbol {
                            unique: var.name.clone(),
                            ty: var.r#type.clone(),
                        },
                    );
                }
                DeclKind::Function(func) => {
                    if Self::is_array_type(&func.return_type) {
                        panic!("function '{}' declared with array return type", func.name);
                    }
                    let signature = FunctionSignature {
                        return_type: func.return_type.clone(),
                        param_types: func
                            .params
                            .iter()
                            .map(|p| self.adjust_parameter_type(p.r#type.clone()))
                            .collect(),
                    };
                    self.functions.insert(func.name.clone(), signature);
                }
            }
        }

        let decls = program
            .0
            .into_iter()
            .map(|decl| self.analyze_decl(decl)?)
            .collect();
        self.exit_scope();
        Ok(Program(decls))
    }

    fn analyze_decl(&mut self, decl: Decl) -> Result<Decl, SemanticError> {
        let Decl {
            kind,
            start,
            end,
            source,
        } = decl;

        let kind = match kind {
            DeclKind::Function(func) => DeclKind::Function(self.analyze_function(func)?),
            DeclKind::Variable(var) => DeclKind::Variable(self.analyze_global_variable(var)?),
        };

        Ok(Decl {
            kind,
            start,
            end,
            source,
        })
    }

    fn analyze_function(&mut self, decl: FunctionDecl) -> Result<FunctionDecl, SemanticError> {
        let FunctionDecl {
            name,
            params,
            body,
            storage_class,
            return_type,
        } = decl;

        match body {
            Some(body_stmts) => {
                let prev_return = self.current_return_type.replace(return_type.clone());
                self.enter_scope();

                let mut unique_params = Vec::new();
                for param in params {
                    if param.r#type == Type::Void {
                        return Err(SemanticError::InvalidParamType(param.name, param.r#type));
                    }

                    let adjusted_type = self.adjust_parameter_type(param.r#type.clone());
                    let unique = self.fresh_name(&param.name);
                    self.insert_symbol(
                        param.name.clone(),
                        Symbol {
                            unique: unique.clone(),
                            ty: adjusted_type.clone(),
                        },
                    );
                    unique_params.push(ParameterDecl {
                        name: unique,
                        r#type: adjusted_type,
                    });
                }

                let body = body_stmts
                    .into_iter()
                    .map(|stmt| self.analyze_stmt(stmt)?)
                    .collect();

                self.exit_scope();
                self.current_return_type = prev_return;

                Ok(FunctionDecl {
                    name,
                    params: unique_params,
                    body: Some(body),
                    storage_class,
                    return_type,
                })
            }
            None => Ok(FunctionDecl {
                name,
                params: params
                    .into_iter()
                    .map(|param| ParameterDecl {
                        name: param.name,
                        r#type: self.adjust_parameter_type(param.r#type),
                    })
                    .collect(),
                body: None,
                storage_class,
                return_type,
            }),
        }
    }

    fn analyze_global_variable(
        &mut self,
        mut decl: VariableDecl,
    ) -> Result<VariableDecl, SemanticError> {
        if decl.r#type == Type::Void {
            return Err(SemanticError::InvalidVarType(decl.name, Type::Void));
        }

        let init = decl.init.take().map(|expr| {
            let analyzed = self.analyze_expr(expr)?;
            match (&decl.r#type, &analyzed.kind) {
                (Type::Array(elem, _), ExprKind::String(_))
                    if Self::is_char_type(elem.as_ref()) =>
                {
                    // handled during IR generation
                }
                (Type::Array(_, _), _) => {
                    panic!(
                        "array variable '{}' requires a string literal initializer",
                        decl.name
                    );
                }
                (_, ExprKind::String(_)) => {
                    if !matches!(
                        &decl.r#type,
                        Type::Pointer(inner) if Self::is_char_type(inner.as_ref())
                    ) {
                        return Err(SemanticError::InvalidStringLiteral(decl.r#type));
                    }
                }
                _ => {
                    self.ensure_assignable(
                        &decl.r#type,
                        &analyzed.r#type,
                        "variable initialization",
                    );
                }
            }
            Ok(analyzed)
        });
        decl.init = init?;

        Ok(decl)
    }

    fn analyze_local_variable_decl(
        &mut self,
        mut decl: VariableDecl,
    ) -> Result<VariableDecl, SemanticError> {
        if decl.r#type == Type::Void {
            return Err(SemanticError::InvalidVarType(decl.name, Type::Void));
        }

        let init = decl.init.take().map(|expr| {
            let analyzed = self.analyze_expr(expr)?;
            match (&decl.r#type, &analyzed.kind) {
                (Type::Array(elem, _), ExprKind::String(_))
                    if Self::is_char_type(elem.as_ref()) =>
                {
                    // handled later during IR lowering
                }
                (Type::Array(_, _), _) => {
                    // Err(SemanticError::)
                    // panic!(
                    //     "array variable '{}' requires a string literal initializer",
                    //     decl.name
                    // );
                }
                (_, ExprKind::String(_)) => {
                    if !matches!(
                        &decl.r#type,
                        Type::Pointer(inner) if Self::is_char_type(inner.as_ref())
                    ) {
                        return Err(SemanticError::InvalidStringLiteral(decl.r#type));
                    }
                }
                _ => {
                    self.ensure_assignable(
                        &decl.r#type,
                        &analyzed.r#type,
                        "variable initialization",
                    );
                }
            }
            Ok(analyzed)
        });
        decl.init = Some(init.unwrap()?);

        match decl.storage_class {
            Some(StorageClass::Extern) => {
                self.insert_symbol(
                    decl.name.clone(),
                    Symbol {
                        unique: decl.name.clone(),
                        ty: decl.r#type.clone(),
                    },
                );
                Ok(decl)
            }
            Some(StorageClass::Static) => {
                return Err(SemanticError::Unsupported(
                    "static local variables are not supported yet".to_string(),
                ));
            }
            None => {
                let original = decl.name.clone();
                let unique = self.fresh_name(&original);
                self.insert_symbol(
                    original,
                    Symbol {
                        unique: unique.clone(),
                        ty: decl.r#type.clone(),
                    },
                );
                decl.name = unique;
                Ok(decl)
            }
        }
    }

    fn analyze_stmt(&mut self, stmt: Stmt) -> Result<Stmt, SemanticError> {
        let Stmt {
            kind,
            start,
            end,
            source,
            r#type: _,
        } = stmt;

        let kind = match kind {
            StmtKind::Return(expr) => {
                let expr = self.analyze_expr(expr)?;
                if let Some(expected) = &self.current_return_type {
                    self.ensure_assignable(expected, &expr.r#type, "return statement");
                }
                StmtKind::Return(expr)
            }
            StmtKind::Expr(expr) => StmtKind::Expr(self.analyze_expr(expr)?),
            StmtKind::Compound(stmts) => {
                self.enter_scope();
                let stmts = stmts.into_iter().map(|s| self.analyze_stmt(s)).collect();
                self.exit_scope();
                StmtKind::Compound(stmts)
            }
            StmtKind::Declaration(decl) => {
                let decl = self.analyze_local_variable_decl(decl)?;
                StmtKind::Declaration(decl)
            }
            StmtKind::Null => StmtKind::Null,
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let condition = self.analyze_expr(condition)?;
                self.ensure_condition_type(&condition.r#type, "if condition");
                StmtKind::If {
                    condition,
                    then_branch: Box::new(self.analyze_stmt(*then_branch)?),
                    else_branch: else_branch.map(|stmt| Box::new(self.analyze_stmt(*stmt))),
                }
            }
            StmtKind::While {
                condition,
                body,
                loop_id,
            } => {
                let condition = self.analyze_expr(condition)?;
                self.ensure_condition_type(&condition.r#type, "while condition");
                StmtKind::While {
                    condition,
                    body: Box::new(self.analyze_stmt(*body)?),
                    loop_id,
                }
            }
            StmtKind::DoWhile {
                body,
                condition,
                loop_id,
            } => {
                let condition = self.analyze_expr(condition)?;
                self.ensure_condition_type(&condition.r#type, "do-while condition");
                StmtKind::DoWhile {
                    body: Box::new(self.analyze_stmt(*body)?),
                    condition,
                    loop_id,
                }
            }
            StmtKind::For {
                init,
                condition,
                post,
                body,
                loop_id,
            } => {
                self.enter_scope();
                let init = self.analyze_for_init(init);
                let condition = condition.map(|expr| {
                    let analyzed = self.analyze_expr(expr)?;
                    self.ensure_condition_type(&analyzed.r#type, "for condition");
                    analyzed
                });
                let post = post.map(|expr| self.analyze_expr(expr)?);
                let body = Box::new(self.analyze_stmt(*body)?);
                self.exit_scope();
                StmtKind::For {
                    init,
                    condition,
                    post,
                    body,
                    loop_id,
                }
            }
            StmtKind::Break { loop_id } => StmtKind::Break { loop_id },
            StmtKind::Continue { loop_id } => StmtKind::Continue { loop_id },
        };

        Ok(Stmt {
            kind,
            start,
            end,
            source,
            r#type: Type::Void,
        })
    }

    fn analyze_expr(&mut self, expr: Expr) -> Result<Expr, SemanticError> {
        self.analyze_expr_internal(expr, true)
    }

    fn analyze_expr_no_decay(&mut self, expr: Expr) -> Result<Expr, SemanticError> {
        self.analyze_expr_internal(expr, false)
    }

    fn analyze_expr_internal(
        &mut self,
        expr: Expr,
        decay_arrays: bool,
    ) -> Result<Expr, SemanticError> {
        let Expr {
            kind,
            start,
            end,
            source,
            r#type: _,
        } = expr;

        let result = match kind {
            ExprKind::Constant(c) => {
                let ty = match &c {
                    Const::Char(_) => Type::Int,
                    Const::UChar(_) => Type::UInt,
                    Const::Int(_) => Type::Int,
                    Const::Long(_) => Type::Long,
                    Const::UInt(_) => Type::UInt,
                    Const::ULong(_) => Type::ULong,
                    Const::Double(_) => Type::Double,
                };
                Ok(Expr {
                    kind: ExprKind::Constant(c),
                    start,
                    end,
                    source,
                    r#type: ty,
                })
            }
            ExprKind::String(value) => Ok(Expr {
                kind: ExprKind::String(value),
                start,
                end,
                source,
                r#type: Type::Pointer(Box::new(Type::Char)),
            }),
            ExprKind::Var(name) => {
                let symbol = self.lookup_symbol(&name)?;
                Ok(Expr {
                    kind: ExprKind::Var(symbol.unique.clone()),
                    start,
                    end,
                    source,
                    r#type: symbol.ty.clone(),
                })
            }
            ExprKind::FunctionCall(name, args) => {
                let signature = self
                    .functions
                    .get(&name)
                    .cloned()
                    .unwrap_or_else(|| panic!("call to undeclared function {name}"));
                if args.len() != signature.param_types.len() {
                    panic!(
                        "function '{}' called with wrong number of arguments (expected {}, got {})",
                        name,
                        signature.param_types.len(),
                        args.len()
                    );
                }

                let analyzed_args = args
                    .into_iter()
                    .zip(signature.param_types.iter())
                    .map(|(arg, expected)| {
                        let expr = self.analyze_expr(arg)?;
                        self.ensure_assignable(expected, &expr.r#type, "function argument");
                        expr
                    })
                    .collect();

                Ok(Expr {
                    kind: ExprKind::FunctionCall(name, analyzed_args),
                    start,
                    end,
                    source,
                    r#type: signature.return_type,
                })
            }
            ExprKind::Cast(target, expr) => {
                if target == Type::Void {
                    return Err(SemanticError::CastTo(Type::Void));
                }
                if Self::is_array_type(&target) {
                    panic!("cannot cast to array type");
                }

                let expr = self.analyze_expr(*expr)?;
                self.ensure_castable(&expr.r#type, &target);
                let ty = target.clone();
                Ok(Expr {
                    kind: ExprKind::Cast(ty.clone(), Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                })
            }
            ExprKind::Neg(expr) => {
                self.analyze_arithmetic_unary(*expr, start, end, source, ExprKind::Neg)
            }
            ExprKind::BitNot(expr) => {
                Ok(self.analyze_integer_unary(*expr, start, end, source, ExprKind::BitNot))
            }
            ExprKind::Not(expr) => {
                let expr = self.analyze_expr(*expr)?;
                self.ensure_condition_type(&expr.r#type, "logical not");
                Expr {
                    kind: ExprKind::Not(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                }
            }
            ExprKind::AddrOf(expr) => {
                let expr = self.analyze_expr_no_decay(*expr)?;
                self.ensure_lvalue(&expr, "address-of operand");
                let ty = Type::Pointer(Box::new(expr.r#type.clone()));
                Expr {
                    kind: ExprKind::AddrOf(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::Dereference(expr) => {
                let expr = self.analyze_expr(*expr)?;
                let ty = match &expr.r#type {
                    Type::Pointer(inner) => {
                        if matches!(inner.as_ref(), Type::Void) {
                            panic!("cannot dereference void pointer");
                        }
                        inner.as_ref().clone()
                    }
                    other => panic!("cannot dereference non-pointer type {other:?}"),
                };
                Expr {
                    kind: ExprKind::Dereference(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::Conditional(cond, then_expr, else_expr) => {
                let cond = self.analyze_expr(*cond)?;
                self.ensure_condition_type(&cond.r#type, "conditional condition");
                let then_expr = self.analyze_expr(*then_expr)?;
                let else_expr = self.analyze_expr(*else_expr)?;
                let then_type = then_expr.r#type.clone();
                let else_type = else_expr.r#type.clone();
                let result_type = if Self::is_numeric_type(&then_type)
                    && Self::is_numeric_type(&else_type)
                {
                    self.numeric_result_type(&then_type, &else_type)
                } else if Self::is_pointer_type(&then_type) && Self::is_pointer_type(&else_type) {
                    if !Self::pointer_types_compatible(&then_type, &else_type) {
                        panic!(
                            "conditional operator requires compatible pointer types ({:?} vs {:?})",
                            then_type, else_type
                        );
                    }
                    then_type.clone()
                } else {
                    panic!(
                        "conditional operator requires numeric or pointer operands ({:?} vs {:?})",
                        then_type, else_type
                    );
                };

                Expr {
                    kind: ExprKind::Conditional(
                        Box::new(cond),
                        Box::new(then_expr),
                        Box::new(else_expr),
                    ),
                    start,
                    end,
                    source,
                    r#type: result_type,
                }
            }
            ExprKind::Add(lhs, rhs) => self.analyze_add(*lhs, *rhs, start, end, source),
            ExprKind::Sub(lhs, rhs) => self.analyze_sub(*lhs, *rhs, start, end, source),
            ExprKind::Mul(lhs, rhs) => {
                self.analyze_arithmetic_binary(*lhs, *rhs, start, end, source, ExprKind::Mul)
            }
            ExprKind::Div(lhs, rhs) => {
                self.analyze_arithmetic_binary(*lhs, *rhs, start, end, source, ExprKind::Div)
            }
            ExprKind::Rem(lhs, rhs) => {
                self.analyze_integer_binary(*lhs, *rhs, start, end, source, ExprKind::Rem)
            }
            ExprKind::Equal(lhs, rhs) => {
                self.analyze_equality(*lhs, *rhs, start, end, source, ExprKind::Equal)
            }
            ExprKind::NotEqual(lhs, rhs) => {
                self.analyze_equality(*lhs, *rhs, start, end, source, ExprKind::NotEqual)
            }
            ExprKind::LessThan(lhs, rhs) => {
                self.analyze_comparison(*lhs, *rhs, start, end, source, ExprKind::LessThan)
            }
            ExprKind::LessThanEqual(lhs, rhs) => {
                self.analyze_comparison(*lhs, *rhs, start, end, source, ExprKind::LessThanEqual)
            }
            ExprKind::GreaterThan(lhs, rhs) => {
                self.analyze_comparison(*lhs, *rhs, start, end, source, ExprKind::GreaterThan)
            }
            ExprKind::GreaterThanEqual(lhs, rhs) => {
                self.analyze_comparison(*lhs, *rhs, start, end, source, ExprKind::GreaterThanEqual)
            }
            ExprKind::Or(lhs, rhs) => {
                self.analyze_logical_binary(*lhs, *rhs, start, end, source, ExprKind::Or)
            }
            ExprKind::And(lhs, rhs) => {
                self.analyze_logical_binary(*lhs, *rhs, start, end, source, ExprKind::And)
            }
            ExprKind::BitAnd(lhs, rhs) => {
                self.analyze_integer_binary(*lhs, *rhs, start, end, source, ExprKind::BitAnd)
            }
            ExprKind::Xor(lhs, rhs) => {
                self.analyze_integer_binary(*lhs, *rhs, start, end, source, ExprKind::Xor)
            }
            ExprKind::BitOr(lhs, rhs) => {
                self.analyze_integer_binary(*lhs, *rhs, start, end, source, ExprKind::BitOr)
            }
            ExprKind::LeftShift(lhs, rhs) => {
                let lhs = self.analyze_expr(*lhs)?;
                let rhs = self.analyze_expr(*rhs)?;
                self.ensure_integer_type(&lhs.r#type, "shift operand");
                self.ensure_integer_type(&rhs.r#type, "shift amount");
                let ty = lhs.r#type.clone();
                Expr {
                    kind: ExprKind::LeftShift(Box::new(lhs), Box::new(rhs)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::RightShift(lhs, rhs) => {
                let lhs = self.analyze_expr(*lhs)?;
                let rhs = self.analyze_expr(*rhs)?;
                self.ensure_integer_type(&lhs.r#type, "shift operand");
                self.ensure_integer_type(&rhs.r#type, "shift amount");
                let ty = lhs.r#type.clone();
                Expr {
                    kind: ExprKind::RightShift(Box::new(lhs), Box::new(rhs)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::Assignment(lhs, rhs) => {
                let lhs = self.analyze_expr_no_decay(*lhs)?;
                self.ensure_lvalue(&lhs, "assignment left-hand side");
                if Self::is_array_type(&lhs.r#type) {
                    panic!("assignment to array type is not allowed");
                }
                let lhs_type = lhs.r#type.clone();
                let rhs = self.analyze_expr(*rhs)?;
                self.ensure_assignable(&lhs_type, &rhs.r#type, "assignment");
                Expr {
                    kind: ExprKind::Assignment(Box::new(lhs), Box::new(rhs)),
                    start,
                    end,
                    source,
                    r#type: lhs_type,
                }
            }
            ExprKind::PreIncrement(expr) => {
                let expr = self.analyze_expr_no_decay(*expr)?;
                self.ensure_lvalue(&expr, "pre-increment target");
                self.ensure_numeric_type(&expr.r#type, "pre-increment operand");
                let ty = expr.r#type.clone();
                Expr {
                    kind: ExprKind::PreIncrement(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::PreDecrement(expr) => {
                let expr = self.analyze_expr_no_decay(*expr)?;
                self.ensure_lvalue(&expr, "pre-decrement target");
                self.ensure_numeric_type(&expr.r#type, "pre-decrement operand");
                let ty = expr.r#type.clone();
                Expr {
                    kind: ExprKind::PreDecrement(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::PostIncrement(expr) => {
                let expr = self.analyze_expr_no_decay(*expr)?;
                self.ensure_lvalue(&expr, "post-increment target");
                self.ensure_numeric_type(&expr.r#type, "post-increment operand");
                let ty = expr.r#type.clone();
                Expr {
                    kind: ExprKind::PostIncrement(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::PostDecrement(expr) => {
                let expr = self.analyze_expr_no_decay(*expr)?;
                self.ensure_lvalue(&expr, "post-decrement target");
                self.ensure_numeric_type(&expr.r#type, "post-decrement operand");
                let ty = expr.r#type.clone();
                Expr {
                    kind: ExprKind::PostDecrement(Box::new(expr)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
        };

        Ok(self.apply_array_decay(result, decay_arrays))
    }

    fn analyze_for_init(&mut self, init: ForInit) -> ForInit {
        match init {
            ForInit::Declaration(stmt) => ForInit::Declaration(Box::new(self.analyze_stmt(*stmt)?)),
            ForInit::Expr(expr) => ForInit::Expr(
                expr.map(|e| self.analyze_expr(e).map_err(|e| e.to_string()))
                    .iter()
                    .collect(),
            ),
        }
    }

    fn ensure_lvalue(&self, expr: &Expr, context: &str) {
        match &expr.kind {
            ExprKind::Var(_) | ExprKind::Dereference(_) => {}
            _ => panic!("{context} requires an lvalue"),
        }
    }

    fn analyze_integer_unary<F>(
        &mut self,
        expr: Expr,
        start: usize,
        end: usize,
        source: String,
        constructor: F,
    ) -> Result<Expr, SemanticError>
    where
        F: FnOnce(Box<Expr>) -> ExprKind,
    {
        let expr = self.analyze_expr(expr)?;
        self.ensure_integer_type(&expr.r#type, "unary operator");
        let ty = expr.r#type.clone();
        Ok(Expr {
            kind: constructor(Box::new(expr)),
            start,
            end,
            source,
            r#type: ty,
        })
    }

    fn analyze_arithmetic_unary<F>(
        &mut self,
        expr: Expr,
        start: usize,
        end: usize,
        source: String,
        constructor: F,
    ) -> Result<Expr, SemanticError>
    where
        F: FnOnce(Box<Expr>) -> ExprKind,
    {
        let expr = self.analyze_expr(expr)?;
        self.ensure_numeric_type(&expr.r#type, "unary operator");
        let ty = expr.r#type.clone();
        Ok(Expr {
            kind: constructor(Box::new(expr)),
            start,
            end,
            source,
            r#type: ty,
        })
    }

    fn analyze_add(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        start: usize,
        end: usize,
        source: String,
    ) -> Result<Expr, SemanticError> {
        let lhs = self.analyze_expr(lhs)?;
        let rhs = self.analyze_expr(rhs)?;
        let lhs_type = lhs.r#type.clone();
        let rhs_type = rhs.r#type.clone();

        let result_type = if Self::is_numeric_type(&lhs_type) && Self::is_numeric_type(&rhs_type) {
            self.numeric_result_type(&lhs_type, &rhs_type)
        } else if Self::is_pointer_type(&lhs_type) && Self::is_integer_type(&rhs_type) {
            self.pointer_arithmetic_base(&lhs_type, "pointer addition");
            lhs_type.clone()
        } else if Self::is_integer_type(&lhs_type) && Self::is_pointer_type(&rhs_type) {
            self.pointer_arithmetic_base(&rhs_type, "pointer addition");
            rhs_type.clone()
        } else {
            return Err(SemanticError::InvalidBinaryExpr(lhs, rhs, TokenKind::Plus));
        };

        Ok(Expr {
            kind: ExprKind::Add(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source,
            r#type: result_type,
        })
    }

    fn analyze_sub(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        start: usize,
        end: usize,
        source: String,
    ) -> Result<Expr, SemanticError> {
        let lhs = self.analyze_expr(lhs)?;
        let rhs = self.analyze_expr(rhs)?;
        let lhs_type = lhs.r#type.clone();
        let rhs_type = rhs.r#type.clone();

        let result_type = if Self::is_numeric_type(&lhs_type) && Self::is_numeric_type(&rhs_type) {
            self.numeric_result_type(&lhs_type, &rhs_type)
        } else if Self::is_pointer_type(&lhs_type) && Self::is_integer_type(&rhs_type) {
            self.pointer_arithmetic_base(&lhs_type, "pointer subtraction");
            lhs_type.clone()
        } else if Self::is_pointer_type(&lhs_type) && Self::is_pointer_type(&rhs_type) {
            let lhs_base = self.pointer_arithmetic_base(&lhs_type, "pointer subtraction");
            let rhs_base = self.pointer_arithmetic_base(&rhs_type, "pointer subtraction");
            if lhs_base != rhs_base {
                panic!(
                    "pointer subtraction requires identical pointer types ({:?} vs {:?})",
                    lhs_type, rhs_type
                );
            }
            Type::Long
        } else {
            panic!(
                "unsupported operand types {:?} and {:?} in subtraction",
                lhs_type, rhs_type
            );
        };

        Ok(Expr {
            kind: ExprKind::Sub(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source,
            r#type: result_type,
        })
    }

    fn analyze_integer_binary<F>(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        start: usize,
        end: usize,
        source: String,
        constructor: F,
    ) -> Result<Expr, SemanticError>
    where
        F: FnOnce(Box<Expr>, Box<Expr>) -> ExprKind,
    {
        let lhs = self.analyze_expr(lhs)?;
        let rhs = self.analyze_expr(rhs)?;
        self.ensure_integer_type(&lhs.r#type, "binary operand");
        self.ensure_integer_type(&rhs.r#type, "binary operand");
        let ty = self.numeric_result_type(&lhs.r#type, &rhs.r#type);
        Ok(Expr {
            kind: constructor(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source,
            r#type: ty,
        })
    }

    fn analyze_arithmetic_binary<F>(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        start: usize,
        end: usize,
        source: String,
        constructor: F,
    ) -> Result<Expr, SemanticError>
    where
        F: FnOnce(Box<Expr>, Box<Expr>) -> ExprKind,
    {
        let lhs = self.analyze_expr(lhs)?;
        let rhs = self.analyze_expr(rhs)?;
        self.ensure_numeric_type(&lhs.r#type, "binary operand");
        self.ensure_numeric_type(&rhs.r#type, "binary operand");
        let ty = self.numeric_result_type(&lhs.r#type, &rhs.r#type);
        Ok(Expr {
            kind: constructor(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source,
            r#type: ty,
        })
    }

    fn analyze_equality<F>(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        start: usize,
        end: usize,
        source: String,
        constructor: F,
    ) -> Result<Expr, SemanticError>
    where
        F: FnOnce(Box<Expr>, Box<Expr>) -> ExprKind,
    {
        let lhs = self.analyze_expr(lhs)?;
        let rhs = self.analyze_expr(rhs)?;
        if Self::is_pointer_type(&lhs.r#type) || Self::is_pointer_type(&rhs.r#type) {
            self.ensure_pointer_comparable(&lhs.r#type, &rhs.r#type, "equality comparison");
        } else {
            self.ensure_numeric_type(&lhs.r#type, "comparison operand");
            self.ensure_numeric_type(&rhs.r#type, "comparison operand");
        }
        Ok(Expr {
            kind: constructor(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source,
            r#type: Type::Int,
        })
    }

    fn analyze_comparison<F>(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        start: usize,
        end: usize,
        source: String,
        constructor: F,
    ) -> Result<Expr, SemanticError>
    where
        F: FnOnce(Box<Expr>, Box<Expr>) -> ExprKind,
    {
        let lhs = self.analyze_expr(lhs)?;
        let rhs = self.analyze_expr(rhs)?;
        self.ensure_numeric_type(&lhs.r#type, "comparison operand");
        self.ensure_numeric_type(&rhs.r#type, "comparison operand");
        Ok(Expr {
            kind: constructor(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source,
            r#type: Type::Int,
        })
    }

    fn analyze_logical_binary<F>(
        &mut self,
        lhs: Expr,
        rhs: Expr,
        start: usize,
        end: usize,
        source: String,
        constructor: F,
    ) -> Result<Expr, SemanticError>
    where
        F: FnOnce(Box<Expr>, Box<Expr>) -> ExprKind,
    {
        let lhs = self.analyze_expr(lhs)?;
        let rhs = self.analyze_expr(rhs)?;
        self.ensure_condition_type(&lhs.r#type, "logical operand");
        self.ensure_condition_type(&rhs.r#type, "logical operand");
        Ok(Expr {
            kind: constructor(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source,
            r#type: Type::Int,
        })
    }

    fn numeric_result_type(&self, lhs: &Type, rhs: &Type) -> Type {
        if !Self::is_numeric_type(lhs) || !Self::is_numeric_type(rhs) {
            panic!(
                "unsupported operand types {:?} and {:?} in numeric expression",
                lhs, rhs
            );
        }

        if Self::is_floating_type(lhs) || Self::is_floating_type(rhs) {
            return Type::Double;
        }

        if Self::type_rank(lhs) >= Self::type_rank(rhs) {
            lhs.clone()
        } else {
            rhs.clone()
        }
    }

    fn ensure_integer_type(&self, ty: &Type, context: &str) {
        let ty = self.decay_array_type(ty);
        if !Self::is_integer_type(&ty) {
            panic!("{context} requires integer type, found {:?}", ty);
        }
    }

    fn ensure_condition_type(&self, ty: &Type, context: &str) {
        let ty = self.decay_array_type(ty);
        if Self::is_integer_type(&ty) || Self::is_pointer_type(&ty) {
            return;
        }
        panic!("{context} requires scalar type, found {:?}", ty);
    }

    fn ensure_numeric_type(&self, ty: &Type, context: &str) {
        let ty = self.decay_array_type(ty);
        if !Self::is_numeric_type(&ty) {
            panic!("{context} requires numeric type, found {:?}", ty);
        }
    }

    fn ensure_assignable(&self, target: &Type, value: &Type, context: &str) {
        if Self::is_array_type(target) {
            panic!("{context} cannot assign to array type");
        }

        let value_type = self.decay_array_type(value);

        if target == &value_type {
            return;
        }

        if Self::is_numeric_type(target) && Self::is_numeric_type(&value_type) {
            return;
        }

        if Self::is_pointer_type(target) {
            if Self::is_pointer_type(&value_type)
                && Self::pointer_types_compatible(target, &value_type)
            {
                return;
            }
            if Self::is_integer_type(&value_type) {
                return;
            }
        }

        panic!(
            "{context} requires compatible types ({:?} <- {:?})",
            target, value_type
        );
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

    fn is_pointer_type(ty: &Type) -> bool {
        matches!(ty, Type::Pointer(_))
    }

    fn is_floating_type(ty: &Type) -> bool {
        matches!(ty, Type::Double)
    }

    fn is_numeric_type(ty: &Type) -> bool {
        Self::is_integer_type(ty) || Self::is_floating_type(ty)
    }

    fn is_char_type(ty: &Type) -> bool {
        matches!(ty, Type::Char | Type::SChar | Type::UChar)
    }

    fn ensure_castable(&self, from: &Type, to: &Type) {
        if Self::is_array_type(to) {
            panic!("cannot cast to array type");
        }

        let from_type = self.decay_array_type(from);
        let to_type = self.decay_array_type(to);

        // if both are numeric types
        // or if both are pointers
        // or if one is pointer or one is integer
        if Self::is_numeric_type(&from_type) && Self::is_numeric_type(&to_type) {
            return;
        }
        if Self::is_pointer_type(&from_type) && Self::is_pointer_type(&to_type) {
            return;
        }
        if Self::is_pointer_type(&from_type) && Self::is_integer_type(&to_type) {
            return;
        }
        if Self::is_integer_type(&from_type) && Self::is_pointer_type(&to_type) {
            return;
        }
        panic!("unsupported cast {:?} -> {:?}", from_type, to_type);
    }

    fn type_rank(ty: &Type) -> usize {
        match ty {
            Type::Char | Type::SChar => 0,
            Type::UChar => 1,
            Type::Int => 2,
            Type::UInt => 3,
            Type::Long => 4,
            Type::ULong => 5,
            Type::Void => panic!("void type has no integer rank"),
            Type::Double => panic!("double type has no integer rank"),
            Type::Pointer(_) | Type::FunType(_, _) => panic!("pointer type has no integer rank"),
            Type::Array(_, _) => panic!("array type has no integer rank"),
        }
    }

    fn pointer_types_compatible(lhs: &Type, rhs: &Type) -> bool {
        if lhs == rhs {
            return true;
        }

        match (lhs, rhs) {
            (Type::Pointer(l), Type::Pointer(r)) => {
                matches!(l.as_ref(), Type::Void) || matches!(r.as_ref(), Type::Void)
            }
            _ => false,
        }
    }

    fn ensure_pointer_comparable(&self, lhs: &Type, rhs: &Type, context: &str) {
        let lhs_type = self.decay_array_type(lhs);
        let rhs_type = self.decay_array_type(rhs);

        if Self::is_pointer_type(&lhs_type) && Self::is_pointer_type(&rhs_type) {
            if Self::pointer_types_compatible(&lhs_type, &rhs_type) {
                return;
            }
        } else if (Self::is_pointer_type(&lhs_type) && Self::is_integer_type(&rhs_type))
            || (Self::is_integer_type(&lhs_type) && Self::is_pointer_type(&rhs_type))
        {
            return;
        }

        panic!(
            "{context} requires compatible pointer types ({:?} vs {:?})",
            lhs_type, rhs_type
        );
    }

    fn pointer_arithmetic_base<'a>(&self, ty: &'a Type, context: &str) -> &'a Type {
        match ty {
            Type::Pointer(inner) => match inner.as_ref() {
                Type::Void => panic!("{context} requires pointer to sized type"),
                Type::FunType(_, _) => panic!("{context} requires pointer to sized type"),
                base => base,
            },
            _ => panic!("{context} requires pointer operand"),
        }
    }

    fn adjust_parameter_type(&self, ty: Type) -> Type {
        match ty {
            Type::Array(inner, _) => Type::Pointer(inner),
            Type::FunType(params, ret) => Type::Pointer(Box::new(Type::FunType(params, ret))),
            other => other,
        }
    }

    fn decay_array_type(&self, ty: &Type) -> Type {
        match ty {
            Type::Array(inner, _) => Type::Pointer(inner.clone()),
            Type::FunType(params, ret) => {
                Type::Pointer(Box::new(Type::FunType(params.clone(), ret.clone())))
            }
            other => other.clone(),
        }
    }

    fn apply_array_decay(&self, mut expr: Expr, decay_arrays: bool) -> Expr {
        if decay_arrays {
            expr.r#type = match expr.r#type.clone() {
                Type::Array(inner, _) => Type::Pointer(inner),
                Type::FunType(params, ret) => Type::Pointer(Box::new(Type::FunType(params, ret))),
                other => other,
            };
        }
        expr
    }

    fn is_array_type(ty: &Type) -> bool {
        matches!(ty, Type::Array(_, _))
    }

    fn insert_symbol(&mut self, name: String, symbol: Symbol) {
        self.scopes
            .last_mut()
            .expect("scope stack empty")
            .insert(name, symbol);
    }

    fn lookup_symbol(&self, name: &str) -> Result<&Symbol, SemanticError> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Ok(symbol);
            }
        }
        Err(SemanticError::UndeclaredIdentifier(name.to_string()))
    }

    fn enter_scope(&mut self) {
        self.scopes.push(BTreeMap::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn fresh_name(&mut self, base: &str) -> String {
        let name = format!("{base}{}", self.counter);
        self.counter += 1;
        name
    }
}
