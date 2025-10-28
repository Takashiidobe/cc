#![allow(clippy::result_large_err)]
pub(crate) mod loop_label;
pub(crate) mod struct_label;

use std::collections::BTreeMap;

use thiserror::Error;

use crate::{
    parse::{
        Const, Decl, DeclKind, Expr, ExprKind, ForInit, FunctionDecl, ParameterDecl, Program, Stmt,
        StmtKind, StorageClass, StructDeclaration, Type, VariableDecl,
    },
    structs::{self, StructLayout, StructMemberLayout},
    tokenize::TokenKind,
};

#[derive(Error, Debug)]
pub(crate) enum SemanticError {
    #[error("Cannot cast from type `{0}` to `{1}`")]
    CastToError(Type, Type),

    #[error("{0}")]
    Unsupported(String),

    #[error("Variable `{0}` cannot be type `{1}`")]
    InvalidVarType(String, Type),

    #[error("Param `{0}` cannot be type `{1}`")]
    InvalidParamType(String, Type),

    #[error("Cannot dereference type: `{0}`")]
    DereferenceError(Type),

    #[error("String literal cannot initialize type {0:?}")]
    InvalidStringLiteral(Type),

    #[error("Undeclared identifier: {0}")]
    UndeclaredIdentifier(String),

    #[error("Invalid Binary Expression: {0:?} {1:?} {2:?}")]
    InvalidBinaryExpr(Expr, TokenKind, Expr),

    #[error("function '{0}' declared with array return type")]
    FunctionArrayReturnType(String),

    #[error("array variable '{0}' requires a string literal initializer")]
    ArrayVarRequiresStringInitializer(String),

    #[error("{0} requires integer type, found {1}")]
    IntegerTypeRequired(&'static str, Type),

    #[error("{0} requires numeric type, found {1}")]
    NumericTypeRequired(&'static str, Type),

    #[error("{0} requires scalar (int or pointer) type, found {1}")]
    ScalarTypeRequired(&'static str, Type),

    #[error("{0} requires compatible types ({1} <- {2})")]
    IncompatibleForContext(&'static str, Type, Type),

    #[error("{0} requires pointer operand, found {1}")]
    PointerOperandRequired(&'static str, Type),

    #[error("{0} requires pointer to sized type, found {1}")]
    PointerSizedBaseRequired(&'static str, Type),

    #[error("conditional operator requires compatible types ({0} vs {1})")]
    ConditionalTypeMismatch(Type, Type),

    #[error("pointer subtraction requires identical pointer types ({0} vs {1})")]
    PointerSubTypeMismatch(Type, Type),

    #[error("function '{0}' called with wrong number of arguments (expected {1}, got {2})")]
    WrongArgCount(String, usize, usize),

    #[error("{0} requires lvalue")]
    LValueRequired(&'static str),

    #[error("struct '{0}' is incomplete")]
    IncompleteStruct(String),

    #[error("struct '{0}' has no member named '{1}'")]
    UnknownStructMember(String, String),

    #[error("member access requires struct type")]
    MemberAccessOnNonStruct,
}

#[derive(Clone)]
pub(crate) struct Symbol {
    unique: String,
    ty: Type,
}

#[derive(Clone)]
pub(crate) struct FunctionSignature {
    return_type: Type,
    param_types: Vec<Type>,
}

pub(crate) struct SemanticAnalyzer {
    counter: usize,
    scopes: Vec<BTreeMap<String, Symbol>>,
    functions: BTreeMap<String, FunctionSignature>,
    current_return_type: Option<Type>,
    struct_defs: BTreeMap<String, StructDeclaration>,
    struct_layouts: BTreeMap<String, StructLayout>,
}

impl SemanticAnalyzer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn analyze_program(mut self, program: Program) -> Result<Program, SemanticError> {
        self.enter_scope();

        self.collect_struct_definitions(&program)?;
        self.initialize_struct_layouts()?;

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
                        return Err(SemanticError::FunctionArrayReturnType(func.name.clone()));
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
                DeclKind::Struct(_) => {}
            }
        }

        let decls = program
            .0
            .into_iter()
            .map(|decl| self.analyze_decl(decl))
            .collect::<Result<Vec<_>, _>>()?;
        self.exit_scope();
        Ok(Program(decls))
    }

    fn collect_struct_definitions(&mut self, program: &Program) -> Result<(), SemanticError> {
        self.struct_defs.clear();
        for decl in &program.0 {
            if let DeclKind::Struct(struct_decl) = &decl.kind {
                if !struct_decl.members.is_empty() {
                    self.struct_defs
                        .insert(struct_decl.tag.clone(), struct_decl.clone());
                }
            }
        }
        Ok(())
    }

    fn initialize_struct_layouts(&mut self) -> Result<(), SemanticError> {
        self.struct_layouts.clear();
        structs::clear_struct_layouts();
        let tags: Vec<String> = self.struct_defs.keys().cloned().collect();
        for tag in tags {
            let layout = self.struct_layout(&tag)?;
            structs::set_struct_layout(tag, layout);
        }
        Ok(())
    }

    fn struct_layout(&mut self, tag: &str) -> Result<StructLayout, SemanticError> {
        if let Some(layout) = self.struct_layouts.get(tag) {
            return Ok(layout.clone());
        }

        let members_decl = {
            let decl = self
                .struct_defs
                .get(tag)
                .ok_or_else(|| SemanticError::IncompleteStruct(tag.to_string()))?;
            decl.members.clone()
        };

        let mut offset = 0usize;
        let mut max_align = 1usize;
        let mut members = BTreeMap::new();

        for member in members_decl {
            let (size, align) = self.type_size_align(&member.member_type)?;
            if align == 0 {
                return Err(SemanticError::Unsupported(format!(
                    "type `{}` has zero alignment",
                    member.member_type
                )));
            }
            if offset % align != 0 {
                offset += align - (offset % align);
            }
            members.insert(
                member.member_name.clone(),
                StructMemberLayout {
                    offset,
                    ty: member.member_type.clone(),
                },
            );
            offset += size;
            max_align = max_align.max(align);
        }

        if max_align > 0 && offset % max_align != 0 {
            offset += max_align - (offset % max_align);
        }

        let layout = StructLayout {
            size: offset,
            align: max_align.max(1),
            members,
        };
        self.struct_layouts.insert(tag.to_string(), layout.clone());
        Ok(layout)
    }

    fn type_size_align(&mut self, ty: &Type) -> Result<(usize, usize), SemanticError> {
        match ty {
            Type::Struct(tag) => {
                let layout = self.struct_layout(tag)?;
                Ok((layout.size, layout.align))
            }
            Type::Array(inner, len) => {
                let (size, align) = self.type_size_align(inner)?;
                Ok((size * len, align))
            }
            Type::IncompleteArray(inner) => {
                let (_, align) = self.type_size_align(inner)?;
                Err(SemanticError::Unsupported(format!(
                    "incomplete array type `{}` has no size",
                    ty
                )))
            }
            Type::Pointer(_) => Ok((8, 8)),
            Type::FunType(_, _) => Err(SemanticError::Unsupported(format!(
                "cannot determine size of function type `{}`",
                ty
            ))),
            _ => Ok((ty.byte_size(), ty.byte_align())),
        }
    }

    fn struct_member_layout(
        &mut self,
        tag: &str,
        member: &str,
    ) -> Result<StructMemberLayout, SemanticError> {
        let layout = self.struct_layout(tag)?;
        layout
            .members
            .get(member)
            .cloned()
            .ok_or_else(|| SemanticError::UnknownStructMember(tag.to_string(), member.to_string()))
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
            DeclKind::Struct(decl) => DeclKind::Struct(decl),
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
                        return Err(SemanticError::InvalidParamType(param.name, Type::Void));
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
                    .map(|stmt| self.analyze_stmt(stmt))
                    .collect::<Result<Vec<_>, _>>()?;

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
                (Type::Array(elem, _), ExprKind::String(_)) if elem.as_ref().is_char() => {
                    // handled during IR generation
                }
                (Type::Array(_, _), _) => {
                    return Err(SemanticError::ArrayVarRequiresStringInitializer(
                        decl.name.clone(),
                    ));
                }
                (Type::IncompleteArray(elem), ExprKind::String(value))
                    if elem.as_ref().is_char() =>
                {
                    let size = value.len() + 1;
                    decl.r#type = Type::Array(elem.clone(), size);
                }
                (Type::IncompleteArray(_), _) => {
                    return Err(SemanticError::ArrayVarRequiresStringInitializer(
                        decl.name.clone(),
                    ));
                }
                (_, ExprKind::String(_)) => {
                    if !matches!(&decl.r#type, Type::Pointer(inner) if inner.as_ref().is_char()) {
                        return Err(SemanticError::InvalidStringLiteral(decl.r#type.clone()));
                    }
                }
                _ => {
                    self.ensure_assignable(
                        &decl.r#type,
                        &analyzed.r#type,
                        "variable initialization",
                    )?;
                }
            }
            Ok(analyzed)
        });
        decl.init = init.transpose()?;

        if decl.is_definition && matches!(decl.r#type, Type::IncompleteArray(_)) {
            return Err(SemanticError::ArrayVarRequiresStringInitializer(decl.name));
        }

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
                (Type::Array(elem, _), ExprKind::String(_)) if elem.as_ref().is_char() => {
                    // handled later during IR lowering
                }
                (Type::Array(_, _), _) => {
                    return Err(SemanticError::ArrayVarRequiresStringInitializer(
                        decl.name.clone(),
                    ));
                }
                (Type::IncompleteArray(elem), ExprKind::String(value))
                    if elem.as_ref().is_char() =>
                {
                    let size = value.len() + 1;
                    decl.r#type = Type::Array(elem.clone(), size);
                }
                (Type::IncompleteArray(_), _) => {
                    return Err(SemanticError::ArrayVarRequiresStringInitializer(
                        decl.name.clone(),
                    ));
                }
                (_, ExprKind::String(_)) => {
                    if !matches!(&decl.r#type, Type::Pointer(inner) if inner.as_ref().is_char()) {
                        return Err(SemanticError::InvalidStringLiteral(decl.r#type.clone()));
                    }
                }
                _ => {
                    self.ensure_assignable(
                        &decl.r#type,
                        &analyzed.r#type,
                        "variable initialization",
                    )?;
                }
            }
            Ok(analyzed)
        });
        if let Some(r) = init {
            decl.init = Some(r?);
        }

        if matches!(decl.r#type, Type::IncompleteArray(_)) {
            return Err(SemanticError::ArrayVarRequiresStringInitializer(decl.name));
        }

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
            Some(StorageClass::Static) => Err(SemanticError::Unsupported(
                "static local variables are not supported yet".to_string(),
            )),
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
                    self.ensure_assignable(expected, &expr.r#type, "return statement")?;
                }
                StmtKind::Return(expr)
            }
            StmtKind::Expr(expr) => StmtKind::Expr(self.analyze_expr(expr)?),
            StmtKind::Compound(stmts) => {
                self.enter_scope();
                let stmts = stmts
                    .into_iter()
                    .map(|s| self.analyze_stmt(s))
                    .collect::<Result<Vec<_>, _>>()?;
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
                self.ensure_condition_type(&condition.r#type, "if condition")?;
                StmtKind::If {
                    condition,
                    then_branch: Box::new(self.analyze_stmt(*then_branch)?),
                    else_branch: else_branch
                        .map(|stmt| self.analyze_stmt(*stmt))
                        .transpose()?
                        .map(Box::new),
                }
            }
            StmtKind::While {
                condition,
                body,
                loop_id,
            } => {
                let condition = self.analyze_expr(condition)?;
                self.ensure_condition_type(&condition.r#type, "while condition")?;
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
                self.ensure_condition_type(&condition.r#type, "do-while condition")?;
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
                let init = self.analyze_for_init(init)?;
                let condition = condition
                    .map(|expr| {
                        let analyzed = self.analyze_expr(expr)?;
                        self.ensure_condition_type(&analyzed.r#type, "for condition")?;
                        Ok(analyzed)
                    })
                    .transpose()?;
                let post = post.map(|expr| self.analyze_expr(expr)).transpose()?;
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

    pub(crate) fn analyze_expr(&mut self, expr: Expr) -> Result<Expr, SemanticError> {
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

        let result: Expr = match kind {
            ExprKind::Constant(c) => {
                let ty = match &c {
                    Const::Char(_) => Type::Int,
                    Const::Short(_) => Type::Short,
                    Const::UShort(_) => Type::UShort,
                    Const::UChar(_) => Type::UInt,
                    Const::Int(_) => Type::Int,
                    Const::Long(_) => Type::Long,
                    Const::UInt(_) => Type::UInt,
                    Const::ULong(_) => Type::ULong,
                    Const::Double(_) => Type::Double,
                };
                Expr {
                    kind: ExprKind::Constant(c),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::String(value) => Expr {
                kind: ExprKind::String(value),
                start,
                end,
                source,
                r#type: Type::Pointer(Box::new(Type::Char)),
            },
            ExprKind::Var(name) => {
                let symbol = self.lookup_symbol(&name)?;
                Expr {
                    kind: ExprKind::Var(symbol.unique.clone()),
                    start,
                    end,
                    source,
                    r#type: symbol.ty.clone(),
                }
            }
            ExprKind::SizeOf(expr0) => {
                let analyzed = self.analyze_expr_no_decay(*expr0)?;
                let size = self.size_of_type(&analyzed.r#type)?;
                Expr {
                    kind: ExprKind::Constant(Const::ULong(size)),
                    start,
                    end,
                    source,
                    r#type: Type::ULong,
                }
            }
            ExprKind::SizeOfType(ty) => {
                let size = self.size_of_type(&ty)?;
                Expr {
                    kind: ExprKind::Constant(Const::ULong(size)),
                    start,
                    end,
                    source,
                    r#type: Type::ULong,
                }
            }
            ExprKind::FunctionCall(name, args) => {
                let Some(signature) = self.functions.get(&name).cloned() else {
                    return Err(SemanticError::UndeclaredIdentifier(name));
                };
                if args.len() != signature.param_types.len() {
                    return Err(SemanticError::WrongArgCount(
                        name.clone(),
                        signature.param_types.len(),
                        args.len(),
                    ));
                }

                let analyzed_args = args
                    .into_iter()
                    .zip(signature.param_types.iter())
                    .map(|(arg, expected)| {
                        let expr = self.analyze_expr(arg)?;
                        self.ensure_assignable(expected, &expr.r#type, "function argument")?;
                        Ok(expr)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                Expr {
                    kind: ExprKind::FunctionCall(name, analyzed_args),
                    start,
                    end,
                    source,
                    r#type: signature.return_type,
                }
            }
            ExprKind::Cast(target, expr0) => {
                // Analyze source first so we can give a precise from->to in the error.
                let expr1 = self.analyze_expr(*expr0)?;
                // Forbid casting to arrays
                if Self::is_array_type(&target) {
                    return Err(SemanticError::CastToError(
                        self.decay_array_type(&expr1.r#type),
                        target,
                    ));
                }
                self.ensure_castable(&expr1.r#type, &target)?;
                let ty = target.clone();
                Expr {
                    kind: ExprKind::Cast(ty.clone(), Box::new(expr1)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::Neg(expr0) => {
                self.analyze_arithmetic_unary(*expr0, start, end, source.clone(), ExprKind::Neg)?
            }
            ExprKind::BitNot(expr0) => {
                self.analyze_integer_unary(*expr0, start, end, source.clone(), ExprKind::BitNot)?
            }
            ExprKind::Not(expr0) => {
                let expr1 = self.analyze_expr(*expr0)?;
                self.ensure_condition_type(&expr1.r#type, "logical not")?;
                Expr {
                    kind: ExprKind::Not(Box::new(expr1)),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                }
            }
            ExprKind::AddrOf(expr0) => {
                let expr1 = self.analyze_expr_no_decay(*expr0)?;
                self.ensure_lvalue(&expr1, "address-of operand")?;
                let ty = Type::Pointer(Box::new(expr1.r#type.clone()));
                Expr {
                    kind: ExprKind::AddrOf(Box::new(expr1)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::Dereference(expr0) => {
                let expr1 = self.analyze_expr(*expr0)?;
                let ty = match &expr1.r#type {
                    Type::Pointer(inner) => match inner.as_ref() {
                        Type::Void => {
                            return Err(SemanticError::PointerSizedBaseRequired(
                                "dereference",
                                (*inner).as_ref().clone(),
                            ));
                        }
                        other => other.clone(),
                    },
                    other => return Err(SemanticError::DereferenceError(other.clone())),
                };
                Expr {
                    kind: ExprKind::Dereference(Box::new(expr1)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::Member {
                base,
                member,
                offset: _,
                is_arrow,
            } => {
                let (base_expr, struct_tag) = if is_arrow {
                    let analyzed_base = self.analyze_expr(*base)?;
                    match analyzed_base.r#type.clone() {
                        Type::Pointer(inner) => match *inner {
                            Type::Struct(tag) => (analyzed_base, tag),
                            other => {
                                return Err(SemanticError::Unsupported(format!(
                                    "operator '->' requires pointer to struct, found `{other}`"
                                )));
                            }
                        },
                        other => {
                            return Err(SemanticError::Unsupported(format!(
                                "operator '->' requires pointer to struct, found `{other}`"
                            )));
                        }
                    }
                } else {
                    let analyzed_base = self.analyze_expr_no_decay(*base)?;
                    self.ensure_lvalue(&analyzed_base, "member access base")?;
                    match analyzed_base.r#type.clone() {
                        Type::Struct(tag) => (analyzed_base, tag),
                        _ => return Err(SemanticError::MemberAccessOnNonStruct),
                    }
                };

                let member_layout = self.struct_member_layout(&struct_tag, &member)?;
                let member_type = member_layout.ty.clone();
                Expr {
                    kind: ExprKind::Member {
                        base: Box::new(base_expr),
                        member,
                        offset: member_layout.offset,
                        is_arrow,
                    },
                    start,
                    end,
                    source,
                    r#type: member_type,
                }
            }
            ExprKind::Conditional(cond, then_expr, else_expr) => {
                let cond = self.analyze_expr(*cond)?;
                self.ensure_condition_type(&cond.r#type, "conditional condition")?;
                let then_expr = self.analyze_expr(*then_expr)?;
                let else_expr = self.analyze_expr(*else_expr)?;
                let then_type = then_expr.r#type.clone();
                let else_type = else_expr.r#type.clone();
                let result_type = if then_type.is_numeric() && else_type.is_numeric() {
                    self.numeric_result_type(&then_type, &else_type)?
                } else if then_type.is_pointer() && else_type.is_pointer() {
                    if !Self::pointer_types_compatible(&then_type, &else_type) {
                        return Err(SemanticError::ConditionalTypeMismatch(then_type, else_type));
                    }
                    then_type.clone()
                } else {
                    return Err(SemanticError::ConditionalTypeMismatch(then_type, else_type));
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
            ExprKind::Add(lhs, rhs) => self.analyze_add(*lhs, *rhs, start, end, source.clone())?,
            ExprKind::Sub(lhs, rhs) => self.analyze_sub(*lhs, *rhs, start, end, source.clone())?,
            ExprKind::Mul(lhs, rhs) => self.analyze_arithmetic_binary(
                *lhs,
                *rhs,
                start,
                end,
                source.clone(),
                ExprKind::Mul,
            )?,
            ExprKind::Div(lhs, rhs) => self.analyze_arithmetic_binary(
                *lhs,
                *rhs,
                start,
                end,
                source.clone(),
                ExprKind::Div,
            )?,
            ExprKind::Rem(lhs, rhs) => {
                self.analyze_integer_binary(*lhs, *rhs, start, end, source.clone(), ExprKind::Rem)?
            }
            ExprKind::Equal(lhs, rhs) => {
                self.analyze_equality(*lhs, *rhs, start, end, source.clone(), ExprKind::Equal)?
            }
            ExprKind::NotEqual(lhs, rhs) => {
                self.analyze_equality(*lhs, *rhs, start, end, source.clone(), ExprKind::NotEqual)?
            }
            ExprKind::LessThan(lhs, rhs) => {
                self.analyze_comparison(*lhs, *rhs, start, end, source.clone(), ExprKind::LessThan)?
            }
            ExprKind::LessThanEqual(lhs, rhs) => self.analyze_comparison(
                *lhs,
                *rhs,
                start,
                end,
                source.clone(),
                ExprKind::LessThanEqual,
            )?,
            ExprKind::GreaterThan(lhs, rhs) => self.analyze_comparison(
                *lhs,
                *rhs,
                start,
                end,
                source.clone(),
                ExprKind::GreaterThan,
            )?,
            ExprKind::GreaterThanEqual(lhs, rhs) => self.analyze_comparison(
                *lhs,
                *rhs,
                start,
                end,
                source.clone(),
                ExprKind::GreaterThanEqual,
            )?,
            ExprKind::Or(lhs, rhs) => {
                self.analyze_logical_binary(*lhs, *rhs, start, end, source.clone(), ExprKind::Or)?
            }
            ExprKind::And(lhs, rhs) => {
                self.analyze_logical_binary(*lhs, *rhs, start, end, source.clone(), ExprKind::And)?
            }
            ExprKind::BitAnd(lhs, rhs) => self.analyze_integer_binary(
                *lhs,
                *rhs,
                start,
                end,
                source.clone(),
                ExprKind::BitAnd,
            )?,
            ExprKind::Xor(lhs, rhs) => {
                self.analyze_integer_binary(*lhs, *rhs, start, end, source.clone(), ExprKind::Xor)?
            }
            ExprKind::BitOr(lhs, rhs) => self.analyze_integer_binary(
                *lhs,
                *rhs,
                start,
                end,
                source.clone(),
                ExprKind::BitOr,
            )?,
            ExprKind::LeftShift(lhs, rhs) => {
                let lhs = self.analyze_expr(*lhs)?;
                let rhs = self.analyze_expr(*rhs)?;
                self.ensure_integer_type(&lhs.r#type, "shift operand")?;
                self.ensure_integer_type(&rhs.r#type, "shift amount")?;
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
                self.ensure_integer_type(&lhs.r#type, "shift operand")?;
                self.ensure_integer_type(&rhs.r#type, "shift amount")?;
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
                self.ensure_lvalue(&lhs, "assignment left-hand side")?;
                if Self::is_array_type(&lhs.r#type) {
                    return Err(SemanticError::Unsupported(
                        "assignment to array type is not allowed".to_string(),
                    ));
                }
                let lhs_type = lhs.r#type.clone();
                let rhs = self.analyze_expr(*rhs)?;
                self.ensure_assignable(&lhs_type, &rhs.r#type, "assignment")?;
                Expr {
                    kind: ExprKind::Assignment(Box::new(lhs), Box::new(rhs)),
                    start,
                    end,
                    source,
                    r#type: lhs_type,
                }
            }
            ExprKind::PreIncrement(expr0) => {
                let expr1 = self.analyze_expr_no_decay(*expr0)?;
                self.ensure_lvalue(&expr1, "pre-increment target")?;
                self.ensure_numeric_type(&expr1.r#type, "pre-increment operand")?;
                let ty = expr1.r#type.clone();
                Expr {
                    kind: ExprKind::PreIncrement(Box::new(expr1)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::PreDecrement(expr0) => {
                let expr1 = self.analyze_expr_no_decay(*expr0)?;
                self.ensure_lvalue(&expr1, "pre-decrement target")?;
                self.ensure_numeric_type(&expr1.r#type, "pre-decrement operand")?;
                let ty = expr1.r#type.clone();
                Expr {
                    kind: ExprKind::PreDecrement(Box::new(expr1)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::PostIncrement(expr0) => {
                let expr1 = self.analyze_expr_no_decay(*expr0)?;
                self.ensure_lvalue(&expr1, "post-increment target")?;
                self.ensure_numeric_type(&expr1.r#type, "post-increment operand")?;
                let ty = expr1.r#type.clone();
                Expr {
                    kind: ExprKind::PostIncrement(Box::new(expr1)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
            ExprKind::PostDecrement(expr0) => {
                let expr1 = self.analyze_expr_no_decay(*expr0)?;
                self.ensure_lvalue(&expr1, "post-decrement target")?;
                self.ensure_numeric_type(&expr1.r#type, "post-decrement operand")?;
                let ty = expr1.r#type.clone();
                Expr {
                    kind: ExprKind::PostDecrement(Box::new(expr1)),
                    start,
                    end,
                    source,
                    r#type: ty,
                }
            }
        };

        Ok(self.apply_array_decay(result, decay_arrays))
    }

    fn analyze_for_init(&mut self, init: ForInit) -> Result<ForInit, SemanticError> {
        match init {
            ForInit::Declaration(stmt) => {
                Ok(ForInit::Declaration(Box::new(self.analyze_stmt(*stmt)?)))
            }
            ForInit::Expr(expr_opt) => Ok(ForInit::Expr(
                expr_opt.map(|e| self.analyze_expr(e)).transpose()?,
            )),
        }
    }

    fn ensure_lvalue(&self, expr: &Expr, context: &'static str) -> Result<(), SemanticError> {
        match &expr.kind {
            ExprKind::Var(_) | ExprKind::Dereference(_) | ExprKind::Member { .. } => Ok(()),
            _ => Err(SemanticError::LValueRequired(context)),
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
        self.ensure_integer_type(&expr.r#type, "unary operator")?;
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
        self.ensure_numeric_type(&expr.r#type, "unary operator")?;
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

        let result_type = if lhs_type.is_numeric() && rhs_type.is_numeric() {
            self.numeric_result_type(&lhs_type, &rhs_type)?
        } else if lhs_type.is_pointer() && rhs_type.is_integer() {
            self.pointer_arithmetic_base(&lhs_type, "pointer addition")?;
            lhs_type.clone()
        } else if lhs_type.is_integer() && rhs_type.is_pointer() {
            self.pointer_arithmetic_base(&rhs_type, "pointer addition")?;
            rhs_type.clone()
        } else {
            return Err(SemanticError::InvalidBinaryExpr(lhs, TokenKind::Plus, rhs));
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

        let result_type = if lhs_type.is_numeric() && rhs_type.is_numeric() {
            self.numeric_result_type(&lhs_type, &rhs_type)?
        } else if lhs_type.is_pointer() && rhs_type.is_integer() {
            self.pointer_arithmetic_base(&lhs_type, "pointer subtraction")?;
            lhs_type.clone()
        } else if lhs_type.is_pointer() && rhs_type.is_pointer() {
            let lhs_base = self.pointer_arithmetic_base(&lhs_type, "pointer subtraction")?;
            let rhs_base = self.pointer_arithmetic_base(&rhs_type, "pointer subtraction")?;
            if lhs_base != rhs_base {
                return Err(SemanticError::PointerSubTypeMismatch(lhs_type, rhs_type));
            }
            Type::Long
        } else {
            return Err(SemanticError::InvalidBinaryExpr(lhs, TokenKind::Minus, rhs));
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
        self.ensure_integer_type(&lhs.r#type, "binary operand")?;
        self.ensure_integer_type(&rhs.r#type, "binary operand")?;
        let ty = self.numeric_result_type(&lhs.r#type, &rhs.r#type)?;
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
        self.ensure_numeric_type(&lhs.r#type, "binary operand")?;
        self.ensure_numeric_type(&rhs.r#type, "binary operand")?;
        let ty = self.numeric_result_type(&lhs.r#type, &rhs.r#type)?;
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
        if lhs.r#type.is_pointer() || rhs.r#type.is_pointer() {
            self.ensure_pointer_comparable(&lhs.r#type, &rhs.r#type, "equality comparison")?;
        } else {
            self.ensure_numeric_type(&lhs.r#type, "comparison operand")?;
            self.ensure_numeric_type(&rhs.r#type, "comparison operand")?;
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
        self.ensure_numeric_type(&lhs.r#type, "comparison operand")?;
        self.ensure_numeric_type(&rhs.r#type, "comparison operand")?;
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
        self.ensure_condition_type(&lhs.r#type, "logical operand")?;
        self.ensure_condition_type(&rhs.r#type, "logical operand")?;
        Ok(Expr {
            kind: constructor(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source,
            r#type: Type::Int,
        })
    }

    fn numeric_result_type(&self, lhs: &Type, rhs: &Type) -> Result<Type, SemanticError> {
        if !lhs.is_numeric() || !rhs.is_numeric() {
            return Err(SemanticError::IncompatibleForContext(
                "numeric expression",
                lhs.clone(),
                rhs.clone(),
            ));
        }

        if lhs.is_floating() || rhs.is_floating() {
            return Ok(Type::Double);
        }

        let lr = lhs.type_rank();
        let rr = rhs.type_rank();
        Ok(if lr >= rr { lhs.clone() } else { rhs.clone() })
    }

    fn ensure_integer_type(&self, ty: &Type, context: &'static str) -> Result<(), SemanticError> {
        let ty = self.decay_array_type(ty);
        if ty.is_integer() {
            Ok(())
        } else {
            Err(SemanticError::IntegerTypeRequired(context, ty))
        }
    }

    fn ensure_condition_type(&self, ty: &Type, context: &'static str) -> Result<(), SemanticError> {
        let ty = self.decay_array_type(ty);
        if ty.is_integer() || ty.is_pointer() {
            Ok(())
        } else {
            Err(SemanticError::ScalarTypeRequired(context, ty))
        }
    }

    fn ensure_numeric_type(&self, ty: &Type, context: &'static str) -> Result<(), SemanticError> {
        let ty = self.decay_array_type(ty);
        if ty.is_numeric() {
            Ok(())
        } else {
            Err(SemanticError::NumericTypeRequired(context, ty))
        }
    }

    fn ensure_assignable(
        &self,
        target: &Type,
        value: &Type,
        context: &'static str,
    ) -> Result<(), SemanticError> {
        if Self::is_array_type(target) {
            return Err(SemanticError::Unsupported(format!(
                "{context} cannot assign to array type"
            )));
        }

        let value_type = self.decay_array_type(value);

        if target == &value_type {
            return Ok(());
        }

        if target.is_numeric() && value_type.is_numeric() {
            return Ok(());
        }

        if target.is_pointer() {
            if value_type.is_pointer() && Self::pointer_types_compatible(target, &value_type) {
                return Ok(());
            }
            if value_type.is_integer() {
                return Ok(());
            }
        }

        Err(SemanticError::IncompatibleForContext(
            context,
            target.clone(),
            value_type,
        ))
    }

    fn ensure_castable(&self, from: &Type, to: &Type) -> Result<(), SemanticError> {
        let from_type = self.decay_array_type(from);
        let to_type = self.decay_array_type(to);

        if from_type.is_numeric() && to_type.is_numeric() {
            return Ok(());
        }

        if from_type.is_pointer() && to_type.is_pointer() {
            if Self::pointer_types_compatible(&from_type, &to_type) {
                return Ok(());
            }
        }

        if (from_type.is_pointer() && to_type.is_integer())
            || (from_type.is_integer() && to_type.is_pointer())
        {
            return Ok(());
        }

        Err(SemanticError::CastToError(from_type, to_type))
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

    fn ensure_pointer_comparable(
        &self,
        lhs: &Type,
        rhs: &Type,
        context: &'static str,
    ) -> Result<(), SemanticError> {
        let lhs_type = self.decay_array_type(lhs);
        let rhs_type = self.decay_array_type(rhs);

        if lhs_type.is_pointer() && rhs_type.is_pointer() {
            if Self::pointer_types_compatible(&lhs_type, &rhs_type) {
                return Ok(());
            }
            return Err(SemanticError::IncompatibleForContext(
                context, lhs_type, rhs_type,
            ));
        } else if (lhs_type.is_pointer() && rhs_type.is_integer())
            || (lhs_type.is_integer() && rhs_type.is_pointer())
        {
            return Ok(());
        }

        Err(SemanticError::IncompatibleForContext(
            context, lhs_type, rhs_type,
        ))
    }

    fn pointer_arithmetic_base<'a>(
        &self,
        ty: &'a Type,
        context: &'static str,
    ) -> Result<&'a Type, SemanticError> {
        match ty {
            Type::Pointer(inner) => match inner.as_ref() {
                Type::Void | Type::FunType(_, _) => Err(SemanticError::PointerSizedBaseRequired(
                    context,
                    (*inner).as_ref().clone(),
                )),
                base => Ok(base),
            },
            other => Err(SemanticError::PointerOperandRequired(
                context,
                other.clone(),
            )),
        }
    }

    fn adjust_parameter_type(&self, ty: Type) -> Type {
        match ty {
            Type::Array(inner, _) => Type::Pointer(inner),
            Type::IncompleteArray(inner) => Type::Pointer(inner),
            Type::FunType(params, ret) => Type::Pointer(Box::new(Type::FunType(params, ret))),
            other => other,
        }
    }

    fn size_of_type(&self, ty: &Type) -> Result<u64, SemanticError> {
        match ty {
            Type::FunType(_, _) => Err(SemanticError::Unsupported(format!(
                "cannot apply sizeof to function type `{ty}`"
            ))),
            Type::IncompleteArray(_) => Err(SemanticError::Unsupported(format!(
                "cannot apply sizeof to incomplete type `{ty}`"
            ))),
            _ => {
                let size = ty.byte_size();
                Ok(size as u64)
            }
        }
    }

    fn decay_array_type(&self, ty: &Type) -> Type {
        match ty {
            Type::Array(inner, _) => Type::Pointer(inner.clone()),
            Type::IncompleteArray(inner) => Type::Pointer(inner.clone()),
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
                Type::IncompleteArray(inner) => Type::Pointer(inner),
                Type::FunType(params, ret) => Type::Pointer(Box::new(Type::FunType(params, ret))),
                other => other,
            };
        }
        expr
    }

    fn is_array_type(ty: &Type) -> bool {
        matches!(ty, Type::Array(_, _) | Type::IncompleteArray(_))
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

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self {
            counter: 0,
            scopes: Vec::new(),
            functions: BTreeMap::new(),
            current_return_type: None,
            struct_defs: BTreeMap::new(),
            struct_layouts: BTreeMap::new(),
        }
    }
}
