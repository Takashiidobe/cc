pub(crate) mod error;

use std::collections::{BTreeMap, BTreeSet};

use crate::parse::{
    Decl, DeclKind, Expr, ExprKind, ForInit, FunctionDecl, MemberDeclaration, Program, Stmt,
    StmtKind, StructDeclaration, Type, VariableDecl,
};

use self::error::StructLabelerError;

#[derive(Default)]
pub(crate) struct StructLabeler {
    structs: BTreeMap<String, StructInfo>,
}

#[derive(Default)]
struct StructInfo {
    defined: bool,
}

impl StructLabeler {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn label_program(mut self, program: Program) -> Result<Program, StructLabelerError> {
        for decl in &program.0 {
            self.visit_decl(decl)?;
        }
        Ok(program)
    }

    fn visit_decl(&mut self, decl: &Decl) -> Result<(), StructLabelerError> {
        match &decl.kind {
            DeclKind::Struct(struct_decl) => self.visit_struct_decl(struct_decl),
            DeclKind::Variable(var_decl) => self.visit_variable_decl(var_decl),
            DeclKind::Function(func_decl) => self.visit_function_decl(func_decl),
        }
    }

    fn visit_struct_decl(&mut self, decl: &StructDeclaration) -> Result<(), StructLabelerError> {
        let tag = decl.tag.clone();
        if decl.members.is_empty() {
            self.structs.entry(tag).or_insert_with(StructInfo::default);
            return Ok(());
        }

        {
            let entry = self
                .structs
                .entry(tag.clone())
                .or_insert_with(StructInfo::default);
            if entry.defined {
                return Err(StructLabelerError::DuplicateDefinition(tag));
            }
            entry.defined = true;
        }

        self.validate_struct_members(&decl.tag, &decl.members)
    }

    fn validate_struct_members(
        &mut self,
        struct_name: &str,
        members: &[MemberDeclaration],
    ) -> Result<(), StructLabelerError> {
        let mut seen = BTreeSet::new();
        for member in members {
            if !seen.insert(member.member_name.clone()) {
                return Err(StructLabelerError::DuplicateMember(
                    struct_name.to_string(),
                    member.member_name.clone(),
                ));
            }

            if matches!(
                &member.member_type,
                Type::Struct(member_struct) if member_struct == struct_name
            ) {
                return Err(StructLabelerError::RecursiveStruct(struct_name.to_string()));
            }

            self.note_type(&member.member_type, true)?;
        }
        Ok(())
    }

    fn visit_variable_decl(&mut self, decl: &VariableDecl) -> Result<(), StructLabelerError> {
        self.note_type(&decl.r#type, true)?;
        if let Some(init) = &decl.init {
            self.visit_expr(init)?;
        }
        Ok(())
    }

    fn visit_function_decl(&mut self, decl: &FunctionDecl) -> Result<(), StructLabelerError> {
        self.note_type(&decl.return_type, true)?;
        for param in &decl.params {
            self.note_type(&param.r#type, true)?;
        }

        if let Some(body) = &decl.body {
            for stmt in body {
                self.visit_stmt(stmt)?;
            }
        }
        Ok(())
    }

    fn visit_stmt(&mut self, stmt: &Stmt) -> Result<(), StructLabelerError> {
        match &stmt.kind {
            StmtKind::Expr(expr) | StmtKind::Return(expr) => self.visit_expr(expr),
            StmtKind::Compound(stmts) => {
                for inner in stmts {
                    self.visit_stmt(inner)?;
                }
                Ok(())
            }
            StmtKind::Declaration(var) => self.visit_variable_decl(var),
            StmtKind::Null | StmtKind::Break { .. } | StmtKind::Continue { .. } => Ok(()),
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.visit_expr(condition)?;
                self.visit_stmt(then_branch)?;
                if let Some(else_branch) = else_branch {
                    self.visit_stmt(else_branch)?;
                }
                Ok(())
            }
            StmtKind::While {
                condition, body, ..
            } => {
                self.visit_expr(condition)?;
                self.visit_stmt(body)
            }
            StmtKind::DoWhile {
                body, condition, ..
            } => {
                self.visit_stmt(body)?;
                self.visit_expr(condition)
            }
            StmtKind::For {
                init,
                condition,
                post,
                body,
                ..
            } => {
                self.visit_for_init(init)?;
                if let Some(cond) = condition {
                    self.visit_expr(cond)?;
                }
                if let Some(post) = post {
                    self.visit_expr(post)?;
                }
                self.visit_stmt(body)
            }
        }
    }

    fn visit_for_init(&mut self, init: &ForInit) -> Result<(), StructLabelerError> {
        match init {
            ForInit::Declaration(stmt) => self.visit_stmt(stmt),
            ForInit::Expr(expr_opt) => {
                if let Some(expr) = expr_opt {
                    self.visit_expr(expr)?;
                }
                Ok(())
            }
        }
    }

    fn visit_expr(&mut self, expr: &Expr) -> Result<(), StructLabelerError> {
        self.note_type(&expr.r#type, true)?;
        match &expr.kind {
            ExprKind::Constant(_) | ExprKind::String(_) | ExprKind::Var(_) => Ok(()),
            ExprKind::FunctionCall(_, args) => {
                for arg in args {
                    self.visit_expr(arg)?;
                }
                Ok(())
            }
            ExprKind::Cast(target, inner) => {
                self.note_type(target, true)?;
                self.visit_expr(inner)
            }
            ExprKind::Neg(inner)
            | ExprKind::BitNot(inner)
            | ExprKind::SizeOf(inner)
            | ExprKind::AddrOf(inner)
            | ExprKind::Dereference(inner)
            | ExprKind::Not(inner)
            | ExprKind::PreIncrement(inner)
            | ExprKind::PreDecrement(inner)
            | ExprKind::PostIncrement(inner)
            | ExprKind::PostDecrement(inner) => self.visit_expr(inner),
            ExprKind::Conditional(cond, then_expr, else_expr) => {
                self.visit_expr(cond)?;
                self.visit_expr(then_expr)?;
                self.visit_expr(else_expr)
            }
            ExprKind::Add(lhs, rhs)
            | ExprKind::Sub(lhs, rhs)
            | ExprKind::Mul(lhs, rhs)
            | ExprKind::Div(lhs, rhs)
            | ExprKind::Rem(lhs, rhs)
            | ExprKind::Equal(lhs, rhs)
            | ExprKind::NotEqual(lhs, rhs)
            | ExprKind::LessThan(lhs, rhs)
            | ExprKind::LessThanEqual(lhs, rhs)
            | ExprKind::GreaterThan(lhs, rhs)
            | ExprKind::GreaterThanEqual(lhs, rhs)
            | ExprKind::Or(lhs, rhs)
            | ExprKind::And(lhs, rhs)
            | ExprKind::BitAnd(lhs, rhs)
            | ExprKind::Xor(lhs, rhs)
            | ExprKind::BitOr(lhs, rhs)
            | ExprKind::LeftShift(lhs, rhs)
            | ExprKind::RightShift(lhs, rhs)
            | ExprKind::Assignment(lhs, rhs) => {
                self.visit_expr(lhs)?;
                self.visit_expr(rhs)
            }
            ExprKind::SizeOfType(target) => self.note_type(target, true),
        }
    }

    fn note_type(&mut self, ty: &Type, require_complete: bool) -> Result<(), StructLabelerError> {
        match ty {
            Type::Struct(tag) => {
                let entry = self
                    .structs
                    .entry(tag.clone())
                    .or_insert_with(StructInfo::default);
                if require_complete && !entry.defined {
                    return Err(StructLabelerError::IncompleteStruct(tag.clone()));
                }
            }
            Type::Pointer(inner) => {
                self.note_type(inner, false)?;
            }
            Type::Array(inner, _) | Type::IncompleteArray(inner) => {
                self.note_type(inner, true)?;
            }
            Type::FunType(params, ret) => {
                for param in params {
                    self.note_type(param, true)?;
                }
                self.note_type(ret, true)?;
            }
            _ => {}
        }
        Ok(())
    }
}
