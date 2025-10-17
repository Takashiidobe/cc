use std::collections::HashMap;

use crate::parse::{Expr, ExprKind, Program, Stmt, StmtKind};

#[derive(Default)]
pub struct SemanticAnalyzer {
    counter: usize,
    scopes: Vec<HashMap<String, String>>,
}

impl SemanticAnalyzer {
    pub fn analyze_program(mut self, program: Program) -> Program {
        Program(self.analyze_stmt(program.0))
    }

    fn analyze_stmt(&mut self, stmt: Stmt) -> Stmt {
        let Stmt {
            kind,
            start,
            end,
            source,
            r#type,
        } = stmt;

        let kind = match kind {
            StmtKind::Return(expr) => StmtKind::Return(self.analyze_expr(expr)),
            StmtKind::Expr(expr) => StmtKind::Expr(self.analyze_expr(expr)),
            StmtKind::Block(stmts) => {
                self.enter_scope();
                let stmts = stmts.into_iter().map(|s| self.analyze_stmt(s)).collect();
                self.exit_scope();
                StmtKind::Block(stmts)
            }
            StmtKind::FnDecl(name, body) => {
                self.enter_scope();
                let body = body.into_iter().map(|s| self.analyze_stmt(s)).collect();
                self.exit_scope();
                StmtKind::FnDecl(name, body)
            }
            StmtKind::Declaration { name, init } => {
                let init = init.map(|expr| self.analyze_expr(expr));
                let unique_name = self.fresh_name(&name);
                self.insert(name, unique_name.clone());
                StmtKind::Declaration {
                    name: unique_name,
                    init,
                }
            }
            StmtKind::Null => StmtKind::Null,
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => StmtKind::If {
                condition: self.analyze_expr(condition),
                then_branch: Box::new(self.analyze_stmt(*then_branch)),
                else_branch: else_branch.map(|stmt| Box::new(self.analyze_stmt(*stmt))),
            },
        };

        Stmt {
            kind,
            start,
            end,
            source,
            r#type,
        }
    }

    fn analyze_expr(&mut self, expr: Expr) -> Expr {
        let Expr {
            kind,
            start,
            end,
            source,
            r#type,
        } = expr;

        let kind = match kind {
            ExprKind::Integer(n) => ExprKind::Integer(n),
            ExprKind::Var(name) => {
                let unique = self
                    .lookup(&name)
                    .unwrap_or_else(|| panic!("use of undeclared identifier {name}"));
                ExprKind::Var(unique)
            }
            ExprKind::Neg(expr) => ExprKind::Neg(Box::new(self.analyze_expr(*expr))),
            ExprKind::BitNot(expr) => ExprKind::BitNot(Box::new(self.analyze_expr(*expr))),
            ExprKind::Add(lhs, rhs) => ExprKind::Add(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::Sub(lhs, rhs) => ExprKind::Sub(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::Mul(lhs, rhs) => ExprKind::Mul(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::Div(lhs, rhs) => ExprKind::Div(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::Rem(lhs, rhs) => ExprKind::Rem(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::Equal(lhs, rhs) => ExprKind::Equal(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::NotEqual(lhs, rhs) => ExprKind::NotEqual(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::LessThan(lhs, rhs) => ExprKind::LessThan(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::LessThanEqual(lhs, rhs) => ExprKind::LessThanEqual(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::GreaterThan(lhs, rhs) => ExprKind::GreaterThan(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::GreaterThanEqual(lhs, rhs) => ExprKind::GreaterThanEqual(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::Or(lhs, rhs) => ExprKind::Or(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::And(lhs, rhs) => ExprKind::And(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::Not(expr) => ExprKind::Not(Box::new(self.analyze_expr(*expr))),
            ExprKind::BitAnd(lhs, rhs) => ExprKind::BitAnd(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::Xor(lhs, rhs) => ExprKind::Xor(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::BitOr(lhs, rhs) => ExprKind::BitOr(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::PreIncrement(expr) => {
                ExprKind::PreIncrement(Box::new(self.analyze_expr(*expr)))
            }
            ExprKind::PreDecrement(expr) => {
                ExprKind::PreDecrement(Box::new(self.analyze_expr(*expr)))
            }
            ExprKind::PostIncrement(expr) => {
                ExprKind::PostIncrement(Box::new(self.analyze_expr(*expr)))
            }
            ExprKind::PostDecrement(expr) => {
                ExprKind::PostDecrement(Box::new(self.analyze_expr(*expr)))
            }
            ExprKind::LeftShift(lhs, rhs) => ExprKind::LeftShift(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::RightShift(lhs, rhs) => ExprKind::RightShift(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::Assignment(lhs, rhs) => ExprKind::Assignment(
                Box::new(self.analyze_expr(*lhs)),
                Box::new(self.analyze_expr(*rhs)),
            ),
            ExprKind::Conditional(cond, then_expr, else_expr) => ExprKind::Conditional(
                Box::new(self.analyze_expr(*cond)),
                Box::new(self.analyze_expr(*then_expr)),
                Box::new(self.analyze_expr(*else_expr)),
            ),
        };

        Expr {
            kind,
            start,
            end,
            source,
            r#type,
        }
    }

    fn fresh_name(&mut self, base: &str) -> String {
        let name = format!("{base}{}", self.counter);
        self.counter += 1;
        name
    }

    fn insert(&mut self, base: String, unique: String) {
        self.scopes
            .last_mut()
            .expect("scope stack empty")
            .insert(base, unique);
    }

    fn lookup(&self, name: &str) -> Option<String> {
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.get(name) {
                return Some(value.clone());
            }
        }
        None
    }

    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }
}
