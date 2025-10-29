mod error;

use crate::{
    parse::{Decl, DeclKind, FunctionDecl, Program, Stmt, StmtKind},
    semantic::loop_label::error::LoopLabelerError,
};

#[derive(Default)]
pub(crate) struct LoopLabeler {
    next_loop_id: usize,
    loop_stack: Vec<usize>,
}

impl LoopLabeler {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn label_program(mut self, program: Program) -> Result<Program, LoopLabelerError> {
        let decls: Result<Vec<_>, _> = program
            .0
            .into_iter()
            .map(|decl| self.label_decl(decl))
            .collect();
        Ok(Program(decls?))
    }

    fn label_decl(&mut self, decl: Decl) -> Result<Decl, LoopLabelerError> {
        let Decl { kind, loc } = decl;

        let kind = match kind {
            DeclKind::Function(func) => DeclKind::Function(self.label_function(func)?),
            DeclKind::Variable(var) => DeclKind::Variable(var),
            DeclKind::Struct(decl) => DeclKind::Struct(decl),
        };

        Ok(Decl { kind, loc })
    }

    fn label_function(&mut self, decl: FunctionDecl) -> Result<FunctionDecl, LoopLabelerError> {
        let FunctionDecl {
            name,
            params,
            body,
            storage_class,
            return_type,
        } = decl;

        let body: Option<Vec<Stmt>> = body
            .map(|stmts| {
                stmts
                    .into_iter()
                    .map(|s| self.label_stmt(s))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;

        Ok(FunctionDecl {
            name,
            params,
            body,
            storage_class,
            return_type,
        })
    }

    fn label_stmt(&mut self, stmt: Stmt) -> Result<Stmt, LoopLabelerError> {
        let Stmt { kind, loc, r#type } = stmt;

        let kind = match kind {
            StmtKind::Return(expr) => StmtKind::Return(expr),
            StmtKind::Expr(expr) => StmtKind::Expr(expr),
            StmtKind::Compound(stmts) => {
                let stmts: Result<Vec<_>, _> = stmts
                    .into_iter()
                    .map(|stmt| self.label_stmt(stmt))
                    .collect();
                StmtKind::Compound(stmts?)
            }
            StmtKind::Declaration(decl) => StmtKind::Declaration(decl),
            StmtKind::Null => StmtKind::Null,
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let else_branch = match else_branch {
                    Some(s) => Some(Box::new(self.label_stmt(*s)?)),
                    None => None,
                };

                StmtKind::If {
                    condition,
                    then_branch: Box::new(self.label_stmt(*then_branch)?),
                    else_branch,
                }
            }
            StmtKind::While {
                condition,
                body,
                loop_id: _,
            } => {
                let (loop_id, body) = self.loop_body(body)?;
                StmtKind::While {
                    condition,
                    body,
                    loop_id,
                }
            }
            StmtKind::DoWhile {
                body,
                condition,
                loop_id: _,
            } => {
                let (loop_id, body) = self.loop_body(body)?;
                StmtKind::DoWhile {
                    body,
                    condition,
                    loop_id,
                }
            }
            StmtKind::For {
                init,
                condition,
                post,
                body,
                loop_id: _,
            } => {
                let (loop_id, body) = self.loop_body(body)?;
                StmtKind::For {
                    init,
                    condition,
                    post,
                    body,
                    loop_id,
                }
            }
            StmtKind::Break { loop_id: _ } => {
                let loop_id = *self
                    .loop_stack
                    .last()
                    .ok_or(LoopLabelerError::BreakNotInLoop)?;
                StmtKind::Break {
                    loop_id: Some(loop_id),
                }
            }
            StmtKind::Continue { loop_id: _ } => {
                let loop_id = *self
                    .loop_stack
                    .last()
                    .ok_or(LoopLabelerError::ContinueNotInLoop)?;
                StmtKind::Continue {
                    loop_id: Some(loop_id),
                }
            }
        };

        Ok(Stmt { kind, loc, r#type })
    }

    fn loop_body(
        &mut self,
        body: Box<Stmt>,
    ) -> Result<(Option<usize>, Box<Stmt>), LoopLabelerError> {
        let loop_id = self.next_loop_id();
        self.loop_stack.push(loop_id);
        let body = Box::new(self.label_stmt(*body)?);
        self.loop_stack.pop();
        Ok((Some(loop_id), body))
    }

    fn next_loop_id(&mut self) -> usize {
        let id = self.next_loop_id;
        self.next_loop_id += 1;
        id
    }
}
