use crate::parse::{Decl, DeclKind, ForInit, FunctionDecl, Program, Stmt, StmtKind};

#[derive(Default)]
pub struct LoopLabeler {
    next_loop_id: usize,
    loop_stack: Vec<usize>,
}

impl LoopLabeler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn label_program(mut self, program: Program) -> Program {
        let decls = program
            .0
            .into_iter()
            .map(|decl| self.label_decl(decl))
            .collect();
        Program(decls)
    }

    fn label_decl(&mut self, decl: Decl) -> Decl {
        let Decl {
            kind,
            start,
            end,
            source,
        } = decl;

        let kind = match kind {
            DeclKind::Function(func) => DeclKind::Function(self.label_function(func)),
            DeclKind::Variable(var) => DeclKind::Variable(var),
        };

        Decl {
            kind,
            start,
            end,
            source,
        }
    }

    fn label_function(&mut self, decl: FunctionDecl) -> FunctionDecl {
        let FunctionDecl {
            name,
            params,
            body,
            storage_class,
            return_type,
        } = decl;

        let body = body.map(|stmts| {
            stmts
                .into_iter()
                .map(|stmt| self.label_stmt(stmt))
                .collect()
        });

        FunctionDecl {
            name,
            params,
            body,
            storage_class,
            return_type,
        }
    }

    fn label_stmt(&mut self, stmt: Stmt) -> Stmt {
        let Stmt {
            kind,
            start,
            end,
            source,
            r#type,
        } = stmt;

        let kind = match kind {
            StmtKind::Return(expr) => StmtKind::Return(expr),
            StmtKind::Expr(expr) => StmtKind::Expr(expr),
            StmtKind::Compound(stmts) => StmtKind::Compound(
                stmts
                    .into_iter()
                    .map(|stmt| self.label_stmt(stmt))
                    .collect(),
            ),
            StmtKind::Declaration(decl) => StmtKind::Declaration(decl),
            StmtKind::Null => StmtKind::Null,
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => StmtKind::If {
                condition,
                then_branch: Box::new(self.label_stmt(*then_branch)),
                else_branch: else_branch.map(|stmt| Box::new(self.label_stmt(*stmt))),
            },
            StmtKind::While {
                condition,
                body,
                loop_id: _,
            } => {
                let loop_id = self.next_loop_id();
                self.loop_stack.push(loop_id);
                let body = Box::new(self.label_stmt(*body));
                self.loop_stack.pop();
                StmtKind::While {
                    condition,
                    body,
                    loop_id: Some(loop_id),
                }
            }
            StmtKind::DoWhile {
                body,
                condition,
                loop_id: _,
            } => {
                let loop_id = self.next_loop_id();
                self.loop_stack.push(loop_id);
                let body = Box::new(self.label_stmt(*body));
                self.loop_stack.pop();
                StmtKind::DoWhile {
                    body,
                    condition,
                    loop_id: Some(loop_id),
                }
            }
            StmtKind::For {
                init,
                condition,
                post,
                body,
                loop_id: _,
            } => {
                let init = self.label_for_init(init);
                let loop_id = self.next_loop_id();
                self.loop_stack.push(loop_id);
                let body = Box::new(self.label_stmt(*body));
                self.loop_stack.pop();
                StmtKind::For {
                    init,
                    condition,
                    post,
                    body,
                    loop_id: Some(loop_id),
                }
            }
            StmtKind::Break { loop_id: _ } => {
                let loop_id = *self
                    .loop_stack
                    .last()
                    .expect("break statement not within a loop");
                StmtKind::Break {
                    loop_id: Some(loop_id),
                }
            }
            StmtKind::Continue { loop_id: _ } => {
                let loop_id = *self
                    .loop_stack
                    .last()
                    .expect("continue statement not within a loop");
                StmtKind::Continue {
                    loop_id: Some(loop_id),
                }
            }
        };

        Stmt {
            kind,
            start,
            end,
            source,
            r#type,
        }
    }

    fn label_for_init(&mut self, init: ForInit) -> ForInit {
        match init {
            ForInit::Declaration(decl) => ForInit::Declaration(Box::new(self.label_stmt(*decl))),
            ForInit::Expr(expr) => ForInit::Expr(expr),
        }
    }

    fn next_loop_id(&mut self) -> usize {
        let id = self.next_loop_id;
        self.next_loop_id += 1;
        id
    }
}
