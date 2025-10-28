use std::fmt;

use crate::parse::{
    Decl, DeclKind, ExprKind, ForInit, FunctionDecl, MemberDeclaration, Program, Stmt, StmtKind,
    StorageClass, StructDeclaration, VariableDecl,
};

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for decl in &self.0 {
            write!(f, "{}", decl)?;
        }
        Ok(())
    }
}

impl fmt::Display for Decl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            DeclKind::Function(function_decl) => writeln!(f, "{}", function_decl),
            DeclKind::Variable(variable_decl) => writeln!(f, "{}", variable_decl),
            DeclKind::Struct(struct_decl) => writeln!(f, "{}", struct_decl),
        }
    }
}

impl fmt::Display for StructDeclaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "struct {} {{", self.tag)?;
        for member in &self.members {
            writeln!(f, "    {} {};", member.member_type, member.member_name)?;
        }
        write!(f, "}};")
    }
}

impl fmt::Display for MemberDeclaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.member_type, self.member_name)
    }
}

impl fmt::Display for FunctionDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(storage_class) = &self.storage_class {
            write!(f, "{} ", storage_class)?;
        }

        write!(f, "{} {}(", self.return_type, self.name)?;

        for (idx, param) in self.params.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }

            if param.name.is_empty() {
                write!(f, "{}", param.r#type)?;
            } else {
                write!(f, "{} {}", param.r#type, param.name)?;
            }
        }

        if self.params.is_empty() {
            write!(f, "void")?;
        }

        write!(f, ")")?;

        if let Some(body) = &self.body {
            let body_str = stmts_to_str(body);
            write!(f, " {body_str}")?;
        } else {
            write!(f, ";")?;
        }

        Ok(())
    }
}

impl fmt::Display for VariableDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(storage_class) = &self.storage_class {
            write!(f, "{} ", storage_class)?;
        }

        write!(f, "{} {}", self.r#type, self.name)?;

        if let Some(init) = &self.init {
            write!(f, " = {}", init.kind)?;
        }

        write!(f, ";")
    }
}

// ExprKinds are parenthesized because Neg(Neg(50)) (--50) is not considered legal. It is ok though
// as -(-50) or (-(-(50))) or something similar.
impl fmt::Display for ExprKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;

        f.write_str(&match self {
            ExprKind::Constant(c) => c.to_string(),
            ExprKind::String(s) => format!("\"{s}\""),
            ExprKind::Var(var) => var.clone(),
            ExprKind::FunctionCall(name, exprs) => {
                let mut res = String::from(name);
                res.push('(');

                for expr in exprs {
                    res.push_str(&expr.kind.to_string());
                    res.push(',');
                }
                if res.ends_with(',') {
                    res.pop();
                }
                res.push(')');

                res
            }
            ExprKind::Cast(t, expr) => format!("({t}){}", expr.kind),
            ExprKind::Neg(expr) => format!("-{}", expr.kind),
            ExprKind::BitNot(expr) => format!("~{}", expr.kind),
            ExprKind::Conditional(cond, lhs, rhs) => {
                format!("{} ? {} : {}", cond.kind, lhs.kind, rhs.kind)
            }
            ExprKind::Add(lhs, rhs) => {
                format!("{} + {}", lhs.kind, rhs.kind)
            }
            ExprKind::Sub(lhs, rhs) => {
                format!("{} - {}", lhs.kind, rhs.kind)
            }
            ExprKind::Mul(lhs, rhs) => {
                format!("{} * {}", lhs.kind, rhs.kind)
            }
            ExprKind::Div(lhs, rhs) => {
                format!("{} / {}", lhs.kind, rhs.kind)
            }
            ExprKind::Rem(lhs, rhs) => {
                format!("{} % {}", lhs.kind, rhs.kind)
            }
            ExprKind::Equal(lhs, rhs) => {
                format!("{} == {}", lhs.kind, rhs.kind)
            }
            ExprKind::NotEqual(lhs, rhs) => {
                format!("{} != {}", lhs.kind, rhs.kind)
            }
            ExprKind::LessThan(lhs, rhs) => {
                format!("{} < {}", lhs.kind, rhs.kind)
            }
            ExprKind::LessThanEqual(lhs, rhs) => {
                format!("{} <= {}", lhs.kind, rhs.kind)
            }
            ExprKind::GreaterThan(lhs, rhs) => {
                format!("{} > {}", lhs.kind, rhs.kind)
            }
            ExprKind::GreaterThanEqual(lhs, rhs) => {
                format!("{} >= {}", lhs.kind, rhs.kind)
            }
            ExprKind::Or(lhs, rhs) => {
                format!("{} || {}", lhs.kind, rhs.kind)
            }
            ExprKind::And(lhs, rhs) => {
                format!("{} && {}", lhs.kind, rhs.kind)
            }
            ExprKind::Not(expr) => format!("!{}", expr.kind),
            ExprKind::AddrOf(expr) => format!("&{}", expr.kind),
            ExprKind::Dereference(expr) => format!("*{}", expr.kind),
            ExprKind::Member {
                base,
                member,
                offset: _,
                is_arrow,
            } => {
                if *is_arrow {
                    format!("{}->{}", base.kind, member)
                } else {
                    format!("{}.{}", base.kind, member)
                }
            }
            ExprKind::BitAnd(lhs, rhs) => {
                format!("{} & {}", lhs.kind, rhs.kind)
            }
            ExprKind::Xor(lhs, rhs) => {
                format!("{} ^ {}", lhs.kind, rhs.kind)
            }
            ExprKind::BitOr(lhs, rhs) => {
                format!("{} | {}", lhs.kind, rhs.kind)
            }
            ExprKind::PreIncrement(expr) => format!("++{}", expr.kind),
            ExprKind::PreDecrement(expr) => format!("--{}", expr.kind),
            ExprKind::PostIncrement(expr) => format!("{}++", expr.kind),
            ExprKind::PostDecrement(expr) => format!("{}--", expr.kind),
            ExprKind::LeftShift(lhs, rhs) => {
                format!("{} << {}", lhs.kind, rhs.kind)
            }
            ExprKind::RightShift(lhs, rhs) => {
                format!("{} >> {}", lhs.kind, rhs.kind)
            }
            ExprKind::Assignment(lhs, rhs) => {
                format!("{} = {}", lhs.kind, rhs.kind)
            }
            ExprKind::SizeOf(node) => format!("sizeof({})", node.kind),
            ExprKind::SizeOfType(t) => format!("sizeof({})", t),
        })?;

        write!(f, ")")
    }
}

impl fmt::Display for StorageClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            StorageClass::Static => "static",
            StorageClass::Extern => "extern",
        })
    }
}

fn stmts_to_str(stmts: &[Stmt]) -> String {
    let mut s = String::from("{\n");

    for stmt in stmts {
        s.push_str(&stmt.kind.to_string());
        s.push('\n');
    }

    s.push('}');
    s
}

impl fmt::Display for StmtKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&match self {
            StmtKind::Expr(node) => format!("{};", &node.kind),
            StmtKind::Return(node) => format!("return {};", node.kind),
            StmtKind::Compound(stmts) => stmts_to_str(stmts),
            StmtKind::Declaration(variable_decl) => {
                let VariableDecl {
                    name,
                    init,
                    storage_class,
                    r#type,
                    ..
                } = variable_decl;
                let mut s = String::new();
                if let Some(storage_class) = storage_class {
                    s.push_str(&storage_class.to_string());
                    s.push(' ');
                }
                s.push_str(&r#type.to_string());
                s.push(' ');
                s.push_str(name);
                if let Some(expr) = init {
                    s.push_str(" = ");
                    s.push_str(&expr.kind.to_string());
                }
                s.push(';');
                s
            }
            StmtKind::Null => "".to_string(),
            StmtKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let mut s = String::from("if (");
                s.push_str(&condition.kind.to_string());
                s.push_str(") {\n");
                s.push_str(&then_branch.kind.to_string());
                if let Some(else_branch) = else_branch {
                    s.push_str("\n}");
                    s.push_str("else {\n");
                    s.push_str(&else_branch.kind.to_string());
                }
                s.push_str("\n}");
                s
            }
            StmtKind::While {
                condition, body, ..
            } => {
                let mut s = String::from("while (");
                s.push_str(&condition.kind.to_string());
                s.push_str(") {\n");
                s.push_str(&body.kind.to_string());
                s.push_str("\n}");
                s
            }
            StmtKind::DoWhile {
                body, condition, ..
            } => {
                let mut s = String::from("do {\n");
                s.push_str(&body.kind.to_string());
                s.push_str("\n} while (");
                s.push_str(&condition.kind.to_string());
                s.push(')');
                s
            }
            StmtKind::For {
                init,
                condition,
                post,
                body,
                ..
            } => {
                let mut s = String::from("for (");

                let init = match init {
                    ForInit::Declaration(node) => node.kind.to_string(),
                    ForInit::Expr(node) => {
                        if let Some(e) = node {
                            e.kind.to_string()
                        } else {
                            "".to_string()
                        }
                    }
                };

                s.push_str(&init);
                s.push(';');

                if let Some(cond) = condition {
                    s.push_str(&cond.kind.to_string());
                }
                s.push(';');
                if let Some(post) = post {
                    s.push_str(&post.kind.to_string());
                }

                s.push_str(") {\n");
                s.push_str(&body.kind.to_string());
                s.push_str("\n}");

                s
            }
            StmtKind::Break { .. } => "break".to_string(),
            StmtKind::Continue { .. } => "continue".to_string(),
        })
    }
}
