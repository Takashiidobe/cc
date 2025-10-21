// this module is for creating programs that conform to the parser's rules and which can be lowered
// to a textual output.

use std::{cmp::max, fmt};

use crate::parse::{
    Const, Decl, DeclKind, Expr, ExprKind, ForInit, FunctionDecl, Program, Stmt, StmtKind,
    StorageClass, Type, VariableDecl,
};
use quickcheck::{empty_shrinker, Arbitrary, Gen};

pub(crate) fn generate() -> String {
    let mut qc_gen = Gen::new(16);
    let program = Program::arbitrary(&mut qc_gen);
    program.to_string()
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for decl in &self.0 {
            write!(f, "{}", decl)?;
        }
        Ok(())
    }
}

impl Arbitrary for Program {
    fn arbitrary(g: &mut Gen) -> Self {
        let max_items = max(g.size(), 1);
        let count = usize::arbitrary(g) % max_items + 1;
        let mut decls = Vec::with_capacity(count);
        for idx in 0..count {
            decls.push(random_variable_decl(g, idx));
        }
        Program(decls)
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        empty_shrinker()
    }
}

fn random_variable_decl(g: &mut Gen, index: usize) -> Decl {
    let name = format!("{}_{}", random_identifier(g), index);
    let r#type = random_type(g);
    let storage_class = random_storage_class(g);
    let is_extern = matches!(storage_class, Some(StorageClass::Extern));
    let allow_initializer = !is_extern;
    let init = if allow_initializer && bool::arbitrary(g) {
        Some(random_initializer(&r#type, g))
    } else {
        None
    };
    let variable_decl = VariableDecl {
        name,
        init,
        storage_class: storage_class.clone(),
        r#type: r#type.clone(),
        is_definition: !is_extern,
    };
    let source = variable_decl.to_string();
    Decl {
        kind: DeclKind::Variable(variable_decl),
        start: 0,
        end: source.len(),
        source,
    }
}

fn random_identifier(g: &mut Gen) -> String {
    let len = usize::arbitrary(g) % 6 + 1;
    let mut ident = String::with_capacity(len);
    ident.push(random_letter(g));
    for _ in 1..len {
        ident.push(random_alphanumeric(g));
    }
    ident
}

fn random_letter(g: &mut Gen) -> char {
    (b'a' + (u8::arbitrary(g) % 26)) as char
}

fn random_alphanumeric(g: &mut Gen) -> char {
    match u8::arbitrary(g) % 36 {
        n if n < 26 => (b'a' + n) as char,
        n => (b'0' + (n - 26)) as char,
    }
}

fn random_type(g: &mut Gen) -> Type {
    match u8::arbitrary(g) % 3 {
        0 => Type::Int,
        1 => Type::UInt,
        _ => Type::Long,
    }
}

fn random_storage_class(g: &mut Gen) -> Option<StorageClass> {
    match u8::arbitrary(g) % 3 {
        0 => None,
        1 => Some(StorageClass::Static),
        _ => Some(StorageClass::Extern),
    }
}

fn random_initializer(r#type: &Type, g: &mut Gen) -> Expr {
    match r#type {
        Type::Int => constant_expr(Const::Int(i32::arbitrary(g)), Type::Int),
        Type::UInt => constant_expr(Const::UInt(u32::arbitrary(g)), Type::UInt),
        Type::Long => constant_expr(Const::Long(i64::arbitrary(g)), Type::Long),
        Type::Char | Type::SChar => constant_expr(Const::Char(i8::arbitrary(g)), Type::Char),
        Type::UChar => constant_expr(Const::UChar(u8::arbitrary(g)), Type::UChar),
        Type::Short => constant_expr(Const::Short(i16::arbitrary(g)), Type::Short),
        Type::UShort => constant_expr(Const::UShort(u16::arbitrary(g)), Type::UShort),
        _ => constant_expr(Const::Int(i32::arbitrary(g)), Type::Int),
    }
}

fn constant_expr(constant: Const, r#type: Type) -> Expr {
    Expr {
        kind: ExprKind::Constant(constant),
        start: 0,
        end: 0,
        source: String::new(),
        r#type,
    }
}

impl fmt::Display for Decl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            DeclKind::Function(function_decl) => writeln!(f, "{}", function_decl),
            DeclKind::Variable(variable_decl) => writeln!(f, "{}", variable_decl),
        }
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

impl fmt::Display for ExprKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&match self {
            ExprKind::Constant(c) => c.to_string(),
            ExprKind::String(s) => s.clone(),
            ExprKind::Var(var) => var.clone(),
            ExprKind::FunctionCall(name, exprs) => {
                let mut res = String::from(name);
                res.push('(');

                for expr in exprs {
                    res.push_str(&expr.kind.to_string());
                    res.push(',');
                }
                if !res.ends_with('(') {
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
        })
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
    let mut s = String::from("{");

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
                s.push(';');

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
