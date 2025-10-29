// this module is for creating programs that conform to the parser's rules and which can be lowered
// to a textual output.

pub(crate) mod display;

use crate::parse::{
    Const, Decl, DeclKind, Expr, ExprKind, ForInit, FunctionDecl, Program, SourceLocation, Stmt,
    StmtKind, Type,
};
use quickcheck::{Arbitrary, Gen, empty_shrinker};

// find a way to use variables by keeping variables in a given scope
// if you "create" a function/decl, you can then refer to it as you iterate through with scopes.

pub(crate) fn generate() -> String {
    let mut qc_gen = Gen::new(16);
    let program = Program::arbitrary(&mut qc_gen);
    program.to_string()
}

impl Arbitrary for Program {
    fn arbitrary(g: &mut Gen) -> Self {
        Program(vec![gen_fn(g), gen_main(g)])
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        empty_shrinker()
    }
}

fn gen_fn(g: &mut Gen) -> Decl {
    fn_decl(FunctionDecl {
        name: format!("example_{}", u8::arbitrary(g)),
        params: vec![],
        body: Some(gen_stmts(g, 8)),
        storage_class: None,
        return_type: Type::Int,
    })
}

fn gen_main(g: &mut Gen) -> Decl {
    fn_decl(FunctionDecl {
        name: String::from("main"),
        params: vec![],
        body: Some(gen_stmts(g, 8)),
        storage_class: None,
        return_type: Type::Int,
    })
}

fn gen_stmts(g: &mut Gen, times: usize) -> Vec<Stmt> {
    let mut stmts = vec![];
    for _ in 0..times {
        stmts.push(gen_stmt(g, Type::Int));
    }

    stmts
}

fn fn_decl(function: FunctionDecl) -> Decl {
    Decl {
        kind: DeclKind::Function(function),
        loc: SourceLocation::default(),
    }
}

fn gen_stmt(g: &mut Gen, r#type: Type) -> Stmt {
    Stmt {
        kind: rand_stmt_kind(g, r#type.clone()),
        loc: SourceLocation::default(),
        r#type,
    }
}

fn gen_expr(g: &mut Gen, r#type: Type) -> Expr {
    match u8::arbitrary(g) % 4 {
        0..2 => constant_expr(g, r#type),
        2 => gen_unary(g, r#type),
        // 2 => gen_incr_decr(g, r#type), // this should only be for variables
        _ => gen_binary(g, r#type),
    }
}

fn constant_expr(g: &mut Gen, r#type: Type) -> Expr {
    let constant = match r#type {
        Type::Char | Type::SChar => Const::Char(i8::arbitrary(g)),
        Type::UChar => Const::UChar(u8::arbitrary(g)),
        Type::Short => Const::Short(i16::arbitrary(g)),
        Type::UShort => Const::UShort(u16::arbitrary(g)),
        Type::Int => Const::Int(i32::arbitrary(g)),
        Type::Long => Const::Long(i64::arbitrary(g)),
        Type::UInt => Const::UInt(u32::arbitrary(g)),
        Type::ULong => Const::ULong(u64::arbitrary(g)),
        Type::Double => Const::Double(f64::arbitrary(g)),
        _ => todo!(),
    };

    Expr {
        kind: ExprKind::Constant(constant),
        loc: SourceLocation::default(),
        r#type,
    }
}

// now we want to either gen a unary or a binary expr or a const
fn gen_unary(g: &mut Gen, r#type: Type) -> Expr {
    let expr_kind = gen_expr(g, r#type.clone());

    Expr {
        kind: rand_unary_kind(g, expr_kind),
        loc: SourceLocation::default(),
        r#type,
    }
}

fn gen_binary(g: &mut Gen, r#type: Type) -> Expr {
    let lhs = gen_expr(g, r#type.clone());
    let rhs = gen_expr(g, r#type.clone());

    Expr {
        kind: rand_binary_kind(g, lhs, rhs),
        loc: SourceLocation::default(),
        r#type,
    }
}

// only for assignable values
#[allow(unused)]
fn gen_incr_decr(g: &mut Gen, r#type: Type) -> Expr {
    let expr = gen_expr(g, r#type.clone());

    Expr {
        kind: rand_incr_decr_kind(g, expr),
        loc: SourceLocation::default(),
        r#type,
    }
}

fn rand_stmt_kind(g: &mut Gen, r#type: Type) -> StmtKind {
    match u8::arbitrary(g) % 17 {
        0..8 => StmtKind::Expr(gen_expr(g, r#type)),
        8..13 => StmtKind::Return(gen_expr(g, r#type)),
        13..15 => StmtKind::Compound(gen_stmts(g, 2)),
        15 => StmtKind::If {
            condition: gen_expr(g, r#type.clone()),
            then_branch: Box::new(gen_stmt(g, r#type.clone())),
            else_branch: Some(Box::new(gen_stmt(g, r#type.clone()))),
        },
        _ => StmtKind::For {
            init: ForInit::Expr(Some(gen_expr(g, r#type.clone()))),
            condition: Some(gen_expr(g, r#type.clone())),
            post: Some(gen_expr(g, r#type.clone())),
            body: Box::new(gen_stmt(g, r#type)),
            loop_id: None,
        },
    }
}

fn rand_binary_kind(g: &mut Gen, lhs: Expr, rhs: Expr) -> ExprKind {
    match u8::arbitrary(g) % 18 {
        0 => ExprKind::Add(Box::new(lhs), Box::new(rhs)),
        1 => ExprKind::Sub(Box::new(lhs), Box::new(rhs)),
        2 => ExprKind::Mul(Box::new(lhs), Box::new(rhs)),
        3 => ExprKind::Div(Box::new(lhs), Box::new(rhs)),
        4 => ExprKind::Xor(Box::new(lhs), Box::new(rhs)),
        5 => ExprKind::BitAnd(Box::new(lhs), Box::new(rhs)),
        6 => ExprKind::BitOr(Box::new(lhs), Box::new(rhs)),
        7 => ExprKind::Or(Box::new(lhs), Box::new(rhs)),
        8 => ExprKind::Rem(Box::new(lhs), Box::new(rhs)),
        9 => ExprKind::LeftShift(Box::new(lhs), Box::new(rhs)),
        10 => ExprKind::RightShift(Box::new(lhs), Box::new(rhs)),
        11 => ExprKind::GreaterThan(Box::new(lhs), Box::new(rhs)),
        12 => ExprKind::GreaterThanEqual(Box::new(lhs), Box::new(rhs)),
        13 => ExprKind::LessThan(Box::new(lhs), Box::new(rhs)),
        14 => ExprKind::LessThanEqual(Box::new(lhs), Box::new(rhs)),
        15 => ExprKind::Equal(Box::new(lhs), Box::new(rhs)),
        16 => ExprKind::NotEqual(Box::new(lhs), Box::new(rhs)),
        _ => ExprKind::And(Box::new(lhs), Box::new(rhs)),
    }
}

fn rand_unary_kind(g: &mut Gen, expr: Expr) -> ExprKind {
    match u8::arbitrary(g) % 3 {
        0 => ExprKind::BitNot(Box::new(expr)),
        1 => ExprKind::Neg(Box::new(expr)),
        _ => ExprKind::Not(Box::new(expr)),
    }
}

#[allow(unused)]
fn rand_incr_decr_kind(g: &mut Gen, expr: Expr) -> ExprKind {
    match u8::arbitrary(g) % 4 {
        0 => ExprKind::PreIncrement(Box::new(expr)),
        1 => ExprKind::PreDecrement(Box::new(expr)),
        2 => ExprKind::PostIncrement(Box::new(expr)),
        _ => ExprKind::PostDecrement(Box::new(expr)),
    }
}
