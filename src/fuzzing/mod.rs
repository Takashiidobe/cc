// this module is for creating programs that conform to the parser's rules and which can be lowered
// to a textual output.

pub(crate) mod display;

use crate::parse::{
    Const, Decl, DeclKind, Expr, ExprKind, FunctionDecl, Program, Stmt, StmtKind, Type,
};
use quickcheck::{Arbitrary, Gen, empty_shrinker};

// find a way to use variables by keeping variables in a given scope
// if you "create" a function/decl, you can then refer to it as you iterate through with scopes.
// First create a program that returns int main(void) { return number; }
// Second, create a program that returns int main(void) { return -number | +number or ~number };

pub(crate) fn generate() -> String {
    let mut qc_gen = Gen::new(16);
    let program = Program::arbitrary(&mut qc_gen);
    program.to_string()
}

impl Arbitrary for Program {
    fn arbitrary(g: &mut Gen) -> Self {
        Program(vec![fn_decl(gen_main(g))])
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        empty_shrinker()
    }
}

fn gen_main(g: &mut Gen) -> FunctionDecl {
    let stmt = stmt(StmtKind::Return(gen_expr(g, Type::Int)), Type::Int);

    FunctionDecl {
        name: String::from("main"),
        params: vec![],
        body: Some(vec![stmt]),
        storage_class: None,
        return_type: Type::Int,
    }
}

fn stmt(kind: StmtKind, r#type: Type) -> Stmt {
    Stmt {
        kind,
        start: 0,
        end: 0,
        source: "".to_string(),
        r#type,
    }
}

fn fn_decl(function: FunctionDecl) -> Decl {
    Decl {
        kind: DeclKind::Function(function),
        start: 0,
        end: 0,
        source: "".to_string(),
    }
}

fn gen_expr(g: &mut Gen, r#type: Type) -> Expr {
    match bool::arbitrary(g) {
        true => constant_expr(g, r#type),
        false => gen_unary(g, r#type),
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
        start: 0,
        end: 0,
        source: String::new(),
        r#type,
    }
}

// now we want to either gen a unary or a binary expr or a const
fn gen_unary(g: &mut Gen, r#type: Type) -> Expr {
    let expr_kind = gen_expr(g, r#type.clone());

    Expr {
        kind: rand_unary_kind(g, expr_kind),
        start: 0,
        end: 0,
        source: String::new(),
        r#type,
    }
}

fn gen_binary(g: &mut Gen) -> Expr {
    todo!()
}

fn rand_unary_kind(g: &mut Gen, expr: Expr) -> ExprKind {
    match bool::arbitrary(g) {
        true => ExprKind::BitNot(Box::new(expr)),
        false => ExprKind::Neg(Box::new(expr)),
    }
}
