// this module is for creating programs that conform to the parser's rules and which can be lowered
// to a textual output.

pub(crate) mod display;

use crate::parse::{
    Const, Decl, DeclKind, Expr, ExprKind, FunctionDecl, Program, Stmt, StmtKind, StorageClass,
    Type, VariableDecl,
};
use quickcheck::{Arbitrary, Gen, empty_shrinker};

// find a way to use variables by keeping variables in a given scope
// if you "create" a function/decl, you can then refer to it as you iterate through with scopes.
// First create a program that returns int main(void) { return number; }

pub(crate) fn generate() -> String {
    let mut qc_gen = Gen::new(16);
    let program = Program::arbitrary(&mut qc_gen);
    program.to_string()
}

impl Arbitrary for Program {
    fn arbitrary(g: &mut Gen) -> Self {
        Program(vec![Decl {
            kind: DeclKind::Function(gen_main(g)),
            start: 0,
            end: 0,
            source: "".to_string(),
        }])
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        empty_shrinker()
    }
}

fn gen_main(g: &mut Gen) -> FunctionDecl {
    let name = "main";

    let stmt = Stmt {
        kind: StmtKind::Return(constant_expr(Const::Char(i8::arbitrary(g)), Type::Int)),
        start: 0,
        end: 0,
        source: "".to_string(),
        r#type: Type::Int,
    };

    FunctionDecl {
        name: name.to_string(),
        params: vec![],
        body: Some(vec![stmt]),
        storage_class: None,
        return_type: Type::Int,
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
