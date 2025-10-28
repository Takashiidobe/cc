use thiserror::Error;

use crate::tokenize::{Token, TokenKind};
use std::convert::TryFrom;
use std::{fmt, mem};

pub(crate) type Expr = Node<ExprKind>;
pub(crate) type Stmt = Node<StmtKind>;

#[derive(Error, Debug, Clone)]
pub(crate) enum ParserError {
    #[error("unexpected end of input{0}")]
    UnexpectedEof(&'static str),

    #[error("expected token {0:?}, found {1:?}")]
    ExpectedToken(TokenKind, Option<TokenKind>),

    #[error("not at end of program (next token: {0:?})")]
    NotAtEnd(Option<TokenKind>),

    #[error("storage class specifiers are not allowed here")]
    StorageNotAllowedHere,

    #[error("multiple storage class specifiers in declaration")]
    MultipleStorageClasses,

    #[error("unsupported type specifier {0:?}")]
    UnsupportedTypeSpecifier(TokenKind),

    #[error("duplicate type specifier '{0}'")]
    DuplicateTypeSpecifier(&'static str),

    #[error("conflicting type specifiers '{0}' and '{1}'")]
    ConflictingTypeSpecifiers(&'static str, &'static str),

    #[error("'void' cannot be combined with other type specifiers")]
    VoidCannotCombine,

    #[error("'double' cannot be combined with other type specifiers")]
    DoubleCannotCombine,

    #[error("declaration missing type specifier")]
    MissingTypeSpecifier,

    #[error("expected declaration specifiers")]
    ExpectedDeclSpecifiers,

    #[error("array size must be non-negative")]
    NegativeArraySize,

    #[error("array size does not fit in usize ({0})")]
    ArraySizeTooLarge(i128),

    #[error("expected constant array size, found {0:?}")]
    ExpectedConstArraySize(TokenKind),

    #[error("'void' parameter must be the only parameter")]
    VoidOnlyParameter,

    #[error("variable declared with void type")]
    VariableWithVoidType,

    #[error("function declarations are not allowed in block scope")]
    FunctionDeclInBlockScope,

    #[error("variable declared with function type")]
    VariableWithFunctionType,

    #[error("invalid function call target: {0:?}")]
    InvalidFunctionCallTarget(ExprKind),

    #[error("unsupported compound assignment token: {0:?}")]
    UnsupportedCompoundAssign(TokenKind),

    #[error("unsupported cast target: void")]
    UnsupportedCastTargetVoid,

    #[error("expected identifier or '(' in declarator, found {0:?}")]
    ExpectedIdentOrParen(TokenKind),

    #[error("Expected primary expression, found {0:?}")]
    ExpectedPrimary(TokenKind),

    #[error("expected identifier")]
    ExpectedIdentifier,
}

type PResult<T> = Result<T, ParserError>;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Program(pub(crate) Vec<Decl>);

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StorageClass {
    Static,
    Extern,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExprKind {
    Constant(Const),
    String(String),
    Var(String),
    FunctionCall(String, Vec<Expr>),
    Cast(Type, Box<Expr>),
    Neg(Box<Expr>),
    BitNot(Box<Expr>),
    Conditional(Box<Expr>, Box<Expr>, Box<Expr>),

    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Rem(Box<Expr>, Box<Expr>),

    Equal(Box<Expr>, Box<Expr>),
    NotEqual(Box<Expr>, Box<Expr>),
    LessThan(Box<Expr>, Box<Expr>),
    LessThanEqual(Box<Expr>, Box<Expr>),
    GreaterThan(Box<Expr>, Box<Expr>),
    GreaterThanEqual(Box<Expr>, Box<Expr>),

    Or(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),

    AddrOf(Box<Expr>),
    Dereference(Box<Expr>),

    BitAnd(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),
    BitOr(Box<Expr>, Box<Expr>),

    SizeOf(Box<Expr>),
    SizeOfType(Type),

    PreIncrement(Box<Expr>),
    PreDecrement(Box<Expr>),
    PostIncrement(Box<Expr>),
    PostDecrement(Box<Expr>),

    LeftShift(Box<Expr>, Box<Expr>),
    RightShift(Box<Expr>, Box<Expr>),
    Assignment(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VariableDecl {
    pub(crate) name: String,
    pub(crate) init: Option<Expr>,
    pub(crate) storage_class: Option<StorageClass>,
    pub(crate) r#type: Type,
    pub(crate) is_definition: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParameterDecl {
    pub(crate) name: String,
    pub(crate) r#type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FunctionDecl {
    pub(crate) name: String,
    pub(crate) params: Vec<ParameterDecl>,
    pub(crate) body: Option<Vec<Stmt>>,
    pub(crate) storage_class: Option<StorageClass>,
    pub(crate) return_type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StructDeclaration {
    pub(crate) tag: String,
    pub(crate) members: Vec<MemberDeclaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MemberDeclaration {
    pub(crate) member_name: String,
    pub(crate) member_type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DeclKind {
    Function(FunctionDecl),
    Variable(VariableDecl),
    Struct(StructDeclaration),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Decl {
    pub(crate) kind: DeclKind,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) source: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum StmtKind {
    Expr(Expr),
    Return(Expr),
    Compound(Vec<Stmt>),
    Declaration(VariableDecl),
    Null,
    If {
        condition: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    While {
        condition: Expr,
        body: Box<Stmt>,
        loop_id: Option<usize>,
    },
    DoWhile {
        body: Box<Stmt>,
        condition: Expr,
        loop_id: Option<usize>,
    },
    For {
        init: ForInit,
        condition: Option<Expr>,
        post: Option<Expr>,
        body: Box<Stmt>,
        loop_id: Option<usize>,
    },
    Break {
        loop_id: Option<usize>,
    },
    Continue {
        loop_id: Option<usize>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ForInit {
    Declaration(Box<Stmt>),
    Expr(Option<Expr>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Width {
    W8,
    W16,
    W32,
    W64,
}

impl Width {
    pub(crate) fn suffix(&self) -> char {
        match self {
            Width::W8 => 'b',
            Width::W16 => 'w',
            Width::W32 => 'l',
            Width::W64 => 'q',
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Type {
    Char,
    SChar,
    UChar,
    Short,
    UShort,
    Int,
    Long,
    UInt,
    ULong,
    Double,
    Void,
    Struct(String),
    Pointer(Box<Type>),
    Array(Box<Type>, usize),
    IncompleteArray(Box<Type>),
    FunType(Vec<Type>, Box<Type>),
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Char | Type::SChar => f.write_str("char"),
            Type::UChar => f.write_str("unsigned char"),
            Type::Short => f.write_str("short"),
            Type::UShort => f.write_str("unsigned short"),
            Type::Int => f.write_str("int"),
            Type::Long => f.write_str("long"),
            Type::UInt => f.write_str("unsigned int"),
            Type::ULong => f.write_str("unsigned long"),
            Type::Double => f.write_str("double"),
            Type::Void => f.write_str("void"),
            Type::Struct(tag) => f.write_str(&format!("struct {tag}")),
            Type::Pointer(t) => f.write_str(&format!("*{t}")),
            Type::Array(t, len) => f.write_str(&format!("{t}[{len}]")),
            Type::IncompleteArray(t) => f.write_str(&format!("{t}[]")),
            Type::FunType(_, _) => f.write_str("function"),
        }
    }
}

impl Type {
    pub(crate) fn type_rank(&self) -> usize {
        match self {
            Type::Char | Type::SChar => 0,
            Type::UChar => 1,
            Type::Short => 2,
            Type::UShort => 3,
            Type::Int => 4,
            Type::UInt => 5,
            Type::Long => 6,
            Type::ULong => 7,
            Type::Void
            | Type::Struct(_)
            | Type::Double
            | Type::Pointer(_)
            | Type::FunType(_, _)
            | Type::Array(_, _)
            | Type::IncompleteArray(_) => panic!("Cannot be compared"),
        }
    }
    pub(crate) fn is_void(&self) -> bool {
        matches!(self, Type::Void)
    }

    pub(crate) fn is_unsigned(&self) -> bool {
        matches!(self, Type::UChar | Type::UInt | Type::ULong)
    }

    pub(crate) fn is_signed(&self) -> bool {
        self.is_integer() && !self.is_unsigned()
    }

    pub(crate) fn is_pointer(&self) -> bool {
        matches!(self, Type::Pointer(_))
    }

    pub(crate) fn is_integer(&self) -> bool {
        matches!(
            self,
            Type::Char
                | Type::SChar
                | Type::UChar
                | Type::Short
                | Type::UShort
                | Type::Int
                | Type::UInt
                | Type::Long
                | Type::ULong
        )
    }

    pub(crate) fn width(&self) -> Width {
        use Type::*;
        match self {
            SChar | Char | UChar => Width::W8,
            Short | UShort => Width::W16,
            Int | UInt => Width::W32,
            Long | ULong => Width::W64,
            Double => todo!(),
            Void | Struct(_) | Pointer(_) | Array(_, _) | IncompleteArray(_) | FunType(_, _) => {
                todo!("invalid type size")
            }
        }
    }

    pub(crate) fn bit_width(&self) -> usize {
        use Type::*;
        match self {
            SChar | Char | UChar => 8,
            Short | UShort => 16,
            Int | UInt => 32,
            Long | ULong | Double => 64,
            Void => 0,
            Struct(_) | Pointer(_) | Array(_, _) | IncompleteArray(_) | FunType(_, _) => {
                todo!("invalid type size")
            }
        }
    }

    pub(crate) fn byte_size(&self) -> usize {
        match self {
            Type::Char | Type::SChar | Type::UChar => 1,
            Type::Short | Type::UShort => 2,
            Type::Int | Type::UInt => 4,
            Type::Long | Type::ULong => 8,
            Type::Double => 8,
            Type::Pointer(_) => 8,
            Type::Array(inner, len) => len * inner.byte_size(),
            Type::IncompleteArray(t) => t.byte_size(),
            Type::Void => 1,
            Type::Struct(_) => todo!("struct size not implemented"),
            Type::FunType(_, _) => {
                panic!("No size");
            }
        }
    }

    pub(crate) fn byte_align(&self) -> usize {
        match self {
            Type::Void => 1,
            Type::Char | Type::SChar | Type::UChar => 2,
            Type::Short | Type::UShort => 2,
            Type::Int | Type::UInt => 4,
            Type::Long | Type::ULong => 8,
            Type::Double => 8,
            Type::Pointer(_) => 8,
            Type::Array(inner, _) => inner.byte_align(),
            Type::IncompleteArray(t) => t.byte_align(),
            Type::Struct(_) => todo!("struct alignment not implemented"),
            Type::FunType(_, _) => {
                panic!("No size");
            }
        }
    }

    pub(crate) fn integer_promotion(&self) -> Type {
        use Type::*;
        match self {
            Char | UChar => Int,
            t if t.is_integer() => self.clone(),
            _ => panic!("Invalid integer promotion type"),
        }
    }

    pub(crate) fn working_type(&self, rhs: Type) -> Type {
        use Type::*;

        let a = self.integer_promotion();
        let b = rhs.integer_promotion();
        if a == b {
            return a;
        }

        let wa = a.width();
        let wb = b.width();
        let sa = a.is_signed();
        let sb = b.is_signed();

        let rank = |w: Width| match w {
            Width::W8 => 1,
            Width::W16 => 2,
            Width::W32 => 3,
            Width::W64 => 4,
        };

        if rank(wa) != rank(wb) {
            return match (wa, sa, wb, sb) {
                (Width::W64, true, _, _) => Long,
                (Width::W64, false, _, _) => ULong,
                (_, _, Width::W64, true) => Long,
                (_, _, Width::W64, false) => ULong,
                (Width::W32, true, Width::W16 | Width::W8, _) => Int,
                (Width::W32, false, Width::W16 | Width::W8, _) => UInt,
                (Width::W16 | Width::W8, _, Width::W32, true) => Int,
                (Width::W16 | Width::W8, _, Width::W32, false) => UInt,
                _ => unreachable!(),
            };
        }

        match (wa, sa, sb) {
            (Width::W64, false, _) | (Width::W64, _, false) => ULong,
            (Width::W32, false, _) | (Width::W32, _, false) => UInt,
            _ => Long,
        }
    }

    pub(crate) fn is_char(&self) -> bool {
        matches!(self, Type::Char | Type::SChar | Type::UChar)
    }

    pub(crate) fn is_floating(&self) -> bool {
        matches!(self, Type::Double)
    }

    pub(crate) fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_floating()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Const {
    Char(i8),
    UChar(u8),
    Short(i16),
    UShort(u16),
    Int(i32),
    UInt(u32),
    Long(i64),
    ULong(u64),
    Double(f64),
}

impl fmt::Display for Const {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&match self {
            Const::Char(n) => n.to_string(),
            Const::UChar(n) => n.to_string(),
            Const::Short(n) => n.to_string(),
            Const::UShort(n) => n.to_string(),
            Const::Int(n) => n.to_string(),
            Const::UInt(n) => n.to_string(),
            Const::Long(n) => n.to_string(),
            Const::ULong(n) => n.to_string(),
            Const::Double(n) => n.to_string(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node<Kind> {
    pub(crate) kind: Kind,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) source: String,
    pub(crate) r#type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Parser {
    pub(crate) source: Vec<u8>,
    pub(crate) index: usize,
    pub(crate) tokens: Vec<Token>,
    pub(crate) pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct ParsedDeclarator {
    name: String,
    type_expr: TypeExpr,
}

#[derive(Debug, Clone, PartialEq)]
enum TypeExpr {
    Base,
    Pointer(Box<TypeExpr>),
    Array {
        size: Option<usize>,
        elem: Box<TypeExpr>,
    },
    Function {
        params: Vec<ParameterDecl>,
        ret: Box<TypeExpr>,
    },
}

impl TypeExpr {
    fn add_pointer(self) -> Self {
        match self {
            TypeExpr::Function { params, ret } => TypeExpr::Function {
                params,
                ret: Box::new(ret.add_pointer()),
            },
            TypeExpr::Array { size, elem } => TypeExpr::Array {
                size,
                elem: Box::new(elem.add_pointer()),
            },
            other => TypeExpr::Pointer(Box::new(other)),
        }
    }

    fn add_function(self, params: Vec<ParameterDecl>) -> Self {
        match self {
            TypeExpr::Pointer(inner) => TypeExpr::Pointer(Box::new(inner.add_function(params))),
            TypeExpr::Array { size, elem } => TypeExpr::Array {
                size,
                elem: Box::new(elem.add_function(params)),
            },
            other => TypeExpr::Function {
                params,
                ret: Box::new(other),
            },
        }
    }

    fn add_array(self, size: Option<usize>) -> Self {
        match self {
            TypeExpr::Pointer(inner) => TypeExpr::Pointer(Box::new(inner.add_array(size))),
            other => TypeExpr::Array {
                size,
                elem: Box::new(other),
            },
        }
    }

    fn apply(&self, base: Type) -> Type {
        match self {
            TypeExpr::Base => base,
            TypeExpr::Pointer(inner) => Type::Pointer(Box::new(inner.apply(base))),
            TypeExpr::Array { size, elem } => match size {
                Some(len) => Type::Array(Box::new(elem.apply(base)), *len),
                None => Type::IncompleteArray(Box::new(elem.apply(base))),
            },
            TypeExpr::Function { params, ret } => {
                let param_types = params.iter().map(|p| p.r#type.clone()).collect();
                let return_type = ret.apply(base);
                Type::FunType(param_types, Box::new(return_type))
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
struct TypeSpecifierState {
    saw_void: bool,
    saw_double: bool,
    saw_short: bool,
    saw_int: bool,
    saw_long: bool,
    saw_char: bool,
    signedness: Option<bool>,
    is_const: bool,
}

impl TypeSpecifierState {
    fn add(&mut self, kind: &TokenKind) -> PResult<()> {
        match kind {
            TokenKind::Void => {
                if self.saw_void {
                    return Err(ParserError::DuplicateTypeSpecifier("void"));
                }
                if self.saw_long || self.saw_int || self.signedness.is_some() || self.saw_double {
                    return Err(ParserError::VoidCannotCombine);
                }
                self.saw_void = true;
            }
            TokenKind::Double => {
                if self.saw_double {
                    return Err(ParserError::DuplicateTypeSpecifier("double"));
                }
                if self.saw_void || self.saw_long || self.saw_int || self.signedness.is_some() {
                    return Err(ParserError::DoubleCannotCombine);
                }
                self.saw_double = true;
            }
            TokenKind::Long => {
                if self.saw_void || self.saw_double {
                    return Err(ParserError::ConflictingTypeSpecifiers(
                        "long",
                        "void/double",
                    ));
                }
                self.saw_long = true;
            }
            TokenKind::Int => {
                if self.saw_void || self.saw_double {
                    return Err(ParserError::ConflictingTypeSpecifiers("int", "void/double"));
                }
                if self.saw_int {
                    return Err(ParserError::DuplicateTypeSpecifier("int"));
                }
                self.saw_int = true;
            }
            TokenKind::Short => {
                if self.saw_char
                    || self.saw_void
                    || self.saw_double
                    || self.saw_long
                    || self.saw_int
                {
                    return Err(ParserError::ConflictingTypeSpecifiers(
                        "char",
                        "void/double/long/int",
                    ));
                }
                if self.saw_short {
                    return Err(ParserError::DuplicateTypeSpecifier("short"));
                }
                self.saw_short = true;
            }
            TokenKind::Char => {
                if self.saw_void || self.saw_double || self.saw_long || self.saw_int {
                    return Err(ParserError::ConflictingTypeSpecifiers(
                        "char",
                        "void/double/long/int",
                    ));
                }
                if self.saw_char {
                    return Err(ParserError::DuplicateTypeSpecifier("char"));
                }
                self.saw_char = true;
            }
            TokenKind::Signed => {
                if self.saw_double {
                    return Err(ParserError::ConflictingTypeSpecifiers("signed", "double"));
                }
                if self.signedness == Some(false) {
                    return Err(ParserError::ConflictingTypeSpecifiers("signed", "unsigned"));
                }
                self.signedness = Some(true);
            }
            TokenKind::Unsigned => {
                if self.saw_double {
                    return Err(ParserError::ConflictingTypeSpecifiers("unsigned", "double"));
                }
                if self.signedness == Some(true) {
                    return Err(ParserError::ConflictingTypeSpecifiers("unsigned", "signed"));
                }
                self.signedness = Some(false);
            }
            TokenKind::Const => {
                if self.is_const {
                    return Err(ParserError::DuplicateTypeSpecifier("const"));
                }
                self.is_const = true;
            }
            other => return Err(ParserError::UnsupportedTypeSpecifier(other.clone())),
        }
        Ok(())
    }

    fn has_type_specifier(&self) -> bool {
        self.saw_void
            || self.saw_double
            || self.saw_long
            || self.saw_short
            || self.saw_int
            || self.saw_char
            || self.signedness.is_some()
    }

    fn resolve(&self) -> PResult<Type> {
        if !self.has_type_specifier() {
            return Err(ParserError::MissingTypeSpecifier);
        }

        if self.saw_void {
            return Ok(Type::Void);
        }

        if self.saw_double {
            return Ok(Type::Double);
        }

        if self.saw_char {
            return Ok(match self.signedness {
                Some(true) => Type::SChar,
                Some(false) => Type::UChar,
                None => Type::Char,
            });
        }

        let is_unsigned = matches!(self.signedness, Some(false));
        let ty = if self.saw_long {
            if is_unsigned { Type::ULong } else { Type::Long }
        } else if is_unsigned {
            Type::UInt
        } else {
            Type::Int
        };

        Ok(ty)
    }
}

impl Parser {
    pub(crate) fn new(source: Vec<u8>, tokens: Vec<Token>) -> Self {
        Self {
            source,
            tokens,
            index: 0,
            pos: 0,
        }
    }

    pub(crate) fn parse(&mut self) -> PResult<Program> {
        let mut decls = Vec::new();
        while self.pos < self.tokens.len() {
            decls.push(self.declaration()?);
        }
        self.ensure_done()?;
        Ok(Program(decls))
    }

    fn parse_specifiers(
        &mut self,
    ) -> PResult<(Type, Option<StorageClass>, Option<StructDeclaration>)> {
        self.parse_specifiers_internal(true)
    }

    fn parse_type_specifiers(&mut self) -> PResult<(Type, Option<StructDeclaration>)> {
        let (ty, storage, struct_decl) = self.parse_specifiers_internal(false)?;
        debug_assert!(storage.is_none());
        Ok((ty, struct_decl))
    }

    fn parse_specifiers_internal(
        &mut self,
        allow_storage: bool,
    ) -> PResult<(Type, Option<StorageClass>, Option<StructDeclaration>)> {
        let mut storage: Option<StorageClass> = None;
        let mut state = TypeSpecifierState::default();
        let mut consumed_any = false;
        let mut struct_spec: Option<(Type, Option<StructDeclaration>)> = None;

        while self.pos < self.tokens.len() {
            let kind = self.peek_kind();
            match kind.clone() {
                Some(TokenKind::Static) => {
                    if !allow_storage {
                        return Err(ParserError::StorageNotAllowedHere);
                    }
                    if storage.is_some() {
                        return Err(ParserError::MultipleStorageClasses);
                    }
                    self.advance()?;
                    storage = Some(StorageClass::Static);
                    consumed_any = true;
                }
                Some(TokenKind::Extern) => {
                    if !allow_storage {
                        return Err(ParserError::StorageNotAllowedHere);
                    }
                    if storage.is_some() {
                        return Err(ParserError::MultipleStorageClasses);
                    }
                    self.advance()?;
                    storage = Some(StorageClass::Extern);
                    consumed_any = true;
                }
                Some(TokenKind::Struct) => {
                    if struct_spec.is_some() || state.has_type_specifier() {
                        return Err(ParserError::ConflictingTypeSpecifiers(
                            "struct",
                            "other type specifier",
                        ));
                    }
                    self.advance()?;
                    let parsed = self.parse_struct_specifier()?;
                    struct_spec = Some(parsed);
                    consumed_any = true;
                }
                Some(TokenKind::Int)
                | Some(TokenKind::Short)
                | Some(TokenKind::Long)
                | Some(TokenKind::Void)
                | Some(TokenKind::Signed)
                | Some(TokenKind::Unsigned)
                | Some(TokenKind::Double)
                | Some(TokenKind::Char)
                | Some(TokenKind::Const) => {
                    let k = kind.unwrap();
                    self.advance()?;
                    state.add(&k)?;
                    consumed_any = true;
                }
                _ => break,
            }
        }

        if !consumed_any {
            return Err(ParserError::ExpectedDeclSpecifiers);
        }

        let (ty, struct_decl) = if let Some((ty, decl)) = struct_spec {
            (ty, decl)
        } else {
            (state.resolve()?, None)
        };
        Ok((ty, storage, struct_decl))
    }

    fn parse_struct_specifier(&mut self) -> PResult<(Type, Option<StructDeclaration>)> {
        let tag = match self.peek_kind() {
            Some(TokenKind::Identifier(name)) => {
                self.advance()?;
                name
            }
            _ => return Err(ParserError::ExpectedIdentifier),
        };

        if matches!(self.peek_kind(), Some(TokenKind::LBrace)) {
            self.advance()?;
            let members = self.parse_struct_member_declarations()?;
            self.expect(&TokenKind::RBrace)?;
            let decl = StructDeclaration {
                tag: tag.clone(),
                members,
            };
            Ok((Type::Struct(tag), Some(decl)))
        } else {
            Ok((Type::Struct(tag), None))
        }
    }

    fn parse_struct_member_declarations(&mut self) -> PResult<Vec<MemberDeclaration>> {
        let mut members = Vec::new();

        while !matches!(self.peek_kind(), Some(TokenKind::RBrace)) {
            let (base_type, _) = self.parse_type_specifiers()?;
            let declarator = self.parse_declarator()?;
            let member_type = declarator.type_expr.apply(base_type);
            if matches!(member_type, Type::FunType(_, _)) {
                return Err(ParserError::VariableWithFunctionType);
            }
            self.expect(&TokenKind::Semicolon)?;
            members.push(MemberDeclaration {
                member_name: declarator.name,
                member_type,
            });
        }

        Ok(members)
    }

    fn parse_declarator(&mut self) -> PResult<ParsedDeclarator> {
        if matches!(self.peek_kind(), Some(TokenKind::Star)) {
            self.advance()?;
            let mut inner = self.parse_declarator()?;
            let current = mem::replace(&mut inner.type_expr, TypeExpr::Base);
            inner.type_expr = current.add_pointer();
            Ok(inner)
        } else {
            self.parse_direct_declarator()
        }
    }

    fn parse_direct_declarator(&mut self) -> PResult<ParsedDeclarator> {
        let mut declarator = if let Some(TokenKind::Identifier(name)) = self.peek_kind() {
            self.advance()?;
            ParsedDeclarator {
                name,
                type_expr: TypeExpr::Base,
            }
        } else if matches!(self.peek_kind(), Some(TokenKind::LParen)) {
            self.advance()?;
            let declarator = self.parse_declarator()?;
            self.expect(&TokenKind::RParen)?;
            declarator
        } else {
            return Err(ParserError::ExpectedIdentOrParen(self.peek_kind().unwrap()));
        };

        while self.pos < self.tokens.len() {
            match self.peek_kind() {
                Some(TokenKind::LParen) => {
                    self.advance()?;
                    let params = self.parse_parameter_list()?;
                    self.expect(&TokenKind::RParen)?;
                    let current = mem::replace(&mut declarator.type_expr, TypeExpr::Base);
                    declarator.type_expr = current.add_function(params);
                }
                Some(TokenKind::LBracket) => {
                    self.advance()?;
                    let size = if matches!(self.peek_kind(), Some(TokenKind::RBracket)) {
                        None
                    } else {
                        Some(self.parse_array_size()?)
                    };
                    self.expect(&TokenKind::RBracket)?;
                    let current = mem::replace(&mut declarator.type_expr, TypeExpr::Base);
                    declarator.type_expr = current.add_array(size);
                }
                _ => break,
            }
        }

        Ok(declarator)
    }

    fn parse_array_size(&mut self) -> PResult<usize> {
        match self.peek_kind() {
            Some(TokenKind::Integer(n)) => {
                if n < 0 {
                    return Err(ParserError::NegativeArraySize);
                }
                self.advance()?;
                usize::try_from(n).map_err(|_| ParserError::ArraySizeTooLarge(n as i128))
            }
            Some(TokenKind::LongInteger(n)) => {
                if n < 0 {
                    return Err(ParserError::NegativeArraySize);
                }
                self.advance()?;
                usize::try_from(n).map_err(|_| ParserError::ArraySizeTooLarge(n as i128))
            }
            Some(TokenKind::UnsignedInteger(n)) => {
                self.advance()?;
                usize::try_from(n).map_err(|_| ParserError::ArraySizeTooLarge(n as i128))
            }
            Some(TokenKind::UnsignedLongInteger(n)) => {
                self.advance()?;
                usize::try_from(n).map_err(|_| ParserError::ArraySizeTooLarge(n as i128))
            }
            other => Err(ParserError::ExpectedConstArraySize(other.unwrap())),
        }
    }

    fn parse_parameter_list(&mut self) -> PResult<Vec<ParameterDecl>> {
        if matches!(self.peek_kind(), Some(TokenKind::RParen)) {
            return Ok(Vec::new());
        }

        if matches!(self.peek_kind(), Some(TokenKind::Void))
            && self.pos + 1 < self.tokens.len()
            && matches!(self.tokens[self.pos + 1].kind, TokenKind::RParen)
        {
            self.advance()?;
            return Ok(Vec::new());
        }

        let mut params = Vec::new();

        loop {
            let param = self.parse_parameter()?;
            if param.r#type.is_void() {
                return Err(ParserError::VoidOnlyParameter);
            }
            params.push(param);

            if matches!(self.peek_kind(), Some(TokenKind::Comma)) {
                self.advance()?;
                continue;
            }
            break;
        }

        Ok(params)
    }

    fn parse_parameter(&mut self) -> PResult<ParameterDecl> {
        let (base_type, _) = self.parse_type_specifiers()?;
        let declarator = self.parse_declarator()?;
        let param_type = declarator.type_expr.apply(base_type);
        Ok(ParameterDecl {
            name: declarator.name,
            r#type: param_type,
        })
    }

    fn declaration(&mut self) -> PResult<Decl> {
        if self.pos >= self.tokens.len() {
            return Err(ParserError::UnexpectedEof("while parsing declaration"));
        }

        let Token { start, .. } = self.peek()?;
        let (base_type, storage_class, struct_decl) = self.parse_specifiers()?;

        if matches!(self.peek_kind(), Some(TokenKind::Semicolon)) {
            if let Type::Struct(tag) = &base_type {
                self.advance()?;
                let end = self.index;
                let source = self.source_slice(start, end);
                let decl = match struct_decl {
                    Some(decl) => decl,
                    None => StructDeclaration {
                        tag: tag.clone(),
                        members: Vec::new(),
                    },
                };
                return Ok(Decl {
                    kind: DeclKind::Struct(decl),
                    start,
                    end,
                    source,
                });
            }
        }
        let declarator = self.parse_declarator()?;
        let name = declarator.name.clone();

        match declarator.type_expr {
            TypeExpr::Function { params, ret } => {
                let return_type = ret.apply(base_type);
                self.parse_function_declaration(start, name, params, return_type, storage_class)
            }
            type_expr => {
                let var_type = type_expr.apply(base_type);
                if var_type.is_void() {
                    return Err(ParserError::VariableWithVoidType);
                }
                self.parse_variable_declaration(start, name, var_type, storage_class)
            }
        }
    }

    fn parse_function_declaration(
        &mut self,
        start: usize,
        name: String,
        params: Vec<ParameterDecl>,
        return_type: Type,
        storage_class: Option<StorageClass>,
    ) -> PResult<Decl> {
        let body = if matches!(self.peek_kind(), Some(TokenKind::LBrace)) {
            self.advance()?;
            let stmts = self.stmts()?;
            self.expect(&TokenKind::RBrace)?;
            Some(stmts)
        } else {
            self.expect(&TokenKind::Semicolon)?;
            None
        };

        let end = self.index;
        let source = self.source_slice(start, end);

        Ok(Decl {
            kind: DeclKind::Function(FunctionDecl {
                name,
                params,
                body,
                storage_class,
                return_type,
            }),
            start,
            end,
            source,
        })
    }

    fn parse_variable_declaration(
        &mut self,
        start: usize,
        name: String,
        ty: Type,
        storage_class: Option<StorageClass>,
    ) -> PResult<Decl> {
        if matches!(ty, Type::FunType(_, _)) {
            return Err(ParserError::VariableWithFunctionType);
        }

        let init = if matches!(self.peek_kind(), Some(TokenKind::Equal)) {
            self.advance()?;
            Some(self.expr()?)
        } else {
            None
        };

        self.expect(&TokenKind::Semicolon)?;

        let end = self.index;
        let source = self.source_slice(start, end);

        let is_definition = storage_class != Some(StorageClass::Extern);
        let decl = VariableDecl {
            name,
            init,
            storage_class,
            r#type: ty.clone(),
            is_definition,
        };

        Ok(Decl {
            kind: DeclKind::Variable(decl),
            start,
            end,
            source,
        })
    }

    fn stmts(&mut self) -> PResult<Vec<Stmt>> {
        let mut out = vec![];

        while !matches!(self.peek_kind(), Some(TokenKind::RBrace)) {
            if self.is_declaration_start() {
                out.push(self.variable_declaration_stmt()?);
            } else {
                out.push(self.stmt()?);
            }
        }

        Ok(out)
    }

    fn variable_declaration_stmt(&mut self) -> PResult<Stmt> {
        let Token { start, .. } = self.peek()?;
        let (base_type, storage_class, _) = self.parse_specifiers()?;
        let declarator = self.parse_declarator()?;
        let name = declarator.name.clone();
        let var_type = declarator.type_expr.apply(base_type);
        if var_type.is_void() {
            return Err(ParserError::VariableWithVoidType);
        }
        if matches!(var_type, Type::FunType(_, _)) {
            return Err(ParserError::FunctionDeclInBlockScope);
        }
        let init = if matches!(self.peek_kind(), Some(TokenKind::Equal)) {
            self.advance()?;
            Some(self.expr()?)
        } else {
            None
        };

        self.expect(&TokenKind::Semicolon)?;

        let end = self.index;
        let source = self.source_slice(start, end);

        let is_definition = storage_class != Some(StorageClass::Extern);
        let decl = VariableDecl {
            name,
            init,
            storage_class,
            r#type: var_type.clone(),
            is_definition,
        };

        Ok(Stmt {
            kind: StmtKind::Declaration(decl),
            start,
            end,
            source,
            r#type: var_type,
        })
    }

    fn stmt(&mut self) -> PResult<Stmt> {
        let token = self.peek()?;
        match token.kind.clone() {
            TokenKind::Return => {
                self.advance()?;
                let expr = self.expr()?;
                self.expect(&TokenKind::Semicolon)?;
                Ok(Stmt {
                    kind: StmtKind::Return(expr),
                    start: token.start,
                    end: self.index,
                    source: token.source,
                    r#type: Type::Int,
                })
            }
            TokenKind::LBrace => self.block(),
            TokenKind::Semicolon => {
                self.advance()?;
                Ok(Stmt {
                    kind: StmtKind::Null,
                    start: token.start,
                    end: token.end,
                    source: token.source,
                    r#type: Type::Void,
                })
            }
            TokenKind::If => self.if_stmt(),
            TokenKind::While => self.while_stmt(),
            TokenKind::Do => self.do_while_stmt(),
            TokenKind::For => self.for_stmt(),
            TokenKind::Break => self.break_stmt(),
            TokenKind::Continue => self.continue_stmt(),
            _ => {
                let expr = self.expr()?;
                self.expect(&TokenKind::Semicolon)?;
                let start = expr.start;
                let end = expr.end;
                let source = expr.source.clone();
                Ok(Stmt {
                    kind: StmtKind::Expr(expr),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                })
            }
        }
    }

    fn break_stmt(&mut self) -> PResult<Stmt> {
        let Token { start, .. } = self.peek()?;
        self.advance()?;
        self.expect(&TokenKind::Semicolon)?;
        let end = self.index;
        Ok(Stmt {
            kind: StmtKind::Break { loop_id: None },
            start,
            end,
            source: self.source_slice(start, end),
            r#type: Type::Void,
        })
    }

    fn continue_stmt(&mut self) -> PResult<Stmt> {
        let Token { start, .. } = self.peek()?;
        self.advance()?;
        self.expect(&TokenKind::Semicolon)?;
        let end = self.index;
        Ok(Stmt {
            kind: StmtKind::Continue { loop_id: None },
            start,
            end,
            source: self.source_slice(start, end),
            r#type: Type::Void,
        })
    }

    fn block(&mut self) -> PResult<Stmt> {
        let Token { start, .. } = self.peek()?;
        self.expect(&TokenKind::LBrace)?;
        let stmts = self.stmts()?;
        self.expect(&TokenKind::RBrace)?;
        let end = self.index;
        let source = String::from_utf8_lossy(&self.source[start..end]).to_string();
        Ok(Stmt {
            kind: StmtKind::Compound(stmts),
            start,
            end,
            source,
            r#type: Type::Void,
        })
    }

    fn while_stmt(&mut self) -> PResult<Stmt> {
        let Token { start, .. } = self.peek()?;
        self.advance()?;
        self.expect(&TokenKind::LParen)?;
        let condition = self.expr()?;
        self.expect(&TokenKind::RParen)?;
        let body_stmt = self.stmt()?;
        let end = body_stmt.end;
        Ok(Stmt {
            kind: StmtKind::While {
                condition,
                body: Box::new(body_stmt),
                loop_id: None,
            },
            start,
            end,
            source: self.source_slice(start, end),
            r#type: Type::Void,
        })
    }

    fn do_while_stmt(&mut self) -> PResult<Stmt> {
        let Token { start, .. } = self.peek()?;
        self.advance()?;
        let body_stmt = self.stmt()?;
        self.expect(&TokenKind::While)?;
        self.expect(&TokenKind::LParen)?;
        let condition = self.expr()?;
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::Semicolon)?;
        let end = self.index;
        Ok(Stmt {
            kind: StmtKind::DoWhile {
                body: Box::new(body_stmt),
                condition,
                loop_id: None,
            },
            start,
            end,
            source: self.source_slice(start, end),
            r#type: Type::Void,
        })
    }

    fn for_stmt(&mut self) -> PResult<Stmt> {
        let Token { start, .. } = self.peek()?;
        self.advance()?;
        self.expect(&TokenKind::LParen)?;

        let init = if matches!(self.peek_kind(), Some(TokenKind::Semicolon)) {
            self.advance()?;
            ForInit::Expr(None)
        } else if self.is_declaration_start() {
            let decl = self.variable_declaration_stmt()?;
            ForInit::Declaration(Box::new(decl))
        } else {
            let expr = self.expr()?;
            self.expect(&TokenKind::Semicolon)?;
            ForInit::Expr(Some(expr))
        };

        let condition = if matches!(self.peek_kind(), Some(TokenKind::Semicolon)) {
            self.advance()?;
            None
        } else {
            let expr = self.expr()?;
            self.expect(&TokenKind::Semicolon)?;
            Some(expr)
        };

        let post = if matches!(self.peek_kind(), Some(TokenKind::RParen)) {
            None
        } else {
            Some(self.expr()?)
        };

        self.expect(&TokenKind::RParen)?;
        let body_stmt = self.stmt()?;
        let end = body_stmt.end;

        Ok(Stmt {
            kind: StmtKind::For {
                init,
                condition,
                post,
                body: Box::new(body_stmt),
                loop_id: None,
            },
            start,
            end,
            source: self.source_slice(start, end),
            r#type: Type::Void,
        })
    }

    fn if_stmt(&mut self) -> PResult<Stmt> {
        let Token { start, .. } = self.peek()?;
        self.advance()?; // if
        self.expect(&TokenKind::LParen)?;
        let condition = self.expr()?;
        self.expect(&TokenKind::RParen)?;
        let then_branch = self.stmt()?;
        let else_branch = if matches!(self.peek_kind(), Some(TokenKind::Else)) {
            self.advance()?;
            Some(Box::new(self.stmt()?))
        } else {
            None
        };
        let end = else_branch
            .as_ref()
            .map(|b| b.end)
            .unwrap_or(then_branch.end);
        let source = self.source_slice(start, end);
        Ok(Stmt {
            kind: StmtKind::If {
                condition,
                then_branch: Box::new(then_branch),
                else_branch,
            },
            start,
            end,
            source,
            r#type: Type::Void,
        })
    }

    fn expr(&mut self) -> PResult<Expr> {
        self.assign()
    }

    fn assign(&mut self) -> PResult<Expr> {
        let lhs = self.conditional()?;

        let token_kind = self.peek_kind();
        match token_kind {
            Some(TokenKind::Equal) => {
                self.advance()?;
                let rhs = self.assign()?;
                Ok(self.make_assignment_expr(lhs, rhs))
            }
            Some(
                TokenKind::PlusEqual
                | TokenKind::MinusEqual
                | TokenKind::StarEqual
                | TokenKind::SlashEqual
                | TokenKind::PercentEqual
                | TokenKind::AmpersandEqual
                | TokenKind::OrEqual
                | TokenKind::XorEqual
                | TokenKind::LShiftEqual
                | TokenKind::RShiftEqual,
            ) => {
                let op = self.peek_kind().unwrap();
                self.advance()?;
                let rhs = self.assign()?;
                let compound_rhs = self.compound_assignment_rhs(lhs.clone(), rhs, &op)?;
                Ok(self.make_assignment_expr(lhs, compound_rhs))
            }
            _ => Ok(lhs),
        }
    }

    fn conditional(&mut self) -> PResult<Expr> {
        let condition = self.or()?;

        if matches!(self.peek_kind(), Some(TokenKind::Question)) {
            self.advance()?;
            let then_expr = self.assign()?;
            self.expect(&TokenKind::Colon)?;
            let else_expr = self.conditional()?;
            let start = condition.start;
            let end = else_expr.end;
            Ok(Expr {
                kind: ExprKind::Conditional(
                    Box::new(condition),
                    Box::new(then_expr),
                    Box::new(else_expr),
                ),
                start,
                end,
                source: self.source_slice(start, end),
                r#type: Type::Int,
            })
        } else {
            Ok(condition)
        }
    }

    fn make_assignment_expr(&self, lhs: Expr, rhs: Expr) -> Expr {
        let start = lhs.start;
        let end = rhs.end;
        Expr {
            kind: ExprKind::Assignment(Box::new(lhs), Box::new(rhs)),
            start,
            end,
            source: self.source_slice(start, end),
            r#type: Type::Int,
        }
    }

    fn compound_assignment_rhs(&self, lhs: Expr, rhs: Expr, op: &TokenKind) -> PResult<Expr> {
        let start = lhs.start;
        let end = rhs.end;

        let source = self.source_slice(start, end);

        let kind = match op {
            TokenKind::PlusEqual => ExprKind::Add(Box::new(lhs), Box::new(rhs)),
            TokenKind::MinusEqual => ExprKind::Sub(Box::new(lhs), Box::new(rhs)),
            TokenKind::StarEqual => ExprKind::Mul(Box::new(lhs), Box::new(rhs)),
            TokenKind::SlashEqual => ExprKind::Div(Box::new(lhs), Box::new(rhs)),
            TokenKind::PercentEqual => ExprKind::Rem(Box::new(lhs), Box::new(rhs)),
            TokenKind::AmpersandEqual => ExprKind::BitAnd(Box::new(lhs), Box::new(rhs)),
            TokenKind::OrEqual => ExprKind::BitOr(Box::new(lhs), Box::new(rhs)),
            TokenKind::XorEqual => ExprKind::Xor(Box::new(lhs), Box::new(rhs)),
            TokenKind::LShiftEqual => ExprKind::LeftShift(Box::new(lhs), Box::new(rhs)),
            TokenKind::RShiftEqual => ExprKind::RightShift(Box::new(lhs), Box::new(rhs)),
            other => return Err(ParserError::UnsupportedCompoundAssign(other.clone())),
        };

        Ok(Expr {
            kind,
            start,
            end,
            source,
            r#type: Type::Int,
        })
    }

    fn or(&mut self) -> PResult<Expr> {
        let mut node = self.and()?;

        while matches!(self.peek_kind(), Some(TokenKind::Or)) {
            let Token {
                start, end, source, ..
            } = self.peek()?;
            self.advance()?;

            node = Expr {
                kind: ExprKind::Or(Box::new(node), Box::new(self.and()?)),
                start,
                end,
                source,
                r#type: Type::Int,
            }
        }

        Ok(node)
    }

    fn and(&mut self) -> PResult<Expr> {
        let mut node = self.bit_or()?;

        while matches!(self.peek_kind(), Some(TokenKind::DoubleAmpersand)) {
            let Token {
                start, end, source, ..
            } = self.peek()?;
            self.advance()?;

            node = Expr {
                kind: ExprKind::And(Box::new(node), Box::new(self.bit_or()?)),
                start,
                end,
                source,
                r#type: Type::Int,
            }
        }

        Ok(node)
    }

    fn bit_or(&mut self) -> PResult<Expr> {
        let mut node = self.xor()?;

        while matches!(self.peek_kind(), Some(TokenKind::BitOr)) {
            let Token {
                start, end, source, ..
            } = self.peek()?;
            self.advance()?;

            node = Expr {
                kind: ExprKind::BitOr(Box::new(node), Box::new(self.xor()?)),
                start,
                end,
                source,
                r#type: Type::Int,
            }
        }

        Ok(node)
    }

    fn xor(&mut self) -> PResult<Expr> {
        let mut node = self.bit_and()?;

        while matches!(self.peek_kind(), Some(TokenKind::Xor)) {
            let Token {
                start, end, source, ..
            } = self.peek()?;
            self.advance()?;

            node = Expr {
                kind: ExprKind::Xor(Box::new(node), Box::new(self.bit_and()?)),
                start,
                end,
                source,
                r#type: Type::Int,
            }
        }

        Ok(node)
    }

    fn bit_and(&mut self) -> PResult<Expr> {
        let mut node = self.eq()?;

        while matches!(self.peek_kind(), Some(TokenKind::Ampersand)) {
            let Token {
                start, end, source, ..
            } = self.peek()?;
            self.advance()?;

            node = Expr {
                kind: ExprKind::BitAnd(Box::new(node), Box::new(self.eq()?)),
                start,
                end,
                source,
                r#type: Type::Int,
            }
        }

        Ok(node)
    }

    fn eq(&mut self) -> PResult<Expr> {
        let mut node = self.rel()?;

        while matches!(
            self.peek_kind(),
            Some(TokenKind::DoubleEqual | TokenKind::NotEqual)
        ) {
            let Token {
                start,
                end,
                source,
                kind,
                ..
            } = self.peek()?;
            self.advance()?;

            node = Expr {
                kind: if kind == TokenKind::DoubleEqual {
                    ExprKind::Equal(Box::new(node), Box::new(self.rel()?))
                } else {
                    ExprKind::NotEqual(Box::new(node), Box::new(self.rel()?))
                },
                start,
                end,
                source,
                r#type: Type::Int,
            }
        }

        Ok(node)
    }

    fn rel(&mut self) -> PResult<Expr> {
        let mut node = self.shift()?;

        while matches!(
            self.peek_kind(),
            Some(
                TokenKind::LessThan
                    | TokenKind::LessThanEqual
                    | TokenKind::GreaterThan
                    | TokenKind::GreaterThanEqual
            )
        ) {
            let Token {
                start,
                end,
                source,
                kind,
                ..
            } = self.peek()?;
            self.advance()?;

            node = Expr {
                kind: if kind == TokenKind::LessThan {
                    ExprKind::LessThan(Box::new(node), Box::new(self.shift()?))
                } else if kind == TokenKind::LessThanEqual {
                    ExprKind::LessThanEqual(Box::new(node), Box::new(self.shift()?))
                } else if kind == TokenKind::GreaterThan {
                    ExprKind::GreaterThan(Box::new(node), Box::new(self.shift()?))
                } else {
                    ExprKind::GreaterThanEqual(Box::new(node), Box::new(self.shift()?))
                },
                start,
                end,
                source,
                r#type: Type::Int,
            }
        }

        Ok(node)
    }

    fn shift(&mut self) -> PResult<Expr> {
        let mut node = self.add()?;

        while matches!(
            self.peek_kind(),
            Some(TokenKind::LShift | TokenKind::RShift)
        ) {
            let Token {
                start,
                end,
                source,
                kind,
                ..
            } = self.peek()?;
            self.advance()?;

            node = Expr {
                kind: if kind == TokenKind::LShift {
                    ExprKind::LeftShift(Box::new(node), Box::new(self.add()?))
                } else {
                    ExprKind::RightShift(Box::new(node), Box::new(self.add()?))
                },
                start,
                end,
                source,
                r#type: Type::Int,
            }
        }

        Ok(node)
    }

    fn add(&mut self) -> PResult<Expr> {
        let mut node = self.mul()?;

        while matches!(self.peek_kind(), Some(TokenKind::Plus | TokenKind::Minus)) {
            let Token {
                start,
                end,
                source,
                kind,
                ..
            } = self.peek()?;
            self.advance()?;

            node = Expr {
                kind: if kind == TokenKind::Plus {
                    ExprKind::Add(Box::new(node), Box::new(self.mul()?))
                } else {
                    ExprKind::Sub(Box::new(node), Box::new(self.mul()?))
                },
                start,
                end,
                source,
                r#type: Type::Int,
            }
        }

        Ok(node)
    }

    fn mul(&mut self) -> PResult<Expr> {
        let mut node = self.unary()?;

        while matches!(
            self.peek_kind(),
            Some(TokenKind::Star | TokenKind::Slash | TokenKind::Percent)
        ) {
            let Token {
                start,
                end,
                source,
                kind,
                ..
            } = self.peek()?;
            self.advance()?;

            node = Expr {
                kind: if kind == TokenKind::Star {
                    ExprKind::Mul(Box::new(node), Box::new(self.unary()?))
                } else if kind == TokenKind::Percent {
                    ExprKind::Rem(Box::new(node), Box::new(self.unary()?))
                } else {
                    ExprKind::Div(Box::new(node), Box::new(self.unary()?))
                },
                start,
                end,
                source,
                r#type: Type::Int,
            }
        }

        Ok(node)
    }

    fn unary(&mut self) -> PResult<Expr> {
        if matches!(self.peek_kind(), Some(TokenKind::LParen)) && self.is_cast_expression()? {
            let Token { start, .. } = self.peek()?;
            self.advance()?;
            let (base_type, _) = self.parse_type_specifiers()?;
            let mut cast_type = base_type;
            while matches!(self.peek_kind(), Some(TokenKind::Star)) {
                self.advance()?;
                cast_type = Type::Pointer(Box::new(cast_type));
            }
            if cast_type.is_void() {
                return Err(ParserError::UnsupportedCastTargetVoid);
            }
            self.expect(&TokenKind::RParen)?;
            let expr = self.unary()?;
            let end = expr.end;
            return Ok(Expr {
                kind: ExprKind::Cast(cast_type.clone(), Box::new(expr)),
                start,
                end,
                source: self.source_slice(start, end),
                r#type: cast_type,
            });
        }

        let token = self.peek()?;
        match token.kind.clone() {
            TokenKind::Plus => {
                self.advance()?;
                self.unary()
            }
            TokenKind::Minus => {
                self.advance()?;
                let expr = self.unary()?;
                let start = token.start;
                let end = expr.end;
                Ok(Expr {
                    kind: ExprKind::Neg(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                })
            }
            TokenKind::Tilde => {
                self.advance()?;
                let expr = self.unary()?;
                let start = token.start;
                let end = expr.end;
                Ok(Expr {
                    kind: ExprKind::BitNot(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                })
            }
            TokenKind::Not => {
                self.advance()?;
                let expr = self.unary()?;
                let start = token.start;
                let end = expr.end;
                Ok(Expr {
                    kind: ExprKind::Not(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                })
            }
            TokenKind::Ampersand => {
                self.advance()?;
                let expr = self.unary()?;
                let start = token.start;
                let end = expr.end;
                Ok(Expr {
                    kind: ExprKind::AddrOf(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                })
            }
            TokenKind::Star => {
                self.advance()?;
                let expr = self.unary()?;
                let start = token.start;
                let end = expr.end;
                Ok(Expr {
                    kind: ExprKind::Dereference(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                })
            }
            TokenKind::Increment => {
                self.advance()?;
                let expr = self.unary()?;
                let start = token.start;
                let end = expr.end;
                Ok(Expr {
                    kind: ExprKind::PreIncrement(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                })
            }
            TokenKind::Decrement => {
                self.advance()?;
                let expr = self.unary()?;
                let start = token.start;
                let end = expr.end;
                Ok(Expr {
                    kind: ExprKind::PreDecrement(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                })
            }
            TokenKind::Sizeof => {
                self.advance()?;
                let start = token.start;

                if self.is_sizeof_type()? {
                    self.expect(&TokenKind::LParen)?;
                    let (base_type, _) = self.parse_type_specifiers()?;
                    let mut ty = base_type;
                    while matches!(self.peek_kind(), Some(TokenKind::Star)) {
                        self.advance()?;
                        ty = Type::Pointer(Box::new(ty));
                    }
                    self.expect(&TokenKind::RParen)?;
                    let end = self.index;
                    return Ok(Expr {
                        kind: ExprKind::SizeOfType(ty.clone()),
                        start,
                        end,
                        source: self.source_slice(start, end),
                        r#type: Type::ULong,
                    });
                }

                let expr = self.unary()?;
                let end = expr.end;
                return Ok(Expr {
                    kind: ExprKind::SizeOf(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::ULong,
                });
            }
            _ => self.postfix(),
        }
    }

    fn postfix(&mut self) -> PResult<Expr> {
        let mut node = self.primary()?;

        loop {
            let token = self.peek()?;
            match token.kind {
                TokenKind::Increment => {
                    self.advance()?;
                    let start = node.start;
                    let end = self.index;
                    let source = self.source_slice(start, end);
                    let boxed = Box::new(node);
                    node = Expr {
                        kind: ExprKind::PostIncrement(boxed),
                        start,
                        end,
                        source,
                        r#type: Type::Int,
                    };
                }
                TokenKind::Decrement => {
                    self.advance()?;
                    let start = node.start;
                    let end = self.index;
                    let source = self.source_slice(start, end);
                    let boxed = Box::new(node);
                    node = Expr {
                        kind: ExprKind::PostDecrement(boxed),
                        start,
                        end,
                        source,
                        r#type: Type::Int,
                    };
                }
                TokenKind::LBracket => {
                    self.advance()?;
                    let index_expr = self.expr()?;
                    self.expect(&TokenKind::RBracket)?;
                    let start = node.start;
                    let end = self.index;
                    let source = self.source_slice(start, end);
                    let base_expr = Box::new(node);
                    let index_boxed = Box::new(index_expr);
                    let add_expr = Expr {
                        kind: ExprKind::Add(base_expr, index_boxed),
                        start,
                        end,
                        source: source.clone(),
                        r#type: Type::Int,
                    };
                    let deref_expr = Expr {
                        kind: ExprKind::Dereference(Box::new(add_expr)),
                        start,
                        end,
                        source: source.clone(),
                        r#type: Type::Int,
                    };
                    node = deref_expr;
                }
                TokenKind::LParen => {
                    let start = node.start;
                    let func_name = match &node.kind {
                        ExprKind::Var(name) => name.clone(),
                        kind => return Err(ParserError::InvalidFunctionCallTarget(kind.clone())),
                    };
                    self.advance()?;
                    let args = self.arguments()?;
                    self.expect(&TokenKind::RParen)?;
                    let end = self.index;
                    let source = self.source_slice(start, end);
                    node = Expr {
                        kind: ExprKind::FunctionCall(func_name, args),
                        start,
                        end,
                        source,
                        r#type: Type::Int,
                    };
                }
                _ => break,
            }
        }

        Ok(node)
    }

    fn primary(&mut self) -> PResult<Expr> {
        let Token {
            start,
            end,
            kind,
            source,
        } = self.peek()?;

        match kind.clone() {
            TokenKind::Integer(n) => {
                self.advance()?;
                Ok(Expr {
                    kind: ExprKind::Constant(Const::Int(n as i32)),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                })
            }
            TokenKind::LongInteger(n) => {
                self.advance()?;
                Ok(Expr {
                    kind: ExprKind::Constant(Const::Long(n)),
                    start,
                    end,
                    source,
                    r#type: Type::Long,
                })
            }
            TokenKind::UnsignedInteger(n) => {
                self.advance()?;
                Ok(Expr {
                    kind: ExprKind::Constant(Const::UInt(n)),
                    start,
                    end,
                    source,
                    r#type: Type::UInt,
                })
            }
            TokenKind::UnsignedLongInteger(n) => {
                self.advance()?;
                Ok(Expr {
                    kind: ExprKind::Constant(Const::ULong(n)),
                    start,
                    end,
                    source,
                    r#type: Type::ULong,
                })
            }
            TokenKind::Float(n) => {
                self.advance()?;
                Ok(Expr {
                    kind: ExprKind::Constant(Const::Double(n)),
                    start,
                    end,
                    source,
                    r#type: Type::Double,
                })
            }
            TokenKind::CharConstant(value) => {
                self.advance()?;
                Ok(Expr {
                    kind: ExprKind::Constant(Const::Char(value)),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                })
            }
            TokenKind::String(value) => {
                self.advance()?;
                let mut combined_value = value;
                let mut final_end = end;
                while self.pos < self.tokens.len() {
                    let next_token = self.peek()?;
                    if let TokenKind::String(next_value) = next_token.kind.clone() {
                        combined_value.push_str(&next_value);
                        final_end = next_token.end;
                        self.advance()?;
                    } else {
                        break;
                    }
                }
                let combined_source = self.source_slice(start, final_end);
                Ok(Expr {
                    kind: ExprKind::String(combined_value),
                    start,
                    end: final_end,
                    source: combined_source,
                    r#type: Type::Pointer(Box::new(Type::Char)),
                })
            }
            TokenKind::Identifier(name) => {
                self.advance()?;
                Ok(Expr {
                    kind: ExprKind::Var(name),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                })
            }
            TokenKind::LParen => {
                self.advance()?;
                let expr = self.expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(expr)
            }
            other => Err(ParserError::ExpectedPrimary(other)),
        }
    }

    fn is_cast_expression(&self) -> PResult<bool> {
        if !matches!(self.peek_kind(), Some(TokenKind::LParen)) {
            return Ok(false);
        }

        let mut clone = self.clone();
        clone.advance()?; // consume '('

        let parse_result = clone.parse_type_specifiers();
        let _ = match parse_result {
            Ok(res) => res,
            Err(ParserError::ExpectedDeclSpecifiers) => return Ok(false),
            Err(e) => return Err(e),
        };

        while matches!(clone.peek_kind(), Some(TokenKind::Star)) {
            clone.advance()?;
        }

        Ok(matches!(clone.peek_kind(), Some(TokenKind::RParen)))
    }

    fn is_sizeof_type(&self) -> PResult<bool> {
        if !matches!(self.peek_kind(), Some(TokenKind::LParen)) {
            return Ok(false);
        }

        let mut clone = self.clone();
        clone.advance()?; // consume '('

        let parse_result = clone.parse_type_specifiers();
        let _ = match parse_result {
            Ok(res) => res,
            Err(ParserError::ExpectedDeclSpecifiers) => return Ok(false),
            Err(e) => return Err(e),
        };

        while matches!(clone.peek_kind(), Some(TokenKind::Star)) {
            clone.advance()?;
        }

        Ok(matches!(clone.peek_kind(), Some(TokenKind::RParen)))
    }

    fn arguments(&mut self) -> PResult<Vec<Expr>> {
        let mut args = Vec::new();

        if matches!(self.peek_kind(), Some(TokenKind::RParen)) {
            return Ok(args);
        }

        loop {
            args.push(self.assign()?);
            if matches!(self.peek_kind(), Some(TokenKind::Comma)) {
                self.advance()?;
                continue;
            }
            break;
        }

        Ok(args)
    }

    fn is_declaration_start(&self) -> bool {
        if self.pos >= self.tokens.len() {
            return false;
        }
        matches!(
            self.tokens[self.pos].kind,
            TokenKind::Int
                | TokenKind::Long
                | TokenKind::Double
                | TokenKind::Char
                | TokenKind::Short
                | TokenKind::Unsigned
                | TokenKind::Signed
                | TokenKind::Void
                | TokenKind::Struct
                | TokenKind::Static
                | TokenKind::Extern
                | TokenKind::Const
        )
    }

    fn peek_kind(&self) -> Option<TokenKind> {
        self.tokens.get(self.pos).map(|t| t.kind.clone())
    }

    fn peek(&self) -> PResult<Token> {
        self.tokens
            .get(self.pos)
            .cloned()
            .ok_or(ParserError::UnexpectedEof(""))
    }

    fn advance(&mut self) -> PResult<()> {
        if self.pos >= self.tokens.len() {
            return Err(ParserError::UnexpectedEof(""));
        }
        self.index = self.peek()?.end;
        self.pos += 1;
        Ok(())
    }

    fn expect(&mut self, kind: &TokenKind) -> PResult<()> {
        let found = self.peek_kind();
        if found.as_ref() != Some(kind) {
            return Err(ParserError::ExpectedToken(kind.clone(), found));
        }
        self.advance()
    }

    fn ensure_done(&self) -> PResult<()> {
        if self.pos == self.tokens.len() {
            Ok(())
        } else {
            Err(ParserError::NotAtEnd(self.peek_kind()))
        }
    }

    fn source_slice(&self, start: usize, end: usize) -> String {
        String::from_utf8_lossy(&self.source[start..end]).to_string()
    }
}
