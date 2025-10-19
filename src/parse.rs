use crate::tokenize::{Token, TokenKind};
use std::mem;

pub type Expr = Node<ExprKind>;
pub type Stmt = Node<StmtKind>;

#[derive(Debug, Clone, PartialEq)]
pub enum StorageClass {
    Static,
    Extern,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Constant(Const),
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

    PreIncrement(Box<Expr>),
    PreDecrement(Box<Expr>),
    PostIncrement(Box<Expr>),
    PostDecrement(Box<Expr>),

    LeftShift(Box<Expr>, Box<Expr>),
    RightShift(Box<Expr>, Box<Expr>),
    Assignment(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariableDecl {
    pub name: String,
    pub init: Option<Expr>,
    pub storage_class: Option<StorageClass>,
    pub r#type: Type,
    pub is_definition: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParameterDecl {
    pub name: String,
    pub r#type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDecl {
    pub name: String,
    pub params: Vec<ParameterDecl>,
    pub body: Option<Vec<Stmt>>,
    pub storage_class: Option<StorageClass>,
    pub return_type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeclKind {
    Function(FunctionDecl),
    Variable(VariableDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decl {
    pub kind: DeclKind,
    pub start: usize,
    pub end: usize,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
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
pub enum ForInit {
    Declaration(Box<Stmt>),
    Expr(Option<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Long,
    UInt,
    ULong,
    Double,
    Void,
    Pointer(Box<Type>),
    FunType(Vec<Type>, Box<Type>),
}

fn is_plain_void(ty: &Type) -> bool {
    matches!(ty, Type::Void)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    Int(i64),
    Long(i64),
    UInt(u64),
    ULong(u64),
    Double(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node<Kind> {
    pub kind: Kind,
    pub start: usize,
    pub end: usize,
    pub source: String,
    pub r#type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parser {
    pub source: Vec<char>,
    pub index: usize,
    pub tokens: Vec<Token>,
    pub pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program(pub Vec<Decl>);

#[derive(Debug, Clone, PartialEq)]
struct ParsedDeclarator {
    name: String,
    type_expr: TypeExpr,
}

#[derive(Debug, Clone, PartialEq)]
enum TypeExpr {
    Base,
    Pointer(Box<TypeExpr>),
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
            other => TypeExpr::Pointer(Box::new(other)),
        }
    }

    fn add_function(self, params: Vec<ParameterDecl>) -> Self {
        match self {
            TypeExpr::Pointer(inner) => TypeExpr::Pointer(Box::new(inner.add_function(params))),
            other => TypeExpr::Function {
                params,
                ret: Box::new(other),
            },
        }
    }

    fn apply(&self, base: Type) -> Type {
        match self {
            TypeExpr::Base => base,
            TypeExpr::Pointer(inner) => Type::Pointer(Box::new(inner.apply(base))),
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
    saw_long: bool,
    saw_int: bool,
    signedness: Option<bool>, // Some(true) for signed, Some(false) for unsigned
    is_const: bool,
}

impl TypeSpecifierState {
    fn add(&mut self, kind: &TokenKind) {
        match kind {
            TokenKind::Void => {
                if self.saw_void {
                    panic!("duplicate 'void' specifier in declaration");
                }
                if self.saw_long || self.saw_int || self.signedness.is_some() || self.saw_double {
                    panic!("'void' cannot be combined with other type specifiers");
                }
                self.saw_void = true;
            }
            TokenKind::Double => {
                if self.saw_double {
                    panic!("duplicate 'double' specifier in declaration");
                }
                if self.saw_void || self.saw_long || self.saw_int || self.signedness.is_some() {
                    panic!("'double' cannot be combined with other type specifiers");
                }
                self.saw_double = true;
            }
            TokenKind::Long => {
                if self.saw_void || self.saw_double {
                    panic!("'long' cannot be combined with this type specifier");
                }
                if self.saw_long {
                    panic!("duplicate 'long' specifier in declaration");
                }
                self.saw_long = true;
            }
            TokenKind::Int => {
                if self.saw_void || self.saw_double {
                    panic!("'int' cannot be combined with this type specifier");
                }
                if self.saw_int {
                    panic!("duplicate 'int' specifier in declaration");
                }
                self.saw_int = true;
            }
            TokenKind::Signed => {
                if self.saw_double {
                    panic!("'signed' cannot be combined with 'double'");
                }
                if self.signedness == Some(false) {
                    panic!("conflicting 'signed' and 'unsigned' specifiers");
                }
                if self.signedness == Some(true) {
                    panic!("duplicate 'signed' specifier in declaration");
                }
                self.signedness = Some(true);
            }
            TokenKind::Unsigned => {
                if self.saw_double {
                    panic!("'unsigned' cannot be combined with 'double'");
                }
                if self.signedness == Some(true) {
                    panic!("conflicting 'signed' and 'unsigned' specifiers");
                }
                if self.signedness == Some(false) {
                    panic!("duplicate 'unsigned' specifier in declaration");
                }
                self.signedness = Some(false);
            }
            TokenKind::Const => {
                if self.is_const {
                    panic!("duplicate 'const' qualifier in declaration");
                }
                self.is_const = true;
            }
            _ => panic!("unsupported type specifier"),
        }
    }

    fn has_type_specifier(&self) -> bool {
        self.saw_void
            || self.saw_double
            || self.saw_long
            || self.saw_int
            || self.signedness.is_some()
    }

    fn resolve(&self) -> Type {
        if !self.has_type_specifier() {
            panic!("declaration missing type specifier");
        }

        if self.saw_void {
            return Type::Void;
        }

        if self.saw_double {
            return Type::Double;
        }

        let is_unsigned = matches!(self.signedness, Some(false));
        if self.saw_long {
            if is_unsigned { Type::ULong } else { Type::Long }
        } else {
            if is_unsigned { Type::UInt } else { Type::Int }
        }
    }
}

impl Parser {
    pub fn new(source: Vec<char>, tokens: Vec<Token>) -> Self {
        Self {
            source,
            tokens,
            index: 0,
            pos: 0,
        }
    }

    fn parse_specifiers(&mut self) -> (Type, Option<StorageClass>) {
        self.parse_specifiers_internal(true)
    }

    fn parse_type_specifiers(&mut self) -> Type {
        let (ty, storage) = self.parse_specifiers_internal(false);
        debug_assert!(storage.is_none());
        ty
    }

    fn parse_specifiers_internal(&mut self, allow_storage: bool) -> (Type, Option<StorageClass>) {
        let mut storage: Option<StorageClass> = None;
        let mut state = TypeSpecifierState::default();
        let mut consumed_any = false;

        while self.pos < self.tokens.len() {
            let kind = self.peek().kind.clone();
            match kind {
                TokenKind::Static => {
                    if !allow_storage {
                        panic!("storage class specifiers are not allowed here");
                    }
                    if storage.is_some() {
                        panic!("multiple storage class specifiers in declaration");
                    }
                    self.advance();
                    storage = Some(StorageClass::Static);
                    consumed_any = true;
                }
                TokenKind::Extern => {
                    if !allow_storage {
                        panic!("storage class specifiers are not allowed here");
                    }
                    if storage.is_some() {
                        panic!("multiple storage class specifiers in declaration");
                    }
                    self.advance();
                    storage = Some(StorageClass::Extern);
                    consumed_any = true;
                }
                TokenKind::Int
                | TokenKind::Long
                | TokenKind::Void
                | TokenKind::Signed
                | TokenKind::Unsigned
                | TokenKind::Double
                | TokenKind::Const => {
                    self.advance();
                    state.add(&kind);
                    consumed_any = true;
                }
                _ => break,
            }
        }

        if !consumed_any {
            panic!("expected declaration specifiers");
        }

        let ty = state.resolve();
        (ty, storage)
    }

    fn parse_declarator(&mut self) -> ParsedDeclarator {
        dbg!(self.peek());
        if self.peek().kind == TokenKind::Star {
            self.advance();
            let mut inner = self.parse_declarator();
            let current = mem::replace(&mut inner.type_expr, TypeExpr::Base);
            inner.type_expr = current.add_pointer();
            inner
        } else {
            self.parse_direct_declarator()
        }
    }

    fn parse_direct_declarator(&mut self) -> ParsedDeclarator {
        let mut declarator = if let TokenKind::Identifier(name) = self.peek().kind.clone() {
            self.advance();
            ParsedDeclarator {
                name,
                type_expr: TypeExpr::Base,
            }
        } else if self.peek().kind == TokenKind::LParen {
            self.advance();
            let declarator = self.parse_declarator();
            self.skip(&TokenKind::RParen);
            declarator
        } else {
            panic!(
                "expected identifier or '(' in declarator, found {:?}",
                self.peek()
            );
        };

        while self.pos < self.tokens.len() && self.peek().kind == TokenKind::LParen {
            self.advance();
            let params = self.parse_parameter_list();
            self.skip(&TokenKind::RParen);
            let current = mem::replace(&mut declarator.type_expr, TypeExpr::Base);
            declarator.type_expr = current.add_function(params);
        }

        declarator
    }

    fn parse_parameter_list(&mut self) -> Vec<ParameterDecl> {
        if self.peek().kind == TokenKind::RParen {
            return Vec::new();
        }

        if self.peek().kind == TokenKind::Void {
            if self.pos + 1 < self.tokens.len()
                && self.tokens[self.pos + 1].kind == TokenKind::RParen
            {
                self.advance();
                return Vec::new();
            }
        }

        let mut params = Vec::new();

        loop {
            let param = self.parse_parameter();
            if is_plain_void(&param.r#type) {
                panic!("'void' parameter must be the only parameter");
            }
            params.push(param);

            if self.peek().kind == TokenKind::Comma {
                self.advance();
                continue;
            }
            break;
        }

        params
    }

    fn parse_parameter(&mut self) -> ParameterDecl {
        let base_type = self.parse_type_specifiers();
        let declarator = self.parse_declarator();
        let param_type = declarator.type_expr.apply(base_type);
        ParameterDecl {
            name: declarator.name,
            r#type: param_type,
        }
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
                | TokenKind::Unsigned
                | TokenKind::Signed
                | TokenKind::Void
                | TokenKind::Static
                | TokenKind::Extern
                | TokenKind::Const
        )
    }

    fn variable_declaration_stmt(&mut self) -> Stmt {
        let Token { start, .. } = self.peek();
        let (base_type, storage_class) = self.parse_specifiers();
        let declarator = self.parse_declarator();
        let name = declarator.name.clone();
        let var_type = declarator.type_expr.apply(base_type);
        if is_plain_void(&var_type) {
            panic!("variable declared with void type");
        }
        if matches!(var_type, Type::FunType(_, _)) {
            panic!("function declarations are not allowed in block scope");
        }
        let init = if self.peek().kind == TokenKind::Equal {
            self.advance();
            Some(self.expr())
        } else {
            None
        };

        self.skip(&TokenKind::Semicolon);

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

        Stmt {
            kind: StmtKind::Declaration(decl),
            start,
            end,
            source,
            r#type: var_type,
        }
    }

    pub fn parse(&mut self) -> Program {
        let mut decls = Vec::new();
        while self.pos < self.tokens.len() {
            decls.push(self.declaration());
        }
        self.ensure_done();
        Program(decls)
    }

    fn declaration(&mut self) -> Decl {
        if self.pos >= self.tokens.len() {
            panic!("Unexpected end of input while parsing declaration");
        }

        let Token { start, .. } = self.peek();
        let (base_type, storage_class) = self.parse_specifiers();
        let declarator = self.parse_declarator();
        let name = declarator.name.clone();

        match declarator.type_expr {
            TypeExpr::Function { params, ret } => {
                let return_type = ret.apply(base_type);
                self.parse_function_declaration(start, name, params, return_type, storage_class)
            }
            type_expr => {
                let var_type = type_expr.apply(base_type);
                if is_plain_void(&var_type) {
                    panic!("variable '{name}' declared with void type");
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
    ) -> Decl {
        let body = if self.pos < self.tokens.len() && self.peek().kind == TokenKind::LBrace {
            self.advance();
            let stmts = self.stmts();
            self.skip(&TokenKind::RBrace);
            Some(stmts)
        } else {
            self.skip(&TokenKind::Semicolon);
            None
        };

        let end = self.index;
        let source = self.source_slice(start, end);

        Decl {
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
        }
    }

    fn parse_variable_declaration(
        &mut self,
        start: usize,
        name: String,
        ty: Type,
        storage_class: Option<StorageClass>,
    ) -> Decl {
        if matches!(ty, Type::FunType(_, _)) {
            panic!("variable '{name}' declared with function type");
        }

        let init = if self.peek().kind == TokenKind::Equal {
            self.advance();
            Some(self.expr())
        } else {
            None
        };

        self.skip(&TokenKind::Semicolon);

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

        Decl {
            kind: DeclKind::Variable(decl),
            start,
            end,
            source,
        }
    }

    fn stmts(&mut self) -> Vec<Stmt> {
        let mut stmts = vec![];

        while self.peek().kind != TokenKind::RBrace {
            if self.is_declaration_start() {
                stmts.push(self.variable_declaration_stmt());
            } else {
                stmts.push(self.stmt());
            }
        }

        stmts
    }

    fn stmt(&mut self) -> Stmt {
        let token = self.peek();
        match token.kind {
            TokenKind::Return => {
                self.advance();
                let expr = self.expr();
                self.skip(&TokenKind::Semicolon);
                Stmt {
                    kind: StmtKind::Return(expr),
                    start: token.start,
                    end: self.index,
                    source: token.source,
                    r#type: Type::Int,
                }
            }
            TokenKind::LBrace => self.block(),
            TokenKind::Semicolon => {
                self.advance();
                Stmt {
                    kind: StmtKind::Null,
                    start: token.start,
                    end: token.end,
                    source: token.source,
                    r#type: Type::Void,
                }
            }
            TokenKind::If => self.if_stmt(),
            TokenKind::While => self.while_stmt(),
            TokenKind::Do => self.do_while_stmt(),
            TokenKind::For => self.for_stmt(),
            TokenKind::Break => self.break_stmt(),
            TokenKind::Continue => self.continue_stmt(),
            _ => {
                let expr = self.expr();
                self.skip(&TokenKind::Semicolon);
                let start = expr.start;
                let end = expr.end;
                let source = expr.source.clone();
                Stmt {
                    kind: StmtKind::Expr(expr),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                }
            }
        }
    }

    fn break_stmt(&mut self) -> Stmt {
        let Token { start, .. } = self.peek();
        self.advance();
        self.skip(&TokenKind::Semicolon);
        let end = self.index;
        Stmt {
            kind: StmtKind::Break { loop_id: None },
            start,
            end,
            source: self.source_slice(start, end),
            r#type: Type::Void,
        }
    }

    fn continue_stmt(&mut self) -> Stmt {
        let Token { start, .. } = self.peek();
        self.advance();
        self.skip(&TokenKind::Semicolon);
        let end = self.index;
        Stmt {
            kind: StmtKind::Continue { loop_id: None },
            start,
            end,
            source: self.source_slice(start, end),
            r#type: Type::Void,
        }
    }

    fn block(&mut self) -> Stmt {
        let Token { start, .. } = self.peek();
        self.skip(&TokenKind::LBrace);
        let stmts = self.stmts();
        self.skip(&TokenKind::RBrace);
        let end = self.index;
        let source: String = self.source[start..end].iter().collect();
        Stmt {
            kind: StmtKind::Compound(stmts),
            start,
            end,
            source,
            r#type: Type::Void,
        }
    }

    fn while_stmt(&mut self) -> Stmt {
        let Token { start, .. } = self.peek();
        self.advance(); // consume while
        self.skip(&TokenKind::LParen);
        let condition = self.expr();
        self.skip(&TokenKind::RParen);
        let body_stmt = self.stmt();
        let end = body_stmt.end;
        Stmt {
            kind: StmtKind::While {
                condition,
                body: Box::new(body_stmt),
                loop_id: None,
            },
            start,
            end,
            source: self.source_slice(start, end),
            r#type: Type::Void,
        }
    }

    fn do_while_stmt(&mut self) -> Stmt {
        let Token { start, .. } = self.peek();
        self.advance(); // consume do
        let body_stmt = self.stmt();
        self.skip(&TokenKind::While);
        self.skip(&TokenKind::LParen);
        let condition = self.expr();
        self.skip(&TokenKind::RParen);
        self.skip(&TokenKind::Semicolon);
        let end = self.index;
        Stmt {
            kind: StmtKind::DoWhile {
                body: Box::new(body_stmt),
                condition,
                loop_id: None,
            },
            start,
            end,
            source: self.source_slice(start, end),
            r#type: Type::Void,
        }
    }

    fn for_stmt(&mut self) -> Stmt {
        let Token { start, .. } = self.peek();
        self.advance(); // consume for
        self.skip(&TokenKind::LParen);

        let init = if self.peek().kind == TokenKind::Semicolon {
            self.advance();
            ForInit::Expr(None)
        } else if self.is_declaration_start() {
            let decl = self.variable_declaration_stmt();
            ForInit::Declaration(Box::new(decl))
        } else {
            let expr = self.expr();
            self.skip(&TokenKind::Semicolon);
            ForInit::Expr(Some(expr))
        };

        let condition = if self.peek().kind == TokenKind::Semicolon {
            self.advance();
            None
        } else {
            let expr = self.expr();
            self.skip(&TokenKind::Semicolon);
            Some(expr)
        };

        let post = if self.peek().kind == TokenKind::RParen {
            None
        } else {
            Some(self.expr())
        };

        self.skip(&TokenKind::RParen);
        let body_stmt = self.stmt();
        let end = body_stmt.end;

        Stmt {
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
        }
    }

    fn if_stmt(&mut self) -> Stmt {
        let Token { start, .. } = self.peek();
        self.advance(); // consume if
        self.skip(&TokenKind::LParen);
        let condition = self.expr();
        self.skip(&TokenKind::RParen);
        let then_branch = self.stmt();
        let else_branch = if self.peek().kind == TokenKind::Else {
            self.advance();
            Some(Box::new(self.stmt()))
        } else {
            None
        };
        let end = if let Some(ref else_stmt) = else_branch {
            else_stmt.end
        } else {
            then_branch.end
        };
        let source = self.source_slice(start, end);
        Stmt {
            kind: StmtKind::If {
                condition,
                then_branch: Box::new(then_branch),
                else_branch,
            },
            start,
            end,
            source,
            r#type: Type::Void,
        }
    }

    fn expr(&mut self) -> Expr {
        self.assign()
    }

    fn assign(&mut self) -> Expr {
        let lhs = self.conditional();

        let token_kind = self.peek().kind.clone();
        match token_kind {
            TokenKind::Equal => {
                self.advance();
                let rhs = self.assign();
                self.make_assignment_expr(lhs, rhs)
            }
            TokenKind::PlusEqual
            | TokenKind::MinusEqual
            | TokenKind::StarEqual
            | TokenKind::SlashEqual
            | TokenKind::PercentEqual
            | TokenKind::AmpersandEqual
            | TokenKind::OrEqual
            | TokenKind::XorEqual
            | TokenKind::LShiftEqual
            | TokenKind::RShiftEqual => {
                self.advance();
                let rhs = self.assign();
                let compound_rhs = self.compound_assignment_rhs(lhs.clone(), rhs, &token_kind);
                self.make_assignment_expr(lhs, compound_rhs)
            }
            _ => lhs,
        }
    }

    fn conditional(&mut self) -> Expr {
        let condition = self.or();

        if self.peek().kind == TokenKind::Question {
            self.advance();
            let then_expr = self.assign();
            self.skip(&TokenKind::Colon);
            let else_expr = self.conditional();
            let start = condition.start;
            let end = else_expr.end;
            Expr {
                kind: ExprKind::Conditional(
                    Box::new(condition),
                    Box::new(then_expr),
                    Box::new(else_expr),
                ),
                start,
                end,
                source: self.source_slice(start, end),
                r#type: Type::Int,
            }
        } else {
            condition
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

    fn compound_assignment_rhs(&self, lhs: Expr, rhs: Expr, op: &TokenKind) -> Expr {
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
            kind => panic!("Unsupported compound assignment token: {kind:?}"),
        };

        Expr {
            kind,
            start,
            end,
            source,
            r#type: Type::Int,
        }
    }

    fn or(&mut self) -> Expr {
        let mut node = self.and();

        #[allow(irrefutable_let_patterns)]
        while let Token {
            kind,
            start,
            end,
            source,
        } = self.peek()
        {
            if !matches!(kind, TokenKind::Or) {
                break;
            }

            self.advance();

            node = Expr {
                kind: ExprKind::Or(Box::new(node), Box::new(self.and())),
                start,
                end,
                source: source.clone(),
                r#type: Type::Int,
            }
        }

        node
    }

    fn and(&mut self) -> Expr {
        let mut node = self.bit_or();

        #[allow(irrefutable_let_patterns)]
        while let Token {
            kind,
            start,
            end,
            source,
        } = self.peek()
        {
            if !matches!(kind, TokenKind::DoubleAmpersand) {
                break;
            }

            self.advance();

            node = Expr {
                kind: ExprKind::And(Box::new(node), Box::new(self.bit_or())),
                start,
                end,
                source: source.clone(),
                r#type: Type::Int,
            }
        }

        node
    }

    fn bit_or(&mut self) -> Expr {
        let mut node = self.xor();

        #[allow(irrefutable_let_patterns)]
        while let Token {
            kind,
            start,
            end,
            source,
        } = self.peek()
        {
            if !matches!(kind, TokenKind::BitOr) {
                break;
            }

            self.advance();

            node = Expr {
                kind: ExprKind::BitOr(Box::new(node), Box::new(self.xor())),
                start,
                end,
                source: source.clone(),
                r#type: Type::Int,
            }
        }

        node
    }

    fn xor(&mut self) -> Expr {
        let mut node = self.bit_and();

        #[allow(irrefutable_let_patterns)]
        while let Token {
            kind,
            start,
            end,
            source,
        } = self.peek()
        {
            if !matches!(kind, TokenKind::Xor) {
                break;
            }

            self.advance();

            node = Expr {
                kind: ExprKind::Xor(Box::new(node), Box::new(self.bit_and())),
                start,
                end,
                source: source.clone(),
                r#type: Type::Int,
            }
        }

        node
    }

    fn bit_and(&mut self) -> Expr {
        let mut node = self.eq();

        #[allow(irrefutable_let_patterns)]
        while let Token {
            kind,
            start,
            end,
            source,
        } = self.peek()
        {
            if !matches!(kind, TokenKind::Ampersand) {
                break;
            }

            self.advance();

            node = Expr {
                kind: ExprKind::BitAnd(Box::new(node), Box::new(self.eq())),
                start,
                end,
                source: source.clone(),
                r#type: Type::Int,
            }
        }

        node
    }

    fn eq(&mut self) -> Expr {
        let mut node = self.rel();

        #[allow(irrefutable_let_patterns)]
        while let Token {
            kind,
            start,
            end,
            source,
        } = self.peek()
        {
            if !matches!(kind, TokenKind::DoubleEqual | TokenKind::NotEqual) {
                break;
            }

            self.advance();

            node = Expr {
                kind: if kind == TokenKind::DoubleEqual {
                    ExprKind::Equal(Box::new(node), Box::new(self.rel()))
                } else {
                    ExprKind::NotEqual(Box::new(node), Box::new(self.rel()))
                },
                start,
                end,
                source: source.clone(),
                r#type: Type::Int,
            }
        }

        node
    }

    fn rel(&mut self) -> Expr {
        let mut node = self.shift();

        #[allow(irrefutable_let_patterns)]
        while let Token {
            kind,
            start,
            end,
            source,
        } = self.peek()
        {
            if !matches!(
                kind,
                TokenKind::LessThan
                    | TokenKind::LessThanEqual
                    | TokenKind::GreaterThan
                    | TokenKind::GreaterThanEqual
            ) {
                break;
            }

            self.advance();

            node = Expr {
                kind: if kind == TokenKind::LessThan {
                    ExprKind::LessThan(Box::new(node), Box::new(self.shift()))
                } else if kind == TokenKind::LessThanEqual {
                    ExprKind::LessThanEqual(Box::new(node), Box::new(self.shift()))
                } else if kind == TokenKind::GreaterThan {
                    ExprKind::GreaterThan(Box::new(node), Box::new(self.shift()))
                } else {
                    ExprKind::GreaterThanEqual(Box::new(node), Box::new(self.shift()))
                },
                start,
                end,
                source: source.clone(),
                r#type: Type::Int,
            }
        }

        node
    }

    fn shift(&mut self) -> Expr {
        let mut node = self.add();

        #[allow(irrefutable_let_patterns)]
        while let Token {
            kind,
            start,
            end,
            source,
        } = self.peek()
        {
            if !matches!(kind, TokenKind::LShift | TokenKind::RShift) {
                break;
            }

            self.advance();

            node = Expr {
                kind: if kind == TokenKind::LShift {
                    ExprKind::LeftShift(Box::new(node), Box::new(self.add()))
                } else {
                    ExprKind::RightShift(Box::new(node), Box::new(self.add()))
                },
                start,
                end,
                source: source.clone(),
                r#type: Type::Int,
            }
        }

        node
    }

    fn add(&mut self) -> Expr {
        let mut node = self.mul();

        #[allow(irrefutable_let_patterns)]
        while let Token {
            kind,
            start,
            end,
            source,
        } = self.peek()
        {
            if !matches!(kind, TokenKind::Plus | TokenKind::Minus) {
                break;
            }

            self.advance();

            node = Expr {
                kind: if kind == TokenKind::Plus {
                    ExprKind::Add(Box::new(node), Box::new(self.mul()))
                } else {
                    ExprKind::Sub(Box::new(node), Box::new(self.mul()))
                },
                start,
                end,
                source: source.clone(),
                r#type: Type::Int,
            }
        }

        node
    }

    fn mul(&mut self) -> Expr {
        let mut node = self.unary();

        #[allow(irrefutable_let_patterns)]
        while let Token {
            kind,
            start,
            end,
            source,
        } = self.peek()
        {
            if !matches!(
                kind,
                TokenKind::Star | TokenKind::Slash | TokenKind::Percent
            ) {
                break;
            }

            self.advance();

            node = Expr {
                kind: if kind == TokenKind::Star {
                    ExprKind::Mul(Box::new(node), Box::new(self.unary()))
                } else if kind == TokenKind::Percent {
                    ExprKind::Rem(Box::new(node), Box::new(self.unary()))
                } else {
                    ExprKind::Div(Box::new(node), Box::new(self.unary()))
                },
                start,
                end,
                source: source.clone(),
                r#type: Type::Int,
            }
        }

        node
    }

    fn unary(&mut self) -> Expr {
        if self.peek().kind == TokenKind::LParen && self.is_cast_expression() {
            let Token { start, .. } = self.peek();
            self.advance(); // consume '('
            let base_type = self.parse_type_specifiers();
            let mut cast_type = base_type;
            while self.peek().kind == TokenKind::Star {
                self.advance();
                cast_type = Type::Pointer(Box::new(cast_type));
            }
            if is_plain_void(&cast_type) {
                panic!("Unsupported cast target: void");
            }
            self.skip(&TokenKind::RParen);
            let expr = self.unary();
            let end = expr.end;
            return Expr {
                kind: ExprKind::Cast(cast_type.clone(), Box::new(expr)),
                start,
                end,
                source: self.source_slice(start, end),
                r#type: cast_type,
            };
        }

        let token = self.peek();
        match token.kind.clone() {
            TokenKind::Plus => {
                self.advance();
                self.unary()
            }
            TokenKind::Minus => {
                self.advance();
                let expr = self.unary();
                let start = token.start;
                let end = expr.end;
                Expr {
                    kind: ExprKind::Neg(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                }
            }
            TokenKind::Tilde => {
                self.advance();
                let expr = self.unary();
                let start = token.start;
                let end = expr.end;
                Expr {
                    kind: ExprKind::BitNot(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                }
            }
            TokenKind::Not => {
                self.advance();
                let expr = self.unary();
                let start = token.start;
                let end = expr.end;
                Expr {
                    kind: ExprKind::Not(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                }
            }
            TokenKind::Ampersand => {
                self.advance();
                let expr = self.unary();
                let start = token.start;
                let end = expr.end;
                Expr {
                    kind: ExprKind::AddrOf(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                }
            }
            TokenKind::Star => {
                self.advance();
                let expr = self.unary();
                let start = token.start;
                let end = expr.end;
                Expr {
                    kind: ExprKind::Dereference(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                }
            }
            TokenKind::Increment => {
                self.advance();
                let expr = self.unary();
                let start = token.start;
                let end = expr.end;
                Expr {
                    kind: ExprKind::PreIncrement(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                }
            }
            TokenKind::Decrement => {
                self.advance();
                let expr = self.unary();
                let start = token.start;
                let end = expr.end;
                Expr {
                    kind: ExprKind::PreDecrement(Box::new(expr)),
                    start,
                    end,
                    source: self.source_slice(start, end),
                    r#type: Type::Int,
                }
            }
            _ => self.postfix(),
        }
    }

    fn postfix(&mut self) -> Expr {
        let mut node = self.primary();

        loop {
            let token = self.peek();
            match token.kind {
                TokenKind::Increment => {
                    self.advance();
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
                    self.advance();
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
                TokenKind::LParen => {
                    let start = node.start;
                    let func_name = match &node.kind {
                        ExprKind::Var(name) => name.clone(),
                        kind => panic!("Invalid function call target: {kind:?}"),
                    };
                    self.advance();
                    let args = self.arguments();
                    self.skip(&TokenKind::RParen);
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

        node
    }

    fn primary(&mut self) -> Expr {
        let Token {
            start,
            end,
            kind,
            source,
        } = self.peek();

        match kind.clone() {
            TokenKind::Integer(n) => {
                self.advance();
                Expr {
                    kind: ExprKind::Constant(Const::Int(n)),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                }
            }
            TokenKind::LongInteger(n) => {
                self.advance();
                Expr {
                    kind: ExprKind::Constant(Const::Long(n)),
                    start,
                    end,
                    source,
                    r#type: Type::Long,
                }
            }
            TokenKind::UnsignedInteger(n) => {
                self.advance();
                Expr {
                    kind: ExprKind::Constant(Const::UInt(n)),
                    start,
                    end,
                    source,
                    r#type: Type::UInt,
                }
            }
            TokenKind::UnsignedLongInteger(n) => {
                self.advance();
                Expr {
                    kind: ExprKind::Constant(Const::ULong(n)),
                    start,
                    end,
                    source,
                    r#type: Type::ULong,
                }
            }
            TokenKind::Float(n) => {
                self.advance();
                Expr {
                    kind: ExprKind::Constant(Const::Double(n)),
                    start,
                    end,
                    source,
                    r#type: Type::Double,
                }
            }
            TokenKind::Identifier(name) => {
                self.advance();
                Expr {
                    kind: ExprKind::Var(name),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                }
            }
            TokenKind::LParen => {
                self.advance();
                let expr = self.expr();
                self.skip(&TokenKind::RParen);
                expr
            }
            kind => panic!("Expected primary, found {kind:?}"),
        }
    }

    fn is_cast_expression(&self) -> bool {
        if self.pos + 1 >= self.tokens.len() {
            return false;
        }

        let mut idx = self.pos + 1;
        let mut state = TypeSpecifierState::default();
        let mut consumed_any = false;

        while idx < self.tokens.len() {
            let kind = &self.tokens[idx].kind;
            match kind {
                TokenKind::Int
                | TokenKind::Long
                | TokenKind::Void
                | TokenKind::Signed
                | TokenKind::Unsigned
                | TokenKind::Double
                | TokenKind::Const => {
                    state.add(kind);
                    consumed_any = true;
                    idx += 1;
                }
                TokenKind::Star => {
                    if !consumed_any {
                        return false;
                    }
                    idx += 1;
                }
                TokenKind::RParen => {
                    if !consumed_any {
                        return false;
                    }
                    // ensure the specifier combination is valid
                    let _ = state.resolve();
                    return true;
                }
                _ => return false,
            }
        }

        false
    }

    fn source_slice(&self, start: usize, end: usize) -> String {
        self.source[start..end].iter().collect()
    }

    fn arguments(&mut self) -> Vec<Expr> {
        let mut args = Vec::new();

        if self.peek().kind == TokenKind::RParen {
            return args;
        }

        loop {
            args.push(self.assign());
            if self.peek().kind == TokenKind::Comma {
                self.advance();
                continue;
            }
            break;
        }

        args
    }

    fn peek(&self) -> Token {
        self.tokens[self.pos].clone()
    }

    fn advance(&mut self) {
        if self.pos >= self.tokens.len() {
            panic!("unexpected end of file");
        }

        self.index = self.peek().end;
        self.pos += 1;
    }

    fn skip(&mut self, kind: &TokenKind) {
        if !self.r#match(kind) {
            panic!("Expected {kind:?}, got {:?}", self.peek());
        }

        self.advance();
    }

    fn ensure_done(&self) {
        if self.pos == self.tokens.len() {
            return;
        }
        panic!("Not at end of program");
    }

    fn r#match(&self, kind: &TokenKind) -> bool {
        let peeked = self.peek();
        peeked.kind == *kind
    }
}
