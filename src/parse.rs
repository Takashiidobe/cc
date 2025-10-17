use crate::tokenize::{Token, TokenKind};

pub type Expr = Node<ExprKind>;
pub type Stmt = Node<StmtKind>;

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Integer(i64),
    Var(String),
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
pub enum StmtKind {
    Expr(Expr),
    Return(Expr),
    Compound(Vec<Stmt>),
    Declaration {
        name: String,
        init: Option<Expr>,
    },
    Null,
    If {
        condition: Expr,
        then_branch: Box<Stmt>,
        else_branch: Option<Box<Stmt>>,
    },
    // return type, name, args (not yet), Body
    FnDecl(String, Vec<Stmt>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Void,
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
pub struct Program(pub Stmt);

impl Parser {
    pub fn new(source: Vec<char>, tokens: Vec<Token>) -> Self {
        Self {
            source,
            tokens,
            index: 0,
            pos: 0,
        }
    }

    pub fn parse(&mut self) -> Program {
        let res = self.main();
        self.ensure_done();
        Program(res)
    }

    // int main(void) { Vec<Stmt> }
    fn main(&mut self) -> Stmt {
        let start = self.index;

        // match type int
        self.skip(&TokenKind::Int);
        // match identifier main
        self.skip(&TokenKind::Identifier("main".to_string()));
        // match lparen
        self.skip(&TokenKind::LParen);
        // match type void
        self.skip(&TokenKind::Void);
        // match rparen
        self.skip(&TokenKind::RParen);
        // match lbrace
        self.skip(&TokenKind::LBrace);
        // get all statements
        let stmts = self.stmts();
        // match rbrace
        self.skip(&TokenKind::RBrace);

        let end = self.index;

        let source: String = self.source[start..end].iter().collect();

        Stmt {
            kind: StmtKind::FnDecl("main".to_string(), stmts),
            start,
            end,
            source,
            r#type: Type::Int,
        }
    }

    fn stmts(&mut self) -> Vec<Stmt> {
        let mut stmts = vec![];

        while self.peek().kind != TokenKind::RBrace {
            if self.peek().kind == TokenKind::Int {
                stmts.push(self.declaration());
            } else {
                stmts.push(self.stmt());
            }
        }

        stmts
    }

    fn declaration(&mut self) -> Stmt {
        let Token { start, .. } = self.peek();
        self.skip(&TokenKind::Int);

        let name = match self.peek().kind.clone() {
            TokenKind::Identifier(name) => {
                self.advance();
                name
            }
            kind => panic!("Expected identifier in declaration, found {kind:?}"),
        };

        let init = if self.peek().kind == TokenKind::Equal {
            self.advance();
            Some(self.expr())
        } else {
            None
        };

        self.skip(&TokenKind::Semicolon);

        let end = self.index;
        let source: String = self.source[start..end].iter().collect();

        Stmt {
            kind: StmtKind::Declaration { name, init },
            start,
            end,
            source,
            r#type: Type::Int,
        }
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
            | TokenKind::AndEqual
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
            TokenKind::AndEqual => ExprKind::BitAnd(Box::new(lhs), Box::new(rhs)),
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
            if !matches!(kind, TokenKind::And) {
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
            if !matches!(kind, TokenKind::BitAnd) {
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
                    kind: ExprKind::Integer(n),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
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

    fn source_slice(&self, start: usize, end: usize) -> String {
        self.source[start..end].iter().collect()
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
