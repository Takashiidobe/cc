use crate::tokenize::{Token, TokenKind};

pub type Expr = Node<ExprKind>;
pub type Stmt = Node<StmtKind>;

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Integer(i64),
    Neg(Box<Expr>),
    BitNot(Box<Expr>),

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
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Expr(Expr),
    Return(Expr),
    Block(Vec<Stmt>),
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
            stmts.push(self.stmt());
        }

        stmts
    }

    fn stmt(&mut self) -> Stmt {
        let Token {
            start,
            end,
            kind,
            source,
        } = self.peek();

        let res = match kind {
            TokenKind::Return => {
                self.advance();
                Stmt {
                    kind: StmtKind::Return(self.expr()),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                }
            }
            kind => panic!("Expected return, got {kind:?}"),
        };

        self.skip(&TokenKind::Semicolon);
        res
    }

    fn expr(&mut self) -> Expr {
        self.add()
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
        let Token {
            start,
            end,
            kind,
            source,
        } = self.peek();

        if matches!(kind, TokenKind::Plus | TokenKind::Minus | TokenKind::Tilde) {
            self.advance();

            if kind == TokenKind::Plus {
                return self.unary();
            }

            return Expr {
                kind: if kind == TokenKind::Minus {
                    ExprKind::Neg(Box::new(self.unary()))
                } else {
                    ExprKind::BitNot(Box::new(self.unary()))
                },
                start,
                end,
                source,
                r#type: Type::Int,
            };
        }

        self.primary()
    }

    fn primary(&mut self) -> Expr {
        let Token {
            start,
            end,
            kind,
            source,
        } = self.peek();

        match kind {
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
            TokenKind::LParen => {
                self.advance();
                let expr = self.expr();
                self.skip(&TokenKind::RParen);
                expr
            }
            kind => panic!("Expected primary, found {kind:?}"),
        }
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
            panic!("Expected {kind:?}");
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
