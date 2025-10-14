use crate::tokenize::{Token, TokenKind};

pub type Expr = Node<ExprKind>;
pub type Stmt = Node<StmtKind>;

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Integer(i64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
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
pub struct Program(Stmt);

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
        self.stmts();
        // match rbrace
        self.skip(&TokenKind::RBrace);

        let end = self.index;

        let source: String = self.source[start..end].iter().collect();

        Stmt {
            kind: StmtKind::FnDecl("main".to_string(), vec![]),
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
                    kind: StmtKind::Return(self.primary()),
                    start,
                    end,
                    source,
                    r#type: Type::Int,
                }
            }
            _ => panic!("Expected expression"),
        };

        self.skip(&TokenKind::Semicolon);
        res
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
            _ => panic!("Expected expression"),
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
