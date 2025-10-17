use logos::Logos;

#[derive(Debug, Clone, PartialEq, Logos)]
#[logos(skip r"[ \t\r\n\f]+")]
pub enum TokenKind {
    #[token("false", |_| false)]
    #[token("true", |_| true)]
    Bool(bool),

    #[token("{")]
    LBrace,

    #[token("}")]
    RBrace,

    #[token("[")]
    LBracket,

    #[token("]")]
    RBracket,

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token(":")]
    Colon,

    #[token(",")]
    Comma,

    #[token(";")]
    Semicolon,

    #[token("++")]
    Increment,

    #[token("+=")]
    PlusEqual,

    #[token("+")]
    Plus,

    #[token("--")]
    Decrement,

    #[token("-=")]
    MinusEqual,

    #[token("-")]
    Minus,

    #[token("*")]
    Star,

    #[token("*=")]
    StarEqual,

    #[token("/")]
    Slash,

    #[token("/=")]
    SlashEqual,

    #[token("%")]
    Percent,

    #[token("%=")]
    PercentEqual,

    #[token("~")]
    Tilde,

    #[token("<<=")]
    LShiftEqual,

    #[token(">>=")]
    RShiftEqual,

    #[token("<<")]
    LShift,

    #[token(">>")]
    RShift,

    #[token("&=")]
    AndEqual,

    #[token("&")]
    BitAnd,

    #[token("&&")]
    And,

    #[token("^=")]
    XorEqual,

    #[token("|")]
    BitOr,

    #[token("^")]
    Xor,

    #[token("|=")]
    OrEqual,

    #[token("!")]
    Not,

    #[token("||")]
    Or,

    #[token("=")]
    Equal,

    #[token("==")]
    DoubleEqual,

    #[token("!=")]
    NotEqual,

    #[token("<")]
    LessThan,

    #[token(">")]
    GreaterThan,

    #[token("<=")]
    LessThanEqual,

    #[token(">=")]
    GreaterThanEqual,

    #[token("void")]
    Void,

    #[token("int")]
    Int,

    #[token("return")]
    Return,

    #[token("if")]
    If,

    #[token("else")]
    Else,

    #[token("?")]
    Question,

    #[regex(r"([0-9]+)", |lex| lex.slice().parse::<i64>().unwrap(), priority = 5)]
    Integer(i64),

    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().unwrap(), priority = 4)]
    Float(f64),

    #[regex(r#"[a-zA-Z_]\w*"#, |lex| lex.slice().to_owned(), priority = 2)]
    Identifier(String),

    #[regex(r#""\w*""#, |lex| lex.slice().to_owned())]
    String(String),

    #[regex(r#"//.*"#, |lex| lex.slice().to_owned(), priority = 4)]
    Comment(String),

    #[regex(r#"\\\*.*\*\/"#, |lex| lex.slice().to_owned())]
    MultilineComment(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub start: usize,
    pub end: usize,
    pub source: String,
}

pub fn tokenize(source: &str) -> anyhow::Result<Vec<Token>> {
    let mut lexer = TokenKind::lexer(source);
    let mut tokens = vec![];

    while let Some(t) = lexer.next() {
        let kind = t.map_err(|e| anyhow::anyhow!("Error: {e:?}"))?;

        // ignore all comments
        if matches!(kind, TokenKind::Comment(_) | TokenKind::MultilineComment(_)) {
            continue;
        }

        let (start, end) = (lexer.span().start, lexer.span().end);
        let source = lexer.slice().to_string();
        tokens.push(Token {
            kind,
            start,
            end,
            source,
        });
    }

    Ok(tokens)
}
