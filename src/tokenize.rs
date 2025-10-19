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

    #[token("&")]
    Ampersand,

    #[token("&&")]
    DoubleAmpersand,

    #[token("&=")]
    AmpersandEqual,

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

    #[token("double")]
    Double,

    #[token("long")]
    Long,

    #[token("unsigned")]
    Unsigned,

    #[token("signed")]
    Signed,

    #[token("const")]
    Const,

    #[token("static")]
    Static,

    #[token("extern")]
    Extern,

    #[token("return")]
    Return,

    #[token("if")]
    If,

    #[token("else")]
    Else,

    #[token("while")]
    While,

    #[token("do")]
    Do,

    #[token("for")]
    For,

    #[token("break")]
    Break,

    #[token("continue")]
    Continue,

    #[token("?")]
    Question,

    #[regex(r"([0-9]+)", |lex| lex.slice().parse::<i64>().unwrap(), priority = 5)]
    Integer(i64),

    #[regex(
        r"([0-9]+)[uU]",
        |lex| lex.slice()[..lex.slice().len() - 1].parse::<u64>().unwrap(),
        priority = 7
    )]
    UnsignedInteger(u64),

    #[regex(
        r"([0-9]+)[lL]",
        |lex| lex.slice()[..lex.slice().len() - 1].parse::<i64>().unwrap(),
        priority = 6
    )]
    LongInteger(i64),

    #[regex(
        r"([0-9]+)([lL][uU]|[uU][lL])",
        |lex| lex.slice()[..lex.slice().len() - 2].parse::<u64>().unwrap(),
        priority = 8
    )]
    UnsignedLongInteger(u64),

    #[regex(
        r"([0-9]*\.[0-9]+([Ee][+-]?[0-9]+)?|[0-9]+\.([Ee][+-]?[0-9]+)?|[0-9]+[Ee][+-]?[0-9]+)",
        |lex| {
            lex.slice()
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("invalid floating literal: {}", lex.slice()))
        },
        priority = 4
    )]
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
