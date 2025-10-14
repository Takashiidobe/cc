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

    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Multiply,

    #[token("/")]
    Divide,

    #[token("void")]
    Void,

    #[token("int")]
    Int,

    #[token("return")]
    Return,

    #[regex(r"[1-9][0-9]*", |lex| lex.slice().parse::<i64>().unwrap(), priority = 5)]
    Integer(i64),

    #[regex(r"[1-9][0-9]*\.[0-9]*", |lex| lex.slice().parse::<f64>().unwrap(), priority = 4)]
    Float(f64),

    #[regex(r#"\w*"#, |lex| lex.slice().to_owned(), priority = 2)]
    Identifier(String),

    #[regex(r#""\w*""#, |lex| lex.slice().to_owned())]
    String(String),
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
