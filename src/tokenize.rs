use anyhow::Result;
use logos::Logos;
use thiserror::Error;

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

    #[token("char")]
    Char,

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

    #[regex(r#"'([^'\\\n]|\\['\"?\\abfnrtv])'"#, |lex| parse_char_literal(lex.slice()).unwrap())]
    CharConstant(i8),

    #[regex(r#"[a-zA-Z_]\w*"#, |lex| lex.slice().to_owned(), priority = 2)]
    Identifier(String),

    #[regex(r#""([^"\\\n]|\\['"\\?abfnrtv])*""#, |lex| parse_string_literal(lex.slice()).unwrap())]
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

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("unsupported escape `{0}` at byte {1}")]
    UnsupportedEscape(char, usize),

    #[error("invalid character `{0}`")]
    InvalidCharacter(char),

    #[error("invalid string literal: {0:?}")]
    InvalidStringLiteral(String),

    #[error("unterminated string literal: {0:?}")]
    UnterminatedStringLiteral(String),

    #[error("truncated escape at byte {0} in {1:?}")]
    TruncatedEscapeSequence(usize, String),

    #[error("invalid character literal: {0:?}")]
    InvalidCharLiteral(String),

    #[error("unterminated character literal: {0:?}")]
    UnterminatedCharLiteral(String),

    #[error("empty character literal: {0:?}")]
    EmptyCharLiteral(String),

    #[error("too many chars in char literal: first={0:?}, extra={1:?}")]
    TooManyCharsInCharLiteral(char, String),

    #[error("char does not fit in i8: {0:?}")]
    CharOutOfRange(char),
}

fn parse_escape_with_pos(ch: char, at: usize) -> Result<char, TokenizerError> {
    match ch {
        '\'' => Ok('\''),
        '"' => Ok('"'),
        '?' => Ok('?'),
        '\\' => Ok('\\'),
        'a' => Ok('\x07'),
        'b' => Ok('\x08'),
        'f' => Ok('\x0c'),
        'n' => Ok('\n'),
        'r' => Ok('\r'),
        't' => Ok('\t'),
        'v' => Ok('\x0b'),
        _ => Err(TokenizerError::UnsupportedEscape(ch, at)),
    }
}

fn parse_char_literal(literal: &str) -> Result<i8, TokenizerError> {
    let mut it = literal.char_indices();

    let (_, open) = it
        .next()
        .ok_or_else(|| TokenizerError::InvalidCharLiteral(literal.to_string()))?;
    if open != '\'' {
        return Err(TokenizerError::InvalidCharacter(open));
    }

    let (_, payload_char) = match it.next() {
        None => return Err(TokenizerError::EmptyCharLiteral(literal.to_string())),
        Some((i, '\\')) => {
            let (esc_i, esc) = it
                .next()
                .ok_or_else(|| TokenizerError::TruncatedEscapeSequence(i, literal.to_string()))?;
            let ch = parse_escape_with_pos(esc, esc_i)?;
            (i, ch)
        }
        Some((i, c)) => (i, c),
    };

    match it.next() {
        Some((_, '\'')) => {}
        _ => return Err(TokenizerError::UnterminatedCharLiteral(literal.to_string())),
    }

    if it.next().is_some() {
        let extra = {
            let last_quote = literal.rfind('\'').unwrap_or(literal.len());
            literal[last_quote + 1..].to_string()
        };
        return Err(TokenizerError::TooManyCharsInCharLiteral(
            payload_char,
            extra,
        ));
    }

    if (payload_char as u32) > 0x7F {
        return Err(TokenizerError::CharOutOfRange(payload_char));
    }

    Ok(payload_char as i8)
}

fn parse_string_literal(literal: &str) -> Result<String, TokenizerError> {
    if literal.len() < 2 || !literal.starts_with('"') || !literal.ends_with('"') {
        return Err(TokenizerError::InvalidStringLiteral(literal.to_string()));
    }

    let inner_start = 1;
    let inner_end = literal.len() - 1;
    let mut out = String::with_capacity(inner_end.saturating_sub(inner_start));

    let mut it = literal[inner_start..inner_end].char_indices().peekable();
    while let Some((off, c)) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        let (esc_rel_off, esc) = it.next().ok_or_else(|| {
            TokenizerError::TruncatedEscapeSequence(inner_start + off, literal.to_string())
        })?;
        let ch = parse_escape_with_pos(esc, inner_start + off + esc_rel_off)?;
        out.push(ch);
    }

    Ok(out)
}
