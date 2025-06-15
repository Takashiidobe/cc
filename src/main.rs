use std::env;
use std::fs;
use std::path::PathBuf;

// Simple compiler driver options
struct Options {
    stage: Stage,
    emit_asm: Option<PathBuf>,
    input: PathBuf,
}

#[derive(PartialEq)]
enum Stage {
    Full,
    Codegen,
    Parse,
    Lex,
}

fn parse_args() -> Result<Options, String> {
    let mut args = env::args().skip(1);
    let mut stage = Stage::Full;
    let mut emit_asm = None;
    let mut input = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--lex" => stage = Stage::Lex,
            "--parse" => stage = Stage::Parse,
            "--codegen" => stage = Stage::Codegen,
            "-S" => {
                if let Some(v) = args.next() {
                    emit_asm = Some(PathBuf::from(v));
                } else {
                    return Err("-S expects a file".into());
                }
            }
            _ => {
                if arg.starts_with('-') {
                    return Err(format!("unknown option: {}", arg));
                }
                input = Some(PathBuf::from(arg));
            }
        }
    }
    let input = input.ok_or_else(|| "no input file".to_string())?;
    Ok(Options { stage, emit_asm, input })
}

fn main() {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };
    if let Err(e) = run(opts) {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}

fn run(opt: Options) -> Result<(), String> {
    let source = fs::read_to_string(&opt.input)
        .map_err(|e| format!("failed to read {}: {}", opt.input.display(), e))?;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.lex()?;
    if opt.stage == Stage::Lex {
        return Ok(());
    }

    let mut parser = ParserC::new(tokens);
    let program = parser.parse_program()?;
    if opt.stage == Stage::Parse {
        return Ok(());
    }

    let asm = generate_assembly(&program);
    if opt.stage == Stage::Codegen {
        return Ok(());
    }

    if let Some(out) = opt.emit_asm {
        fs::write(&out, asm).map_err(|e| format!("failed to write {}: {}", out.display(), e))?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    Int,
    Return,
    Void,
    Identifier(String),
    IntConst(i32),
    LParen,
    RParen,
    LBrace,
    RBrace,
    Semicolon,
    Eof,
}

#[derive(Debug, Clone)]
struct Token {
    kind: TokenKind,
}

struct Lexer<'a> {
    chars: Vec<char>,
    pos: usize,
    input: &'a str,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self { chars: input.chars().collect(), pos: 0, input }
    }

    fn lex(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        loop {
            self.skip_ws();
            if self.pos >= self.chars.len() {
                tokens.push(Token { kind: TokenKind::Eof });
                break;
            }
            let ch = self.chars[self.pos];
            if ch.is_ascii_alphabetic() || ch == '_' {
                let ident = self.lex_identifier();
                let kind = match ident.as_str() {
                    "int" => TokenKind::Int,
                    "return" => TokenKind::Return,
                    "void" => TokenKind::Void,
                    _ => TokenKind::Identifier(ident),
                };
                tokens.push(Token { kind });
            } else if ch.is_ascii_digit() {
                let num = self.lex_number()?;
                tokens.push(Token { kind: TokenKind::IntConst(num) });
            } else {
                self.pos += 1;
                let kind = match ch {
                    '(' => TokenKind::LParen,
                    ')' => TokenKind::RParen,
                    '{' => TokenKind::LBrace,
                    '}' => TokenKind::RBrace,
                    ';' => TokenKind::Semicolon,
                    _ => return Err(format!("unexpected character '{}'", ch)),
                };
                tokens.push(Token { kind });
            }
        }
        Ok(tokens)
    }

    fn skip_ws(&mut self) {
        loop {
            while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
                self.pos += 1;
            }
            if self.pos + 1 < self.chars.len() && self.chars[self.pos] == '/' && self.chars[self.pos + 1] == '/' {
                self.pos += 2;
                while self.pos < self.chars.len() && self.chars[self.pos] != '\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn lex_identifier(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.chars.len() && (self.chars[self.pos].is_ascii_alphanumeric() || self.chars[self.pos] == '_') {
            self.pos += 1;
        }
        self.input[start..self.pos].to_string()
    }

    fn lex_number(&mut self) -> Result<i32, String> {
        let start = self.pos;
        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        self.input[start..self.pos]
            .parse::<i32>()
            .map_err(|e| format!("invalid number: {}", e))
    }
}

struct ParserC {
    tokens: Vec<Token>,
    pos: usize,
}

impl ParserC {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn parse_program(&mut self) -> Result<Program, String> {
        let func = self.parse_function()?;
        self.expect(TokenKind::Eof)?;
        Ok(Program { function: func })
    }

    fn parse_function(&mut self) -> Result<Function, String> {
        self.expect(TokenKind::Int)?;
        let name = if let TokenKind::Identifier(id) = self.next().kind.clone() {
            id
        } else {
            return Err("expected identifier".into());
        };
        self.expect(TokenKind::LParen)?;
        self.expect(TokenKind::Void)?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::LBrace)?;
        let stmt = self.parse_statement()?;
        self.expect(TokenKind::RBrace)?;
        Ok(Function { name, body: stmt })
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        self.expect(TokenKind::Return)?;
        let expr = self.parse_exp()?;
        self.expect(TokenKind::Semicolon)?;
        Ok(Statement::Return(expr))
    }

    fn parse_exp(&mut self) -> Result<Expr, String> {
        if let TokenKind::IntConst(v) = self.next().kind.clone() {
            Ok(Expr::Int(v))
        } else {
            Err("expected integer constant".into())
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<(), String> {
        if std::mem::discriminant(&self.peek().kind) == std::mem::discriminant(&kind) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {:?}", kind))
        }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn next(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        self.pos += 1;
        tok
    }
}

struct Program {
    function: Function,
}

struct Function {
    name: String,
    body: Statement,
}

enum Statement {
    Return(Expr),
}

enum Expr {
    Int(i32),
}

fn generate_assembly(prog: &Program) -> String {
    let mut asm = String::new();
    asm.push_str(".section .text\n");
    asm.push_str(&format!(".globl {}\n", prog.function.name));
    asm.push_str(&format!("{}:\n", prog.function.name));
    match &prog.function.body {
        Statement::Return(Expr::Int(v)) => {
            asm.push_str(&format!("    movl ${}, %eax\n", v));
            asm.push_str("    ret\n");
        }
    }
    asm
}
