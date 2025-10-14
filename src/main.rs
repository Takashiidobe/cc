use std::fs;

use anyhow::Error;
use cc::{cli::Args, codegen::Codegen, parse::Parser, tokenize::tokenize};
use clap::Parser as _;

fn main() -> anyhow::Result<(), Error> {
    let args = Args::parse();

    if args.lex {
        let source = fs::read_to_string(&args.input_path)
            .unwrap_or_else(|err| panic!("Failed to read {}: {}", args.input_path.display(), err));
        dbg!(&source);
        let tokens = tokenize(&source)?;
        dbg!(tokens);
    }
    if args.parse {
        let source = fs::read_to_string(&args.input_path)
            .unwrap_or_else(|err| panic!("Failed to read {}: {}", args.input_path.display(), err));
        dbg!(&source);
        let tokens = tokenize(&source)?;
        dbg!(&tokens);
        let mut parser = Parser::new(source.chars().collect(), tokens);
        let ast = parser.parse();
        dbg!(ast);
    }
    if args.codegen {
        let source = fs::read_to_string(&args.input_path)
            .unwrap_or_else(|err| panic!("Failed to read {}: {}", args.input_path.display(), err));
        dbg!(&source);
        let tokens = tokenize(&source)?;
        dbg!(&tokens);
        let mut parser = Parser::new(source.chars().collect(), tokens);
        let ast = parser.parse();
        dbg!(&ast);
        let mut codegen = Codegen::new(source.chars().collect(), ast, std::io::stdout());
        codegen.program()?;
    }
    Ok(())
}
