use std::fs;

use anyhow::Error;
use cc::{
    cli::Args, codegen::Codegen, parse::Parser, semantic::SemanticAnalyzer, tacky::TackyGen,
    tokenize::tokenize,
};
use clap::Parser as _;

fn main() -> anyhow::Result<(), Error> {
    let args = Args::parse();

    if args.lex {
        let source = fs::read_to_string(&args.input_path)
            .unwrap_or_else(|err| panic!("Failed to read {}: {}", args.input_path.display(), err));
        eprintln!("{:?}", &source);
        let tokens = tokenize(&source)?;
        eprintln!("{:?}", tokens);
    }
    if args.parse {
        let source = fs::read_to_string(&args.input_path)
            .unwrap_or_else(|err| panic!("Failed to read {}: {}", args.input_path.display(), err));
        eprintln!("{:?}", &source);
        let tokens = tokenize(&source)?;
        eprintln!("{:?}", &tokens);
        let mut parser = Parser::new(source.chars().collect(), tokens);
        let ast = parser.parse();
        eprintln!("{:?}", ast);
    }
    if args.codegen {
        let source = fs::read_to_string(&args.input_path)
            .unwrap_or_else(|err| panic!("Failed to read {}: {}", args.input_path.display(), err));
        eprintln!("{:?}", &source);
        let tokens = tokenize(&source)?;
        eprintln!("{:?}", &tokens);
        let mut parser = Parser::new(source.chars().collect(), tokens);
        let ast = parser.parse();
        eprintln!("{:?}", &ast);
        let analyzed_ast = SemanticAnalyzer::default().analyze_program(ast);
        eprintln!("{:?}", &analyzed_ast);
        let tacky = TackyGen::new(analyzed_ast);
        let tacky_program = tacky.codegen();
        let mut codegen = Codegen::new(std::io::stdout());
        codegen.lower(&tacky_program)?;
    }
    Ok(())
}
