use std::fs;

use anyhow::Error;
use cc::{
    cli::Args, codegen::Codegen, loop_label::LoopLabeler, parse::Parser,
    semantic::SemanticAnalyzer, tacky::TackyGen, tokenize::tokenize,
};
use clap::Parser as _;

fn main() -> anyhow::Result<(), Error> {
    let args = Args::parse();

    if args.lex {
        let source = fs::read_to_string(&args.input_path)?;
        if args.verbose {
            eprintln!("{:?}", &source);
        }
        let tokens = tokenize(&source)?;
        eprintln!("{:?}", tokens);
    }
    if args.parse {
        let source = fs::read_to_string(&args.input_path)?;
        if args.verbose {
            eprintln!("{:?}", &source);
        }
        let tokens = tokenize(&source)?;
        if args.verbose {
            eprintln!("{:?}", &tokens);
        }
        let mut parser = Parser::new(source.chars().collect(), tokens);
        let ast = parser.parse();
        eprintln!("{:?}", ast);
    }
    if args.codegen {
        let source = fs::read_to_string(&args.input_path)?;
        if args.verbose {
            eprintln!("source: {:?}", &source);
        }
        let tokens = tokenize(&source)?;
        if args.verbose {
            eprintln!("tokens: {:?}", &tokens);
        }
        let mut parser = Parser::new(source.chars().collect(), tokens);
        let ast = parser.parse()?;
        if args.verbose {
            eprintln!("AST: {:?}", &ast);
        }
        let analyzed_ast = SemanticAnalyzer::new().analyze_program(ast)?;
        if args.verbose {
            eprintln!("Analyzed AST: {:?}", &analyzed_ast);
        }

        let labeled_ast = LoopLabeler::new().label_program(analyzed_ast)?;
        if args.verbose {
            eprintln!("Labeled AST: {:?}", &labeled_ast);
        }
        let tacky = TackyGen::new(labeled_ast);
        let tacky_program = tacky.codegen()?;
        let mut codegen = Codegen::new(std::io::stdout());
        codegen.lower(&tacky_program)?;
    }
    Ok(())
}
