pub(crate) mod cli;
pub(crate) mod codegen;
pub(crate) mod fuzzing;
pub(crate) mod parse;
pub(crate) mod preprocess;
pub(crate) mod semantic;
pub(crate) mod tacky;
pub(crate) mod tokenize;

use std::{fs, path::Path};

use crate::{fuzzing::generate, preprocess::Preprocessor};
use anyhow::Error;
use clap::Parser as _;

use {
    cli::Args, codegen::Codegen, parse::Parser, semantic::SemanticAnalyzer,
    semantic::loop_label::LoopLabeler, semantic::struct_label::StructLabeler, tacky::TackyGen,
    tokenize::tokenize,
};

fn main() -> anyhow::Result<(), Error> {
    let args = Args::parse();

    if args.fuzz {
        println!("{}", generate());
        return Ok(());
    }

    let input_path = args.input_path.unwrap();

    if args.lex {
        let source = fs::read_to_string(&input_path)?;
        if args.verbose {
            eprintln!("{:?}", &source);
        }
        let tokens = tokenize(&source, input_path.to_str().unwrap())?;
        eprintln!("{:?}", tokens);
    }
    if args.parse {
        let source = fs::read_to_string(&input_path)?;
        if args.verbose {
            eprintln!("{:?}", &source);
        }
        let tokens = tokenize(&source, input_path.to_str().unwrap())?;
        if args.verbose {
            eprintln!("{:?}", &tokens);
        }
        let mut parser = Parser::new(source.bytes().collect(), tokens);
        let ast = parser.parse();
        eprintln!("{:?}", ast);
    }
    if args.codegen {
        let source = fs::read_to_string(&input_path)?;
        if args.verbose {
            eprintln!("source: {:?}", &source);
        }
        let tokens = tokenize(&source, input_path.to_str().unwrap())?;
        if args.verbose {
            eprintln!("tokens: {:?}", &tokens);
        }
        let mut preprocessor = Preprocessor::new(vec![]);
        let tokens = preprocessor.expand_file(Path::new(&input_path))?;

        let mut parser = Parser::new(source.bytes().collect(), tokens);

        let ast = parser.parse()?;
        if args.verbose {
            eprintln!("AST: {:?}", &ast);
        }
        let struct_checked_ast = StructLabeler::new().label_program(ast)?;
        if args.verbose {
            eprintln!("Struct Checked AST: {:?}", &struct_checked_ast);
        }
        let analyzed_ast = SemanticAnalyzer::new().analyze_program(struct_checked_ast)?;
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
