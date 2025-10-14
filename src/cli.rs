use std::{path::PathBuf, str::FromStr as _};

use clap::{Parser, arg, command};

#[derive(Debug, Parser)]
#[command(version, about)]
pub struct Args {
    #[arg(long, short, group = "flag")]
    pub codegen: bool,
    #[arg(long, short = 'S', group = "flag")]
    pub assembly: bool,
    #[arg(long, short, group = "flag")]
    pub parse: bool,
    #[arg(long, short, group = "flag")]
    pub lex: bool,

    #[arg(value_parser = path_exists)]
    pub input_path: PathBuf,
}

pub fn path_exists(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from_str(s).map_err(|e| format!("Invalid path: {}", e))?;
    if path.exists() {
        Ok(path)
    } else {
        Err(format!("Path does not exist: {}", s))
    }
}
