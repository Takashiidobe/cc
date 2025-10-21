use std::{path::PathBuf, str::FromStr as _};

use clap::{Parser, arg, command};

#[derive(Debug, Parser)]
#[command(version, about)]
pub(crate) struct Args {
    #[arg(long, short = 'S', group = "flag")]
    pub(crate) codegen: bool,
    #[arg(long, short, group = "flag")]
    pub(crate) parse: bool,
    #[arg(long, short, group = "flag")]
    pub(crate) lex: bool,

    #[arg(value_parser = path_exists)]
    pub(crate) input_path: PathBuf,

    #[clap(env, long, short)]
    pub(crate) verbose: bool,
}

pub(crate) fn path_exists(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from_str(s).map_err(|e| format!("Invalid path: {}", e))?;
    if path.exists() {
        Ok(path)
    } else {
        Err(format!("Path does not exist: {}", s))
    }
}
