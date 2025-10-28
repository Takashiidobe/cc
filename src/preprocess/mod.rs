use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::Context as _;

use crate::tokenize::{Token, TokenKind, tokenize};

pub struct Preprocessor {
    search_paths: Vec<PathBuf>,
    active: BTreeSet<PathBuf>,
}

impl Preprocessor {
    pub fn new(search_paths: Vec<PathBuf>) -> Self {
        Self {
            search_paths,
            active: BTreeSet::new(),
        }
    }

    pub fn expand_file(&mut self, file: &Path) -> anyhow::Result<Vec<Token>> {
        let src =
            std::fs::read_to_string(file).context(format!("unable to read {}", file.display()))?;
        let tokens = tokenize(&src, file.to_str().unwrap())?;
        self.expand_tokens(file, tokens)
    }

    fn expand_tokens(&mut self, origin: &Path, tokens: Vec<Token>) -> anyhow::Result<Vec<Token>> {
        let mut output = Vec::with_capacity(tokens.len());
        for tok in tokens {
            match &tok.kind {
                TokenKind::IncludeRelative(file) => {
                    let path = self.resolve_relative(origin, file)?;
                    let included = self.expand_child(&tok, &path)?;
                    output.extend(included);
                }
                TokenKind::IncludeSystem(file) => {
                    let path = self.resolve_system(file)?;
                    let included = self.expand_child(&tok, &path)?;
                    output.extend(included);
                }
                _ => output.push(tok),
            }
        }
        Ok(output)
    }

    fn expand_child(&mut self, site: &Token, path: &Path) -> anyhow::Result<Vec<Token>> {
        let full = path.canonicalize().context(format!(
            "include not found at {}: {}",
            site.filename,
            path.display()
        ))?;

        if !self.active.insert(full.clone()) {
            anyhow::bail!("cyclic include detected: {}", full.display());
        }

        let expanded = self.expand_file(&full);
        self.active.remove(&full);
        expanded
    }

    fn resolve_relative(&self, including_file: &Path, include: &str) -> anyhow::Result<PathBuf> {
        let base = including_file.parent().unwrap_or_else(|| Path::new("."));
        Ok(base.join(include))
    }

    fn resolve_system(&self, include: &str) -> anyhow::Result<PathBuf> {
        for dir in &self.search_paths {
            let candidate = dir.join(include);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
        anyhow::bail!("angle include <{}> not found", include);
    }
}
