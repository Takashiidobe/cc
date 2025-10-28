use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum StructLabelerError {
    #[error("struct '{0}' defined multiple times")]
    DuplicateDefinition(String),

    #[error("struct '{0}' requires a definition")]
    IncompleteStruct(String),

    #[error("duplicate member '{1}' in struct '{0}'")]
    DuplicateMember(String, String),

    #[error("struct '{0}' cannot contain itself directly as a member")]
    RecursiveStruct(String),
}
