use std::collections::BTreeMap;
use std::sync::{OnceLock, RwLock};

use crate::parse::Type;

#[derive(Debug, Clone)]
pub(crate) struct StructMemberLayout {
    pub(crate) offset: usize,
    pub(crate) ty: Type,
}

#[derive(Debug, Clone)]
pub(crate) struct StructLayout {
    pub(crate) size: usize,
    pub(crate) align: usize,
    pub(crate) members: BTreeMap<String, StructMemberLayout>,
}

static STRUCT_LAYOUTS: OnceLock<RwLock<BTreeMap<String, StructLayout>>> = OnceLock::new();

fn layouts() -> &'static RwLock<BTreeMap<String, StructLayout>> {
    STRUCT_LAYOUTS.get_or_init(|| RwLock::new(BTreeMap::new()))
}

pub(crate) fn set_struct_layout(tag: String, layout: StructLayout) {
    layouts().write().unwrap().insert(tag, layout);
}

pub(crate) fn get_struct_layout(tag: &str) -> Option<StructLayout> {
    layouts().read().unwrap().get(tag).cloned()
}

pub(crate) fn clear_struct_layouts() {
    layouts().write().unwrap().clear();
}
