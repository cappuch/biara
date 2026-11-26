//! In-memory buffers (the decompiler parses IR text through these).

#[derive(Clone)]
pub struct MemoryBuffer {
    text: String,
}

impl MemoryBuffer {
    pub fn create_from_memory_range(data: &[u8], len: usize) -> Self {
        Self {
            text: String::from_utf8_lossy(&data[..len.min(data.len())]).into_owned(),
        }
    }

    pub fn create_from_memory_range_copy(data: &[u8], len: usize) -> Self {
        Self::create_from_memory_range(data, len)
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }
}
