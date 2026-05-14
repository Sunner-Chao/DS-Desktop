use std::fs;
use std::path::PathBuf;

/// Returns the memory directory path (~/.ds-code/memory or %APPDATA%/ds-code/memory).
fn get_memory_dir() -> PathBuf {
    crate::get_user_data_dir().join("memory")
}

/// Ensure the memory directory exists.
fn ensure_memory_dir() -> PathBuf {
    let dir = get_memory_dir();
    let _ = fs::create_dir_all(&dir);
    dir
}

/// MEMORY.md path — long-term memory that persists across sessions.
pub fn memory_path() -> PathBuf {
    ensure_memory_dir().join("MEMORY.md")
}

/// USER.md path — user preferences and profile.
pub fn user_path() -> PathBuf {
    ensure_memory_dir().join("USER.md")
}

/// Load the full contents of a memory file.
pub fn load_memory(file_path: &PathBuf) -> String {
    fs::read_to_string(file_path).unwrap_or_default()
}

/// Save content to a memory file (overwrite).
pub fn save_memory(file_path: &PathBuf, content: &str) -> Result<(), String> {
    ensure_memory_dir();
    fs::write(file_path, content).map_err(|e| format!("Failed to save memory: {}", e))
}

/// Append a new memory entry to MEMORY.md with a timestamp.
pub fn append_memory(entry: &str) -> Result<(), String> {
    ensure_memory_dir();
    let path = memory_path();
    let mut existing = load_memory(&path);
    let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M").to_string();
    if !existing.is_empty() {
        existing.push('\n');
    }
    existing.push_str(&format!(
        "### {}\n{}\n",
        timestamp,
        entry.trim()
    ));
    fs::write(&path, existing).map_err(|e| format!("Failed to append memory: {}", e))
}

/// Build the memory injection block that gets inserted into the agent prompt.
///
/// Returns (memory_block, user_block) where:
/// - memory_block is the content from MEMORY.md (long-term memory)
/// - user_block is the content from USER.md (user preferences)
///
/// If both files are empty, returns an empty string.
pub fn read_memory_injection() -> String {
    let mem = load_memory(&memory_path()).trim().to_string();
    let usr = load_memory(&user_path()).trim().to_string();

    if mem.is_empty() && usr.is_empty() {
        return String::new();
    }

    let mut block = String::from("\n## 持久记忆\n");
    if !mem.is_empty() {
        // MEMORY.md may already contain headers; just include it directly
        block.push_str(&mem);
        block.push('\n');
    }
    if !usr.is_empty() {
        block.push_str("\n### 用户偏好\n");
        block.push_str(&usr);
        block.push('\n');
    }
    block.push_str("\n请根据以上记忆和用户偏好调整你的回答和行为。\n");
    block
}
