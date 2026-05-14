mod memory;

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use base64::{engine::general_purpose, Engine as _};
use chrono::{Timelike, Utc};
use dirs::home_dir;
use once_cell::sync::Lazy;
use portable_pty::{native_pty_system, ChildKiller, CommandBuilder, PtySize};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_dialog::DialogExt;
use tracing::{error, info};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// Global state for terminal sessions
static TERMINAL_SESSIONS: Lazy<Arc<Mutex<HashMap<String, TerminalSession>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));
static TERMINAL_RESULTS: Lazy<Arc<Mutex<HashMap<String, TerminalCompletedResult>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));
static RUNNING_TASK_PIDS: Lazy<Arc<Mutex<HashMap<String, u32>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub language: String,
    #[serde(rename = "workspacePath")]
    pub workspace_path: String,
    #[serde(rename = "binaryMode")]
    pub binary_mode: String,
    #[serde(rename = "customBinaryPath")]
    pub custom_binary_path: String,
    pub provider: String,
    pub model: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    #[serde(rename = "mcpConfigPath")]
    pub mcp_config_path: String,
    #[serde(rename = "skillsDir")]
    pub skills_dir: String,
    #[serde(rename = "skillsEnabled")]
    pub skills_enabled: bool,
    #[serde(rename = "mcpEnabled")]
    pub mcp_enabled: bool,
    #[serde(rename = "allowShell")]
    pub allow_shell: bool,
    #[serde(rename = "maxSubagents")]
    pub max_subagents: i32,
    #[serde(rename = "harnessEnabled")]
    pub harness_enabled: bool,
    #[serde(rename = "launchAction")]
    pub launch_action: String,
    #[serde(rename = "rememberWorkspace")]
    pub remember_workspace: bool,
    #[serde(rename = "enabledSkills")]
    pub enabled_skills: Vec<String>,
    #[serde(rename = "enabledMcpServers")]
    pub enabled_mcp_servers: Vec<String>,
    #[serde(rename = "mobileBridgeEnabled")]
    pub mobile_bridge_enabled: bool,
    #[serde(rename = "mobileBridgeHost")]
    pub mobile_bridge_host: String,
    #[serde(rename = "mobileBridgePort")]
    pub mobile_bridge_port: i32,
    #[serde(rename = "mobileBridgeToken")]
    pub mobile_bridge_token: String,
    #[serde(rename = "mobileRemoteControlEnabled")]
    pub mobile_remote_control_enabled: bool,
    #[serde(rename = "updatePushEnabled")]
    pub update_push_enabled: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            language: "zh".to_string(),
            workspace_path: home_dir().unwrap_or_default().to_string_lossy().to_string(),
            binary_mode: "bundled".to_string(),
            custom_binary_path: String::new(),
            provider: "deepseek".to_string(),
            model: "deepseek-v4-pro".to_string(),
            base_url: "https://api.deepseek.com".to_string(),
            mcp_config_path: String::new(),
            skills_dir: String::new(),
            skills_enabled: true,
            mcp_enabled: false,
            allow_shell: false,
            max_subagents: 10,
            harness_enabled: false,
            launch_action: "tui".to_string(),
            remember_workspace: true,
            enabled_skills: vec![
                "superpowers".to_string(),
                "ui-ux-design".to_string(),
                "cron-scheduler".to_string(),
                "skill-downloader".to_string(),
            ],
            enabled_mcp_servers: vec![],
            mobile_bridge_enabled: false,
            mobile_bridge_host: "127.0.0.1".to_string(),
            mobile_bridge_port: 8765,
            mobile_bridge_token: String::new(),
            mobile_remote_control_enabled: false,
            update_push_enabled: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalOptions {
    pub cols: i32,
    pub rows: i32,
    #[serde(rename = "workspacePath")]
    pub workspace_path: String,
    #[serde(rename = "launchAction")]
    pub launch_action: String,
    #[serde(rename = "agentPrompt")]
    pub agent_prompt: String,
}

struct TerminalSession {
    writer: Option<Box<dyn Write + Send>>,
    killer: Option<Box<dyn ChildKiller + Send + Sync>>,
}

impl TerminalSession {
    fn new(writer: Box<dyn Write + Send>, killer: Box<dyn ChildKiller + Send + Sync>) -> Self {
        Self {
            writer: Some(writer),
            killer: Some(killer),
        }
    }

    fn write(&mut self, data: &str) -> Result<(), String> {
        if let Some(writer) = self.writer.as_mut() {
            writer
                .write_all(data.as_bytes())
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn kill(&mut self) -> Result<(), String> {
        if let Some(killer) = self.killer.as_mut() {
            killer.kill().map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

pub fn get_user_data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("DS_CODE_DATA_DIR") {
        return PathBuf::from(dir);
    }
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ds-code")
}

fn get_settings_path() -> PathBuf {
    get_user_data_dir().join("settings.json")
}

fn get_secrets_path() -> PathBuf {
    get_user_data_dir().join("secrets.json")
}

fn get_skills_dir() -> PathBuf {
    get_user_data_dir().join("skills")
}

fn get_log_dir() -> PathBuf {
    get_user_data_dir().join("logs")
}

fn configured_screenshot_dir() -> Option<PathBuf> {
    std::env::var_os("DS_CODE_SCREENSHOT_DIR")
        .or_else(|| std::env::var_os("DS_CODE_PLAYWRIGHT_OUTPUT_DIR"))
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

fn ensure_user_data_dir() -> std::io::Result<()> {
    let dir = get_user_data_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir)?;
    }
    Ok(())
}

// Settings commands
#[tauri::command]
fn get_settings() -> Result<Settings, String> {
    let path = get_settings_path();
    if path.exists() {
        let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        serde_json::from_str(&content).map_err(|e| e.to_string())
    } else {
        Ok(Settings::default())
    }
}

#[tauri::command]
fn save_settings(settings: Settings) -> Result<Settings, String> {
    ensure_user_data_dir().map_err(|e| e.to_string())?;
    let path = get_settings_path();
    let content = serde_json::to_string_pretty(&settings).map_err(|e| e.to_string())?;
    fs::write(&path, content).map_err(|e| e.to_string())?;
    info!("Settings saved");
    Ok(settings)
}

#[tauri::command]
fn read_image_data_url(path: String) -> Result<String, String> {
    let image_path = PathBuf::from(path.trim());
    if !image_path.exists() {
        return Err("Image file does not exist".to_string());
    }
    let extension = image_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();
    let mime = match extension.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        _ => "image/png",
    };
    let bytes = fs::read(&image_path).map_err(|e| e.to_string())?;
    Ok(format!(
        "data:{};base64,{}",
        mime,
        general_purpose::STANDARD.encode(bytes)
    ))
}

#[tauri::command]
fn read_media_data_url(path: String) -> Result<String, String> {
    let media_path = PathBuf::from(path.trim());
    if !media_path.exists() {
        return Err("Media file does not exist".to_string());
    }
    let bytes = fs::read(&media_path).map_err(|e| e.to_string())?;
    let max_bytes = std::env::var("DS_CODE_MEDIA_PREVIEW_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64 * 1024 * 1024);
    if bytes.len() > max_bytes {
        return Err(format!(
            "Media file is too large for inline preview: {} bytes",
            bytes.len()
        ));
    }
    let extension = media_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();
    let mime = match extension.as_str() {
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        _ => "video/webm",
    };
    Ok(format!(
        "data:{};base64,{}",
        mime,
        general_purpose::STANDARD.encode(bytes)
    ))
}

// API Key commands
#[tauri::command]
fn get_api_key(provider: String) -> Result<String, String> {
    let path = get_secrets_path();
    if path.exists() {
        let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        #[derive(Deserialize)]
        struct SecretStore {
            #[serde(rename = "apiKeys")]
            api_keys: Option<HashMap<String, String>>,
        }
        if let Ok(store) = serde_json::from_str::<SecretStore>(&content) {
            if let Some(api_keys) = store.api_keys {
                return Ok(api_keys.get(&provider).cloned().unwrap_or_default());
            }
        }
    }
    Ok(String::new())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyResult {
    pub ok: bool,
    pub error: String,
    #[serde(rename = "hasKey")]
    pub has_key: bool,
}

#[tauri::command]
fn save_api_key(provider: String, api_key: String) -> Result<ApiKeyResult, String> {
    ensure_user_data_dir().map_err(|e| e.to_string())?;
    let path = get_secrets_path();
    #[derive(Serialize, Deserialize)]
    struct SecretStore {
        version: i32,
        #[serde(rename = "apiKeys")]
        api_keys: HashMap<String, String>,
    }
    let mut store = if path.exists() {
        serde_json::from_str::<SecretStore>(&fs::read_to_string(&path).map_err(|e| e.to_string())?)
            .unwrap_or(SecretStore {
                version: 1,
                api_keys: HashMap::new(),
            })
    } else {
        SecretStore {
            version: 1,
            api_keys: HashMap::new(),
        }
    };

    let has_key = !api_key.is_empty();
    if !api_key.is_empty() {
        store.api_keys.insert(provider.clone(), api_key.clone());
    }

    let content = serde_json::to_string_pretty(&store).map_err(|e| e.to_string())?;
    fs::write(&path, content).map_err(|e| e.to_string())?;
    info!("API key saved for provider: {}", provider);
    Ok(ApiKeyResult {
        ok: true,
        error: String::new(),
        has_key,
    })
}

// Dialog commands - using async API
#[tauri::command]
async fn choose_directory(app: AppHandle) -> Result<String, String> {
    let folder = app.dialog().file().blocking_pick_folder();
    Ok(folder.map(|p| p.to_string()).unwrap_or_default())
}

#[tauri::command]
async fn choose_file(app: AppHandle) -> Result<String, String> {
    let file = app.dialog().file().blocking_pick_file();
    Ok(file.map(|p| p.to_string()).unwrap_or_default())
}

// Git commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitStatus {
    pub ok: bool,
    pub error: String,
    #[serde(rename = "workspacePath")]
    pub workspace_path: String,
    #[serde(rename = "repoRoot")]
    pub repo_root: String,
    #[serde(rename = "isRepo")]
    pub is_repo: bool,
    pub branch: String,
    pub upstream: String,
    pub ahead: i32,
    pub behind: i32,
    #[serde(rename = "hasChanges")]
    pub has_changes: bool,
    pub staged: i32,
    pub unstaged: i32,
    pub untracked: i32,
    pub remotes: Vec<GitRemote>,
    #[serde(rename = "originUrl")]
    pub origin_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitRemote {
    pub name: String,
    #[serde(rename = "fetchUrl")]
    pub fetch_url: String,
    #[serde(rename = "pushUrl")]
    pub push_url: String,
}

fn run_git(args: &[&str], cwd: &str) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|e| e.to_string())?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

#[tauri::command]
fn git_status(workspace_path: String) -> GitStatus {
    let workspace_path = if workspace_path.is_empty() {
        home_dir().unwrap_or_default().to_string_lossy().to_string()
    } else {
        workspace_path
    };

    let repo_check = run_git(&["rev-parse", "--show-toplevel"], &workspace_path);
    if repo_check.is_err() {
        return GitStatus {
            ok: true,
            error: String::new(),
            workspace_path,
            repo_root: String::new(),
            is_repo: false,
            branch: String::new(),
            upstream: String::new(),
            ahead: 0,
            behind: 0,
            has_changes: false,
            staged: 0,
            unstaged: 0,
            untracked: 0,
            remotes: vec![],
            origin_url: String::new(),
        };
    }

    let repo_root = repo_check.unwrap();
    let branch = run_git(&["branch", "--show-current"], &repo_root).unwrap_or_default();
    let upstream = run_git(
        &["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        &repo_root,
    )
    .unwrap_or_default();

    let (ahead, behind) = if !upstream.is_empty() {
        let output = run_git(
            &[
                "rev-list",
                "--left-right",
                "--count",
                &format!("HEAD...{}", upstream),
            ],
            &repo_root,
        )
        .unwrap_or_default();
        let parts: Vec<&str> = output.trim().split_whitespace().collect();
        (
            parts.first().and_then(|s| s.parse().ok()).unwrap_or(0),
            parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0),
        )
    } else {
        (0, 0)
    };

    let status_output =
        run_git(&["status", "--porcelain=v1", "-uall"], &repo_root).unwrap_or_default();
    let changes: Vec<&str> = status_output.lines().collect();
    let staged = changes
        .iter()
        .filter(|l| l.starts_with(|c: char| c != ' ' && c != '?'))
        .count() as i32;
    let unstaged = changes
        .iter()
        .filter(|l| l.chars().nth(1).map(|c| c != ' ').unwrap_or(false))
        .count() as i32;
    let untracked = changes.iter().filter(|l| l.starts_with("??")).count() as i32;

    let remotes_output = run_git(&["remote", "-v"], &repo_root).unwrap_or_default();
    let mut remotes: HashMap<String, GitRemote> = HashMap::new();
    for line in remotes_output.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let name = parts[0].to_string();
            let url = parts[1].to_string();
            let kind = parts[2].trim_matches(|c: char| c == '(' || c == ')');
            let remote = remotes.entry(name.clone()).or_insert(GitRemote {
                name,
                fetch_url: String::new(),
                push_url: String::new(),
            });
            if kind == "(fetch)" {
                remote.fetch_url = url;
            } else if kind == "(push)" {
                remote.push_url = url;
            }
        }
    }

    GitStatus {
        ok: true,
        error: String::new(),
        workspace_path,
        repo_root,
        is_repo: true,
        branch: branch.trim().to_string(),
        upstream: upstream.trim().to_string(),
        ahead,
        behind,
        has_changes: !changes.is_empty(),
        staged,
        unstaged,
        untracked,
        remotes: remotes.clone().into_values().collect(),
        origin_url: remotes
            .get("origin")
            .map(|r| r.push_url.clone())
            .unwrap_or_default(),
    }
}

#[tauri::command]
fn git_init(workspace_path: String) -> Result<GitStatus, String> {
    let workspace_path = if workspace_path.is_empty() {
        home_dir().unwrap_or_default().to_string_lossy().to_string()
    } else {
        workspace_path
    };

    run_git(&["init", "-b", "main"], &workspace_path).map_err(|e| e.to_string())?;
    Ok(git_status(workspace_path))
}

#[tauri::command]
fn git_set_remote(workspace_path: String, remote_url: String) -> Result<GitStatus, String> {
    let repo_root =
        run_git(&["rev-parse", "--show-toplevel"], &workspace_path).map_err(|e| e.to_string())?;

    let has_origin = run_git(&["remote", "get-url", "origin"], &repo_root).is_ok();

    if has_origin {
        run_git(&["remote", "set-url", "origin", &remote_url], &repo_root)?;
    } else {
        run_git(&["remote", "add", "origin", &remote_url], &repo_root)?;
    }

    Ok(git_status(workspace_path))
}

#[tauri::command]
fn git_commit(workspace_path: String, message: String) -> Result<GitStatus, String> {
    let repo_root =
        run_git(&["rev-parse", "--show-toplevel"], &workspace_path).map_err(|e| e.to_string())?;

    run_git(&["add", "-A"], &repo_root)?;
    run_git(&["commit", "-m", &message], &repo_root)?;

    Ok(git_status(workspace_path))
}

#[tauri::command]
fn git_fetch(workspace_path: String) -> Result<GitStatus, String> {
    let repo_root =
        run_git(&["rev-parse", "--show-toplevel"], &workspace_path).map_err(|e| e.to_string())?;
    run_git(&["fetch", "--prune", "origin"], &repo_root)?;
    Ok(git_status(workspace_path))
}

#[tauri::command]
fn git_pull(workspace_path: String) -> Result<GitStatus, String> {
    let repo_root =
        run_git(&["rev-parse", "--show-toplevel"], &workspace_path).map_err(|e| e.to_string())?;
    run_git(&["pull", "--ff-only"], &repo_root)?;
    Ok(git_status(workspace_path))
}

#[tauri::command]
fn git_push(workspace_path: String) -> Result<GitStatus, String> {
    let repo_root =
        run_git(&["rev-parse", "--show-toplevel"], &workspace_path).map_err(|e| e.to_string())?;
    run_git(&["push"], &repo_root)?;
    Ok(git_status(workspace_path))
}

// Terminal commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalStartResult {
    pub ok: bool,
    pub error: String,
    pub pid: Option<u32>,
    #[serde(rename = "sessionId")]
    pub session_id: Option<String>,
    #[serde(rename = "finalOutput")]
    pub final_output: Option<String>,
    #[serde(rename = "exitCode")]
    pub exit_code: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
struct BrowserToolPlan {
    #[serde(default)]
    start_url: String,
    #[serde(default)]
    search_query: String,
    #[serde(default)]
    click_target: String,
    #[serde(default)]
    screenshot: bool,
    #[serde(default)]
    record_seconds: u64,
    #[serde(default)]
    login_required: bool,
    #[serde(default)]
    fullscreen: bool,
    #[serde(default)]
    wait_for_user_seconds: u64,
    #[serde(default)]
    action_plan: Vec<String>,
    #[serde(default)]
    save_location: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalCompletedResult {
    #[serde(rename = "sessionId")]
    pub session_id: String,
    #[serde(rename = "exitCode")]
    pub exit_code: i32,
    #[serde(rename = "finalOutput")]
    pub final_output: String,
}

fn get_bundled_binary_path() -> String {
    if let Ok(cargo_manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let deepseek_binary = PathBuf::from(cargo_manifest_dir)
            .parent() // src-tauri
            .and_then(|p| p.parent()) // DeepseekCode
            .map(|p| {
                p.join("node_modules")
                    .join("deepseek-tui")
                    .join("bin")
                    .join("downloads")
                    .join("deepseek.exe")
            });
        if let Some(path) = deepseek_binary {
            if path.exists() {
                return path.to_string_lossy().to_string();
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors() {
            let candidate = ancestor
                .join("node_modules")
                .join("deepseek-tui")
                .join("bin")
                .join("downloads")
                .join(if cfg!(target_os = "windows") {
                    "deepseek.exe"
                } else {
                    "deepseek"
                });
            if candidate.exists() {
                return candidate.to_string_lossy().to_string();
            }
        }
    }
    "deepseek".to_string()
}

fn resolve_workspace_dir(workspace_path: &str) -> PathBuf {
    let trimmed = workspace_path.trim();
    if !trimmed.is_empty() {
        let path = PathBuf::from(trimmed);
        if path.is_dir() {
            return path;
        }
    }
    home_dir().unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn language_instruction(language: &str) -> &'static str {
    if language == "zh" {
        "请使用简体中文回答，除非用户明确要求其他语言。"
    } else {
        "Please answer in English unless the user explicitly asks for another language."
    }
}

fn default_skill_content(id: &str) -> Option<&'static str> {
    match id {
        "superpowers" => Some(
            "# Superpowers\n\nUse this skill to strengthen planning, task decomposition, code editing, verification, and final reporting.\n\n- Start by identifying the user's concrete goal and the workspace scope.\n- Prefer small, reversible edits that match the existing codebase.\n- Verify changes with the narrowest useful command before reporting completion.\n- Surface blockers, assumptions, and residual risk clearly.\n",
        ),
        "ui-ux-design" => Some(
            "# UI/UX Design\n\nUse this skill for product UI work, desktop app polish, and visual interaction checks.\n\n- Keep primary workflows visible and reduce default configuration clutter.\n- Use familiar controls: icon buttons for tools, toggles for binary settings, and compact panels for advanced options.\n- Check spacing, overflow, text fit, empty states, disabled states, and responsive constraints.\n- Prefer restrained, work-focused surfaces for developer tools.\n",
        ),
        "cron-scheduler" => Some(
            "---\nname: cron-scheduler\ndescription: Advanced-only helper for hand-authored crontab files. Normal scheduled tasks are managed by the Scheduled Tasks screen.\n---\n\n# Cron Advanced Scripts\n\nUse this skill only when the user explicitly asks for a raw cron file or crontab snippet. For normal recurring Agent tasks, use the desktop Scheduled Tasks screen.\n\n- Treat this as an advanced escape hatch, not the default scheduled-task workflow.\n- Generate and validate a cron file before discussing installation.\n- Do not run `crontab`, overwrite an existing crontab, or install a task unless the user explicitly asks.\n- Prefer outputs under `.deepseek/cron/` and logs under `.deepseek/logs/`.\n",
        ),
        "skill-downloader" => Some(
            "---\nname: skill-downloader\ndescription: Use when the user asks to download, install, import, fetch, or update a Skill from a URL, GitHub raw file, local path, or archive.\n---\n\n# Skill Downloader\n\nUse this skill when a user asks to download or install a Skill during a desktop Agent conversation.\n\n- Do not synthesize remote Skill content. Download or copy the source bytes first, then verify the saved file.\n- Prefer `curl -fsSL \"<skill-url>\" -o \".deepseek/skills/<skill-id>/SKILL.md\"` for URL sources.\n- Verify with a non-empty file check and inspect the first lines for `name:` and `description:` frontmatter.\n- Report the source URL, destination path, and verification result.\n",
        ),
        _ => None,
    }
}

fn default_skill_ids() -> [&'static str; 4] {
    [
        "superpowers",
        "ui-ux-design",
        "cron-scheduler",
        "skill-downloader",
    ]
}

fn sanitize_id(value: &str, fallback: &str) -> String {
    let id = value
        .trim()
        .to_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    if id.is_empty() {
        fallback.to_string()
    } else {
        id
    }
}

fn frontmatter_value(content: &str, key: &str) -> String {
    if !content.starts_with("---") {
        return String::new();
    }
    let Some(rest) = content.strip_prefix("---") else {
        return String::new();
    };
    let Some((frontmatter, _)) = rest.split_once("---") else {
        return String::new();
    };
    let prefix = format!("{}:", key);
    frontmatter
        .lines()
        .map(str::trim)
        .find_map(|line| {
            line.strip_prefix(&prefix)
                .map(|value| value.trim().trim_matches('"').to_string())
        })
        .unwrap_or_default()
}

fn skill_name_from_content(id: &str, content: &str) -> String {
    let fm_name = frontmatter_value(content, "name");
    if !fm_name.is_empty() {
        return fm_name;
    }
    content
        .lines()
        .find_map(|line| line.trim().strip_prefix("# ").map(str::trim))
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| id.to_string())
}

fn skill_description_from_content(content: &str) -> String {
    let description = frontmatter_value(content, "description");
    if !description.is_empty() {
        return description;
    }
    "Custom agent workflow skill.".to_string()
}

fn skill_from_file(id: String, path: PathBuf, source: &str, origin: &str) -> Option<SkillTemplate> {
    let content = fs::read_to_string(&path).ok()?;
    Some(SkillTemplate {
        id: id.clone(),
        name: skill_name_from_content(&id, &content),
        description: skill_description_from_content(&content),
        source: source.to_string(),
        origin: origin.to_string(),
        path: path.to_string_lossy().to_string(),
        content,
    })
}

fn ensure_default_skills(skill_root: &Path) -> Result<(), String> {
    fs::create_dir_all(skill_root).map_err(|e| e.to_string())?;
    for id in default_skill_ids() {
        let Some(content) = default_skill_content(id) else {
            continue;
        };
        let skill_dir = skill_root.join(id);
        let skill_file = skill_dir.join("SKILL.md");
        if !skill_file.exists() {
            fs::create_dir_all(&skill_dir).map_err(|e| e.to_string())?;
            fs::write(skill_file, content).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn load_skill_templates(skill_root: &Path) -> Result<HashMap<String, SkillTemplate>, String> {
    ensure_default_skills(skill_root)?;
    let mut templates = HashMap::new();
    let entries = fs::read_dir(skill_root).map_err(|e| e.to_string())?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let skill_file = path.join("SKILL.md");
        if !skill_file.exists() {
            continue;
        }
        let id = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if id.is_empty() {
            continue;
        }
        let origin = if default_skill_ids().contains(&id.as_str()) {
            "preset"
        } else {
            "custom"
        };
        if let Some(skill) = skill_from_file(id.clone(), skill_file, "file", origin) {
            templates.insert(id, skill);
        }
    }
    Ok(templates)
}

fn build_skill_prompt_prefix(settings: &Settings) -> String {
    if !settings.skills_enabled {
        return String::new();
    }
    let skill_root = if settings.skills_dir.trim().is_empty() {
        get_skills_dir()
    } else {
        PathBuf::from(settings.skills_dir.trim())
    };
    let Ok(templates) = load_skill_templates(&skill_root) else {
        return String::new();
    };
    let enabled = if settings.enabled_skills.is_empty() {
        default_skill_ids()
            .iter()
            .map(|id| id.to_string())
            .collect()
    } else {
        settings.enabled_skills.clone()
    };
    let mut blocks = Vec::new();
    for id in enabled {
        if let Some(skill) = templates.get(&id) {
            blocks.push(format!("## {}\n{}", skill.name, skill.content.trim()));
        }
    }
    if blocks.is_empty() {
        String::new()
    } else {
        format!(
            "已启用以下 Skills。回答和执行任务时必须按这些指令工作；只有当用户请求明显不相关时才忽略。\n\n{}\n\n---\n\n",
            blocks.join("\n\n")
        )
    }
}

fn mcp_preset_server(id: &str, workspace_dir: &PathBuf) -> Option<Value> {
    let workspace = workspace_dir.to_string_lossy().to_string();
    let playwright_output_dir = configured_screenshot_dir()
        .unwrap_or_else(|| workspace_dir.join(".ds-code").join("playwright-output"))
        .to_string_lossy()
        .to_string();
    match id {
        "filesystem" => Some(serde_json::json!({
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", workspace],
            "env": {},
            "url": null,
            "connect_timeout": null,
            "execute_timeout": null,
            "read_timeout": null,
            "disabled": false,
            "enabled": true,
            "required": false,
            "enabled_tools": [],
            "disabled_tools": []
        })),
        "playwright" => Some(serde_json::json!({
            "command": "npx",
            "args": [
                "-y",
                "@playwright/mcp",
                "--browser",
                "chrome",
                "--headless",
                "--output-dir",
                playwright_output_dir,
                "--timeout-action",
                "15000",
                "--timeout-navigation",
                "90000"
            ],
            "env": {},
            "url": null,
            "disabled": false,
            "enabled": true
        })),
        "context7" => Some(serde_json::json!({
            "command": "npx",
            "args": ["-y", "@upstash/context7-mcp"],
            "env": {},
            "url": null,
            "disabled": false,
            "enabled": true
        })),
        _ => None,
    }
}

fn ensure_mcp_config(
    settings: &Settings,
    workspace_dir: &PathBuf,
) -> Result<Option<String>, String> {
    if !settings.mcp_enabled {
        return Ok(None);
    }
    let custom_path = settings.mcp_config_path.trim();
    if !custom_path.is_empty() {
        return Ok(Some(custom_path.to_string()));
    }
    let mut servers = serde_json::Map::new();
    for id in &settings.enabled_mcp_servers {
        if let Some(server) = mcp_preset_server(id, workspace_dir) {
            servers.insert(id.clone(), server);
        }
    }
    if servers.is_empty() {
        return Ok(None);
    }
    ensure_user_data_dir().map_err(|e| e.to_string())?;
    let default_output_dir = configured_screenshot_dir()
        .unwrap_or_else(|| workspace_dir.join(".ds-code").join("playwright-output"));
    let _ = fs::create_dir_all(default_output_dir);
    let path = get_user_data_dir().join("mcp.presets.json");
    let config = serde_json::json!({
        "timeouts": {
            "connect_timeout": 10,
            "execute_timeout": 300,
            "read_timeout": 300
        },
        "servers": servers
    });
    let content = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
    fs::write(&path, content).map_err(|e| e.to_string())?;
    Ok(Some(path.to_string_lossy().to_string()))
}

fn generated_mcp_config_text(
    settings: &Settings,
    workspace_dir: &PathBuf,
) -> Result<String, String> {
    let mut servers = serde_json::Map::new();
    for id in &settings.enabled_mcp_servers {
        if let Some(server) = mcp_preset_server(id, workspace_dir) {
            servers.insert(id.clone(), server);
        }
    }
    let config = serde_json::json!({
        "timeouts": {
            "connect_timeout": 10,
            "execute_timeout": 300,
            "read_timeout": 300
        },
        "servers": servers
    });
    serde_json::to_string_pretty(&config).map_err(|e| e.to_string())
}

fn mcp_config_text_for_settings(
    settings: &Settings,
    workspace_dir: &PathBuf,
) -> Result<(String, String, String), String> {
    let custom_path = settings.mcp_config_path.trim();
    if !custom_path.is_empty() {
        let path = PathBuf::from(custom_path);
        if !path.exists() {
            return Err(format!("MCP config file does not exist: {}", custom_path));
        }
        let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
        let parsed: Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
        return Ok((
            custom_path.to_string(),
            "custom".to_string(),
            serde_json::to_string_pretty(&parsed).map_err(|e| e.to_string())?,
        ));
    }
    Ok((
        get_user_data_dir()
            .join("mcp.presets.json")
            .to_string_lossy()
            .to_string(),
        "generated".to_string(),
        generated_mcp_config_text(settings, workspace_dir)?,
    ))
}

fn command_exists(command: &str) -> bool {
    let command = command.trim();
    if command.is_empty() {
        return false;
    }
    let path = PathBuf::from(command);
    if path.components().count() > 1 {
        return path.exists();
    }
    if let Some(path_value) = std::env::var_os("PATH") {
        let names = if cfg!(target_os = "windows")
            && !command.ends_with(".exe")
            && !command.ends_with(".cmd")
        {
            vec![
                command.to_string(),
                format!("{}.cmd", command),
                format!("{}.exe", command),
            ]
        } else {
            vec![command.to_string()]
        };
        for dir in std::env::split_paths(&path_value) {
            if names.iter().any(|name| dir.join(name).exists()) {
                return true;
            }
        }
    }
    false
}

fn redact_sensitive_text(value: &str) -> String {
    let mut redacted_parts = Vec::new();
    let mut redact_next = false;
    for part in value.split_whitespace() {
        let lower = part.to_lowercase();
        if redact_next {
            redacted_parts.push("***".to_string());
            redact_next = false;
        } else if part.contains("密码")
            || lower.contains("password")
            || lower.contains("passwd")
            || lower.contains("pwd")
        {
            redacted_parts.push("***".to_string());
            if part.ends_with("是") || part.ends_with(':') || part.ends_with('：') || part == "密码"
            {
                redact_next = true;
            }
        } else {
            redacted_parts.push(part.to_string());
        }
    }
    redacted_parts.join(" ")
}

fn looks_like_web_screenshot_request(prompt: &str) -> bool {
    let request = current_user_request(prompt);
    let lower = request.to_lowercase();
    (request.contains("截图")
        || request.contains("截屏")
        || request.contains("截一张")
        || request.contains("截取")
        || request.contains("图片")
        || lower.contains("screenshot"))
        && (lower.contains("http://")
            || lower.contains("https://")
            || lower.contains("bilibili")
            || lower.contains("b站")
            || lower.contains("baidu")
            || request.contains("哔哩哔哩")
            || request.contains("B站")
            || request.contains("b站")
            || request.contains("百度"))
}

fn looks_like_web_recording_request(prompt: &str) -> bool {
    let request = current_user_request(prompt);
    let lower = request.to_lowercase();
    (request.contains("录制")
        || request.contains("录屏")
        || request.contains("短视频")
        || request.contains("视频")
        || lower.contains("record")
        || lower.contains("video"))
        && (lower.contains("http://")
            || lower.contains("https://")
            || lower.contains("bilibili")
            || lower.contains("b站")
            || lower.contains("baidu")
            || request.contains("哔哩哔哩")
            || request.contains("B站")
            || request.contains("b站")
            || request.contains("百度"))
}

fn looks_like_web_media_request(prompt: &str) -> bool {
    looks_like_web_screenshot_request(prompt) || looks_like_web_recording_request(prompt)
}

fn looks_like_mcp_tool_request(prompt: &str) -> bool {
    let request = current_user_request(prompt);
    let lower = request.to_lowercase();
    request.contains("截图")
        || request.contains("截屏")
        || request.contains("截一张")
        || request.contains("截取")
        || request.contains("浏览器")
        || request.contains("打开网页")
        || request.contains("官网")
        || request.contains("图片")
        || request.contains("文件")
        || request.contains("目录")
        || request.contains("读取")
        || request.contains("保存")
        || request.contains("下载")
        || request.contains("录制")
        || request.contains("录屏")
        || request.contains("短视频")
        || lower.contains("screenshot")
        || lower.contains("browser")
        || lower.contains("record")
        || lower.contains("video")
        || lower.contains("http://")
        || lower.contains("https://")
        || lower.contains("filesystem")
}

fn current_user_request(prompt: &str) -> String {
    for marker in ["当前用户问题：", "Current user request:"] {
        if let Some((_, tail)) = prompt.rsplit_once(marker) {
            return tail.trim().to_string();
        }
    }
    prompt.trim().to_string()
}

fn extract_screenshot_url(prompt: &str) -> Option<String> {
    for token in prompt.split_whitespace() {
        let trimmed = token
            .trim_matches(|c: char| {
                matches!(
                    c,
                    '"' | '\''
                        | '`'
                        | '<'
                        | '>'
                        | '，'
                        | '。'
                        | '、'
                        | ','
                        | '.'
                        | ';'
                        | '；'
                        | ')'
                        | '('
                        | ']'
                        | '['
                )
            })
            .to_string();
        if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
            return Some(trimmed);
        }
    }
    let lower = prompt.to_lowercase();
    if lower.contains("bilibili")
        || lower.contains("b站")
        || prompt.contains("B站")
        || prompt.contains("b站")
        || prompt.contains("哔哩哔哩")
    {
        return Some("https://www.bilibili.com".to_string());
    }
    if lower.contains("baidu") || prompt.contains("百度") {
        return Some("https://www.baidu.com".to_string());
    }
    None
}

fn normalize_browser_start_url(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_matches('"').trim_matches('\'');
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return Some(trimmed.to_string());
    }
    let lower = trimmed.to_lowercase();
    if lower.contains('.') && !trimmed.contains(' ') {
        return Some(format!("https://{}", trimmed.trim_start_matches("//")));
    }
    None
}

fn infer_browser_plan_from_request(request: &str) -> BrowserToolPlan {
    let lower = request.to_lowercase();
    let click_target = if should_click_video_before_screenshot(request) {
        "video".to_string()
    } else if request.contains("视频") || lower.contains("video") {
        "video".to_string()
    } else if request.contains("商品") || lower.contains("product") {
        "product".to_string()
    } else if request.contains("文章") || lower.contains("article") {
        "article".to_string()
    } else if request.contains("链接")
        || lower.contains("link")
        || request.contains("任意")
        || lower.contains("any")
    {
        "link".to_string()
    } else {
        String::new()
    };
    BrowserToolPlan {
        start_url: extract_screenshot_url(request).unwrap_or_default(),
        search_query: extract_search_query(request),
        click_target,
        screenshot: looks_like_web_screenshot_request(request),
        record_seconds: if looks_like_web_recording_request(request) {
            extract_record_seconds(request)
        } else {
            0
        },
        login_required: should_prepare_login(request),
        fullscreen: should_fullscreen_video(request),
        wait_for_user_seconds: if should_prepare_login(request) { 45 } else { 0 },
        action_plan: infer_browser_action_plan(request),
        save_location: if request.contains("桌面") || lower.contains("desktop") {
            "desktop".to_string()
        } else {
            String::new()
        },
    }
}

fn infer_browser_action_plan(request: &str) -> Vec<String> {
    let mut steps = Vec::new();
    if should_prepare_login(request) {
        steps.push("assist_login".to_string());
    }
    if !extract_search_query(request).is_empty() {
        steps.push("search".to_string());
    }
    if should_click_video_before_screenshot(request)
        || request.contains("视频")
        || request.to_lowercase().contains("video")
    {
        steps.push("click_video".to_string());
    } else if request.contains("进入") || request.contains("点击") {
        steps.push("click_result".to_string());
    }
    if should_fullscreen_video(request) {
        steps.push("fullscreen".to_string());
    }
    if looks_like_web_recording_request(request) {
        steps.push("record_video".to_string());
    } else if looks_like_web_screenshot_request(request) {
        steps.push("screenshot".to_string());
    }
    if request.contains("保存")
        || request.contains("桌面")
        || request.to_lowercase().contains("desktop")
    {
        steps.push("save_file".to_string());
    }
    steps
}

fn extract_search_query(request: &str) -> String {
    let lower = request.to_lowercase();
    let has_search = request.contains("搜索") || request.contains("搜") || lower.contains("search");
    if !has_search {
        return String::new();
    }
    if request.contains("今日热点相关新闻") {
        return "今日热点相关新闻".to_string();
    }
    for marker in [
        "任意搜索一些",
        "随便搜索一些",
        "搜索一些",
        "去搜一些",
        "搜一些",
    ] {
        if let Some((_, tail)) = request.split_once(marker) {
            let query = tail
                .split(|ch| matches!(ch, '，' | ',' | '。' | ';' | '；' | '\n'))
                .next()
                .unwrap_or("")
                .split("然后")
                .next()
                .unwrap_or("")
                .trim();
            if !query.is_empty() {
                return query.to_string();
            }
        }
    }
    let markers = ["搜索", "search"];
    for marker in markers {
        if let Some((_, tail)) = request.split_once(marker) {
            let query = tail
                .split(|ch| matches!(ch, '，' | ',' | '。' | ';' | '；' | '\n'))
                .next()
                .unwrap_or("")
                .replace("相关新闻", "相关新闻")
                .trim()
                .to_string();
            if !query.is_empty() {
                return query;
            }
        }
    }
    String::new()
}

fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end <= start {
        return None;
    }
    Some(&text[start..=end])
}

fn request_browser_plan_from_model(
    api_key: &str,
    base_url: &str,
    model: &str,
    request: &str,
) -> Result<BrowserToolPlan, String> {
    if api_key.trim().is_empty() {
        return Err("API key is empty".to_string());
    }
    let planning_prompt = format!(
        "你是桌面浏览器自动化 function-calling 路由器。请只输出一个 JSON 对象，不要 Markdown，不要解释。\n\
你要把用户自然语言转换为可执行浏览器工具计划，字段必须使用下列 schema：\n\
{{\n\
  \"start_url\": string,              // http/https URL；站点名请推断常见官方网址\n\
  \"search_query\": string,           // 页面内搜索词；没有则空字符串\n\
  \"click_target\": string,           // video/product/article/link/search_result/login/button 等；没有则空字符串\n\
  \"screenshot\": boolean,            // 是否截图\n\
  \"record_seconds\": number,         // 是否录制视频，0 表示不录制\n\
  \"login_required\": boolean,        // 用户是否要求登录\n\
  \"fullscreen\": boolean,            // 是否要求放大、最大化、全屏或影院模式\n\
  \"wait_for_user_seconds\": number,  // 登录/验证码/手动操作等待秒数；无需等待则 0\n\
  \"action_plan\": string[],          // 顺序动作，如 open, assist_login, search, click_video, fullscreen, record_video, screenshot, save_file\n\
  \"save_location\": \"desktop\" | \"workspace\"\n\
}}\n\
规则：\n\
- 不要编造账号密码字段，不要输出用户密码。\n\
- 如果用户说“搜一些/搜索一些”，search_query 填后面的主题词。\n\
- 如果用户要求进入任意视频，click_target 填 video。\n\
- 如果用户要求录制 7s/10s，record_seconds 填对应数字。\n\
用户请求：{}",
        request
    );
    let body = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "user", "content": planning_prompt }
        ],
        "stream": false,
        "temperature": 0
    });
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?;
    let response = client
        .post(chat_completions_url(base_url))
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("planning HTTP status {}", response.status()));
    }
    let value: Value = response.json().map_err(|e| e.to_string())?;
    let content = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(|content| content.as_str())
        .ok_or_else(|| "planning response has no message.content".to_string())?;
    let json = extract_json_object(content).ok_or_else(|| {
        format!(
            "planning response is not JSON: {}",
            content.chars().take(160).collect::<String>()
        )
    })?;
    serde_json::from_str::<BrowserToolPlan>(json).map_err(|e| e.to_string())
}

fn screenshot_output_dir_for_request(
    request: &str,
    workspace_dir: &Path,
    plan: Option<&BrowserToolPlan>,
) -> PathBuf {
    let lower = request.to_lowercase();
    let wants_desktop = plan
        .map(|plan| plan.save_location.eq_ignore_ascii_case("desktop"))
        .unwrap_or(false)
        || request.contains("桌面")
        || lower.contains("desktop");
    if wants_desktop {
        if let Some(desktop) = dirs::desktop_dir() {
            return desktop;
        }
    }
    if let Some(configured) = configured_screenshot_dir() {
        return configured;
    }
    workspace_dir.join(".ds-code").join("playwright-output")
}

fn should_click_video_before_screenshot(request: &str) -> bool {
    let lower = request.to_lowercase();
    (request.contains("点击")
        || request.contains("打开")
        || request.contains("进入")
        || lower.contains("click"))
        && (request.contains("视频") || lower.contains("video"))
}

fn should_scroll_before_click(request: &str) -> bool {
    let lower = request.to_lowercase();
    request.contains("下方")
        || request.contains("往下")
        || request.contains("向下")
        || request.contains("下面")
        || lower.contains("scroll")
        || lower.contains("below")
        || lower.contains("down")
}

fn should_prepare_login(request: &str) -> bool {
    let lower = request.to_lowercase();
    request.contains("登录") || lower.contains("login") || lower.contains("sign in")
}

fn should_fullscreen_video(request: &str) -> bool {
    let lower = request.to_lowercase();
    request.contains("放大")
        || request.contains("全屏")
        || request.contains("最大化")
        || lower.contains("fullscreen")
        || lower.contains("maximize")
}

fn extract_record_seconds(request: &str) -> u64 {
    let lower = request.to_lowercase();
    let mut digits = String::new();
    for ch in lower.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else if !digits.is_empty() {
            if matches!(ch, 's' | '秒') {
                if let Ok(value) = digits.parse::<u64>() {
                    return value.clamp(1, 120);
                }
            }
            digits.clear();
        }
    }
    if !digits.is_empty() && (lower.contains("秒") || lower.contains('s')) {
        if let Ok(value) = digits.parse::<u64>() {
            return value.clamp(1, 120);
        }
    }
    10
}

fn find_playwright_node_modules() -> Option<PathBuf> {
    for candidate in [
        PathBuf::from("D:\\pro_sunner\\demo_vscode\\node_modules"),
        std::env::current_dir().ok()?.parent()?.join("node_modules"),
    ] {
        let pw = candidate.join("playwright");
        if pw.exists() || candidate.join("@playwright").exists() {
            return Some(candidate);
        }
    }
    None
}

fn npx_command() -> Command {
    for candidate in [
        "C:\\Program Files\\nodejs\\npx.cmd",
        "C:\\Program Files (x86)\\nodejs\\npx.cmd",
    ] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Command::new(path);
        }
    }
    if let Some(path_value) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_value) {
            for name in ["npx.cmd", "npx.exe", "npx"] {
                let path = dir.join(name);
                if path.exists() {
                    return Command::new(path);
                }
            }
        }
    }
    let mut command = Command::new("cmd");
    command.args(["/C", "npx"]);
    command
}

fn sanitize_filename_piece(value: &str) -> String {
    let mut piece = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            piece.push(ch);
        } else if matches!(ch, '.' | '-' | '_') {
            piece.push(ch);
        }
    }
    if piece.is_empty() {
        "page".to_string()
    } else {
        piece.chars().take(48).collect()
    }
}

fn playwright_node_script(
    output_path: &Path,
    url: &str,
    search_query: &str,
    click_target: &str,
    scroll_before_click: bool,
    record_seconds: u64,
    login_requested: bool,
    fullscreen_requested: bool,
    wait_for_user_seconds: u64,
) -> Result<PathBuf, String> {
    let script_dir = get_user_data_dir().join("runtime");
    fs::create_dir_all(&script_dir).map_err(|e| e.to_string())?;
    let script_path = script_dir.join(format!(
        "playwright-shot-{}.cjs",
        Utc::now().timestamp_millis()
    ));
    let output_json = serde_json::to_string(&output_path.to_string_lossy().to_string())
        .map_err(|e| e.to_string())?;
    let url_json = serde_json::to_string(url).map_err(|e| e.to_string())?;
    let search_query_json = serde_json::to_string(search_query).map_err(|e| e.to_string())?;
    let click_target_json = serde_json::to_string(click_target).map_err(|e| e.to_string())?;
    let scroll_before_click_json = if scroll_before_click { "true" } else { "false" };
    let login_requested_json = if login_requested { "true" } else { "false" };
    let fullscreen_requested_json = if fullscreen_requested {
        "true"
    } else {
        "false"
    };
    let video_temp_dir = output_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(".playwright-video-temp");
    let video_temp_dir_json = serde_json::to_string(&video_temp_dir.to_string_lossy().to_string())
        .map_err(|e| e.to_string())?;
    let script = format!(
        r#"const {{ chromium }} = require("playwright");
const fs = require("fs");
const path = require("path");

const targetUrl = {url_json};
const outputPath = {output_json};
const searchQuery = {search_query_json};
const clickTarget = {click_target_json};
const scrollBeforeClick = {scroll_before_click_json};
const recordSeconds = {record_seconds};
const loginRequested = {login_requested_json};
const fullscreenRequested = {fullscreen_requested_json};
const waitForUserSeconds = {wait_for_user_seconds};
const videoTempDir = {video_temp_dir_json};

function selectorForTarget(target) {{
  const normalized = String(target || "").toLowerCase();
  if (!normalized) return 'a[href]';
  if (normalized.includes("video") || normalized.includes("视频")) {{
    return 'a[href*="/video/"], a[href*="video"], a[href*="watch"], a[href*="play"], a[href]';
  }}
  if (normalized.includes("product") || normalized.includes("商品")) {{
    return 'a[href*="item"], a[href*="product"], a[href*="goods"], a[href]';
  }}
  if (normalized.includes("article") || normalized.includes("文章")) {{
    return 'a[href*="article"], a[href*="post"], a[href*="read"], a[href]';
  }}
  return 'a[href]';
}}

async function runSearch(page, query) {{
  if (!query) return;
  console.log("[tool] 正在调用浏览器工具在页面内搜索: " + query);
  const searchBox = page.locator('textarea[name="wd"]:visible, input[name="wd"]:visible, input[type="search"]:visible, input[name="q"]:visible, input[aria-label*="搜索"]:visible, textarea:visible, input:visible').first();
  await searchBox.waitFor({{ state: "visible", timeout: 20000 }});
  console.log("[tool] 已定位搜索输入框，正在输入关键词");
  await searchBox.fill(query);
  const searchButton = page.locator('input[type="submit"]:visible, button[type="submit"]:visible, #su:visible, button:has-text("搜索"):visible, input[value*="搜索"]:visible, button:visible').first();
  console.log("[tool] 正在提交搜索");
  if (await searchButton.count()) {{
    await searchButton.click({{ timeout: 10000 }}).catch(async () => {{
      await searchBox.press("Enter");
    }});
  }} else {{
    await searchBox.press("Enter");
  }}
  await page.waitForLoadState("domcontentloaded", {{ timeout: 45000 }}).catch(() => {{}});
  await page.waitForTimeout(2500);
  console.log("[tool] 搜索完成，当前页面: " + page.url());
}}

async function clickSearchResult(page) {{
  console.log("[tool] 正在查找搜索结果链接");
  const selectors = [
    '#content_left h3 a',
    '.result h3 a',
    'h3 a[href]',
    'a[href^="http"]:visible',
    'a[href]:visible'
  ];
  for (const selector of selectors) {{
    const candidates = page.locator(selector);
    const count = await candidates.count();
    console.log("[tool] 搜索结果选择器 " + selector + " 候选数量: " + count);
    for (let index = 0; index < Math.min(count, 12); index += 1) {{
      const candidate = candidates.nth(index);
      try {{
        const box = await candidate.boundingBox();
        if (!box || box.width < 40 || box.height < 12) continue;
        const href = await candidate.getAttribute("href").catch(() => "");
        const title = (await candidate.innerText({{ timeout: 1000 }}).catch(() => "")).replace(/\s+/g, " ").slice(0, 100);
        if (!href || /javascript:|#/.test(href)) continue;
        console.log("[tool] 正在进入搜索结果 #" + (index + 1) + " href=" + href + " title=" + (title || "-"));
        const popupPromise = page.waitForEvent("popup", {{ timeout: 10000 }}).catch(() => null);
        await candidate.click({{ timeout: 15000, force: true }});
        const popup = await popupPromise;
        const nextPage = popup || page;
        await nextPage.waitForLoadState("domcontentloaded", {{ timeout: 45000 }}).catch(() => {{}});
        await nextPage.waitForTimeout(3500);
        console.log("[tool] 已进入搜索结果页面: " + nextPage.url());
        return nextPage;
      }} catch (error) {{
        console.log("[tool] 搜索结果候选跳过: " + error.message);
      }}
    }}
  }}
  console.log("[tool] 未能进入搜索结果，将截取当前搜索结果页");
  return page;
}}

async function assistLogin(page) {{
  if (!loginRequested) return;
  console.log("[tool] 用户要求登录，正在打开登录入口。为避免把密码写入脚本或日志，请在可见浏览器窗口中完成登录/验证码。");
  const loginSelectors = [
    'text=/登录|登陆|Sign in|Log in/i',
    '.header-login-entry',
    '.login-entry',
    'a[href*="login"]',
    'button:has-text("登录")'
  ];
  for (const selector of loginSelectors) {{
    const entry = page.locator(selector).first();
    if (await entry.count()) {{
      await entry.click({{ timeout: 5000 }}).catch(() => {{}});
      break;
    }}
  }}
  const waitSeconds = Math.max(1, waitForUserSeconds || 45);
  console.log("[tool] 等待用户在浏览器窗口中完成登录/验证码，最多等待 " + waitSeconds + " 秒");
  await page.waitForTimeout(waitSeconds * 1000);
  console.log("[tool] 登录等待结束，继续执行后续搜索与录制步骤");
}}

async function enlargeVideo(page) {{
  if (!fullscreenRequested) return;
  console.log("[tool] 用户要求放大视频页面，正在尝试聚焦播放器并进入全屏/影院模式");
  await page.bringToFront().catch(() => {{}});
  const video = page.locator('video').first();
  if (await video.count()) {{
    await video.click({{ timeout: 5000, force: true }}).catch(() => {{}});
  }}
  const buttons = [
    '[aria-label*="全屏"]',
    '[title*="全屏"]',
    '.bpx-player-ctrl-full',
    '.bpx-player-ctrl-web',
    'button:has-text("全屏")'
  ];
  for (const selector of buttons) {{
    const button = page.locator(selector).first();
    if (await button.count()) {{
      await button.click({{ timeout: 3000, force: true }}).catch(() => {{}});
      await page.waitForTimeout(1200);
      break;
    }}
  }}
  await page.keyboard.press("f").catch(() => {{}});
  await page.waitForTimeout(1800);
}}

(async () => {{
  console.log("[tool] Launching Chrome");
  const browser = await chromium.launch({{ channel: "chrome", headless: false, args: ["--start-maximized"] }});
  fs.mkdirSync(path.dirname(outputPath), {{ recursive: true }});
  let context = null;
  let page;
  if (recordSeconds > 0) {{
    fs.mkdirSync(videoTempDir, {{ recursive: true }});
    console.log("[tool] 已启用浏览器录制，时长 " + recordSeconds + " 秒");
    context = await browser.newContext({{
      viewport: {{ width: 1365, height: 900 }},
      recordVideo: {{ dir: videoTempDir, size: {{ width: 1365, height: 900 }} }}
    }});
    page = await context.newPage();
  }} else {{
    page = await browser.newPage({{ viewport: {{ width: 1365, height: 900 }} }});
  }}
  page.setDefaultTimeout(30000);
  console.log("[tool] Opening " + targetUrl);
  await page.goto(targetUrl, {{ waitUntil: "domcontentloaded", timeout: 90000 }});
  await page.waitForTimeout(3500);
  let shotPage = page;
  await assistLogin(page);
  if (searchQuery) {{
    await runSearch(page, searchQuery);
    shotPage = await clickSearchResult(page);
  }}
  if (clickTarget) {{
    if (scrollBeforeClick) {{
      console.log("[tool] 用户要求导航到页面下方，正在向下滚动以加载下方视频区域");
      for (let step = 0; step < 4; step += 1) {{
        await page.mouse.wheel(0, 720);
        await page.waitForTimeout(900);
      }}
    }}
    console.log("[tool] 正在查找可点击目标: " + clickTarget);
    const candidates = page.locator(selectorForTarget(clickTarget));
    const count = await candidates.count();
    console.log("[tool] 找到候选链接数量: " + count);
    let clicked = false;
    for (let index = 0; index < Math.min(count, 36); index += 1) {{
      const candidate = candidates.nth(index);
      try {{
        const box = await candidate.boundingBox();
        if (!box || box.width < 40 || box.height < 30) continue;
        if (scrollBeforeClick && box.y < 120) continue;
        const href = await candidate.getAttribute("href").catch(() => "");
        const title = (await candidate.innerText({{ timeout: 1000 }}).catch(() => "")).replace(/\s+/g, " ").slice(0, 80);
        console.log("[tool] 正在点击候选 #" + (index + 1) + " href=" + (href || "-") + " title=" + (title || "-"));
        const popupPromise = page.waitForEvent("popup", {{ timeout: 10000 }}).catch(() => null);
        await candidate.click({{ timeout: 15000, force: true }});
        const popup = await popupPromise;
        if (popup) {{
          shotPage = popup;
          console.log("[tool] 目标在新标签页打开");
        }} else {{
          shotPage = page;
          console.log("[tool] 目标在当前标签页打开");
        }}
        clicked = true;
        break;
      }} catch (error) {{
        console.log("[tool] 候选跳过: " + error.message);
      }}
    }}
    if (!clicked) {{
      console.log("[tool] 没有找到匹配的可点击目标，将截取当前页面");
    }}
    await shotPage.waitForLoadState("domcontentloaded", {{ timeout: 45000 }}).catch(() => {{}});
    await shotPage.waitForTimeout(5000);
  }}
  await enlargeVideo(shotPage);
  if (recordSeconds > 0) {{
    console.log("[tool] 正在录制目标页面，等待 " + recordSeconds + " 秒");
    await shotPage.waitForTimeout(recordSeconds * 1000);
    const video = shotPage.video();
    if (context) {{
      await context.close();
    }}
    if (video) {{
      await video.saveAs(outputPath);
      console.log("[tool] 视频已写入: " + outputPath);
    }} else {{
      const files = fs.readdirSync(videoTempDir).filter((name) => name.endsWith(".webm"));
      if (!files.length) throw new Error("No Playwright video file was produced");
      const latest = files
        .map((name) => path.join(videoTempDir, name))
        .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs)[0];
      fs.copyFileSync(latest, outputPath);
      console.log("[tool] 视频已写入: " + outputPath);
    }}
    await browser.close();
  }} else {{
    console.log("[tool] 正在截图");
    await shotPage.screenshot({{ path: outputPath, fullPage: false }});
    console.log("[tool] 截图已写入: " + outputPath);
    await browser.close();
  }}
}})().catch((error) => {{
  console.error("[tool] Playwright automation failed: " + (error && error.stack ? error.stack : error));
  process.exit(1);
}});
"#
    );
    fs::write(&script_path, script).map_err(|e| e.to_string())?;
    Ok(script_path)
}

fn run_playwright_browser_media(
    app: AppHandle,
    session_id: String,
    workspace_dir: PathBuf,
    prompt: String,
    api_key: String,
    base_url: String,
    model: String,
) -> (String, i32) {
    let request = current_user_request(&prompt);
    let requested_record_seconds = if looks_like_web_recording_request(&request) {
        extract_record_seconds(&request)
    } else {
        0
    };
    let action_name = if requested_record_seconds > 0 {
        "录制"
    } else {
        "截图"
    };
    emit_terminal_data(
        &app,
        &session_id,
        format!(
            "[tool] 正在请求模型生成结构化浏览器{}计划...\r\n",
            action_name
        ),
    );
    let plan = match request_browser_plan_from_model(&api_key, &base_url, &model, &request) {
        Ok(plan) => {
            emit_terminal_data(
                &app,
                &session_id,
                "[tool] 模型已返回工具计划，准备执行。\r\n".to_string(),
            );
            plan
        }
        Err(error) => {
            emit_terminal_data(
                &app,
                &session_id,
                format!(
                    "[tool] 模型工具计划不可用，使用本地兜底计划。原因：{}\r\n",
                    redact_sensitive_text(&error)
                ),
            );
            infer_browser_plan_from_request(&request)
        }
    };
    let record_seconds = if plan.record_seconds > 0 {
        plan.record_seconds.clamp(1, 120)
    } else {
        requested_record_seconds
    };
    let action_name = if record_seconds > 0 {
        "录制"
    } else {
        "截图"
    };
    let planned_url =
        normalize_browser_start_url(&plan.start_url).or_else(|| extract_screenshot_url(&request));
    let Some(url) = planned_url else {
        return (
            "没有找到可截图的网址。请提供 http 或 https 开头的网址。".to_string(),
            -1,
        );
    };
    emit_terminal_data(
        &app,
        &session_id,
        format!(
            "[tool] 浏览器计划：start_url={}, search_query={}, click_target={}, screenshot={}, save_location={}, record_seconds={}, login_required={}, fullscreen={}, wait_for_user_seconds={}, actions={}\r\n",
            url,
            if plan.search_query.trim().is_empty() { "-" } else { plan.search_query.trim() },
            if plan.click_target.trim().is_empty() { "-" } else { plan.click_target.trim() },
            plan.screenshot,
            if plan.save_location.trim().is_empty() { "workspace" } else { plan.save_location.trim() },
            record_seconds,
            plan.login_required,
            plan.fullscreen,
            plan.wait_for_user_seconds,
            if plan.action_plan.is_empty() { "-".to_string() } else { plan.action_plan.join(" -> ") }
        ),
    );
    let output_dir = screenshot_output_dir_for_request(&request, &workspace_dir, Some(&plan));
    if let Err(error) = fs::create_dir_all(&output_dir) {
        return (format!("创建截图输出目录失败：{}", error), -1);
    }
    let url_name = sanitize_filename_piece(
        url.trim_start_matches("https://")
            .trim_start_matches("http://")
            .trim_start_matches("www."),
    );
    let extension = if record_seconds > 0 { "webm" } else { "png" };
    let output_path = output_dir.join(format!(
        "{}-{}.{}",
        url_name,
        Utc::now().timestamp_millis(),
        extension
    ));
    emit_terminal_data(
        &app,
        &session_id,
        format!(
            "[tool] 已解析浏览器目标：{}\r\n[tool] 输出位置：{}\r\n[tool] 正在启动 Playwright/Chrome...\r\n",
            url,
            output_path.to_string_lossy()
        ),
    );
    let output_path_string = output_path.to_string_lossy().to_string();
    let click_target = if !plan.click_target.trim().is_empty() {
        plan.click_target.trim().to_string()
    } else if !plan.search_query.trim().is_empty() || request.contains("搜索") {
        if request.contains("视频") || request.to_lowercase().contains("video") {
            "video".to_string()
        } else {
            "search_result".to_string()
        }
    } else if should_click_video_before_screenshot(&request) {
        "video".to_string()
    } else {
        String::new()
    };
    let scroll_before_click = should_scroll_before_click(&request);
    let login_requested = plan.login_required || should_prepare_login(&request);
    let fullscreen_requested = plan.fullscreen || should_fullscreen_video(&request);
    let wait_for_user_seconds = if plan.wait_for_user_seconds > 0 {
        plan.wait_for_user_seconds.clamp(1, 180)
    } else if login_requested {
        45
    } else {
        0
    };
    let mut command = npx_command();
    // Inject NODE_PATH so the node script can resolve `playwright` from D:\pro_sunner\demo_vscode\node_modules
    if let Some(nm) = find_playwright_node_modules() {
        command.env("NODE_PATH", nm.to_string_lossy().to_string());
    }
    if !click_target.is_empty() || !plan.search_query.trim().is_empty() {
        emit_terminal_data(
            &app,
            &session_id,
            format!(
                "[tool] 已识别为“浏览器自动化后{}”，search_query={}, click_target={}, scroll_before_click={}, login_requested={}, fullscreen_requested={}, wait_for_user_seconds={}，将执行浏览器自动化脚本...\r\n",
                action_name,
                if plan.search_query.trim().is_empty() { "-" } else { plan.search_query.trim() },
                if click_target.is_empty() { "-" } else { click_target.as_str() },
                scroll_before_click,
                login_requested,
                fullscreen_requested,
                wait_for_user_seconds
            ),
        );
    }
    let spawn_result = if !click_target.is_empty() || !plan.search_query.trim().is_empty() {
        match playwright_node_script(
            &output_path,
            &url,
            plan.search_query.trim(),
            &click_target,
            scroll_before_click,
            record_seconds,
            login_requested,
            fullscreen_requested,
            wait_for_user_seconds,
        ) {
            Ok(script_path) => command
                .args([
                    "--yes",
                    "--package=playwright",
                    "--",
                    "node",
                    &script_path.to_string_lossy().to_string(),
                ])
                .current_dir(&workspace_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn(),
            Err(error) => return (format!("创建 Playwright 自动化脚本失败：{}", error), -1),
        }
    } else {
        if record_seconds > 0 {
            match playwright_node_script(
                &output_path,
                &url,
                "",
                "",
                false,
                record_seconds,
                login_requested,
                fullscreen_requested,
                wait_for_user_seconds,
            ) {
                Ok(script_path) => command
                    .args([
                        "--yes",
                        "--package=playwright",
                        "--",
                        "node",
                        &script_path.to_string_lossy().to_string(),
                    ])
                    .current_dir(&workspace_dir)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn(),
                Err(error) => return (format!("创建 Playwright 录制脚本失败：{}", error), -1),
            }
        } else {
            command
                .args([
                    "playwright",
                    "screenshot",
                    "--browser=chromium",
                    "--channel=chrome",
                    "--timeout=90000",
                    &url,
                    &output_path_string,
                ])
                .current_dir(&workspace_dir)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
        }
    };

    let mut child = match spawn_result {
        Ok(child) => child,
        Err(error) => {
            return (
                format!("启动 Playwright {}失败：{}", action_name, error),
                -1,
            )
        }
    };

    fn spawn_pipe_reader<R: Read + Send + 'static>(
        mut reader: R,
        app: AppHandle,
        session_id: String,
        output_buffer: Arc<Mutex<String>>,
    ) {
        std::thread::spawn(move || {
            let mut buffer = [0u8; 1024];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        let data = String::from_utf8_lossy(&buffer[..n]).to_string();
                        if let Ok(mut output) = output_buffer.lock() {
                            output.push_str(&data);
                        }
                        emit_terminal_data(&app, &session_id, data);
                    }
                    Err(_) => break,
                }
            }
        });
    }

    let output_buffer = Arc::new(Mutex::new(String::new()));
    if let Some(reader) = child.stdout.take() {
        spawn_pipe_reader(
            reader,
            app.clone(),
            session_id.clone(),
            output_buffer.clone(),
        );
    }
    if let Some(reader) = child.stderr.take() {
        spawn_pipe_reader(
            reader,
            app.clone(),
            session_id.clone(),
            output_buffer.clone(),
        );
    }

    let mut ticks = 0;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let combined = output_buffer
                    .lock()
                    .map(|output| output.clone())
                    .unwrap_or_default();
                if status.success() && output_path.exists() {
                    emit_terminal_data(
                        &app,
                        &session_id,
                        format!(
                            "[tool] 页面{}完成，文件已写入：{}\r\n",
                            action_name,
                            output_path.to_string_lossy()
                        ),
                    );
                    if record_seconds > 0 {
                        return (
                            format!(
                                "视频已保存：{}\n视频所在文件夹：{}",
                                output_path.to_string_lossy(),
                                output_dir.to_string_lossy()
                            ),
                            0,
                        );
                    }
                    return (
                        format!(
                            "截图已保存：{}\n截图所在文件夹：{}",
                            output_path.to_string_lossy(),
                            output_dir.to_string_lossy()
                        ),
                        0,
                    );
                }
                return (
                    format!(
                        "{}失败，退出码：{}。\n{}",
                        action_name,
                        status.code().unwrap_or(-1),
                        combined
                    ),
                    status.code().unwrap_or(-1),
                );
            }
            Ok(None) => {
                ticks += 1;
                if ticks == 1 {
                    emit_terminal_data(
                        &app,
                        &session_id,
                        "[tool] Chrome 已启动，正在打开页面并等待加载...\r\n".to_string(),
                    );
                } else if ticks % 4 == 0 {
                    emit_terminal_data(
                        &app,
                        &session_id,
                        format!("[tool] 页面仍在加载或{}处理中...\r\n", action_name),
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(750));
            }
            Err(error) => {
                return (
                    format!("等待 Playwright {}失败：{}", action_name, error),
                    -1,
                )
            }
        }
    }
}

fn chat_completions_url(base_url: &str) -> String {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{}/chat/completions", trimmed.trim_end_matches("/v1"))
    }
}

fn extract_stream_delta(value: &Value) -> (String, String) {
    let Some(choice) = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .and_then(|choices| choices.first())
    else {
        return (String::new(), String::new());
    };
    let Some(delta) = choice.get("delta") else {
        return (String::new(), String::new());
    };
    let content = delta
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let reasoning = delta
        .get("reasoning_content")
        .or_else(|| delta.get("reasoning"))
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    (content, reasoning)
}

fn emit_terminal_data(app: &AppHandle, session_id: &str, data: String) {
    if data.is_empty() {
        return;
    }
    let emit_chunk = |text: String| {
        let session_json = serde_json::to_string(session_id).unwrap_or_else(|_| "\"\"".to_string());
        let text_json = serde_json::to_string(&text).unwrap_or_else(|_| "\"\"".to_string());
        if let Some(window) = app.get_webview_window("main") {
            let script = format!(
                "window.__deepseekDesktopStreamPush && window.__deepseekDesktopStreamPush({}, {});",
                session_json, text_json
            );
            let _ = window.eval(&script);
        } else {
            let _ = app.emit(
                "terminal:data",
                serde_json::json!({
                    "sessionId": session_id,
                    "data": text,
                }),
            );
        }
    };
    let chars: Vec<char> = data.chars().collect();
    if chars.len() <= 120 {
        emit_chunk(data);
        return;
    }
    for chunk in chars.chunks(64) {
        let text: String = chunk.iter().collect();
        emit_chunk(text);
        std::thread::sleep(std::time::Duration::from_millis(18));
    }
}

fn emit_stream_text(app: &AppHandle, session_id: &str, data: &str) {
    if data.is_empty() {
        return;
    }
    let chars: Vec<char> = data.chars().collect();
    for chunk in chars.chunks(32) {
        let text: String = chunk.iter().collect();
        let session_json = serde_json::to_string(session_id).unwrap_or_else(|_| "\"\"".to_string());
        let text_json = serde_json::to_string(&text).unwrap_or_else(|_| "\"\"".to_string());
        let script = format!(
            "window.__deepseekDesktopStreamPush && window.__deepseekDesktopStreamPush({}, {});",
            session_json, text_json
        );
        if let Some(window) = app.get_webview_window("main") {
            let _ = window.eval(&script);
        } else {
            let _ = app.emit(
                "terminal:data",
                serde_json::json!({
                    "sessionId": session_id,
                    "data": text,
                }),
            );
        }
    }
}

fn handle_stream_event(
    app: &AppHandle,
    session_id: &str,
    event: &str,
    content: &mut String,
    reasoning: &mut String,
) -> Result<bool, String> {
    for line in event.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            return Ok(true);
        }
        let value = serde_json::from_str::<Value>(data).map_err(|e| e.to_string())?;
        let (delta, reasoning_delta) = extract_stream_delta(&value);
        if !reasoning_delta.is_empty() {
            if reasoning.is_empty() {
                emit_terminal_data(app, session_id, "\r\n[reasoning]\r\n".to_string());
            }
            reasoning.push_str(&reasoning_delta);
            emit_stream_text(app, session_id, &reasoning_delta);
        }
        if !delta.is_empty() {
            content.push_str(&delta);
            emit_stream_text(app, session_id, &delta);
        }
    }
    Ok(false)
}

fn run_deepseek_streaming(
    app: &AppHandle,
    session_id: &str,
    api_key: &str,
    base_url: &str,
    model: &str,
    prompt: &str,
) -> Result<(String, i32), String> {
    if api_key.trim().is_empty() {
        return Err("DeepSeek API Key is empty".to_string());
    }
    let url = chat_completions_url(base_url);
    let body = serde_json::json!({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": true
    });
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?;
    let response = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .map_err(|e| e.to_string())?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().unwrap_or_default();
        return Err(format!("DeepSeek HTTP {}: {}", status.as_u16(), text));
    }

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut stream = response;
    let mut buffer = [0u8; 2048];
    let mut pending = String::new();
    loop {
        let n = stream.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        pending.push_str(&String::from_utf8_lossy(&buffer[..n]));
        while let Some(index) = pending.find("\n\n").or_else(|| pending.find("\r\n\r\n")) {
            let delimiter_len = if pending[index..].starts_with("\r\n\r\n") {
                4
            } else {
                2
            };
            let event = pending[..index].to_string();
            pending = pending[index + delimiter_len..].to_string();
            if handle_stream_event(app, session_id, &event, &mut content, &mut reasoning)? {
                return if content.trim().is_empty() {
                    Ok((reasoning, 0))
                } else {
                    Ok((content, 0))
                };
            }
        }
    }
    if !pending.trim().is_empty() {
        let _ = handle_stream_event(app, session_id, &pending, &mut content, &mut reasoning)?;
    }
    if content.trim().is_empty() {
        return Ok((reasoning, 0));
    }
    Ok((content, 0))
}

#[tauri::command]
fn terminal_start(
    app: AppHandle,
    options: TerminalOptions,
    settings: Settings,
) -> Result<TerminalStartResult, String> {
    let session_id = format!("session_{}", Utc::now().timestamp_millis());
    let workspace_dir = resolve_workspace_dir(&options.workspace_path);
    let workspace_dir_string = workspace_dir.to_string_lossy().to_string();

    let mut binary_path = if settings.binary_mode == "system" {
        "deepseek".to_string()
    } else if settings.binary_mode == "custom" && !settings.custom_binary_path.is_empty() {
        settings.custom_binary_path.clone()
    } else {
        // bundled mode: use the bundled binary from node_modules/deepseek-tui/bin/downloads
        let bundled_path = get_bundled_binary_path();
        info!("Using bundled binary: {}", bundled_path);
        bundled_path
    };
    let selected_path = PathBuf::from(&binary_path);
    if selected_path
        .file_name()
        .map(|name| {
            name.to_string_lossy()
                .eq_ignore_ascii_case("deepseek-tui.exe")
        })
        .unwrap_or(false)
    {
        let facade_path = selected_path.with_file_name("deepseek.exe");
        if facade_path.exists() {
            binary_path = facade_path.to_string_lossy().to_string();
            info!("Using deepseek facade for command mode: {}", binary_path);
        }
    }

    let api_key = get_api_key(settings.provider.clone()).unwrap_or_default();
    let mcp_workspace_dir = if options.workspace_path.trim().is_empty() {
        std::env::current_dir().unwrap_or_else(|_| workspace_dir.clone())
    } else {
        workspace_dir.clone()
    };
    let mcp_config_path = ensure_mcp_config(&settings, &mcp_workspace_dir)?;
    let normalized_base_url = settings
        .base_url
        .trim()
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .to_string();
    let model = settings.model.trim().to_string();
    let provider = settings.provider.trim().to_string();
    let skill_prompt_prefix = build_skill_prompt_prefix(&settings);
    let memory_injection = memory::read_memory_injection();
    let skills_dir_for_env = if settings.skills_dir.trim().is_empty() {
        get_skills_dir().to_string_lossy().to_string()
    } else {
        settings.skills_dir.trim().to_string()
    };
    let mut args: Vec<String> = Vec::new();

    if !provider.is_empty() {
        args.push("--provider".to_string());
        args.push(provider.clone());
    }
    if !model.is_empty() {
        args.push("--model".to_string());
        args.push(model.clone());
    }
    if !normalized_base_url.is_empty() {
        args.push("--base-url".to_string());
        args.push(normalized_base_url.clone());
    }

    match options.launch_action.as_str() {
        "continue" => {
            args.extend([
                "run".to_string(),
                "--workspace".to_string(),
                workspace_dir_string.clone(),
                "--continue".to_string(),
            ]);
        }
        "doctor" => args.push("doctor".to_string()),
        "setup" => args.push("setup".to_string()),
        "exec" | "plan" | "yolo" => {
            let language_instruction = language_instruction(&settings.language);
            let prompt = match options.launch_action.as_str() {
                "plan" => format!(
                    "{}\n请只输出实施计划和风险点，不要修改文件或执行破坏性操作。\n\n{}",
                    language_instruction,
                    format!("{}{}{}", skill_prompt_prefix, memory_injection, options.agent_prompt)
                ),
                "yolo" => format!(
                    "{}\n在当前 workspace 中完成用户请求。可以进行必要的代码修改和验证；遇到高风险或破坏性操作时先说明原因。\n\n{}",
                    language_instruction,
                    format!("{}{}{}", skill_prompt_prefix, memory_injection, options.agent_prompt)
                ),
                _ => format!(
                    "{}\n\n{}{}{}",
                    language_instruction, skill_prompt_prefix, memory_injection, options.agent_prompt
                ),
            };
            args.extend(["exec".to_string(), "--auto".to_string(), prompt]);
            if settings.mcp_enabled {
                args.insert(args.len() - 2, "--enable".to_string());
                args.insert(args.len() - 2, "mcp".to_string());
            }
        }
        _ => {
            args.extend([
                "run".to_string(),
                "--workspace".to_string(),
                workspace_dir_string.clone(),
            ]);
            if settings.mcp_enabled {
                args.push("--enable".to_string());
                args.push("mcp".to_string());
            }
        }
    };

    let redacted_args: Vec<String> = args
        .iter()
        .scan(false, |redact_next, arg| {
            if *redact_next {
                *redact_next = false;
                Some("***".to_string())
            } else {
                *redact_next = arg == "--api-key";
                Some(redact_sensitive_text(arg))
            }
        })
        .collect();
    info!(
        "Starting deepseek with binary: {}, args: {:?}",
        binary_path, redacted_args
    );

    if !api_key.is_empty() {
        info!("API key set for provider: {}", settings.provider);
    }
    if !normalized_base_url.is_empty() {
        info!("Base URL set: {}", normalized_base_url);
    }
    if !model.is_empty() {
        info!("Model set: {}", model);
    }

    if matches!(options.launch_action.as_str(), "exec" | "plan" | "yolo")
        && settings.mcp_enabled
        && settings
            .enabled_mcp_servers
            .iter()
            .any(|id| id == "playwright")
        && looks_like_web_media_request(&options.agent_prompt)
    {
        info!(
            "Starting direct Playwright browser media fallback: {}",
            session_id
        );
        let shot_app = app.clone();
        let shot_session_id = session_id.clone();
        let shot_workspace_dir = mcp_workspace_dir.clone();
        let shot_prompt = options.agent_prompt.clone();
        let shot_api_key = api_key.clone();
        let shot_base_url = normalized_base_url.clone();
        let shot_model = model.clone();
        std::thread::spawn(move || {
            let (output, exit_code) = run_playwright_browser_media(
                shot_app.clone(),
                shot_session_id.clone(),
                shot_workspace_dir,
                shot_prompt,
                shot_api_key,
                shot_base_url,
                shot_model,
            );
            emit_terminal_data(&shot_app, &shot_session_id, format!("\r\n{}\r\n", output));
            if let Ok(mut results) = TERMINAL_RESULTS.lock() {
                results.insert(
                    shot_session_id.clone(),
                    TerminalCompletedResult {
                        session_id: shot_session_id.clone(),
                        exit_code,
                        final_output: output.clone(),
                    },
                );
            }
            let _ = shot_app.emit(
                "terminal:exit",
                serde_json::json!({
                    "sessionId": shot_session_id,
                    "exitCode": exit_code,
                    "finalOutput": output,
                }),
            );
        });
        return Ok(TerminalStartResult {
            ok: true,
            error: String::new(),
            pid: None,
            session_id: Some(session_id),
            final_output: None,
            exit_code: None,
        });
    }

    let should_use_mcp_cli =
        settings.mcp_enabled && looks_like_mcp_tool_request(&options.agent_prompt);
    if matches!(options.launch_action.as_str(), "exec" | "plan" | "yolo") && should_use_mcp_cli {
        emit_terminal_data(
            &app,
            &session_id,
            format!(
                "[tool] MCP 路由：该请求需要外部工具，已选择 deepseek CLI + MCP 执行。\r\n[tool] MCP servers: {}\r\n[tool] MCP config: {}\r\n",
                if settings.enabled_mcp_servers.is_empty() {
                    "(custom config)".to_string()
                } else {
                    settings.enabled_mcp_servers.join(",")
                },
                mcp_config_path.clone().unwrap_or_else(|| "(none)".to_string())
            ),
        );
    }
    if matches!(options.launch_action.as_str(), "exec" | "plan" | "yolo")
        && provider == "deepseek"
        && !should_use_mcp_cli
    {
        info!(
            "Starting DeepSeek HTTP streaming session: {}, model={}, url={}, processStream={}",
            session_id,
            model,
            chat_completions_url(&normalized_base_url),
            settings.harness_enabled
        );
        let prompt = args.last().cloned().unwrap_or_default();
        let stream_app = app.clone();
        let stream_session_id = session_id.clone();
        let stream_api_key = api_key.clone();
        let stream_base_url = normalized_base_url.clone();
        let stream_model = model.clone();
        let show_process_banner = settings.harness_enabled;
        std::thread::spawn(move || {
            if show_process_banner {
                let _ = stream_app.emit(
                    "terminal:data",
                    serde_json::json!({
                        "sessionId": stream_session_id.clone(),
                        "data": format!(
                            "DeepSeek stream started: model={}, url={}\r\n\r\n",
                            stream_model,
                            chat_completions_url(&stream_base_url)
                        ),
                    }),
                );
            }
            let (output, exit_code) = match run_deepseek_streaming(
                &stream_app,
                &stream_session_id,
                &stream_api_key,
                &stream_base_url,
                &stream_model,
                &prompt,
            ) {
                Ok((output, exit_code)) => {
                    info!(
                        "DeepSeek HTTP streaming session completed: {}, exitCode={}, outputBytes={}",
                        stream_session_id,
                        exit_code,
                        output.len()
                    );
                    (output, exit_code)
                }
                Err(error) => {
                    error!("DeepSeek HTTP streaming failed: {}", error);
                    (format!("DeepSeek 流式请求失败：{}", error), -1)
                }
            };
            if exit_code != 0 {
                emit_terminal_data(&stream_app, &stream_session_id, output.clone());
            }
            if let Ok(mut results) = TERMINAL_RESULTS.lock() {
                results.insert(
                    stream_session_id.clone(),
                    TerminalCompletedResult {
                        session_id: stream_session_id.clone(),
                        exit_code,
                        final_output: output.clone(),
                    },
                );
            }
            let _ = stream_app.emit(
                "terminal:exit",
                serde_json::json!({
                    "sessionId": stream_session_id,
                    "exitCode": exit_code,
                    "finalOutput": output,
                }),
            );
        });
        return Ok(TerminalStartResult {
            ok: true,
            error: String::new(),
            pid: None,
            session_id: Some(session_id),
            final_output: None,
            exit_code: None,
        });
    }

    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(PtySize {
            rows: options.rows as u16,
            cols: options.cols as u16,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| e.to_string())?;

    let mut cmd = CommandBuilder::new(&binary_path);
    cmd.args(args.iter().map(String::as_str));
    cmd.cwd(&workspace_dir_string);
    cmd.env("TERM", "xterm-256color");
    cmd.env("COLORTERM", "truecolor");

    if !api_key.is_empty() {
        if provider == "nvidia-nim" {
            cmd.env("NVIDIA_API_KEY", &api_key);
            cmd.env("NVIDIA_NIM_API_KEY", &api_key);
        } else {
            cmd.env("DEEPSEEK_API_KEY", &api_key);
        }
    }
    if !normalized_base_url.is_empty() {
        cmd.env("DEEPSEEK_BASE_URL", &normalized_base_url);
    }
    if !model.is_empty() {
        cmd.env("DEEPSEEK_MODEL", &model);
    }
    if !provider.is_empty() {
        cmd.env("DEEPSEEK_PROVIDER", &provider);
    }
    if let Some(path) = &mcp_config_path {
        cmd.env("DEEPSEEK_MCP_CONFIG", path);
        cmd.env(
            "DEEPSEEK_DESKTOP_ENABLED_MCP",
            settings.enabled_mcp_servers.join(","),
        );
        info!("MCP config set: {}", path);
    }
    if settings.skills_enabled {
        cmd.env("DEEPSEEK_SKILLS_DIR", &skills_dir_for_env);
        cmd.env(
            "DEEPSEEK_DESKTOP_ENABLED_SKILLS",
            settings.enabled_skills.join(","),
        );
    }

    if matches!(options.launch_action.as_str(), "exec" | "plan" | "yolo") {
        if settings.harness_enabled && provider == "deepseek" && !should_use_mcp_cli {
            info!(
                "Starting DeepSeek HTTP streaming session: {}, model={}, url={}",
                session_id,
                model,
                chat_completions_url(&normalized_base_url)
            );
            let prompt = args.last().cloned().unwrap_or_default();
            let stream_app = app.clone();
            let stream_session_id = session_id.clone();
            let stream_api_key = api_key.clone();
            let stream_base_url = normalized_base_url.clone();
            let stream_model = model.clone();
            std::thread::spawn(move || {
                let _ = stream_app.emit(
                    "terminal:data",
                    serde_json::json!({
                        "sessionId": stream_session_id.clone(),
                        "data": format!(
                            "DeepSeek stream started: model={}, url={}\r\n\r\n",
                            stream_model,
                            chat_completions_url(&stream_base_url)
                        ),
                    }),
                );
                let (output, exit_code) = match run_deepseek_streaming(
                    &stream_app,
                    &stream_session_id,
                    &stream_api_key,
                    &stream_base_url,
                    &stream_model,
                    &prompt,
                ) {
                    Ok((output, exit_code)) => {
                        info!(
                            "DeepSeek HTTP streaming session completed: {}, exitCode={}, outputBytes={}",
                            stream_session_id,
                            exit_code,
                            output.len()
                        );
                        (output, exit_code)
                    }
                    Err(error) => {
                        error!("DeepSeek HTTP streaming failed: {}", error);
                        (format!("DeepSeek 流式请求失败：{}", error), -1)
                    }
                };
                if exit_code != 0 {
                    let _ = stream_app.emit(
                        "terminal:data",
                        serde_json::json!({
                            "sessionId": stream_session_id.clone(),
                            "data": output,
                        }),
                    );
                }
                let _ = stream_app.emit(
                    "terminal:exit",
                    serde_json::json!({
                        "sessionId": stream_session_id,
                        "exitCode": exit_code,
                        "finalOutput": output,
                    }),
                );
            });
            return Ok(TerminalStartResult {
                ok: true,
                error: String::new(),
                pid: None,
                session_id: Some(session_id),
                final_output: None,
                exit_code: None,
            });
        }

        let mut child = pair.slave.spawn_command(cmd).map_err(|e| {
            error!("Failed to spawn deepseek exec PTY: {}", e);
            e.to_string()
        })?;
        if should_use_mcp_cli {
            emit_terminal_data(
                &app,
                &session_id,
                "[tool] 已启动 MCP CLI 进程，正在等待模型规划和工具调用输出...\r\n".to_string(),
            );
        }
        let pid = child.process_id();
        let killer = child.clone_killer();
        let writer = pair.master.take_writer().map_err(|e| e.to_string())?;
        {
            let mut sessions = TERMINAL_SESSIONS.lock().unwrap();
            sessions.insert(session_id.clone(), TerminalSession::new(writer, killer));
        }

        let exec_app = app.clone();
        let exec_session_id = session_id.clone();
        let output_buffer = Arc::new(Mutex::new(String::new()));
        let reader_output = output_buffer.clone();
        let reader_app = app.clone();
        let reader_session_id = session_id.clone();
        std::thread::spawn(move || {
            let mut reader = pair.master.try_clone_reader().unwrap();
            let mut buffer = [0u8; 4096];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        let data = String::from_utf8_lossy(&buffer[..n]).to_string();
                        if let Ok(mut output) = reader_output.lock() {
                            output.push_str(&data);
                            if output.len() > 120000 {
                                let keep_from = output.len().saturating_sub(120000);
                                *output = output[keep_from..].to_string();
                            }
                        }
                        emit_terminal_data(&reader_app, &reader_session_id, data);
                    }
                    Err(_) => break,
                }
            }
        });

        std::thread::spawn(move || {
            let exit_code = child
                .wait()
                .ok()
                .map(|status| status.exit_code() as i32)
                .unwrap_or(-1);
            {
                let mut sessions = TERMINAL_SESSIONS.lock().unwrap();
                sessions.remove(&exec_session_id);
            }
            let output = output_buffer
                .lock()
                .map(|output| output.clone())
                .unwrap_or_default();
            if let Ok(mut results) = TERMINAL_RESULTS.lock() {
                results.insert(
                    exec_session_id.clone(),
                    TerminalCompletedResult {
                        session_id: exec_session_id.clone(),
                        exit_code,
                        final_output: output.clone(),
                    },
                );
            }
            let _ = exec_app.emit(
                "terminal:exit",
                serde_json::json!({
                    "sessionId": exec_session_id,
                    "exitCode": exit_code,
                    "finalOutput": output,
                }),
            );
        });

        return Ok(TerminalStartResult {
            ok: true,
            error: String::new(),
            pid,
            session_id: Some(session_id),
            final_output: None,
            exit_code: None,
        });
    }

    let mut child = pair.slave.spawn_command(cmd).map_err(|e| {
        error!("Failed to spawn deepseek: {}", e);
        e.to_string()
    })?;
    let pid = child.process_id();
    let killer = child.clone_killer();

    let writer = pair.master.take_writer().map_err(|e| e.to_string())?;

    {
        let mut sessions = TERMINAL_SESSIONS.lock().unwrap();
        sessions.insert(session_id.clone(), TerminalSession::new(writer, killer));
    }

    let reader_session_id = session_id.clone();
    let reader_app = app.clone();

    std::thread::spawn(move || {
        let mut reader = pair.master.try_clone_reader().unwrap();
        let mut buffer = [0u8; 4096];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(n) => {
                    if let Ok(data) = String::from_utf8(buffer[..n].to_vec()) {
                        let _ = reader_app.emit(
                            "terminal:data",
                            serde_json::json!({
                                "sessionId": reader_session_id,
                                "data": data,
                            }),
                        );
                    }
                }
                Err(_) => break,
            }
        }
    });

    let wait_session_id = session_id.clone();
    let wait_app = app.clone();
    std::thread::spawn(move || {
        let exit_code = child.wait().ok().map(|status| status.exit_code() as i32);
        {
            let mut sessions = TERMINAL_SESSIONS.lock().unwrap();
            sessions.remove(&wait_session_id);
        }
        let _ = wait_app.emit(
            "terminal:exit",
            serde_json::json!({
                "sessionId": wait_session_id,
                "exitCode": exit_code.unwrap_or(-1),
            }),
        );
    });

    info!("Terminal session started: {}", session_id);

    Ok(TerminalStartResult {
        ok: true,
        error: String::new(),
        pid,
        session_id: Some(session_id),
        final_output: None,
        exit_code: None,
    })
}

#[tauri::command]
fn terminal_input(session_id: String, data: String) -> Result<(), String> {
    let mut sessions = TERMINAL_SESSIONS.lock().unwrap();
    if let Some(session) = sessions.get_mut(&session_id) {
        session.write(&data)?;
    }
    Ok(())
}

#[tauri::command]
fn terminal_resize(_session_id: String, _cols: i32, _rows: i32) -> Result<(), String> {
    info!("Terminal resize requested");
    Ok(())
}

#[tauri::command]
fn terminal_stop(session_id: String) -> Result<(), String> {
    let mut sessions = TERMINAL_SESSIONS.lock().unwrap();
    if let Some(mut session) = sessions.remove(&session_id) {
        session.kill()?;
    }
    info!("Terminal session stopped: {}", session_id);
    Ok(())
}

#[tauri::command]
fn terminal_result(session_id: String) -> Option<TerminalCompletedResult> {
    if session_id.is_empty() {
        return None;
    }
    TERMINAL_RESULTS
        .lock()
        .ok()
        .and_then(|mut results| results.remove(&session_id))
}

// Skills commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillTemplate {
    pub id: String,
    pub name: String,
    pub description: String,
    pub source: String,
    pub origin: String,
    pub path: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillTemplateActionResult {
    pub ok: bool,
    pub error: String,
    pub skill: Option<SkillTemplate>,
    #[serde(rename = "skillRoot")]
    pub skill_root: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillImportActionResult {
    pub ok: bool,
    pub error: String,
    pub skills: Vec<SkillTemplate>,
    #[serde(rename = "skillRoot")]
    pub skill_root: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomizationResult {
    #[serde(rename = "skillRoot")]
    pub skill_root: String,
    #[serde(rename = "skillTemplates")]
    pub skill_templates: HashMap<String, SkillTemplate>,
    #[serde(rename = "mcpConfigPath")]
    pub mcp_config_path: String,
    #[serde(rename = "mcpConfigSource")]
    pub mcp_config_source: String,
    #[serde(rename = "mcpConfigText")]
    pub mcp_config_text: String,
    #[serde(rename = "mcpConfigError")]
    pub mcp_config_error: String,
}

#[tauri::command]
fn get_customization(settings: Settings) -> Result<CustomizationResult, String> {
    let skill_root = if settings.skills_dir.is_empty() {
        get_skills_dir().to_string_lossy().to_string()
    } else {
        settings.skills_dir.clone()
    };
    let skill_templates = load_skill_templates(&PathBuf::from(&skill_root))?;
    let workspace_dir = resolve_workspace_dir(&settings.workspace_path);
    let (mcp_config_path, mcp_config_source, mcp_config_text, mcp_config_error) =
        match mcp_config_text_for_settings(&settings, &workspace_dir) {
            Ok((path, source, text)) => (path, source, text, String::new()),
            Err(error) => (
                settings.mcp_config_path.clone(),
                "missing".to_string(),
                "{}".to_string(),
                error,
            ),
        };

    Ok(CustomizationResult {
        skill_root,
        skill_templates,
        mcp_config_path,
        mcp_config_source,
        mcp_config_text,
        mcp_config_error,
    })
}

#[tauri::command]
fn create_skill_template(
    skill_id: String,
    name: String,
    description: String,
    content: String,
) -> Result<SkillTemplateActionResult, String> {
    let skills_dir = get_skills_dir();
    let id = sanitize_id(
        if skill_id.trim().is_empty() {
            &name
        } else {
            &skill_id
        },
        "custom-skill",
    );
    let display_name = if name.trim().is_empty() {
        id.clone()
    } else {
        name.trim().to_string()
    };
    let description = if description.trim().is_empty() {
        format!("Use when {} guidance is needed.", display_name)
    } else {
        description.trim().to_string()
    };
    let skill_dir = skills_dir.join(&id);
    let skill_file = skill_dir.join("SKILL.md");

    fs::create_dir_all(&skill_dir).map_err(|e| e.to_string())?;

    let body = if content.trim().is_empty() {
        "## Overview\n\nDescribe the reusable workflow, trigger conditions, and verification steps for this skill.\n"
            .to_string()
    } else {
        content
    };
    let frontmatter = format!(
        "---\nname: {}\ndescription: {}\n---\n\n# {}\n\n{}",
        id, description, display_name, body
    );

    let content_for_return = frontmatter.clone();
    fs::write(&skill_file, &frontmatter).map_err(|e| e.to_string())?;

    let skill = SkillTemplate {
        id,
        name: display_name,
        description,
        source: "file".to_string(),
        origin: "custom".to_string(),
        path: skill_file.to_string_lossy().to_string(),
        content: content_for_return,
    };

    Ok(SkillTemplateActionResult {
        ok: true,
        error: String::new(),
        skill: Some(skill.clone()),
        skill_root: skills_dir.to_string_lossy().to_string(),
        path: skill.path,
    })
}

#[tauri::command]
fn import_skill_directory(source_path: String) -> Result<SkillImportActionResult, String> {
    let source = PathBuf::from(&source_path);
    if !source.exists() {
        return Err("Source directory does not exist".to_string());
    }

    let skills_dir = get_skills_dir();
    fs::create_dir_all(&skills_dir).map_err(|e| e.to_string())?;

    let mut imported = vec![];

    if let Ok(entries) = fs::read_dir(&source) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("SKILL.md").exists() {
                let skill_file = path.join("SKILL.md");
                if let Ok(content) = fs::read_to_string(&skill_file) {
                    let id = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let id = sanitize_id(&id, "imported-skill");
                    let skill_dir = skills_dir.join(&id);
                    let copied_file = skill_dir.join("SKILL.md");
                    fs::create_dir_all(&skill_dir).map_err(|e| e.to_string())?;
                    fs::copy(&skill_file, &copied_file).map_err(|e| e.to_string())?;

                    imported.push(SkillTemplate {
                        id: id.clone(),
                        name: skill_name_from_content(&id, &content),
                        description: skill_description_from_content(&content),
                        source: "file".to_string(),
                        origin: "custom".to_string(),
                        path: copied_file.to_string_lossy().to_string(),
                        content,
                    });
                }
            }
        }
    }

    Ok(SkillImportActionResult {
        ok: true,
        error: String::new(),
        skills: imported,
        skill_root: skills_dir.to_string_lossy().to_string(),
        path: skills_dir.to_string_lossy().to_string(),
    })
}

// MCP commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfigSaveResult {
    pub ok: bool,
    pub error: String,
    pub path: String,
    pub content: String,
}

#[tauri::command]
fn save_mcp_config(content: String) -> Result<McpConfigSaveResult, String> {
    ensure_user_data_dir().map_err(|e| e.to_string())?;
    let parsed: Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
    let formatted = serde_json::to_string_pretty(&parsed).map_err(|e| e.to_string())?;
    let path = get_user_data_dir().join("mcp.custom.json");
    fs::write(&path, &formatted).map_err(|e| e.to_string())?;
    Ok(McpConfigSaveResult {
        ok: true,
        error: String::new(),
        path: path.to_string_lossy().to_string(),
        content: formatted,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerTestResult {
    pub ok: bool,
    #[serde(rename = "testedAt")]
    pub tested_at: String,
    #[serde(rename = "configPath")]
    pub config_path: String,
    pub servers: Vec<McpServerStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerStatus {
    pub id: String,
    pub command: String,
    pub args: Vec<String>,
    pub url: String,
    pub ok: bool,
    #[serde(rename = "commandFound")]
    pub command_found: bool,
    #[serde(rename = "missingEnv")]
    pub missing_env: Vec<String>,
    pub warnings: Vec<String>,
    pub error: String,
}

#[tauri::command]
fn test_mcp_servers(settings: Settings) -> McpServerTestResult {
    let workspace_dir = resolve_workspace_dir(&settings.workspace_path);
    let config_result = mcp_config_text_for_settings(&settings, &workspace_dir);
    let (config_path, config_text) = match config_result {
        Ok((path, _, text)) => (path, text),
        Err(error) => {
            return McpServerTestResult {
                ok: false,
                tested_at: Utc::now().to_rfc3339(),
                config_path: settings.mcp_config_path,
                servers: vec![McpServerStatus {
                    id: "config".to_string(),
                    command: String::new(),
                    args: vec![],
                    url: String::new(),
                    ok: false,
                    command_found: false,
                    missing_env: vec![],
                    warnings: vec![error.clone()],
                    error,
                }],
            }
        }
    };
    let parsed: Value = match serde_json::from_str(&config_text) {
        Ok(value) => value,
        Err(error) => {
            return McpServerTestResult {
                ok: false,
                tested_at: Utc::now().to_rfc3339(),
                config_path,
                servers: vec![McpServerStatus {
                    id: "config".to_string(),
                    command: String::new(),
                    args: vec![],
                    url: String::new(),
                    ok: false,
                    command_found: false,
                    missing_env: vec![],
                    warnings: vec![format!("Invalid JSON: {}", error)],
                    error: error.to_string(),
                }],
            }
        }
    };
    let servers = parsed
        .get("servers")
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();
    let mut statuses = Vec::new();
    for (id, server) in servers {
        let disabled = server
            .get("disabled")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let enabled = server
            .get("enabled")
            .and_then(|value| value.as_bool())
            .unwrap_or(true);
        if disabled || !enabled {
            continue;
        }
        let command = server
            .get("command")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let args = server
            .get("args")
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let url = server
            .get("url")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let mut missing_env = Vec::new();
        if let Some(env) = server.get("env").and_then(|value| value.as_object()) {
            for (key, value) in env {
                let configured_value = value.as_str().unwrap_or_default().trim();
                let missing = configured_value.is_empty() && std::env::var_os(key).is_none();
                if missing {
                    missing_env.push(key.clone());
                }
            }
        }
        let command_found = !url.trim().is_empty() || command_exists(&command);
        let mut warnings = Vec::new();
        if !command_found {
            warnings.push(format!("Command not found: {}", command));
        }
        if !missing_env.is_empty() {
            warnings.push(format!(
                "Missing environment variables: {}",
                missing_env.join(", ")
            ));
        }
        statuses.push(McpServerStatus {
            id,
            command,
            args,
            url,
            ok: command_found && missing_env.is_empty(),
            command_found,
            missing_env,
            warnings: warnings.clone(),
            error: warnings.join("; "),
        });
    }
    let ok = statuses.iter().all(|server| server.ok);
    McpServerTestResult {
        ok,
        tested_at: Utc::now().to_rfc3339(),
        config_path,
        servers: statuses,
    }
}

// Runtime check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeCheck {
    pub selected: String,
    #[serde(rename = "selectedExists")]
    pub selected_exists: bool,
    pub bundled: String,
    #[serde(rename = "bundledExists")]
    pub bundled_exists: bool,
    pub system: String,
    #[serde(rename = "systemExists")]
    pub system_exists: bool,
    pub version: String,
}

#[tauri::command]
fn check_runtime(settings: Settings) -> RuntimeCheck {
    RuntimeCheck {
        selected: if settings.binary_mode == "custom" {
            settings.custom_binary_path
        } else {
            "deepseek".to_string()
        },
        selected_exists: true,
        bundled: String::new(),
        bundled_exists: false,
        system: "deepseek".to_string(),
        system_exists: Command::new("deepseek").arg("--version").output().is_ok(),
        version: String::new(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    pub status: String,
    pub source: String,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    pub mode: String,
    #[serde(rename = "workspacePath")]
    pub workspace_path: String,
    pub pid: u32,
    pub command: String,
    pub args: Vec<String>,
    #[serde(rename = "startedAt")]
    pub started_at: String,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
    #[serde(rename = "lastExit")]
    pub last_exit: Option<serde_json::Value>,
    pub agents: Vec<serde_json::Value>,
    pub counts: serde_json::Value,
    pub events: Vec<serde_json::Value>,
}

#[tauri::command]
fn get_runtime_snapshot() -> RuntimeSnapshot {
    RuntimeSnapshot {
        status: "idle".to_string(),
        source: "none".to_string(),
        session_id: String::new(),
        mode: String::new(),
        workspace_path: String::new(),
        pid: 0,
        command: String::new(),
        args: vec![],
        started_at: String::new(),
        updated_at: Utc::now().to_rfc3339(),
        last_exit: None,
        agents: vec![],
        counts: serde_json::json!({
            "total": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }),
        events: vec![],
    }
}

// Editor open
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorOpenResult {
    pub ok: bool,
    pub error: String,
    pub editor: Option<String>,
    pub path: Option<String>,
    pub command: Option<String>,
}

#[tauri::command]
fn open_workspace_editor(editor: String, workspace_path: String) -> EditorOpenResult {
    let cmd = match editor.as_str() {
        "cursor" => {
            if cfg!(target_os = "windows") {
                vec!["cursor.cmd", "code"]
            } else {
                vec!["cursor"]
            }
        }
        "vscode" => {
            if cfg!(target_os = "windows") {
                vec!["code.cmd", "code"]
            } else {
                vec!["code"]
            }
        }
        _ => {
            return EditorOpenResult {
                ok: false,
                error: "Unsupported editor".to_string(),
                editor: None,
                path: None,
                command: None,
            }
        }
    };

    for command in &cmd {
        let output = Command::new(command).arg(&workspace_path).output();

        if let Ok(output) = output {
            if output.status.success() {
                return EditorOpenResult {
                    ok: true,
                    error: String::new(),
                    editor: Some(editor),
                    path: Some(workspace_path.clone()),
                    command: Some(format!("{} {}", command, workspace_path)),
                };
            }
        }
    }

    EditorOpenResult {
        ok: false,
        error: format!("{} command is not available", editor),
        editor: None,
        path: None,
        command: None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryResult {
    pub ok: bool,
    pub content: String,
}

// ── Automation types ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationTask {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub prompt: String,
    #[serde(rename = "workspacePath")]
    pub workspace_path: String,
    pub frequency: String,
    pub minute: i32,
    pub hour: i32,
    pub weekday: i32,
    #[serde(rename = "customSchedule")]
    pub custom_schedule: String,
    pub schedule: String,
    pub rrule: String,
    pub timezone: String,
    pub status: String,
    pub enabled: bool,
    pub installed: bool,
    #[serde(rename = "cronPath")]
    pub cron_path: String,
    #[serde(rename = "logPath")]
    pub log_path: String,
    #[serde(rename = "commandPreview")]
    pub command_preview: String,
    #[serde(rename = "runtimePath")]
    pub runtime_path: String,
    #[serde(rename = "runnerPath")]
    pub runner_path: String,
    #[serde(rename = "runArgs")]
    pub run_args: Vec<String>,
    pub provider: String,
    pub model: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    #[serde(rename = "mcpConfigPath")]
    pub mcp_config_path: String,
    #[serde(rename = "skillsDir")]
    pub skills_dir: String,
    #[serde(rename = "enabledSkills")]
    pub enabled_skills: Vec<String>,
    #[serde(rename = "mcpEnabled")]
    pub mcp_enabled: bool,
    #[serde(rename = "enabledMcpServers")]
    pub enabled_mcp_servers: Vec<String>,
    #[serde(rename = "allowShell")]
    pub allow_shell: bool,
    #[serde(rename = "maxSubagents")]
    pub max_subagents: i32,
    pub error: Option<String>,
    #[serde(rename = "createdAt")]
    pub created_at: String,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
    #[serde(rename = "lastGeneratedAt")]
    pub last_generated_at: String,
    #[serde(rename = "lastInstalledAt")]
    pub last_installed_at: String,
    #[serde(rename = "lastRunAt")]
    #[serde(default)]
    pub last_run_at: String,
    #[serde(rename = "lastRunResult")]
    #[serde(default)]
    pub last_run_result: String,
    #[serde(rename = "lastRunOutput")]
    #[serde(default)]
    pub last_run_output: String,
    #[serde(rename = "lastRunExitCode")]
    #[serde(default)]
    pub last_run_exit_code: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationStore {
    pub version: i32,
    pub tasks: Vec<AutomationTask>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationActionResult {
    pub ok: bool,
    pub error: Option<String>,
    pub task: Option<AutomationTask>,
    pub tasks: Vec<AutomationTask>,
}

fn get_automations_path() -> PathBuf {
    get_user_data_dir().join("automations.json")
}

fn load_automations() -> AutomationStore {
    let path = get_automations_path();
    match fs::read_to_string(&path) {
        Ok(json) => serde_json::from_str(&json).unwrap_or(AutomationStore {
            version: 1,
            tasks: vec![],
        }),
        Err(_) => AutomationStore {
            version: 1,
            tasks: vec![],
        },
    }
}

fn save_automations(store: &AutomationStore) -> Result<(), String> {
    let path = get_automations_path();
    let json =
        serde_json::to_string_pretty(store).map_err(|e| format!("Failed to serialize: {}", e))?;
    fs::write(&path, json).map_err(|e| format!("Failed to write automations: {}", e))
}

// ── Automation commands ──

#[tauri::command]
fn get_automations() -> AutomationStore {
    load_automations()
}

#[tauri::command]
fn save_automation(task: Value) -> AutomationActionResult {
    let mut store = load_automations();
    let now = chrono::Utc::now().to_rfc3339();

    // Accept partial task data (AutomationDraft) and fill in defaults
    let id = task["id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("automation-{}", std::time::UNIX_EPOCH.elapsed().unwrap().as_millis()));

    // Find existing task to merge with
    let existing = store.tasks.iter().find(|t| t.id == id);

    let name = task["name"].as_str().unwrap_or("").to_string();
    let prompt = task["prompt"].as_str().unwrap_or("").to_string();
    let workspace_path = task["workspacePath"]
        .as_str()
        .unwrap_or(&existing.map(|t| t.workspace_path.as_str()).unwrap_or(""))
        .to_string();
    let minute = task["minute"].as_i64().unwrap_or(existing.map(|t| t.minute as i64).unwrap_or(0)) as i32;
    let hour = task["hour"].as_i64().unwrap_or(existing.map(|t| t.hour as i64).unwrap_or(9)) as i32;
    let frequency = task["frequency"]
        .as_str()
        .unwrap_or(&existing.map(|t| t.frequency.as_str()).unwrap_or("daily"))
        .to_string();
    let status = task["status"]
        .as_str()
        .unwrap_or("PAUSED")
        .to_string();
    let enabled = status == "ACTIVE" || task["enabled"].as_bool().unwrap_or(false);
    let rrule = task["rrule"]
        .as_str()
        .unwrap_or(&format!("FREQ=DAILY;BYHOUR={};BYMINUTE={}", hour, minute))
        .to_string();
    let schedule = task["schedule"]
        .as_str()
        .unwrap_or(&format!("Daily {:02}:{:02}", hour, minute))
        .to_string();
    let timezone = task["timezone"]
        .as_str()
        .unwrap_or(existing.map(|t| t.timezone.as_str()).unwrap_or("Asia/Shanghai"))
        .to_string();
    let custom_schedule = task["customSchedule"]
        .as_str()
        .unwrap_or(&format!("{} {} * * *", minute, hour))
        .to_string();
    let cron_path = task["cronPath"]
        .as_str()
        .unwrap_or(&format!(".deepseek/cron/{}.cron", id))
        .to_string();
    let log_path = task["logPath"]
        .as_str()
        .unwrap_or(&format!(".deepseek/logs/{}.log", id))
        .to_string();

    let created_at = existing
        .map(|t| t.created_at.clone())
        .unwrap_or_else(|| now.clone());

    let last_installed_at = if enabled {
        now.clone()
    } else {
        existing.map(|t| t.last_installed_at.clone()).unwrap_or_default()
    };

    let provider = task["provider"]
        .as_str()
        .unwrap_or(existing.map(|t| t.provider.as_str()).unwrap_or("deepseek"))
        .to_string();
    let model = task["model"]
        .as_str()
        .unwrap_or(existing.map(|t| t.model.as_str()).unwrap_or("deepseek-v4-pro"))
        .to_string();
    let base_url = task["baseUrl"]
        .as_str()
        .unwrap_or(existing.map(|t| t.base_url.as_str()).unwrap_or("https://api.deepseek.com"))
        .to_string();

    let updated = AutomationTask {
        id: id.clone(),
        kind: "cron".to_string(),
        name: if name.is_empty() { "Scheduled Task".to_string() } else { name },
        prompt,
        workspace_path,
        frequency,
        minute,
        hour,
        weekday: task["weekday"].as_i64().unwrap_or(existing.map(|t| t.weekday as i64).unwrap_or(1)) as i32,
        custom_schedule,
        schedule,
        rrule,
        timezone,
        status: if enabled { "ACTIVE".to_string() } else { "PAUSED".to_string() },
        enabled,
        installed: enabled,
        cron_path,
        log_path,
        command_preview: format!("deepseek exec --auto '{}'", task["prompt"].as_str().unwrap_or("")),
        runtime_path: existing.map(|t| t.runtime_path.clone()).unwrap_or_default(),
        runner_path: existing.map(|t| t.runner_path.clone()).unwrap_or_default(),
        run_args: existing
            .map(|t| t.run_args.clone())
            .unwrap_or_default(),
        provider,
        model,
        base_url,
        mcp_config_path: task["mcpConfigPath"]
            .as_str()
            .unwrap_or(existing.map(|t| t.mcp_config_path.as_str()).unwrap_or(""))
            .to_string(),
        skills_dir: task["skillsDir"]
            .as_str()
            .unwrap_or(existing.map(|t| t.skills_dir.as_str()).unwrap_or(""))
            .to_string(),
        enabled_skills: task["enabledSkills"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_else(|| existing.map(|t| t.enabled_skills.clone()).unwrap_or_default()),
        mcp_enabled: task["mcpEnabled"]
            .as_bool()
            .unwrap_or(existing.map(|t| t.mcp_enabled).unwrap_or(false)),
        enabled_mcp_servers: task["enabledMcpServers"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_else(|| existing.map(|t| t.enabled_mcp_servers.clone()).unwrap_or_default()),
        allow_shell: task["allowShell"]
            .as_bool()
            .unwrap_or(existing.map(|t| t.allow_shell).unwrap_or(false)),
        max_subagents: task["maxSubagents"]
            .as_i64()
            .unwrap_or(existing.map(|t| t.max_subagents as i64).unwrap_or(10)) as i32,
        error: None,
        created_at,
        updated_at: now.clone(),
        last_generated_at: now.clone(),
        last_installed_at,
        last_run_at: existing
            .map(|t| t.last_run_at.clone())
            .unwrap_or_default(),
        last_run_result: existing
            .map(|t| t.last_run_result.clone())
            .unwrap_or_default(),
        last_run_output: existing
            .map(|t| t.last_run_output.clone())
            .unwrap_or_default(),
        last_run_exit_code: existing
            .map(|t| t.last_run_exit_code)
            .unwrap_or(0),
    };

    if let Some(pos) = store.tasks.iter().position(|t| t.id == id) {
        store.tasks[pos] = updated.clone();
    } else {
        store.tasks.insert(0, updated.clone());
    }

    match save_automations(&store) {
        Ok(()) => AutomationActionResult {
            ok: true,
            error: None,
            task: Some(updated),
            tasks: store.tasks,
        },
        Err(e) => AutomationActionResult {
            ok: false,
            error: Some(e),
            task: None,
            tasks: store.tasks,
        },
    }
}

#[tauri::command]
fn get_task_log(task_id: String) -> String {
    let log_path = get_user_data_dir().join("logs").join(format!("task_{}.log", task_id));
    fs::read_to_string(&log_path).unwrap_or_default()
}

#[tauri::command]
fn delete_automation(id: String) -> AutomationActionResult {
    let mut store = load_automations();
    store.tasks.retain(|t| t.id != id);

    match save_automations(&store) {
        Ok(()) => AutomationActionResult {
            ok: true,
            error: None,
            task: None,
            tasks: store.tasks,
        },
        Err(e) => AutomationActionResult {
            ok: false,
            error: Some(e),
            task: None,
            tasks: store.tasks,
        },
    }
}

#[tauri::command]
fn stop_automation_task(id: String) -> AutomationActionResult {
    // Kill the running child process if it exists
    if let Some(pid) = RUNNING_TASK_PIDS.lock().unwrap().remove(&id) {
        let pid_str = pid.to_string();
        let _ = Command::new("taskkill")
            .args(["/F", "/PID", &pid_str])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
    }

    // Update the task status
    let mut store = load_automations();
    if let Some(task) = store.tasks.iter_mut().find(|t| t.id == id) {
        if task.last_run_result == "running" {
            task.last_run_result = "error".to_string();
            task.last_run_exit_code = -1;
            // Append stop notice to existing output
            let log_path = get_user_data_dir()
                .join("logs")
                .join(format!("task_{}.log", id));
            if let Ok(log_output) = fs::read_to_string(&log_path) {
                let truncated = if log_output.len() > 8000 {
                    format!(
                        "{}...\n[ truncated, total {} chars ]",
                        &log_output[..8000],
                        log_output.len()
                    )
                } else {
                    log_output
                };
                task.last_run_output = format!(
                    "{}\n\n[Task was stopped by user]",
                    truncated
                );
            } else if task.last_run_output.is_empty() {
                task.last_run_output = "[Task was stopped by user]".to_string();
            } else {
                task.last_run_output =
                    format!("{}\n\n[Task was stopped by user]", task.last_run_output);
            }
        }
    }

    match save_automations(&store) {
        Ok(()) => AutomationActionResult {
            ok: true,
            error: None,
            task: store.tasks.iter().find(|t| t.id == id).cloned(),
            tasks: store.tasks,
        },
        Err(e) => AutomationActionResult {
            ok: false,
            error: Some(e),
            task: None,
            tasks: store.tasks,
        },
    }
}

// ── Automation scheduler ──

fn start_automation_scheduler() {
    std::thread::spawn(|| {
        // Run first check after 10s to let the app initialize
        std::thread::sleep(std::time::Duration::from_secs(10));

        // Recover orphaned tasks: tasks that were "running" when the app last shut down
        {
            let mut store = load_automations();
            let mut recovered = false;
            for task in &mut store.tasks {
                if task.last_run_result == "running" {
                    task.last_run_result = "error".to_string();
                    task.last_run_exit_code = -1;
                    // Try to recover the log file content
                    let log_path = get_user_data_dir()
                        .join("logs")
                        .join(format!("task_{}.log", task.id));
                    if let Ok(log_output) = fs::read_to_string(&log_path) {
                        let truncated = if log_output.len() > 8000 {
                            format!(
                                "{}...\n[ truncated, total {} chars ]",
                                &log_output[..8000],
                                log_output.len()
                            )
                        } else {
                            log_output
                        };
                        task.last_run_output = format!(
                            "[App was restarted while task was running]\n\n{}",
                            truncated
                        );
                    } else {
                        task.last_run_output =
                            "[App was restarted while this task was running — output was lost]"
                                .to_string();
                    }
                    recovered = true;
                }
            }
            if recovered {
                let _ = save_automations(&store);
            }
        }

        loop {
            let now = chrono::Local::now();
            let current_hour = now.hour() as i32;
            let current_minute = now.minute() as i32;
            let now_str = now.format("%Y-%m-%d %H:%M").to_string();

            let store = load_automations();
            let mut modified = false;
            let mut new_store = store;
            new_store.version += 0; // keep same version

            for task in &mut new_store.tasks {
                if task.status != "ACTIVE" {
                    continue;
                }
                // Check if it's time to run (within the current minute window)
                if task.hour == current_hour && task.minute == current_minute {
                    // Don't re-run if already ran in this minute window
                    if task.last_run_at == now_str {
                        continue;
                    }
                    info!(
                        "Scheduler: running automation task '{}' ({})",
                        task.name, task.id
                    );

                    // Execute the task
                    let provider = task.provider.clone();
                    let api_key = get_api_key(provider.clone()).unwrap_or_default();
                    let model = task.model.clone();
                    let base_url = task.base_url.trim().trim_end_matches('/').to_string();
                    let prompt = task.prompt.clone();

                    // Fallback to a valid workspace directory if none specified
                    let workspace = if task.workspace_path.trim().is_empty() {
                        dirs::desktop_dir()
                            .or_else(|| dirs::home_dir())
                            .unwrap_or_else(|| PathBuf::from("."))
                    } else {
                        PathBuf::from(&task.workspace_path)
                    };
                    let workspace_str = workspace.to_string_lossy().to_string();

                    let mut args: Vec<String> = Vec::new();
                    if !provider.is_empty() {
                        args.push("--provider".to_string());
                        args.push(provider.clone());
                    }
                    if !model.is_empty() {
                        args.push("--model".to_string());
                        args.push(model.clone());
                    }
                    if !base_url.is_empty() {
                        args.push("--base-url".to_string());
                        args.push(base_url.clone());
                    }
                    args.push("exec".to_string());
                    args.push("--auto".to_string());
                    args.push(prompt);

                    let binary_path = get_bundled_binary_path();
                    info!(
                        "Scheduler: binary={} workspace={} args={:?}",
                        binary_path, workspace_str, args
                    );

                    // Mark as running immediately (non-blocking)
                    task.last_run_at = now_str.clone();
                    task.last_run_result = "running".to_string();
                    task.last_run_output = String::new();
                    modified = true;

                    // Create a log file for real-time output viewing
                    let log_dir = get_user_data_dir().join("logs");
                    let _ = fs::create_dir_all(&log_dir);
                    let log_path = log_dir.join(format!("task_{}.log", task.id));
                    let log_file = match fs::File::create(&log_path) {
                        Ok(f) => f,
                        Err(e) => {
                            error!("Scheduler: cannot create log file {:?}: {}", log_path, e);
                            continue;
                        }
                    };

                    // Duplicate file handle for stderr so both streams go to the same file
                    let stderr_file = match log_file.try_clone() {
                        Ok(f) => f,
                        Err(e) => {
                            error!("Scheduler: cannot clone log file: {}", e);
                            continue;
                        }
                    };

                    // Spawn without blocking the scheduler loop
                    match Command::new(&binary_path)
                        .args(&args)
                        .current_dir(&workspace)
                        .env("DEEPSEEK_API_KEY", &api_key)
                        .env("DEEPSEEK_BASE_URL", &base_url)
                        .stdout(Stdio::from(log_file))
                        .stderr(Stdio::from(stderr_file))
                        .spawn()
                    {
                        Ok(mut child) => {
                            let task_id = task.id.clone();
                            let task_name = task.name.clone();
                            let task_log_path = log_path.clone();

                            // Store PID so the task can be stopped on demand
                            let pid = child.id();
                            RUNNING_TASK_PIDS
                                .lock()
                                .unwrap()
                                .insert(task_id.clone(), pid);

                            // Wait for completion in a background thread
                            std::thread::spawn(move || {
                                let result = child.wait();
                                // Remove PID from running tasks
                                RUNNING_TASK_PIDS
                                    .lock()
                                    .unwrap()
                                    .remove(&task_id);
                                let mut store = load_automations();
                                if let Some(t) = store.tasks.iter_mut().find(|t| t.id == task_id)
                                {
                                    // Read the log file to capture output
                                    let log_output = fs::read_to_string(&task_log_path)
                                        .unwrap_or_default();
                                    let truncated = if log_output.len() > 8000 {
                                        format!(
                                            "{}...\n[ truncated, total {} chars ]",
                                            &log_output[..8000],
                                            log_output.len()
                                        )
                                    } else {
                                        log_output
                                    };

                                    match result {
                                        Ok(status) => {
                                            let exit_code = status.code().unwrap_or(-1);
                                            info!(
                                                "Scheduler task '{}' completed. exit={} log_bytes={}",
                                                task_name, exit_code, truncated.len()
                                            );
                                            t.last_run_result = if exit_code == 0 {
                                                "success".to_string()
                                            } else {
                                                "failed".to_string()
                                            };
                                            t.last_run_exit_code = exit_code;
                                            t.last_run_output = truncated;
                                        }
                                        Err(e) => {
                                            error!(
                                                "Scheduler task '{}' wait failed: {}",
                                                task_name, e
                                            );
                                            t.last_run_result = "error".to_string();
                                            t.last_run_exit_code = -1;
                                            t.last_run_output = format!(
                                                "Process wait error: {}\n\n--- log ---\n{}",
                                                e, truncated
                                            );
                                        }
                                    }
                                    let _ = save_automations(&store);
                                }
                            });
                        }
                        Err(e) => {
                            let err_msg = format!(
                                "{} (binary={}, workspace={})",
                                e, binary_path, workspace_str
                            );
                            error!("Scheduler task '{}' spawn failed: {}", task.name, err_msg);
                            task.last_run_result = "error".to_string();
                            task.last_run_exit_code = -1;
                            task.last_run_output = err_msg;
                        }
                    }
                }
            }

            if modified {
                let _ = save_automations(&new_store);
            }

            std::thread::sleep(std::time::Duration::from_secs(30));
        }
    });
}

// ── Memory commands ──

#[tauri::command]
fn get_memory(file_type: String) -> MemoryResult {
    let path = match file_type.as_str() {
        "memory" => memory::memory_path(),
        "user" => memory::user_path(),
        _ => {
            return MemoryResult {
                ok: false,
                content: "Invalid file_type. Use 'memory' or 'user'.".to_string(),
            }
        }
    };
    let content = memory::load_memory(&path);
    MemoryResult { ok: true, content }
}

#[tauri::command]
fn save_memory(file_type: String, content: String) -> MemoryResult {
    let path = match file_type.as_str() {
        "memory" => memory::memory_path(),
        "user" => memory::user_path(),
        _ => {
            return MemoryResult {
                ok: false,
                content: "Invalid file_type. Use 'memory' or 'user'.".to_string(),
            }
        }
    };
    match memory::save_memory(&path, &content) {
        Ok(()) => MemoryResult {
            ok: true,
            content: "Saved.".to_string(),
        },
        Err(e) => MemoryResult {
            ok: false,
            content: e,
        },
    }
}

#[tauri::command]
fn append_memory_entry(entry: String) -> MemoryResult {
    match memory::append_memory(&entry) {
        Ok(()) => MemoryResult {
            ok: true,
            content: "Appended.".to_string(),
        },
        Err(e) => MemoryResult {
            ok: false,
            content: e,
        },
    }
}

fn setup_logging(log_dir: PathBuf) {
    let _ = fs::create_dir_all(&log_dir);

    let file_appender = RollingFileAppender::new(Rotation::DAILY, &log_dir, "ds-code.log");

    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(file_appender))
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    info!("DS-Code starting up");
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let log_dir = get_log_dir();
    setup_logging(log_dir);

    start_automation_scheduler();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_os::init())
        .setup(|app| {
            info!("Tauri app setup complete");
            let _ = app;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_settings,
            save_settings,
            get_api_key,
            read_image_data_url,
            read_media_data_url,
            save_api_key,
            choose_directory,
            choose_file,
            git_status,
            git_init,
            git_set_remote,
            git_commit,
            git_fetch,
            git_pull,
            git_push,
            terminal_start,
            terminal_input,
            terminal_resize,
            terminal_stop,
            terminal_result,
            get_customization,
            create_skill_template,
            import_skill_directory,
            save_mcp_config,
            test_mcp_servers,
            check_runtime,
            get_runtime_snapshot,
            open_workspace_editor,
            get_automations,
            save_automation,
            delete_automation,
            stop_automation_task,
            get_memory,
            save_memory,
            append_memory_entry,
            get_task_log,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
