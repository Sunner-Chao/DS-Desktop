use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};

use chrono::Utc;
use dirs::home_dir;
use once_cell::sync::Lazy;
use portable_pty::{native_pty_system, ChildKiller, CommandBuilder, PtySize};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Emitter};
use tauri_plugin_dialog::DialogExt;
use tracing::{error, info};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// Global state for terminal sessions
static TERMINAL_SESSIONS: Lazy<Arc<Mutex<HashMap<String, TerminalSession>>>> =
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));
static TERMINAL_RESULTS: Lazy<Arc<Mutex<HashMap<String, TerminalCompletedResult>>>> =
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

fn get_user_data_dir() -> PathBuf {
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
    // Get the path to the bundled deepseek binary from node_modules/deepseek-tui/bin/downloads
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
    // Fallback: try common locations
    let common_paths = [
        "D:\\pro_sunner\\demo_vscode\\DeepseekCode\\node_modules\\deepseek-tui\\bin\\downloads\\deepseek.exe",
    ];
    for path in &common_paths {
        if PathBuf::from(path).exists() {
            return path.to_string();
        }
    }
    // Last fallback: just "deepseek" in PATH
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

fn mcp_preset_server(id: &str, workspace_dir: &PathBuf) -> Option<Value> {
    let workspace = workspace_dir.to_string_lossy().to_string();
    let playwright_output_dir = workspace_dir
        .join(".ds-code")
        .join("playwright-output")
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
    let _ = fs::create_dir_all(workspace_dir.join(".ds-code").join("playwright-output"));
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

fn looks_like_web_screenshot_request(prompt: &str) -> bool {
    let request = current_user_request(prompt);
    let lower = request.to_lowercase();
    (request.contains("截图")
        || request.contains("截屏")
        || request.contains("截一张")
        || lower.contains("screenshot"))
        && (lower.contains("http://")
            || lower.contains("https://")
            || lower.contains("bilibili")
            || request.contains("哔哩哔哩"))
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
    if lower.contains("bilibili") || prompt.contains("哔哩哔哩") {
        return Some("https://www.bilibili.com".to_string());
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

fn run_playwright_screenshot(
    app: AppHandle,
    session_id: String,
    workspace_dir: PathBuf,
    prompt: String,
) -> (String, i32) {
    let request = current_user_request(&prompt);
    let Some(url) = extract_screenshot_url(&request) else {
        return (
            "没有找到可截图的网址。请提供 http 或 https 开头的网址。".to_string(),
            -1,
        );
    };
    let output_dir = workspace_dir.join(".ds-code").join("playwright-output");
    if let Err(error) = fs::create_dir_all(&output_dir) {
        return (format!("创建截图输出目录失败：{}", error), -1);
    }
    let url_name = sanitize_filename_piece(
        url.trim_start_matches("https://")
            .trim_start_matches("http://")
            .trim_start_matches("www."),
    );
    let output_path = output_dir.join(format!(
        "{}-{}.png",
        url_name,
        Utc::now().timestamp_millis()
    ));
    emit_terminal_data(
        &app,
        &session_id,
        format!(
            "正在使用 Playwright 打开 {}\r\n截图将保存到 {}\r\n",
            url,
            output_path.to_string_lossy()
        ),
    );
    let output_path_string = output_path.to_string_lossy().to_string();
    let mut command = npx_command();
    let result = command
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
        .output();

    match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!("{}{}", stdout, stderr);
            if !combined.trim().is_empty() {
                emit_terminal_data(&app, &session_id, combined);
            }
            if output.status.success() && output_path.exists() {
                (format!("截图已保存：{}", output_path.to_string_lossy()), 0)
            } else {
                (
                    format!(
                        "截图失败，退出码：{}。\n{}",
                        output.status.code().unwrap_or(-1),
                        stderr
                    ),
                    output.status.code().unwrap_or(-1),
                )
            }
        }
        Err(error) => (format!("启动 Playwright 截图失败：{}", error), -1),
    }
}

fn is_only_progress_output(output: &str) -> bool {
    let normalized = output.replace('\r', "\n");
    let clean = normalized
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .filter(|line| {
            let lower = line.to_lowercase();
            !lower.contains("正在请求") && !lower.contains("requesting")
        })
        .collect::<Vec<_>>();
    clean.is_empty()
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
    let chars: Vec<char> = data.chars().collect();
    if chars.len() <= 120 {
        let _ = app.emit(
            "terminal:data",
            serde_json::json!({
                "sessionId": session_id,
                "data": data,
            }),
        );
        return;
    }
    for chunk in chars.chunks(64) {
        let text: String = chunk.iter().collect();
        let _ = app.emit(
            "terminal:data",
            serde_json::json!({
                "sessionId": session_id,
                "data": text,
            }),
        );
        std::thread::sleep(std::time::Duration::from_millis(18));
    }
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
    let reader = BufReader::new(response);
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let value = serde_json::from_str::<Value>(data).map_err(|e| e.to_string())?;
        let (delta, reasoning_delta) = extract_stream_delta(&value);
        if !reasoning_delta.is_empty() {
            if reasoning.is_empty() {
                let marker = "\r\n[reasoning]\r\n";
                let _ = app.emit(
                    "terminal:data",
                    serde_json::json!({
                        "sessionId": session_id,
                        "data": marker,
                    }),
                );
            }
            reasoning.push_str(&reasoning_delta);
            emit_terminal_data(app, session_id, reasoning_delta);
        }
        if !delta.is_empty() {
            content.push_str(&delta);
            emit_terminal_data(app, session_id, delta);
        }
    }
    if content.trim().is_empty() {
        return Ok((reasoning, 0));
    }
    Ok((content, 0))
}

fn collect_exec_process(app: AppHandle, session_id: String, mut child: Child) -> (String, i32) {
    let started_at = std::time::Instant::now();
    let child_pid = child.id();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let (chunk_tx, chunk_rx) = mpsc::channel::<String>();
    if let Some(mut stream) = stdout {
        let chunk_tx = chunk_tx.clone();
        std::thread::spawn(move || {
            let mut buffer = [0u8; 4096];
            loop {
                match stream.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        let _ = chunk_tx.send(String::from_utf8_lossy(&buffer[..n]).to_string());
                    }
                    Err(_) => break,
                }
            }
        });
    }
    if let Some(mut stream) = stderr {
        let chunk_tx = chunk_tx.clone();
        std::thread::spawn(move || {
            let mut buffer = [0u8; 4096];
            loop {
                match stream.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        let _ = chunk_tx.send(String::from_utf8_lossy(&buffer[..n]).to_string());
                    }
                    Err(_) => break,
                }
            }
        });
    }
    drop(chunk_tx);
    let mut output = String::new();
    let exit_code = loop {
        while let Ok(chunk) = chunk_rx.try_recv() {
            output.push_str(&chunk);
            emit_terminal_data(&app, &session_id, chunk);
        }
        if started_at.elapsed() > std::time::Duration::from_secs(300) {
            let _ = Command::new("taskkill")
                .args(["/PID", &child_pid.to_string(), "/T", "/F"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let _ = child.kill();
            break -1;
        }
        match child.try_wait() {
            Ok(Some(status)) => break status.code().unwrap_or(-1),
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(100)),
            Err(_) => break -1,
        }
    };
    let drain_until = std::time::Instant::now() + std::time::Duration::from_millis(750);
    while std::time::Instant::now() < drain_until {
        match chunk_rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(chunk) => {
                output.push_str(&chunk);
                emit_terminal_data(&app, &session_id, chunk);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    if exit_code == -1 && is_only_progress_output(&output) {
        output.push_str(
            "\r\nDeepSeek 请求超时：300 秒内没有收到模型响应。请检查网络、代理、Base URL 或 MCP 服务启动状态。\r\n",
        );
    } else if exit_code == 0 && is_only_progress_output(&output) {
        output.push_str("\r\nDeepSeek 进程已退出，但没有返回模型内容；只收到了请求进度。请检查 API Key、网络代理、Base URL 和 MCP 配置。\r\n");
    }
    let output_preview = output.replace('\r', "\\r").replace('\n', "\\n");
    let output_preview = if output_preview.chars().count() > 240 {
        format!(
            "{}...",
            output_preview.chars().take(240).collect::<String>()
        )
    } else {
        output_preview
    };
    info!(
        "Terminal exec session completed: {}, exitCode={}, outputBytes={}, outputPreview={}",
        session_id,
        exit_code,
        output.len(),
        output_preview
    );
    (output, exit_code)
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
                    language_instruction, options.agent_prompt
                ),
                "yolo" => format!(
                    "{}\n在当前 workspace 中完成用户请求。可以进行必要的代码修改和验证；遇到高风险或破坏性操作时先说明原因。\n\n{}",
                    language_instruction, options.agent_prompt
                ),
                _ => format!("{}\n\n{}", language_instruction, options.agent_prompt),
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
                Some(arg.clone())
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
        && looks_like_web_screenshot_request(&options.agent_prompt)
    {
        info!(
            "Starting direct Playwright screenshot fallback: {}",
            session_id
        );
        let shot_app = app.clone();
        let shot_session_id = session_id.clone();
        let shot_workspace_dir = mcp_workspace_dir.clone();
        let shot_prompt = options.agent_prompt.clone();
        std::thread::spawn(move || {
            let (output, exit_code) = run_playwright_screenshot(
                shot_app.clone(),
                shot_session_id.clone(),
                shot_workspace_dir,
                shot_prompt,
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

    if matches!(options.launch_action.as_str(), "exec" | "plan" | "yolo")
        && provider == "deepseek"
        && !settings.mcp_enabled
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

    if matches!(options.launch_action.as_str(), "exec" | "plan" | "yolo") {
        if settings.harness_enabled && provider == "deepseek" && !settings.mcp_enabled {
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

        let mut process_cmd = Command::new(&binary_path);
        process_cmd
            .args(&args)
            .current_dir(&workspace_dir_string)
            .env("TERM", "xterm-256color")
            .env("COLORTERM", "truecolor")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if !api_key.is_empty() {
            if provider == "nvidia-nim" {
                process_cmd.env("NVIDIA_API_KEY", &api_key);
                process_cmd.env("NVIDIA_NIM_API_KEY", &api_key);
            } else {
                process_cmd.env("DEEPSEEK_API_KEY", &api_key);
            }
        }
        if !normalized_base_url.is_empty() {
            process_cmd.env("DEEPSEEK_BASE_URL", &normalized_base_url);
        }
        if !model.is_empty() {
            process_cmd.env("DEEPSEEK_MODEL", &model);
        }
        if !provider.is_empty() {
            process_cmd.env("DEEPSEEK_PROVIDER", &provider);
        }
        if let Some(path) = &mcp_config_path {
            process_cmd.env("DEEPSEEK_MCP_CONFIG", path);
            process_cmd.env(
                "DEEPSEEK_DESKTOP_ENABLED_MCP",
                settings.enabled_mcp_servers.join(","),
            );
        }

        let child = process_cmd.spawn().map_err(|e| {
            error!("Failed to spawn deepseek exec: {}", e);
            e.to_string()
        })?;
        let pid = child.id();
        let exec_app = app.clone();
        let exec_session_id = session_id.clone();
        std::thread::spawn(move || {
            let (output, exit_code) =
                collect_exec_process(exec_app.clone(), exec_session_id.clone(), child);
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
            pid: Some(pid),
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
    pub path: String,
    pub content: String,
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

    Ok(CustomizationResult {
        skill_root,
        skill_templates: HashMap::new(),
        mcp_config_path: settings.mcp_config_path,
        mcp_config_source: "generated".to_string(),
        mcp_config_text: "{}".to_string(),
        mcp_config_error: String::new(),
    })
}

#[tauri::command]
fn create_skill_template(
    skill_id: String,
    name: String,
    description: String,
    content: String,
) -> Result<SkillTemplate, String> {
    let skills_dir = get_skills_dir();
    let skill_dir = skills_dir.join(&skill_id);
    let skill_file = skill_dir.join("SKILL.md");

    fs::create_dir_all(&skill_dir).map_err(|e| e.to_string())?;

    let frontmatter = format!(
        "---\nname: {}\ndescription: {}\n---\n\n# {}\n\n{}",
        skill_id, description, name, content
    );

    let content_for_return = frontmatter.clone();
    fs::write(&skill_file, &frontmatter).map_err(|e| e.to_string())?;

    Ok(SkillTemplate {
        id: skill_id,
        name,
        description,
        source: "custom".to_string(),
        path: skill_file.to_string_lossy().to_string(),
        content: content_for_return,
    })
}

#[tauri::command]
fn import_skill_directory(source_path: String) -> Result<Vec<SkillTemplate>, String> {
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
                    let skill_dir = skills_dir.join(&id);
                    fs::create_dir_all(&skill_dir).map_err(|e| e.to_string())?;
                    fs::copy(&skill_file, skill_dir.join("SKILL.md")).map_err(|e| e.to_string())?;

                    imported.push(SkillTemplate {
                        id: id.clone(),
                        name: id.clone(),
                        description: String::new(),
                        source: "imported".to_string(),
                        path: skill_file.to_string_lossy().to_string(),
                        content,
                    });
                }
            }
        }
    }

    Ok(imported)
}

// MCP commands
#[tauri::command]
fn save_mcp_config(content: String) -> Result<String, String> {
    ensure_user_data_dir().map_err(|e| e.to_string())?;
    let path = get_user_data_dir().join("mcp.custom.json");
    fs::write(&path, content).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().to_string())
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
    pub ok: bool,
    #[serde(rename = "commandFound")]
    pub command_found: bool,
    #[serde(rename = "missingEnv")]
    pub missing_env: Vec<String>,
    pub warnings: Vec<String>,
}

#[tauri::command]
fn test_mcp_servers(settings: Settings) -> McpServerTestResult {
    McpServerTestResult {
        ok: true,
        tested_at: Utc::now().to_rfc3339(),
        config_path: settings.mcp_config_path,
        servers: vec![],
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
