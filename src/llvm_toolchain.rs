use std::collections::HashSet;
use std::env;
use std::path::PathBuf;

const COMMON_LLVM_BIN_DIRS: &[&str] = &[
    "/root/LLVM/bin",
    "/usr/lib/llvm-21/bin",
    "/usr/lib/llvm-20/bin",
    "/usr/lib/llvm-19/bin",
    "/usr/lib/llvm-18/bin",
    "/usr/lib/llvm-17/bin",
    "/usr/local/llvm/bin",
    "/usr/local/opt/llvm/bin",
    "/opt/homebrew/opt/llvm/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/opt/rocm-7.1.1/llvm/bin",
    "/opt/rocm-7.1.1/bin",
    "/opt/rocm/llvm/bin",
    "/opt/rocm/bin",
];

pub(crate) fn find_clang() -> Result<PathBuf, String> {
    find_tool(
        &["T0_CLANG"],
        &["T0_LLVM_BIN", "LLVM_BIN"],
        &[
            "clang",
            "clang-21",
            "clang-20",
            "clang-19",
            "clang-18",
            "clang-17",
        ],
        "clang",
    )
}

pub(crate) fn find_ld_lld() -> Result<PathBuf, String> {
    find_tool_global_name_priority(
        &["T0_LD_LLD", "T0_LLD"],
        &["T0_LLVM_BIN", "LLVM_BIN"],
        &[
            "ld.lld",
            "ld.lld-21",
            "ld.lld-20",
            "ld.lld-19",
            "ld.lld-18",
            "ld.lld-17",
            
        ],
        "ld.lld",
    )
}

fn find_tool(
    exact_path_vars: &[&str],
    bin_dir_vars: &[&str],
    names: &[&str],
    display_name: &str,
) -> Result<PathBuf, String> {
    let mut checked = Vec::new();

    for var in exact_path_vars {
        if let Some(path) = env_path(var) {
            checked.push(format!("{var}={}", path.display()));
            if path.is_file() {
                return Ok(path);
            }
        }
    }

    let mut dirs = Vec::new();
    let mut seen = HashSet::new();

    for var in bin_dir_vars {
        if let Some(path) = env_path(var) {
            push_dir(&mut dirs, &mut seen, path);
        }
    }

    if let Some(path_var) = env::var_os("PATH") {
        for dir in env::split_paths(&path_var) {
            push_dir(&mut dirs, &mut seen, dir);
        }
    }

    for dir in COMMON_LLVM_BIN_DIRS {
        push_dir(&mut dirs, &mut seen, PathBuf::from(dir));
    }

    for dir in dirs {
        for name in names {
            let candidate = dir.join(name);
            checked.push(candidate.display().to_string());
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }

    Err(format!(
        "LLVM tool '{}' not found. Set {} or T0_LLVM_BIN to your LLVM bin directory, or ensure it is on PATH. Checked: {}",
        display_name,
        exact_path_vars.join(" / "),
        checked.join(", "),
    ))
}

fn find_tool_global_name_priority(
    exact_path_vars: &[&str],
    bin_dir_vars: &[&str],
    names: &[&str],
    display_name: &str,
) -> Result<PathBuf, String> {
    let mut checked = Vec::new();

    for var in exact_path_vars {
        if let Some(path) = env_path(var) {
            checked.push(format!("{var}={}", path.display()));
            if path.is_file() {
                return Ok(path);
            }
        }
    }

    let dirs = collect_dirs(bin_dir_vars);

    for name in names {
        for dir in &dirs {
            let candidate = dir.join(name);
            checked.push(candidate.display().to_string());
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }

    Err(format!(
        "LLVM tool '{}' not found. Set {} or T0_LLVM_BIN to your LLVM bin directory, or ensure it is on PATH. Checked: {}",
        display_name,
        exact_path_vars.join(" / "),
        checked.join(", "),
    ))
}

fn collect_dirs(bin_dir_vars: &[&str]) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut seen = HashSet::new();

    for var in bin_dir_vars {
        if let Some(path) = env_path(var) {
            push_dir(&mut dirs, &mut seen, path);
        }
    }

    for dir in COMMON_LLVM_BIN_DIRS {
        push_dir(&mut dirs, &mut seen, PathBuf::from(dir));
    }

    if let Some(path_var) = env::var_os("PATH") {
        for dir in env::split_paths(&path_var) {
            push_dir(&mut dirs, &mut seen, dir);
        }
    }

    dirs
}

fn env_path(var: &str) -> Option<PathBuf> {
    let value = env::var_os(var)?;
    if value.is_empty() {
        return None;
    }
    Some(PathBuf::from(value))
}

fn push_dir(dirs: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, dir: PathBuf) {
    if !dir.as_os_str().is_empty() && seen.insert(dir.clone()) {
        dirs.push(dir);
    }
}
