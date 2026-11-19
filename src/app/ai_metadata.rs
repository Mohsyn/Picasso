use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default)]
pub(super) struct AiMetadata {
    pub format: String,
    pub text: String,
    pub workflow: Option<String>,
}

pub(super) fn read(path: &Path) -> AiMetadata {
    let Ok(bytes) = std::fs::read(path) else { return AiMetadata::default() };
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let lower = text.to_ascii_lowercase();
    let format = if lower.contains("comfyui") || lower.contains("\"workflow\"") || lower.contains("\"prompt\"") {
        "ComfyUI"
    } else if lower.contains("steps:") || lower.contains("cfg scale:") || lower.contains("negative prompt:") {
        "Stable Diffusion parameters"
    } else {
        ""
    };
    if format.is_empty() {
        return AiMetadata::default();
    }
    let workflow = text.find('{').and_then(|start| {
        let candidate = &text[start..];
        serde_json::from_str::<serde_json::Value>(candidate).ok().map(|value| serde_json::to_string_pretty(&value).unwrap_or_else(|_| candidate.to_string()))
    });
    let text = if text.len() > 32_000 { text[..32_000].to_string() } else { text };
    AiMetadata { format: format.to_string(), text, workflow }
}

fn sidecar(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_os_string();
    sidecar.push(".ai.json");
    PathBuf::from(sidecar)
}

pub(super) fn save(path: &Path, metadata: &AiMetadata) -> std::io::Result<()> {
    let output = serde_json::json!({ "format": metadata.format, "text": metadata.text, "workflow": metadata.workflow });
    std::fs::write(sidecar(path), serde_json::to_vec_pretty(&output).unwrap_or_default())
}

pub(super) fn delete(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(sidecar(path)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_data_is_not_ai_metadata() {
        assert!(read(Path::new("this-file-does-not-exist")).format.is_empty());
    }
}
