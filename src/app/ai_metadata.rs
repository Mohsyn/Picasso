//! AI-generation metadata embedded in image containers.
//!
//! PNG pixel data is compressed binary, so metadata must be read from the
//! container's text chunks rather than by treating the whole file as text.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use flate2::read::ZlibDecoder;
use serde_json::Value;

const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
const MAX_TEXT_CHUNK: u64 = 16 * 1024 * 1024;
const MAX_TEXT_VALUE: u64 = 4 * 1024 * 1024;
const MAX_TOTAL_TEXT: usize = 16 * 1024 * 1024;

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct WorkflowInput {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct WorkflowNode {
    pub id: String,
    pub name: String,
    pub inputs: Vec<WorkflowInput>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct AiMetadata {
    pub format: String,
    /// Recognized text-chunk values only; never image bytes.
    pub text: String,
    /// A pretty-printed workflow or prompt JSON document when one was found.
    pub workflow: Option<String>,
    pub fields: Vec<(String, String)>,
    pub nodes: Vec<WorkflowNode>,
    pub warning: Option<String>,
}

impl AiMetadata {
    pub fn has_data(&self) -> bool {
        !self.format.is_empty()
    }

    pub fn copy_text(&self) -> String {
        self.workflow.clone().filter(|text| !text.is_empty()).unwrap_or_else(|| self.text.clone())
    }
}

#[derive(Debug)]
struct TextEntry {
    key: String,
    value: String,
}

pub(super) fn read(path: &Path) -> AiMetadata {
    let (entries, extraction_warning) = match read_png_text(path) {
        Ok(result) => result,
        Err(_) => return AiMetadata::default(),
    };
    let mut metadata = parse_entries(entries);
    if metadata.warning.is_none() {
        metadata.warning = extraction_warning;
    }
    metadata
}

/// Reads PNG textual chunks after checking the file signature. Unknown chunks,
/// including IDAT pixel data, are skipped without allocating their contents.
fn read_png_text(path: &Path) -> std::io::Result<(Vec<TextEntry>, Option<String>)> {
    let mut file = File::open(path)?;
    let mut signature = [0; 8];
    file.read_exact(&mut signature)?;
    if &signature != PNG_SIGNATURE {
        return Ok((Vec::new(), None));
    }

    let mut entries = Vec::new();
    let mut total = 0usize;
    let mut warning = None;
    loop {
        let mut length = [0; 4];
        if file.read_exact(&mut length).is_err() {
            warning.get_or_insert_with(|| "The PNG ended before its IEND chunk.".to_string());
            break;
        }
        let length = u32::from_be_bytes(length) as u64;
        let mut kind = [0; 4];
        if file.read_exact(&mut kind).is_err() {
            warning.get_or_insert_with(|| "The PNG contains a truncated chunk header.".to_string());
            break;
        }
        let textual = matches!(&kind, b"tEXt" | b"zTXt" | b"iTXt");
        if textual && length <= MAX_TEXT_CHUNK {
            let mut data = vec![0; length as usize];
            if file.read_exact(&mut data).is_err() {
                warning.get_or_insert_with(|| "The PNG contains a truncated text chunk.".to_string());
                break;
            }
            match decode_text_chunk(&kind, &data) {
                Ok(Some(entry)) if total.saturating_add(entry.value.len()) <= MAX_TOTAL_TEXT => {
                    total += entry.value.len();
                    entries.push(entry);
                }
                Ok(Some(_)) => {
                    warning.get_or_insert_with(|| "Some PNG metadata was omitted because it is too large.".to_string());
                }
                Ok(None) => {}
                Err(message) => {
                    warning.get_or_insert(message);
                }
            }
        } else {
            if textual {
                warning.get_or_insert_with(|| "A PNG metadata chunk was omitted because it is too large.".to_string());
            }
            if file.seek(SeekFrom::Current(length as i64)).is_err() {
                warning.get_or_insert_with(|| "The PNG contains an invalid chunk length.".to_string());
                break;
            }
        }
        // CRC belongs to the container, not the text value. Image decoding
        // validates the PNG separately; a missing CRC here is still reported.
        if file.seek(SeekFrom::Current(4)).is_err() {
            warning.get_or_insert_with(|| "The PNG contains a truncated chunk checksum.".to_string());
            break;
        }
        if &kind == b"IEND" {
            break;
        }
    }
    Ok((entries, warning))
}

fn decode_text_chunk(kind: &[u8; 4], data: &[u8]) -> Result<Option<TextEntry>, String> {
    match kind {
        b"tEXt" => {
            let Some(split) = data.iter().position(|byte| *byte == 0) else { return Ok(None) };
            Ok(Some(TextEntry { key: latin1(&data[..split]), value: png_text(&data[split + 1..]) }))
        }
        b"zTXt" => {
            let Some(split) = data.iter().position(|byte| *byte == 0) else { return Ok(None) };
            if data.get(split + 1) != Some(&0) {
                return Err("A compressed PNG text chunk uses an unsupported compression method.".into());
            }
            let value = inflate(&data[split + 2..])?;
            Ok(Some(TextEntry { key: latin1(&data[..split]), value }))
        }
        b"iTXt" => decode_itxt(data).map(Some),
        _ => Ok(None),
    }
}

fn decode_itxt(data: &[u8]) -> Result<TextEntry, String> {
    let Some(keyword_end) = data.iter().position(|byte| *byte == 0) else { return Err("A PNG iTXt chunk has no keyword terminator.".into()) };
    let key = latin1(&data[..keyword_end]);
    let rest = data.get(keyword_end + 1..).ok_or_else(|| "A PNG iTXt chunk is truncated.".to_string())?;
    let (&compressed, rest) = rest.split_first().ok_or_else(|| "A PNG iTXt chunk has no compression flag.".to_string())?;
    let (&method, rest) = rest.split_first().ok_or_else(|| "A PNG iTXt chunk has no compression method.".to_string())?;
    let (_, rest) = split_nul(rest).ok_or_else(|| "A PNG iTXt chunk has no language terminator.".to_string())?;
    let (_, text) = split_nul(rest).ok_or_else(|| "A PNG iTXt chunk has no translated-keyword terminator.".to_string())?;
    let value = match (compressed, method) {
        (0, _) => String::from_utf8_lossy(text).into_owned(),
        (1, 0) => inflate(text)?,
        _ => return Err("A PNG iTXt chunk uses unsupported compression.".into()),
    };
    Ok(TextEntry { key, value })
}

fn split_nul(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let split = bytes.iter().position(|byte| *byte == 0)?;
    Some((&bytes[..split], &bytes[split + 1..]))
}

fn inflate(bytes: &[u8]) -> Result<String, String> {
    let decoder = ZlibDecoder::new(bytes);
    let mut limited = decoder.take(MAX_TEXT_VALUE + 1);
    let mut output = Vec::new();
    limited.read_to_end(&mut output).map_err(|_| "A compressed PNG text chunk could not be read.".to_string())?;
    if output.len() as u64 > MAX_TEXT_VALUE {
        return Err("A compressed PNG text chunk expands beyond the metadata limit.".into());
    }
    Ok(String::from_utf8_lossy(&output).into_owned())
}

fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| char::from(*byte)).collect()
}

/// PNG tEXt is specified as Latin-1, but AI image tools commonly write UTF-8
/// there. Prefer valid UTF-8 and retain standards-compatible Latin-1 as the
/// fallback for older writers.
fn png_text(bytes: &[u8]) -> String {
    std::str::from_utf8(bytes).map(str::to_string).unwrap_or_else(|_| latin1(bytes))
}

fn parse_entries(entries: Vec<TextEntry>) -> AiMetadata {
    let mut metadata = AiMetadata::default();
    let mut recognized = Vec::new();
    let mut prompt_json = None;
    let mut workflow_json = None;
    let mut parameters = None;

    for entry in entries {
        let key = entry.key.trim().to_ascii_lowercase();
        let (kind, value) = if matches!(key.as_str(), "prompt" | "workflow" | "parameters") {
            (key.as_str(), entry.value.trim())
        } else if let Some(value) = strip_prefix_ci(entry.value.trim(), "prompt:") {
            ("prompt", value.trim())
        } else if let Some(value) = strip_prefix_ci(entry.value.trim(), "workflow:") {
            ("workflow", value.trim())
        } else if let Some(value) = strip_prefix_ci(entry.value.trim(), "parameters:") {
            ("parameters", value.trim())
        } else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        recognized.push(format!("{}: {}", if entry.key.trim().is_empty() { kind } else { entry.key.trim() }, value));
        match kind {
            "prompt" if looks_like_json(value) => {
                prompt_json.get_or_insert_with(|| value.to_string());
            }
            "workflow" if looks_like_json(value) => {
                workflow_json.get_or_insert_with(|| value.to_string());
            }
            "parameters" | "prompt" => {
                parameters.get_or_insert_with(|| value.to_string());
            }
            _ => {}
        }
    }

    metadata.text = recognized.join("\n\n");
    if prompt_json.is_some() || workflow_json.is_some() {
        metadata.format = "ComfyUI".into();
        let preferred = prompt_json.as_deref().or(workflow_json.as_deref()).unwrap_or_default();
        match parse_json_compat(preferred) {
            Ok(document) => {
                metadata.workflow = serde_json::to_string_pretty(&document).ok().or_else(|| Some(preferred.to_string()));
                let normalized = unwrap_json(document);
                metadata.nodes = parse_nodes(&normalized);
                metadata.fields = common_fields(&metadata.nodes);
            }
            Err(error) => {
                metadata.workflow = Some(preferred.to_string());
                metadata.warning = Some(format!("ComfyUI metadata could not be parsed: {error}"));
            }
        }
        // The API prompt usually has named inputs. If it was malformed but a
        // valid UI workflow exists, still offer the latter's node list.
        if metadata.nodes.is_empty() {
            if let Some(other) = workflow_json.as_deref().filter(|other| *other != preferred) {
                if let Ok(document) = parse_json_compat(other) {
                    metadata.nodes = parse_nodes(&unwrap_json(document));
                    metadata.fields = common_fields(&metadata.nodes);
                }
            }
        }
    } else if let Some(parameters) = parameters {
        metadata.format = "Stable Diffusion parameters".into();
        metadata.fields = parse_parameters(&parameters);
    }
    metadata
}

fn strip_prefix_ci<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    text.get(..prefix.len()).filter(|head| head.eq_ignore_ascii_case(prefix)).map(|_| &text[prefix.len()..])
}

fn looks_like_json(text: &str) -> bool {
    matches!(text.trim_start().as_bytes().first(), Some(b'{') | Some(b'[') | Some(b'"'))
}

fn parse_json_compat(text: &str) -> Result<Value, serde_json::Error> {
    serde_json::from_str(text).or_else(|_| serde_json::from_str(&text.replace("NaN", "null")))
}

fn unwrap_json(mut value: Value) -> Value {
    for _ in 0..3 {
        value = match value {
            Value::Object(mut object) if object.contains_key("prompt") => object.remove("prompt").unwrap_or(Value::Null),
            Value::String(text) => match parse_json_compat(&text) {
                Ok(parsed) => parsed,
                Err(_) => return Value::String(text),
            },
            other => return other,
        };
    }
    value
}

fn parse_nodes(root: &Value) -> Vec<WorkflowNode> {
    if let Some(nodes) = root.get("nodes").and_then(Value::as_array) {
        return nodes.iter().filter_map(parse_ui_node).collect();
    }
    let Some(object) = root.as_object() else { return Vec::new() };
    object
        .iter()
        .filter_map(|(id, value)| {
            let node = value.as_object()?;
            let name = node.get("class_type").and_then(Value::as_str).unwrap_or("Node").to_string();
            let inputs = node
                .get("inputs")
                .and_then(Value::as_object)
                .map(|inputs| {
                    inputs.iter().map(|(name, value)| WorkflowInput { name: name.clone(), value: display_json(value) }).collect()
                })
                .unwrap_or_default();
            Some(WorkflowNode { id: id.clone(), name, inputs })
        })
        .collect()
}

fn parse_ui_node(value: &Value) -> Option<WorkflowNode> {
    let object = value.as_object()?;
    let id = object.get("id").map(display_json).unwrap_or_else(|| "?".into());
    let name = object.get("type").and_then(Value::as_str).unwrap_or("Node").to_string();
    let inputs = object
        .get("widgets_values")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .enumerate()
                .map(|(index, value)| WorkflowInput { name: format!("Widget {}", index + 1), value: display_json(value) })
                .collect()
        })
        .unwrap_or_default();
    Some(WorkflowNode { id, name, inputs })
}

fn display_json(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => "null".into(),
        _ => serde_json::to_string(value).unwrap_or_default(),
    }
}

fn common_fields(nodes: &[WorkflowNode]) -> Vec<(String, String)> {
    let mut fields = Vec::new();
    let sampler = nodes.iter().find(|node| node.name.to_ascii_lowercase().contains("ksampler"));
    if let Some(node) = sampler {
        for (input, label) in [
            ("seed", "Seed"),
            ("steps", "Steps"),
            ("cfg", "CFG scale"),
            ("sampler_name", "Sampler"),
            ("scheduler", "Scheduler"),
            ("denoise", "Denoise"),
        ] {
            if let Some(value) = input_value(node, input) {
                push_field(&mut fields, label, value);
            }
        }
        for (input, label) in [("positive", "Prompt"), ("negative", "Negative prompt")] {
            if let Some(link) = input_value(node, input).and_then(link_id) {
                if let Some(value) = nodes.iter().find(|candidate| candidate.id == link).and_then(|candidate| input_value(candidate, "text")) {
                    push_field(&mut fields, label, value);
                }
            }
        }
    }
    for node in nodes {
        for (input, label) in [("ckpt_name", "Model"), ("model_name", "Model"), ("width", "Width"), ("height", "Height")] {
            if let Some(value) = input_value(node, input) {
                push_field(&mut fields, label, value);
            }
        }
    }
    fields
}

fn input_value<'a>(node: &'a WorkflowNode, name: &str) -> Option<&'a str> {
    node.inputs.iter().find(|input| input.name == name).map(|input| input.value.as_str())
}

fn link_id(value: &str) -> Option<&str> {
    let value = value.trim();
    if !value.starts_with('[') {
        return None;
    }
    value.get(1..)?.split([',', ']']).next().map(str::trim).map(|id| id.trim_matches('"'))
}

fn push_field(fields: &mut Vec<(String, String)>, label: &str, value: &str) {
    if !value.is_empty() && !fields.iter().any(|(existing, _)| existing == label) {
        fields.push((label.to_string(), value.to_string()));
    }
}

fn parse_parameters(text: &str) -> Vec<(String, String)> {
    let mut fields = Vec::new();
    let (body, settings) = text.rsplit_once('\n').filter(|(_, line)| line.contains("Steps:")).unwrap_or((text, ""));
    let (prompt, negative) = body.split_once("Negative prompt:").unwrap_or((body, ""));
    push_field(&mut fields, "Prompt", prompt.trim());
    push_field(&mut fields, "Negative prompt", negative.trim());
    for part in settings.split(',') {
        let Some((name, value)) = part.split_once(':') else { continue };
        let label = match name.trim().to_ascii_lowercase().as_str() {
            "cfg scale" => "CFG scale",
            "model" | "model hash" => "Model",
            "size" => "Size",
            "sampler" => "Sampler",
            "seed" => "Seed",
            "steps" => "Steps",
            _ => continue,
        };
        push_field(&mut fields, label, value.trim());
    }
    fields
}

fn sidecar(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_os_string();
    sidecar.push(".ai.json");
    PathBuf::from(sidecar)
}

pub(super) fn sidecar_exists(path: &Path) -> bool {
    sidecar(path).is_file()
}

pub(super) fn save(path: &Path, metadata: &AiMetadata) -> std::io::Result<()> {
    if !metadata.has_data() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "no AI metadata to save"));
    }
    let fields: Vec<Value> = metadata.fields.iter().map(|(name, value)| serde_json::json!({ "name": name, "value": value })).collect();
    let nodes: Vec<Value> = metadata
        .nodes
        .iter()
        .map(|node| {
            let inputs: Vec<Value> = node.inputs.iter().map(|input| serde_json::json!({ "name": input.name, "value": input.value })).collect();
            serde_json::json!({ "id": node.id, "name": node.name, "inputs": inputs })
        })
        .collect();
    let output = serde_json::json!({
        "format": metadata.format,
        "text": metadata.text,
        "workflow": metadata.workflow,
        "fields": fields,
        "nodes": nodes,
        "warning": metadata.warning,
    });
    let bytes = serde_json::to_vec_pretty(&output).map_err(std::io::Error::other)?;
    std::fs::write(sidecar(path), bytes)
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
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn chunk(kind: &[u8; 4], data: &[u8], output: &mut Vec<u8>) {
        output.extend_from_slice(&(data.len() as u32).to_be_bytes());
        output.extend_from_slice(kind);
        output.extend_from_slice(data);
        output.extend_from_slice(&[0; 4]); // The metadata reader does not need CRC validation.
    }

    fn png(name: &str, chunks: &[(&[u8; 4], Vec<u8>)]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("picasso-ai-{name}-{}.png", std::process::id()));
        let mut bytes = PNG_SIGNATURE.to_vec();
        for (kind, data) in chunks {
            chunk(kind, data, &mut bytes);
        }
        chunk(b"IEND", &[], &mut bytes);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn text(key: &str, value: &str) -> Vec<u8> {
        let mut data = key.as_bytes().to_vec();
        data.push(0);
        data.extend_from_slice(value.as_bytes());
        data
    }

    fn ztext(key: &str, value: &str) -> Vec<u8> {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(value.as_bytes()).unwrap();
        let mut data = key.as_bytes().to_vec();
        data.extend_from_slice(&[0, 0]);
        data.extend_from_slice(&encoder.finish().unwrap());
        data
    }

    fn itext(key: &str, value: &str, compressed: bool) -> Vec<u8> {
        let body = if compressed {
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(value.as_bytes()).unwrap();
            encoder.finish().unwrap()
        } else {
            value.as_bytes().to_vec()
        };
        let mut data = key.as_bytes().to_vec();
        data.extend_from_slice(&[0, u8::from(compressed), 0, 0, 0]);
        data.extend_from_slice(&body);
        data
    }

    #[test]
    fn empty_or_non_png_data_is_not_ai_metadata() {
        assert!(!read(Path::new("this-file-does-not-exist")).has_data());
        let path = std::env::temp_dir().join(format!("picasso-ai-not-png-{}", std::process::id()));
        std::fs::write(&path, b"binary prompt: {\"workflow\":true}").unwrap();
        assert!(!read(&path).has_data());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn comfy_api_prompt_reads_only_text_chunk_metadata_and_normalizes_nodes() {
        let prompt = r#"{"3":{"class_type":"KSampler","inputs":{"seed":42,"steps":25,"cfg":7.5,"sampler_name":"euler","scheduler":"normal","positive":["6",0],"negative":["7",0]}},"4":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"model.safetensors"}},"5":{"class_type":"EmptyLatentImage","inputs":{"width":1024,"height":768}},"6":{"class_type":"CLIPTextEncode","inputs":{"text":"a café"}},"7":{"class_type":"CLIPTextEncode","inputs":{"text":"blurry"}}}"#;
        let path = png("prompt", &[(b"IDAT", b"pixel prompt: not metadata".to_vec()), (b"tEXt", text("prompt", prompt))]);
        let metadata = read(&path);
        assert_eq!(metadata.format, "ComfyUI");
        assert_eq!(metadata.nodes.len(), 5);
        assert!(metadata.text.contains("a café") && !metadata.text.contains("pixel prompt"));
        assert!(metadata.fields.contains(&("Prompt".into(), "a café".into())));
        assert!(metadata.fields.contains(&("Negative prompt".into(), "blurry".into())));
        assert!(metadata.fields.contains(&("Model".into(), "model.safetensors".into())));
        assert!(metadata.fields.contains(&("Width".into(), "1024".into())));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn international_and_compressed_text_chunks_are_supported() {
        let ui = r#"{"nodes":[{"id":1,"type":"KSampler","widgets_values":[123,20,7.0,"euler"]}]}"#;
        for (name, kind, data) in [
            ("itxt", b"iTXt", itext("workflow", ui, false)),
            ("itxt-z", b"iTXt", itext("workflow", ui, true)),
            ("ztxt", b"zTXt", ztext("workflow", ui)),
        ] {
            let path = png(name, &[(kind, data)]);
            let metadata = read(&path);
            assert_eq!(metadata.format, "ComfyUI", "{name}");
            assert_eq!(metadata.nodes[0].name, "KSampler", "{name}");
            assert_eq!(metadata.nodes[0].inputs[0].value, "123", "{name}");
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn wrapped_string_prompt_and_complex_inputs_are_preserved() {
        let wrapped = serde_json::json!({ "prompt": serde_json::json!({
            "1": { "class_type": "Custom", "inputs": { "link": ["2", 0], "options": { "x": true }, "amount": "NaN" } }
        }).to_string() });
        let entries = vec![TextEntry { key: "prompt".into(), value: wrapped.to_string() }];
        let metadata = parse_entries(entries);
        assert_eq!(metadata.nodes.len(), 1);
        assert_eq!(input_value(&metadata.nodes[0], "link"), Some("[\"2\",0]"));
        assert_eq!(input_value(&metadata.nodes[0], "options"), Some("{\"x\":true}"));
    }

    #[test]
    fn automatic1111_parameters_become_common_fields() {
        let parameters = "a lighthouse at dusk\nNegative prompt: low quality\nSteps: 30, Sampler: DPM++ 2M, CFG scale: 6.5, Seed: 99, Size: 832x1216, Model: coast";
        let path = png("parameters", &[(b"tEXt", text("parameters", parameters))]);
        let metadata = read(&path);
        assert_eq!(metadata.format, "Stable Diffusion parameters");
        assert!(metadata.fields.contains(&("Prompt".into(), "a lighthouse at dusk".into())));
        assert!(metadata.fields.contains(&("Steps".into(), "30".into())));
        assert!(metadata.fields.contains(&("Model".into(), "coast".into())));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn malformed_json_and_chunks_warn_without_losing_raw_metadata() {
        let path = png("bad-json", &[(b"tEXt", text("prompt", "{broken"))]);
        let metadata = read(&path);
        assert!(metadata.has_data() && metadata.workflow.as_deref() == Some("{broken"));
        assert!(metadata.warning.is_some());
        let _ = std::fs::remove_file(path);

        let path = png("bad-itxt", &[(b"iTXt", b"prompt\0".to_vec())]);
        let metadata = read(&path);
        assert!(!metadata.has_data());
        assert!(metadata.warning.is_some());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sidecar_actions_are_safe_and_store_structured_metadata() {
        let path = png("sidecar", &[(b"tEXt", text("parameters", "cat\nSteps: 10, Seed: 2"))]);
        let metadata = read(&path);
        save(&path, &metadata).unwrap();
        assert!(sidecar_exists(&path));
        let saved: Value = serde_json::from_slice(&std::fs::read(sidecar(&path)).unwrap()).unwrap();
        assert_eq!(saved["format"], "Stable Diffusion parameters");
        delete(&path).unwrap();
        delete(&path).unwrap();
        assert!(!sidecar_exists(&path));
        assert!(save(&path, &AiMetadata::default()).is_err());
        let _ = std::fs::remove_file(path);
    }
}
