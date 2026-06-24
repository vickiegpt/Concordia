use serde::Deserialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

const BUILTIN_GPU_MARKERS: &[&str] = &[
    "attention",
    "attn",
    "flash",
    "softmax",
    "soft_max",
    "rope",
    "kq",
    "qk",
    "qkv",
    "query",
    "key",
    "value",
    "kv_cache",
];

const BUILTIN_FFN_MARKERS: &[&str] = &[
    "ffn",
    "feed_forward",
    "mlp",
    "gate_proj",
    "up_proj",
    "down_proj",
    "expert",
    "moe",
    "mul_mat_vec_q",
    "mul_mat_q",
    "mul_mat_f",
];

static ROUTE_LOG_MUTEX: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BitnetRoute {
    CxlTmatmul,
    GpuNative,
    Fallback,
    Reject,
}

impl BitnetRoute {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::CxlTmatmul => "cxl_tmatmul",
            Self::GpuNative => "gpu",
            Self::Fallback => "fallback",
            Self::Reject => "reject",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BitnetRouteSource {
    Disabled,
    ExplicitGpuEnv,
    Manifest,
    ExplicitCxlEnv,
    BuiltinGpuMarker,
    BuiltinFfnMarker,
    Default,
}

impl BitnetRouteSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::ExplicitGpuEnv => "explicit_gpu_env",
            Self::Manifest => "manifest",
            Self::ExplicitCxlEnv => "explicit_cxl_env",
            Self::BuiltinGpuMarker => "builtin_gpu_marker",
            Self::BuiltinFfnMarker => "builtin_ffn_marker",
            Self::Default => "default",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BitnetRouteDecision {
    pub(crate) kernel: String,
    pub(crate) route: BitnetRoute,
    pub(crate) source: BitnetRouteSource,
    pub(crate) matched: Option<String>,
    pub(crate) reason: Option<String>,
    pub(crate) strict: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct BitnetRouteConfig {
    pub(crate) enabled: bool,
    pub(crate) strict: bool,
    pub(crate) gpu_markers: Vec<String>,
    pub(crate) cxl_markers: Vec<String>,
    pub(crate) manifest: Option<BitnetRouteManifest>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub(crate) struct BitnetRouteManifest {
    pub(crate) version: u32,
    #[serde(default = "default_manifest_route")]
    pub(crate) default: ManifestRouteName,
    #[serde(default)]
    pub(crate) routes: Vec<BitnetRouteManifestEntry>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub(crate) struct BitnetRouteManifestEntry {
    #[serde(rename = "match")]
    pub(crate) match_text: String,
    pub(crate) route: ManifestRouteName,
    pub(crate) reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ManifestRouteName {
    #[serde(alias = "cxl")]
    CxlTmatmul,
    #[serde(alias = "gpu_native")]
    Gpu,
    Fallback,
    Reject,
}

impl Default for ManifestRouteName {
    fn default() -> Self {
        Self::Fallback
    }
}

impl ManifestRouteName {
    fn to_route(self) -> BitnetRoute {
        match self {
            Self::CxlTmatmul => BitnetRoute::CxlTmatmul,
            Self::Gpu => BitnetRoute::GpuNative,
            Self::Fallback => BitnetRoute::Fallback,
            Self::Reject => BitnetRoute::Reject,
        }
    }
}

fn default_manifest_route() -> ManifestRouteName {
    ManifestRouteName::Fallback
}

pub(crate) fn load_manifest_file(path: &Path) -> Result<BitnetRouteManifest, String> {
    let text =
        std::fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let manifest: BitnetRouteManifest =
        serde_json::from_str(&text).map_err(|err| format!("parse {}: {err}", path.display()))?;
    if manifest.version != 1 {
        return Err(format!(
            "unsupported manifest version {}, expected 1",
            manifest.version
        ));
    }
    if manifest.default == ManifestRouteName::CxlTmatmul {
        return Err(
            "cxl_tmatmul is not allowed as manifest default; use fallback, gpu, or reject"
                .to_string(),
        );
    }
    Ok(manifest)
}

pub(crate) fn classify_kernel_name(
    kernel_name: &str,
    config: &BitnetRouteConfig,
) -> BitnetRouteDecision {
    let name_lower = normalize_kernel_name(kernel_name);
    if !config.enabled {
        return decision(
            kernel_name,
            BitnetRoute::Fallback,
            BitnetRouteSource::Disabled,
            None,
            None,
            config.strict,
        );
    }

    if let Some(marker) = find_marker(&name_lower, config.gpu_markers.iter().map(String::as_str)) {
        return decision(
            kernel_name,
            BitnetRoute::GpuNative,
            BitnetRouteSource::ExplicitGpuEnv,
            Some(marker),
            None,
            config.strict,
        );
    }

    if let Some(manifest_decision) = classify_manifest(kernel_name, &name_lower, config) {
        return manifest_decision;
    }

    if let Some(marker) = find_marker(&name_lower, config.cxl_markers.iter().map(String::as_str)) {
        return decision(
            kernel_name,
            BitnetRoute::CxlTmatmul,
            BitnetRouteSource::ExplicitCxlEnv,
            Some(marker),
            None,
            config.strict,
        );
    }

    if let Some(marker) = find_marker(&name_lower, BUILTIN_GPU_MARKERS.iter().copied()) {
        return decision(
            kernel_name,
            BitnetRoute::GpuNative,
            BitnetRouteSource::BuiltinGpuMarker,
            Some(marker),
            None,
            config.strict,
        );
    }

    if let Some(marker) = find_marker(&name_lower, BUILTIN_FFN_MARKERS.iter().copied()) {
        return decision(
            kernel_name,
            BitnetRoute::CxlTmatmul,
            BitnetRouteSource::BuiltinFfnMarker,
            Some(marker),
            None,
            config.strict,
        );
    }

    let route = if config.strict {
        BitnetRoute::Reject
    } else {
        BitnetRoute::Fallback
    };
    decision(
        kernel_name,
        route,
        BitnetRouteSource::Default,
        None,
        None,
        config.strict,
    )
}

fn normalize_kernel_name(kernel_name: &str) -> String {
    kernel_name.trim().to_ascii_lowercase()
}

fn find_marker<'a>(name_lower: &str, mut markers: impl Iterator<Item = &'a str>) -> Option<String> {
    markers.find_map(|marker| normalized_marker_match(name_lower, marker))
}

fn normalized_marker_match(name_lower: &str, marker: &str) -> Option<String> {
    let marker = marker.trim();
    if marker.is_empty() {
        return None;
    }

    if marker_is_ascii_lowercase(marker) {
        if name_lower.contains(marker) {
            return Some(marker.to_string());
        }
        return None;
    }

    let marker_lower = marker.to_ascii_lowercase();
    if name_lower.contains(marker_lower.as_str()) {
        Some(marker_lower)
    } else {
        None
    }
}

fn marker_is_ascii_lowercase(marker: &str) -> bool {
    marker.bytes().all(|byte| !byte.is_ascii_uppercase())
}

fn classify_manifest(
    kernel_name: &str,
    name_lower: &str,
    config: &BitnetRouteConfig,
) -> Option<BitnetRouteDecision> {
    let manifest = config.manifest.as_ref()?;
    for entry in &manifest.routes {
        if let Some(marker) = normalized_marker_match(name_lower, entry.match_text.as_str()) {
            return Some(decision(
                kernel_name,
                entry.route.to_route(),
                BitnetRouteSource::Manifest,
                Some(marker),
                entry.reason.clone(),
                config.strict,
            ));
        }
    }

    if manifest.default == ManifestRouteName::CxlTmatmul {
        return Some(decision(
            kernel_name,
            BitnetRoute::Fallback,
            BitnetRouteSource::Manifest,
            None,
            Some("cxl_tmatmul manifest default is not allowed; falling back".to_string()),
            config.strict,
        ));
    }

    if manifest.default != ManifestRouteName::Fallback {
        return Some(decision(
            kernel_name,
            manifest.default.to_route(),
            BitnetRouteSource::Manifest,
            None,
            None,
            config.strict,
        ));
    }
    None
}

pub(crate) fn append_route_log(
    path: &Path,
    decision: &BitnetRouteDecision,
    cxl_enabled: bool,
    hardware_matmul_enabled: bool,
) -> Result<(), String> {
    let record = serde_json::json!({
        "kernel": decision.kernel.as_str(),
        "route": decision.route.as_str(),
        "source": decision.source.as_str(),
        "matched": decision.matched.as_deref(),
        "reason": decision.reason.as_deref(),
        "strict": decision.strict,
        "cxl_enabled": cxl_enabled,
        "hardware_matmul_enabled": hardware_matmul_enabled,
    });
    let mut line = serde_json::to_string(&record)
        .map_err(|err| format!("serialize route log {}: {err}", path.display()))?;
    line.push('\n');

    let _guard = ROUTE_LOG_MUTEX
        .lock()
        .map_err(|_| format!("lock route log {}: mutex poisoned", path.display()))?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| format!("open route log {}: {err}", path.display()))?;
    file.write_all(line.as_bytes())
        .map_err(|err| format!("write route log {}: {err}", path.display()))
}

fn decision(
    kernel_name: &str,
    route: BitnetRoute,
    source: BitnetRouteSource,
    matched: Option<String>,
    reason: Option<String>,
    strict: bool,
) -> BitnetRouteDecision {
    BitnetRouteDecision {
        kernel: kernel_name.to_string(),
        route,
        source,
        matched,
        reason,
        strict,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(enabled: bool) -> BitnetRouteConfig {
        BitnetRouteConfig {
            enabled,
            strict: false,
            gpu_markers: Vec::new(),
            cxl_markers: Vec::new(),
            manifest: None,
        }
    }

    #[test]
    fn disabled_router_falls_back() {
        let decision = classify_kernel_name("layer_0_ffn_gate_mul_mat", &config(false));
        assert_eq!(decision.route, BitnetRoute::Fallback);
        assert_eq!(decision.source, BitnetRouteSource::Disabled);
    }

    #[test]
    fn builtin_attention_markers_route_to_gpu() {
        let decision = classify_kernel_name("_z13flash_attn_mul_mat_q", &config(true));
        assert_eq!(decision.route, BitnetRoute::GpuNative);
        assert_eq!(decision.source, BitnetRouteSource::BuiltinGpuMarker);
        assert_eq!(decision.matched.as_deref(), Some("attn"));
    }

    #[test]
    fn builtin_ffn_markers_route_to_cxl() {
        let decision = classify_kernel_name("layer_03_ffn_gate_mul_mat", &config(true));
        assert_eq!(decision.route, BitnetRoute::CxlTmatmul);
        assert_eq!(decision.source, BitnetRouteSource::BuiltinFfnMarker);
        assert_eq!(decision.matched.as_deref(), Some("ffn"));
    }

    #[test]
    fn explicit_gpu_markers_override_explicit_cxl_markers() {
        let mut cfg = config(true);
        cfg.gpu_markers = vec!["force_gpu".to_string()];
        cfg.cxl_markers = vec!["ffn_gate".to_string(), "force_gpu".to_string()];

        let decision = classify_kernel_name("layer_04_force_gpu_ffn_gate_mul_mat", &cfg);
        assert_eq!(decision.route, BitnetRoute::GpuNative);
        assert_eq!(decision.source, BitnetRouteSource::ExplicitGpuEnv);
        assert_eq!(decision.matched.as_deref(), Some("force_gpu"));
    }

    #[test]
    fn explicit_gpu_markers_are_trimmed_and_lowercased() {
        let mut cfg = config(true);
        cfg.gpu_markers = vec![" FORCE_GPU ".to_string()];
        cfg.cxl_markers = vec!["ffn_gate".to_string()];

        let decision = classify_kernel_name("layer_force_gpu_ffn_gate", &cfg);
        assert_eq!(decision.route, BitnetRoute::GpuNative);
        assert_eq!(decision.source, BitnetRouteSource::ExplicitGpuEnv);
        assert_eq!(decision.matched.as_deref(), Some("force_gpu"));
    }

    #[test]
    fn explicit_cxl_markers_override_builtin_gpu_markers_only_after_gpu_env() {
        let mut cfg = config(true);
        cfg.cxl_markers = vec!["attention_ffn_probe".to_string()];

        let decision = classify_kernel_name("attention_ffn_probe_mul_mat", &cfg);
        assert_eq!(decision.route, BitnetRoute::CxlTmatmul);
        assert_eq!(decision.source, BitnetRouteSource::ExplicitCxlEnv);
        assert_eq!(decision.matched.as_deref(), Some("attention_ffn_probe"));
    }

    #[test]
    fn unknown_enabled_kernel_falls_back_by_default() {
        let decision = classify_kernel_name("layer_07_layernorm", &config(true));
        assert_eq!(decision.route, BitnetRoute::Fallback);
        assert_eq!(decision.source, BitnetRouteSource::Default);
    }

    #[test]
    fn unknown_enabled_kernel_rejects_in_strict_mode() {
        let mut cfg = config(true);
        cfg.strict = true;

        let decision = classify_kernel_name("layer_07_layernorm", &cfg);
        assert_eq!(decision.route, BitnetRoute::Reject);
        assert_eq!(decision.source, BitnetRouteSource::Default);
    }

    #[test]
    fn manifest_routes_kernel_by_substring() {
        let manifest = BitnetRouteManifest {
            version: 1,
            default: ManifestRouteName::Fallback,
            routes: vec![BitnetRouteManifestEntry {
                match_text: "ffn_gate".to_string(),
                route: ManifestRouteName::CxlTmatmul,
                reason: Some("manifest ffn".to_string()),
            }],
        };
        let mut cfg = config(true);
        cfg.manifest = Some(manifest);

        let decision = classify_kernel_name("layer_0_ffn_gate_mul_mat", &cfg);
        assert_eq!(decision.route, BitnetRoute::CxlTmatmul);
        assert_eq!(decision.source, BitnetRouteSource::Manifest);
        assert_eq!(decision.matched.as_deref(), Some("ffn_gate"));
        assert_eq!(decision.reason.as_deref(), Some("manifest ffn"));
    }

    #[test]
    fn manifest_default_rejects_when_requested() {
        let manifest = BitnetRouteManifest {
            version: 1,
            default: ManifestRouteName::Reject,
            routes: Vec::new(),
        };
        let mut cfg = config(true);
        cfg.manifest = Some(manifest);

        let decision = classify_kernel_name("layer_0_unknown", &cfg);
        assert_eq!(decision.route, BitnetRoute::Reject);
        assert_eq!(decision.source, BitnetRouteSource::Manifest);
    }

    #[test]
    fn loads_manifest_from_json_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.json");
        std::fs::write(
            &path,
            r#"{
            "version": 1,
            "default": "fallback",
            "routes": [
                {"match": "flash_attn", "route": "gpu", "reason": "attention"}
            ]
        }"#,
        )
        .unwrap();

        let manifest = load_manifest_file(&path).unwrap();
        assert_eq!(manifest.routes.len(), 1);
        assert_eq!(manifest.routes[0].match_text, "flash_attn");
        assert_eq!(manifest.routes[0].route, ManifestRouteName::Gpu);
    }

    #[test]
    fn rejects_cxl_manifest_default_from_json() {
        for default in ["cxl_tmatmul", "cxl"] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("routes.json");
            std::fs::write(
                &path,
                format!(r#"{{"version": 1, "default": "{default}", "routes": []}}"#),
            )
            .unwrap();

            let err = load_manifest_file(&path).unwrap_err();
            assert!(
                err.contains("cxl_tmatmul is not allowed as manifest default"),
                "{err}"
            );
        }
    }

    #[test]
    fn allows_manifest_route_aliases_from_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.json");
        std::fs::write(
            &path,
            r#"{
            "version": 1,
            "default": "fallback",
            "routes": [
                {"match": "ffn_gate", "route": "cxl", "reason": "alias cxl"},
                {"match": "flash_attn", "route": "gpu_native", "reason": "alias gpu"}
            ]
        }"#,
        )
        .unwrap();

        let manifest = load_manifest_file(&path).unwrap();
        assert_eq!(manifest.routes[0].route, ManifestRouteName::CxlTmatmul);
        assert_eq!(manifest.routes[1].route, ManifestRouteName::Gpu);

        let mut cfg = config(true);
        cfg.manifest = Some(manifest);

        let cxl = classify_kernel_name("layer_0_ffn_gate_mul_mat", &cfg);
        assert_eq!(cxl.route, BitnetRoute::CxlTmatmul);
        assert_eq!(cxl.source, BitnetRouteSource::Manifest);
        assert_eq!(cxl.matched.as_deref(), Some("ffn_gate"));
        assert_eq!(cxl.reason.as_deref(), Some("alias cxl"));

        let gpu = classify_kernel_name("layer_0_flash_attn_mul_mat", &cfg);
        assert_eq!(gpu.route, BitnetRoute::GpuNative);
        assert_eq!(gpu.source, BitnetRouteSource::Manifest);
        assert_eq!(gpu.matched.as_deref(), Some("flash_attn"));
        assert_eq!(gpu.reason.as_deref(), Some("alias gpu"));
    }

    #[test]
    fn omitted_manifest_default_is_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.json");
        std::fs::write(&path, r#"{"version": 1, "routes": []}"#).unwrap();

        let manifest = load_manifest_file(&path).unwrap();
        assert_eq!(manifest.default, ManifestRouteName::Fallback);
    }

    #[test]
    fn manual_manifest_default_cxl_does_not_route_unmatched_kernel_to_cxl() {
        let manifest = BitnetRouteManifest {
            version: 1,
            default: ManifestRouteName::CxlTmatmul,
            routes: Vec::new(),
        };
        let mut cfg = config(true);
        cfg.manifest = Some(manifest);

        let decision = classify_kernel_name("layer_0_unknown", &cfg);
        assert_eq!(decision.route, BitnetRoute::Fallback);
        assert_eq!(decision.source, BitnetRouteSource::Manifest);
    }

    #[test]
    fn rejects_manifest_version_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.json");
        std::fs::write(
            &path,
            r#"{"version": 2, "default": "fallback", "routes": []}"#,
        )
        .unwrap();

        let err = load_manifest_file(&path).unwrap_err();
        assert!(err.contains("unsupported manifest version"));
    }

    #[test]
    fn appends_jsonl_route_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("routes.jsonl");
        let decision = BitnetRouteDecision {
            kernel: "layer_0_ffn_gate_mul_mat".to_string(),
            route: BitnetRoute::CxlTmatmul,
            source: BitnetRouteSource::BuiltinFfnMarker,
            matched: Some("ffn".to_string()),
            reason: Some("FFN ternary matmul candidate".to_string()),
            strict: false,
        };

        append_route_log(&path, &decision, true, true).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["kernel"], "layer_0_ffn_gate_mul_mat");
        assert_eq!(value["route"], "cxl_tmatmul");
        assert_eq!(value["source"], "builtin_ffn_marker");
        assert_eq!(value["matched"], "ffn");
        assert_eq!(value["reason"], "FFN ternary matmul candidate");
        assert_eq!(value["strict"], false);
        assert_eq!(value["cxl_enabled"], true);
        assert_eq!(value["hardware_matmul_enabled"], true);

        let fallback = BitnetRouteDecision {
            kernel: "layer_0_layernorm".to_string(),
            route: BitnetRoute::Fallback,
            source: BitnetRouteSource::Default,
            matched: None,
            reason: None,
            strict: true,
        };
        append_route_log(&path, &fallback, false, false).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let values = text
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 2);
        assert!(values[1]["matched"].is_null());
        assert!(values[1]["reason"].is_null());
        assert_eq!(values[1]["strict"], true);
        assert_eq!(values[1]["hardware_matmul_enabled"], false);
    }

    #[test]
    fn route_and_source_as_str_values_are_stable() {
        assert_eq!(BitnetRoute::CxlTmatmul.as_str(), "cxl_tmatmul");
        assert_eq!(BitnetRoute::GpuNative.as_str(), "gpu");
        assert_eq!(BitnetRoute::Fallback.as_str(), "fallback");
        assert_eq!(BitnetRoute::Reject.as_str(), "reject");

        assert_eq!(BitnetRouteSource::Disabled.as_str(), "disabled");
        assert_eq!(
            BitnetRouteSource::ExplicitGpuEnv.as_str(),
            "explicit_gpu_env"
        );
        assert_eq!(BitnetRouteSource::Manifest.as_str(), "manifest");
        assert_eq!(
            BitnetRouteSource::ExplicitCxlEnv.as_str(),
            "explicit_cxl_env"
        );
        assert_eq!(
            BitnetRouteSource::BuiltinGpuMarker.as_str(),
            "builtin_gpu_marker"
        );
        assert_eq!(
            BitnetRouteSource::BuiltinFfnMarker.as_str(),
            "builtin_ffn_marker"
        );
        assert_eq!(BitnetRouteSource::Default.as_str(), "default");
    }
}
