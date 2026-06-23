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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BitnetRouteManifest;

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

fn find_marker<'a>(name_lower: &str, markers: impl Iterator<Item = &'a str>) -> Option<String> {
    markers
        .filter(|marker| !marker.is_empty())
        .find(|marker| name_lower.contains(*marker))
        .map(str::to_string)
}

fn classify_manifest(
    _kernel_name: &str,
    _name_lower: &str,
    _config: &BitnetRouteConfig,
) -> Option<BitnetRouteDecision> {
    None
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
}
