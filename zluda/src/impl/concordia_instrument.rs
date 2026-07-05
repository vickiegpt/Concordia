#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SafePointKind {
    Entry,
    Barrier,
    Exit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SafePoint {
    pub kind: SafePointKind,
    pub line: usize,
    pub label: String,
}

pub(crate) const SASS_LIVE_STATE_MAGIC: u32 = 0x4353_5353;
pub(crate) const SASS_LIVE_STATE_VERSION: u16 = 1;
pub(crate) const SASS_LIVE_STATE_HEADER_BYTES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SassLiveRegisterSlot {
    pub name: String,
    pub offset: usize,
    pub bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SassLiveStateLayout {
    pub magic: u32,
    pub version: u16,
    pub header_bytes: usize,
    pub record_bytes: usize,
    pub slots: Vec<SassLiveRegisterSlot>,
}

pub(crate) const CTX_RESUME_MAGIC: u32 = 0x3058_5443;
pub(crate) const CTX_RESUME_VERSION: u16 = 1;
pub(crate) const CTX_RESUME_HEADER_BYTES: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CtxResumeTarget {
    NvidiaPtx,
    AmdRocm,
    IntelSpirv,
    TenstorrentTosa,
}

impl CtxResumeTarget {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "nvidia" | "nvidia-ptx" | "ptx" => Ok(Self::NvidiaPtx),
            "amd" | "amd-rocm" | "rocm" => Ok(Self::AmdRocm),
            "intel" | "intel-spirv" | "spirv" | "level-zero" => Ok(Self::IntelSpirv),
            "tenstorrent" | "tenstorrent-tosa" | "tosa" => Ok(Self::TenstorrentTosa),
            other => Err(format!("unsupported CTX resume target '{other}'")),
        }
    }

    fn code(self) -> u16 {
        match self {
            Self::NvidiaPtx => 1,
            Self::AmdRocm => 2,
            Self::IntelSpirv => 3,
            Self::TenstorrentTosa => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CtxResumeHeader {
    pub magic: u32,
    pub version: u16,
    pub header_bytes: usize,
    pub target: CtxResumeTarget,
    pub kernel_id: u64,
    pub safe_point_id: u64,
    pub register_bytes: usize,
    pub shared_memory_bytes: usize,
    pub record_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CtxResumeRecord {
    pub header: CtxResumeHeader,
    pub bytes: Vec<u8>,
}

pub(crate) fn build_ctx_resume_record(
    target: CtxResumeTarget,
    kernel_id: u64,
    safe_point_id: u64,
    register_payload: &[u8],
    shared_memory_payload: &[u8],
) -> Result<CtxResumeRecord, String> {
    let register_bytes = register_payload.len();
    let shared_memory_bytes = shared_memory_payload.len();
    let payload_bytes = register_bytes
        .checked_add(shared_memory_bytes)
        .ok_or_else(|| "CTX resume payload length overflow".to_string())?;
    let record_bytes = CTX_RESUME_HEADER_BYTES
        .checked_add(payload_bytes)
        .ok_or_else(|| "CTX resume record length overflow".to_string())?;
    let register_bytes_u32 = u32::try_from(register_bytes)
        .map_err(|_| "CTX register payload is too large".to_string())?;
    let shared_memory_bytes_u32 = u32::try_from(shared_memory_bytes)
        .map_err(|_| "CTX shared-memory payload is too large".to_string())?;
    let record_bytes_u32 =
        u32::try_from(record_bytes).map_err(|_| "CTX resume record is too large".to_string())?;

    let header = CtxResumeHeader {
        magic: CTX_RESUME_MAGIC,
        version: CTX_RESUME_VERSION,
        header_bytes: CTX_RESUME_HEADER_BYTES,
        target,
        kernel_id,
        safe_point_id,
        register_bytes,
        shared_memory_bytes,
        record_bytes,
    };
    let mut bytes = Vec::with_capacity(record_bytes);
    bytes.extend_from_slice(&CTX_RESUME_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&CTX_RESUME_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(CTX_RESUME_HEADER_BYTES as u16).to_le_bytes());
    bytes.extend_from_slice(&target.code().to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&kernel_id.to_le_bytes());
    bytes.extend_from_slice(&safe_point_id.to_le_bytes());
    bytes.extend_from_slice(&register_bytes_u32.to_le_bytes());
    bytes.extend_from_slice(&shared_memory_bytes_u32.to_le_bytes());
    bytes.extend_from_slice(&record_bytes_u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(bytes.len(), CTX_RESUME_HEADER_BYTES);
    bytes.extend_from_slice(register_payload);
    bytes.extend_from_slice(shared_memory_payload);

    Ok(CtxResumeRecord { header, bytes })
}

pub(crate) fn build_sass_live_state_layout(
    registers: &[&str],
) -> Result<SassLiveStateLayout, String> {
    let mut slots = Vec::new();
    let mut seen = Vec::<String>::new();
    let mut offset = SASS_LIVE_STATE_HEADER_BYTES;

    for raw in registers {
        let name = canonical_sass_live_register(raw)?;
        if seen.iter().any(|existing| existing == &name) {
            continue;
        }
        seen.push(name.clone());
        let bytes = sass_live_register_bytes(&name)
            .ok_or_else(|| format!("unsupported SASS live register '{raw}'"))?;
        offset = align_up(offset, bytes.min(8));
        slots.push(SassLiveRegisterSlot {
            name,
            offset,
            bytes,
        });
        offset = offset.saturating_add(bytes);
    }

    Ok(SassLiveStateLayout {
        magic: SASS_LIVE_STATE_MAGIC,
        version: SASS_LIVE_STATE_VERSION,
        header_bytes: SASS_LIVE_STATE_HEADER_BYTES,
        record_bytes: align_up(offset, 8),
        slots,
    })
}

fn canonical_sass_live_register(raw: &str) -> Result<String, String> {
    let name = raw.trim().to_ascii_uppercase();
    if sass_live_register_bytes(&name).is_some() {
        Ok(name)
    } else {
        Err(format!("unsupported SASS live register '{raw}'"))
    }
}

fn sass_live_register_bytes(name: &str) -> Option<usize> {
    if let Some(rest) = name.strip_prefix("UR") {
        return register_number(rest).map(|_| 8);
    }
    if let Some(rest) = name.strip_prefix("UP") {
        return register_number(rest).map(|_| 1);
    }
    if let Some(rest) = name.strip_prefix('R') {
        return register_number(rest).map(|_| 8);
    }
    if let Some(rest) = name.strip_prefix('P') {
        return register_number(rest).map(|_| 1);
    }
    None
}

fn register_number(text: &str) -> Option<u32> {
    if text.is_empty() {
        return None;
    }
    text.parse::<u32>().ok()
}

fn align_up(value: usize, align: usize) -> usize {
    let align = align.max(1);
    value.div_ceil(align) * align
}

pub(crate) fn discover_ptx_safe_points(ptx: &str) -> Vec<SafePoint> {
    let mut safe_points = Vec::new();
    let mut in_entry = false;
    let mut entry_index = 0usize;
    let mut barrier_index = 0usize;
    let mut exit_index = 0usize;

    for (line, text) in ptx.lines().enumerate() {
        let trimmed = text.trim_start();
        if trimmed.contains(".entry ") || trimmed.starts_with(".entry") {
            in_entry = true;
            safe_points.push(SafePoint {
                kind: SafePointKind::Entry,
                line,
                label: format!("__concordia_safe_entry_{entry_index}"),
            });
            entry_index += 1;
        }

        if !in_entry {
            continue;
        }
        if trimmed.starts_with("bar.") || trimmed.contains("bar.sync") {
            safe_points.push(SafePoint {
                kind: SafePointKind::Barrier,
                line,
                label: format!("__concordia_safe_barrier_{barrier_index}"),
            });
            barrier_index += 1;
        }
        if trimmed.starts_with("ret") {
            safe_points.push(SafePoint {
                kind: SafePointKind::Exit,
                line,
                label: format!("__concordia_safe_exit_{exit_index}"),
            });
            exit_index += 1;
        }
        if trimmed.starts_with('}') {
            in_entry = false;
        }
    }

    safe_points
}

pub(crate) fn annotate_ptx_with_concordia_safe_points(ptx: &str) -> String {
    let mut annotated = String::with_capacity(ptx.len() + 256);
    let mut in_entry = false;
    let mut pending_entry_label: Option<String> = None;
    let mut entry_index = 0usize;
    let mut barrier_index = 0usize;
    let mut exit_index = 0usize;

    for text in ptx.lines() {
        let trimmed = text.trim_start();
        let indent = &text[..text.len() - trimmed.len()];
        if trimmed.contains(".entry ") || trimmed.starts_with(".entry") {
            in_entry = true;
            pending_entry_label = Some(format!("__concordia_safe_entry_{entry_index}"));
            entry_index += 1;
        }

        if in_entry && (trimmed.starts_with("bar.") || trimmed.contains("bar.sync")) {
            annotated.push_str(indent);
            annotated.push_str(&format!("__concordia_safe_barrier_{barrier_index}:\n"));
            barrier_index += 1;
        }
        if in_entry && trimmed.starts_with("ret") {
            annotated.push_str(indent);
            annotated.push_str(&format!("__concordia_safe_exit_{exit_index}:\n"));
            exit_index += 1;
        }

        annotated.push_str(text);
        annotated.push('\n');

        if let Some(label) = pending_entry_label.take() {
            if text.contains('{') {
                annotated.push_str(indent);
                annotated.push_str(&label);
                annotated.push_str(":\n");
            } else {
                pending_entry_label = Some(label);
            }
        }
        if in_entry && trimmed.starts_with('}') {
            in_entry = false;
            pending_entry_label = None;
        }
    }

    annotated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_safe_point_discovery_marks_entry_barrier_and_exit() {
        let ptx = r#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry kernel() {
    .reg .u32 %r<2>;
    mov.u32 %r1, %tid.x;
    bar.sync 0;
    ret;
}
"#;

        let safe_points = discover_ptx_safe_points(ptx);

        assert!(safe_points
            .iter()
            .any(|point| point.kind == SafePointKind::Entry));
        assert!(safe_points
            .iter()
            .any(|point| point.kind == SafePointKind::Barrier));
        assert!(safe_points
            .iter()
            .any(|point| point.kind == SafePointKind::Exit));
    }

    #[test]
    fn ptx_annotation_inserts_stable_safe_point_labels() {
        let ptx = r#"
.visible .entry kernel() {
    bar.sync 0;
    ret;
}
"#;

        let annotated = annotate_ptx_with_concordia_safe_points(ptx);

        assert!(annotated.contains("__concordia_safe_entry_0:"));
        assert!(annotated.contains("__concordia_safe_barrier_"));
        assert!(annotated.contains("__concordia_safe_exit_"));
        assert!(annotated.contains("bar.sync 0;"));
        assert!(annotated.contains("ret;"));
    }

    #[test]
    fn sass_live_state_layout_assigns_stable_aligned_slots() {
        let layout = build_sass_live_state_layout(&["R2", "P0", "R2", "UR4", "UP1"]).unwrap();

        assert_eq!(layout.header_bytes, SASS_LIVE_STATE_HEADER_BYTES);
        assert_eq!(layout.slots.len(), 4);
        assert_eq!(layout.slots[0].name, "R2");
        assert_eq!(layout.slots[0].offset, SASS_LIVE_STATE_HEADER_BYTES);
        assert_eq!(layout.slots[0].bytes, 8);
        assert_eq!(layout.slots[1].name, "P0");
        assert_eq!(layout.slots[1].bytes, 1);
        assert_eq!(layout.slots[2].name, "UR4");
        assert_eq!(layout.slots[2].bytes, 8);
        assert_eq!(layout.slots[3].name, "UP1");
        assert_eq!(layout.slots[3].bytes, 1);
        assert_eq!(layout.record_bytes % 8, 0);
    }

    #[test]
    fn sass_live_state_layout_rejects_unknown_register_names() {
        let err = build_sass_live_state_layout(&["R1", "bad"]).unwrap_err();

        assert!(err.contains("unsupported SASS live register"), "{err}");
    }

    #[test]
    fn ctx_resume_record_serializes_stable_header_and_payloads() {
        let record =
            build_ctx_resume_record(CtxResumeTarget::NvidiaPtx, 17, 3, &[1, 2, 3, 4], &[5, 6, 7])
                .unwrap();

        assert_eq!(record.header.magic, CTX_RESUME_MAGIC);
        assert_eq!(record.header.version, CTX_RESUME_VERSION);
        assert_eq!(record.header.target, CtxResumeTarget::NvidiaPtx);
        assert_eq!(record.header.kernel_id, 17);
        assert_eq!(record.header.safe_point_id, 3);
        assert_eq!(record.header.register_bytes, 4);
        assert_eq!(record.header.shared_memory_bytes, 3);
        assert_eq!(record.bytes.len(), CTX_RESUME_HEADER_BYTES + 7);
        assert_eq!(&record.bytes[0..4], b"CTX0");
        assert_eq!(
            &record.bytes[CTX_RESUME_HEADER_BYTES..],
            &[1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn ctx_resume_target_parser_accepts_supported_backend_names() {
        assert_eq!(
            CtxResumeTarget::parse("nvidia-ptx").unwrap(),
            CtxResumeTarget::NvidiaPtx
        );
        assert_eq!(
            CtxResumeTarget::parse("amd_rocm").unwrap(),
            CtxResumeTarget::AmdRocm
        );
        assert_eq!(
            CtxResumeTarget::parse("intel_spirv").unwrap(),
            CtxResumeTarget::IntelSpirv
        );
        assert_eq!(
            CtxResumeTarget::parse("tenstorrent_tosa").unwrap(),
            CtxResumeTarget::TenstorrentTosa
        );
    }
}
