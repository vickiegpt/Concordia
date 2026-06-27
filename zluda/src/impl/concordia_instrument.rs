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
}
