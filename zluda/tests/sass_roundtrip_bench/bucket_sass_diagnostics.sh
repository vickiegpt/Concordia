#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: bucket_sass_diagnostics.sh [--output diagnostics.csv] [LOG_OR_DIR ...]

Reads hetGPU SASS diagnostic stderr logs and writes:
opcode,message,count,sample_instruction

When no LOG_OR_DIR is provided, logs are read from stdin.
EOF
}

output=""
inputs=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            output="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            inputs+=("$1")
            shift
            ;;
    esac
done

tmp="$(mktemp /tmp/hetgpu-sass-diagnostic-buckets.XXXXXX)"
trap 'rm -f "${tmp}"' EXIT

awk_program='
function trim(s) {
    sub(/^[[:space:]]+/, "", s)
    sub(/[[:space:]]+$/, "", s)
    return s
}
function csv(s) {
    gsub(/"/, "\"\"", s)
    return "\"" s "\""
}
/\[hetGPU SASS\] diagnostic / {
    line = $0
    if (!match(line, /opcode=[^ ]+/)) {
        next
    }
    opcode = substr(line, RSTART + 7, RLENGTH - 7)
    rest = substr(line, RSTART + RLENGTH + 1)
    inst = ""
    inst_pos = index(rest, " inst=")
    if (inst_pos > 0) {
        inst = substr(rest, inst_pos + 6)
        message = substr(rest, 1, inst_pos - 1)
    } else {
        message = rest
    }
    message = trim(message)
    key = opcode SUBSEP message
    counts[key] += 1
    if (!(key in samples) && inst != "") {
        samples[key] = inst
    }
}
END {
    for (key in counts) {
        split(key, parts, SUBSEP)
        print csv(parts[1]) "," csv(parts[2]) "," counts[key] "," csv(samples[key])
    }
}
'

if [[ ${#inputs[@]} -eq 0 ]]; then
    awk "${awk_program}" >"${tmp}"
else
    files=()
    for input in "${inputs[@]}"; do
        if [[ -d "${input}" ]]; then
            while IFS= read -r file; do
                files+=("${file}")
            done < <(find "${input}" -type f \( -name '*.log' -o -name '*.stderr' -o -name '*.err' -o -name '*.txt' \) | sort)
        else
            files+=("${input}")
        fi
    done
    if [[ ${#files[@]} -eq 0 ]]; then
        :
    else
        awk "${awk_program}" "${files[@]}" >"${tmp}"
    fi
fi

emit_csv() {
    printf '%s\n' "opcode,message,count,sample_instruction"
    sort -t, -k3,3nr -k1,1 -k2,2 "${tmp}"
}

if [[ -n "${output}" ]]; then
    emit_csv >"${output}"
    printf '[sass-roundtrip] wrote diagnostic buckets: %s\n' "${output}" >&2
else
    emit_csv
fi
