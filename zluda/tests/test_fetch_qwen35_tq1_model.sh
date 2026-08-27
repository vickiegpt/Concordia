#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
script="${repo_root}/tools/fetch_qwen35_tq1_model.sh"

grep -Fq 'Qwen3.5-397B-A17B-UD-TQ1_0.gguf' "${script}"
grep -Fq '94155830880' "${script}"
grep -Fq '0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568' "${script}"
bash -n "${script}"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
mkdir -p "${work}/bin" "${work}/model"

cat >"${work}/bin/curl" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
output=""
while (($#)); do
    if [[ "$1" == "--output" ]]; then
        output="$2"
        shift 2
    else
        shift
    fi
done
[[ "${output}" == *.partial ]] || {
    echo "download did not target a partial file: ${output}" >&2
    exit 90
}
printf 'fixture' >"${output}"
SH
chmod +x "${work}/bin/curl"

cat >"${work}/bin/sha256sum" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s  %s\n' "${HETGPU_TEST_SHA:?}" "$1"
SH
chmod +x "${work}/bin/sha256sum"

expected_sha='0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568'
output="$({
    HETGPU_MODEL_FETCH_TESTING=1 \
    HETGPU_TEST_MODEL_SIZE=7 \
    HETGPU_TEST_SHA="${expected_sha}" \
    PATH="${work}/bin:${PATH}" \
        "${script}" "${work}/model"
})"
final="${work}/model/Qwen3.5-397B-A17B-UD-TQ1_0.gguf"
[[ -f "${final}" ]]
[[ ! -e "${final}.partial" ]]
grep -Fq "verified_model=${final}" <<<"${output}"
grep -Fq 'verified_size=7' <<<"${output}"
grep -Fq "verified_sha256=${expected_sha}" <<<"${output}"

# A previously verified final file must not be downloaded again.
cat >"${work}/bin/curl" <<'SH'
#!/usr/bin/env bash
echo "unexpected download" >&2
exit 91
SH
chmod +x "${work}/bin/curl"
HETGPU_MODEL_FETCH_TESTING=1 \
HETGPU_TEST_MODEL_SIZE=7 \
HETGPU_TEST_SHA="${expected_sha}" \
PATH="${work}/bin:${PATH}" \
    "${script}" "${work}/model" >/dev/null

# Capacity failure must occur before curl and must not publish a final file.
mkdir -p "${work}/lowbin" "${work}/low-space"
cat >"${work}/lowbin/df" <<'SH'
#!/usr/bin/env bash
printf 'Filesystem 1-blocks Used Available Use%% Mounted on\n'
printf 'fixture 100 0 100 0%% /\n'
SH
cat >"${work}/lowbin/curl" <<'SH'
#!/usr/bin/env bash
touch "${HETGPU_CURL_CALLED:?}"
exit 92
SH
chmod +x "${work}/lowbin/df" "${work}/lowbin/curl"
if HETGPU_MODEL_FETCH_TESTING=1 \
   HETGPU_TEST_MODEL_SIZE=7 \
   HETGPU_TEST_SHA="${expected_sha}" \
   HETGPU_CURL_CALLED="${work}/curl-called" \
   PATH="${work}/lowbin:${work}/bin:${PATH}" \
       "${script}" "${work}/low-space" >"${work}/low.out" 2>"${work}/low.err"; then
    echo "capacity failure unexpectedly succeeded" >&2
    exit 1
fi
grep -Fq 'insufficient free bytes for verified Qwen model' "${work}/low.err"
[[ ! -e "${work}/curl-called" ]]

# Existing non-directory destinations are rejected without mutation.
printf 'not-a-directory' >"${work}/bad-destination"
if HETGPU_MODEL_FETCH_TESTING=1 \
   HETGPU_TEST_MODEL_SIZE=7 \
   HETGPU_TEST_SHA="${expected_sha}" \
   PATH="${work}/bin:${PATH}" \
       "${script}" "${work}/bad-destination" >"${work}/bad.out" 2>"${work}/bad.err"; then
    echo "non-directory destination unexpectedly succeeded" >&2
    exit 1
fi
grep -Fq 'model destination exists and is not a directory' "${work}/bad.err"
