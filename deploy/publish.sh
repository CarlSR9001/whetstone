#!/usr/bin/env bash
set -euo pipefail

# Build and publish exactly one reviewed Git commit.
# Usage: bash deploy/publish.sh [ssh-host] [git-ref]

ssh_host="${1:-vps2}"
git_ref="${2:-HEAD}"
rollback_forge_state="${WHETSTONE_EXPECT_FORGE_ACTIVE:-auto}"
repo="$(git -c safe.directory="${PWD}" rev-parse --show-toplevel)"
git_cmd=(git -c safe.directory="${repo}" -C "${repo}")
branch="$("${git_cmd[@]}" branch --show-current)"
commit="$("${git_cmd[@]}" rev-parse --verify "${git_ref}^{commit}")"

if [[ "${rollback_forge_state}" != "auto" && "${rollback_forge_state}" != "active" && "${rollback_forge_state}" != "inactive" ]]; then
  printf 'error: WHETSTONE_EXPECT_FORGE_ACTIVE must be auto, active, or inactive\n' >&2
  exit 64
fi

if [[ "${branch}" != "main" ]]; then
  printf 'error: releases must be cut from main (current branch: %s)\n' "${branch}" >&2
  exit 64
fi
if [[ -n "$("${git_cmd[@]}" status --porcelain=v1 --untracked-files=all)" ]]; then
  printf 'error: working tree is not clean\n' >&2
  exit 65
fi
if ! "${git_cmd[@]}" merge-base --is-ancestor "${commit}" main; then
  printf 'error: target commit is not on main: %s\n' "${commit}" >&2
  exit 65
fi
"${git_cmd[@]}" fetch --quiet origin main
if ! "${git_cmd[@]}" merge-base --is-ancestor "${commit}" origin/main; then
  printf 'error: target commit is not present on public origin/main: %s\n' "${commit}" >&2
  exit 65
fi

temp_root="$(mktemp -d)"
archive="${temp_root}/whetstone-${commit}.tar"
extract="${temp_root}/extract"
remote_archive="/tmp/whetstone-${commit}.tar"
remote_installer="/tmp/whetstone-install-${commit}.sh"
remote_uploaded=0

cleanup() {
  rm -rf -- "${temp_root}"
  if (( remote_uploaded )); then
    ssh "${ssh_host}" rm -f -- "${remote_archive}" "${remote_installer}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

mkdir -p "${extract}"
"${git_cmd[@]}" archive --format=tar --output="${archive}" "${commit}"
tar -xf "${archive}" -C "${extract}"

embedded_commit="$(
  python3 - "${extract}/src/bcv/_version.py" <<'PY'
import importlib.util
import os
import sys

os.environ.pop("WHETSTONE_BUILD_COMMIT", None)
spec = importlib.util.spec_from_file_location("_whetstone_release", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.build_commit())
PY
)"
if [[ "${embedded_commit}" != "${commit}" ]]; then
  printf 'error: local archive provenance %q does not match %s\n' "${embedded_commit}" "${commit}" >&2
  exit 65
fi

printf 'commit=%s\narchive_sha256=' "${commit}"
sha256sum "${archive}" | awk '{print $1}'
scp "${archive}" "${ssh_host}:${remote_archive}"
remote_uploaded=1
scp "${extract}/deploy/install-release.sh" "${ssh_host}:${remote_installer}"
ssh "${ssh_host}" sudo bash "${remote_installer}" "${commit}" "${remote_archive}" "${rollback_forge_state}"
