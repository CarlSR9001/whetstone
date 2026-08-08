#!/usr/bin/env bash
set -Eeuo pipefail

# Server-side installer for one immutable, commit-named public release.
# Usage: sudo bash install-release.sh <40-char-commit> <git-archive.tar> [auto|active|inactive]

commit="${1:-}"
archive="${2:-}"
rollback_forge_state="${3:-auto}"
release_root="/opt/whetstone-tools/releases"
current_link="/opt/whetstone-tools/current"
unit_root="/etc/systemd/system"
release_env="/etc/whetstone-tools/release.env"
nginx_site="/etc/nginx/sites-available/whetstone-tools"

if [[ "${EUID}" -ne 0 ]]; then
  printf 'error: install-release.sh must run as root\n' >&2
  exit 64
fi
if [[ ! "${commit}" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'error: expected a full lowercase Git commit, got %q\n' "${commit}" >&2
  exit 64
fi
if [[ ! -f "${archive}" ]]; then
  printf 'error: archive does not exist: %s\n' "${archive}" >&2
  exit 66
fi
if [[ "${rollback_forge_state}" != "auto" && "${rollback_forge_state}" != "active" && "${rollback_forge_state}" != "inactive" ]]; then
  printf 'error: rollback forge state must be auto, active, or inactive\n' >&2
  exit 64
fi
if ! command -v flock >/dev/null 2>&1; then
  printf 'error: flock is required for serialized activation\n' >&2
  exit 69
fi
if ! command -v ssh-keygen >/dev/null 2>&1; then
  printf 'error: ssh-keygen with SSHSIG support is required for receipt signing\n' >&2
  exit 69
fi

exec 9>/run/lock/whetstone-release.lock
if ! flock -n 9; then
  printf 'error: another Whetstone activation is already running\n' >&2
  exit 75
fi

target="${release_root}/${commit}"
incoming="${release_root}/.incoming-${commit}-$$"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_root="/var/backups/whetstone-tools/${timestamp}-${commit}"
previous="$(readlink -f "${current_link}" 2>/dev/null || true)"
tools_was_active="$(systemctl is-active whetstone-tools.service 2>/dev/null || true)"
forge_was_active="$(systemctl is-active whetstone-forge.service 2>/dev/null || true)"
tools_was_enabled="$(systemctl is-enabled whetstone-tools.service 2>/dev/null || true)"
forge_was_enabled="$(systemctl is-enabled whetstone-forge.service 2>/dev/null || true)"
if [[ "${rollback_forge_state}" != "auto" ]]; then
  forge_was_active="${rollback_forge_state}"
fi
rollback_needed=0
state_validator=""

require_no_report_card_sessions() {
  if [[ "$(systemctl is-active whetstone-tools.service 2>/dev/null || true)" != "active" ]]; then
    return
  fi
  local health
  health="$(curl --fail --silent --show-error --max-time 3 http://127.0.0.1:8988/api/health)"
  if ! printf '%s' "${health}" | python3 "${state_validator}" sessions --quiet; then
    printf 'error: live report-card sessions are still active; retry after they expire or submit\n' >&2
    printf '%s' "${health}" | python3 "${state_validator}" sessions || true
    exit 75
  fi
}

cleanup() {
  rm -rf -- "${incoming}"
  rm -f -- "${archive}"
  if [[ -n "${state_validator}" ]]; then
    rm -f -- "${state_validator}"
  fi
}

restore_file() {
  local backup="$1"
  local destination="$2"
  if [[ -f "${backup}" ]]; then
    install -m 0644 "${backup}" "${destination}"
  elif [[ -f "${backup}.absent" ]]; then
    rm -f -- "${destination}"
  fi
}

rollback() {
  trap - ERR
  printf 'activation failed; restoring previous release %s\n' "${previous:-<none>}" >&2
  systemctl stop whetstone-forge.service 2>/dev/null || true
  systemctl stop whetstone-tools.service 2>/dev/null || true

  restore_file "${backup_root}/whetstone-tools.service" "${unit_root}/whetstone-tools.service"
  restore_file "${backup_root}/whetstone-forge.service" "${unit_root}/whetstone-forge.service"
  restore_file "${backup_root}/release.env" "${release_env}"
  restore_file "${backup_root}/nginx-whetstone" "${nginx_site}"

  if [[ -n "${previous}" && -d "${previous}" ]]; then
    ln -s "${previous}" "${current_link}.rollback-$$"
    mv -Tf "${current_link}.rollback-$$" "${current_link}"
  else
    rm -f -- "${current_link}"
  fi
  systemctl daemon-reload
  if [[ "${tools_was_enabled}" == "enabled" ]]; then
    systemctl enable whetstone-tools.service 2>/dev/null || true
  else
    systemctl disable whetstone-tools.service 2>/dev/null || true
  fi
  if [[ "${forge_was_enabled}" == "enabled" ]]; then
    systemctl enable whetstone-forge.service 2>/dev/null || true
  else
    systemctl disable whetstone-forge.service 2>/dev/null || true
  fi
  if [[ "${tools_was_active}" == "active" ]]; then
    systemctl start whetstone-tools.service || true
  fi
  if [[ "${forge_was_active}" == "active" ]]; then
    systemctl start whetstone-forge.service || true
  fi
  nginx -t && systemctl reload nginx.service || true
}

on_error() {
  local rc=$?
  local line="$1"
  if (( rollback_needed )); then
    rollback
  fi
  printf 'error: release activation failed at line %s (exit %s)\n' "${line}" "${rc}" >&2
  exit "${rc}"
}

trap 'on_error "${LINENO}"' ERR
trap cleanup EXIT

state_validator="$(mktemp /tmp/whetstone-release-state.XXXXXX.py)"
tar -xOf "${archive}" scripts/check_release_state.py > "${state_validator}"
chmod 0500 "${state_validator}"
require_no_report_card_sessions
forge_pid="$(systemctl show whetstone-forge.service --property=MainPID --value 2>/dev/null || true)"
if [[ "${forge_pid}" =~ ^[0-9]+$ ]] && (( forge_pid > 0 )); then
  forge_children="$(pgrep -P "${forge_pid}" 2>/dev/null || true)"
  if [[ -n "${forge_children}" ]]; then
    printf 'error: forge cycle is active under PID %s; retry at the next idle boundary\n' "${forge_pid}" >&2
    exit 75
  fi
fi

install -d -m 0755 "${release_root}"
if [[ ! -d "${target}" ]]; then
  install -d -m 0755 "${incoming}"
  tar --no-same-owner --no-same-permissions -xf "${archive}" -C "${incoming}"

  embedded_commit="$(
    python3 - "${incoming}/src/bcv/_version.py" <<'PY'
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
    printf 'error: archive provenance %q does not match target %s\n' "${embedded_commit}" "${commit}" >&2
    exit 65
  fi
  chown -R root:root "${incoming}"
  chmod -R go-w "${incoming}"
  mv "${incoming}" "${target}"
fi

target_commit="$(
  env -u WHETSTONE_BUILD_COMMIT python3 - "${target}/src/bcv/_version.py" <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("_whetstone_release", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.build_commit())
PY
)"
target_version="$(
  python3 - "${target}/src/bcv/_version.py" <<'PY'
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("_whetstone_release", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.__version__)
PY
)"
if [[ "${target_commit}" != "${commit}" ]]; then
  printf 'error: release directory failed provenance check: %s\n' "${target_commit}" >&2
  exit 65
fi
systemd-analyze verify \
  "${target}/deploy/whetstone-tools.service" \
  "${target}/deploy/whetstone-forge.service"

install -d -m 0700 "$(dirname "${backup_root}")"
state_bytes=0
for state_path in /var/lib/whetstone-tools /home/shankatsu/whetstone-forge/state; do
  if [[ -d "${state_path}" ]]; then
    size="$(du -sb "${state_path}" | awk '{print $1}')"
    state_bytes=$((state_bytes + size))
  fi
done
available_bytes="$(df -PB1 "$(dirname "${backup_root}")" | awk 'NR == 2 {print $4}')"
required_bytes=$((state_bytes + 104857600))
if (( available_bytes < required_bytes )); then
  printf 'error: insufficient backup space (need %s bytes, have %s)\n' \
    "${required_bytes}" "${available_bytes}" >&2
  exit 70
fi
install -d -m 0700 "${backup_root}"
for unit in whetstone-tools.service whetstone-forge.service; do
  if [[ -f "${unit_root}/${unit}" ]]; then
    cp -a "${unit_root}/${unit}" "${backup_root}/${unit}"
  else
    : > "${backup_root}/${unit}.absent"
  fi
done
if [[ -f "${release_env}" ]]; then
  cp -a "${release_env}" "${backup_root}/release.env"
else
  : > "${backup_root}/release.env.absent"
fi
if [[ -f "${nginx_site}" ]]; then
  cp -a "${nginx_site}" "${backup_root}/nginx-whetstone"
else
  : > "${backup_root}/nginx-whetstone.absent"
fi

cycle_lock="/home/shankatsu/whetstone-forge/state/forge_cycle.lock"
install -d -o shankatsu -g shankatsu -m 0750 "$(dirname "${cycle_lock}")"
touch "${cycle_lock}"
chown shankatsu:shankatsu "${cycle_lock}"
chmod 0600 "${cycle_lock}"
exec 8<>"${cycle_lock}"
if ! flock -n 8; then
  printf 'error: forge cycle lock is held; retry at the next idle boundary\n' >&2
  exit 75
fi
forge_pid="$(systemctl show whetstone-forge.service --property=MainPID --value 2>/dev/null || true)"
if [[ "${forge_pid}" =~ ^[0-9]+$ ]] && (( forge_pid > 0 )); then
  forge_children="$(pgrep -P "${forge_pid}" 2>/dev/null || true)"
  if [[ -n "${forge_children}" ]]; then
    printf 'error: forge cycle began before activation lock; retry later\n' >&2
    exit 75
  fi
fi

require_no_report_card_sessions
rollback_needed=1
if [[ "$(systemctl show whetstone-forge.service --property=LoadState --value 2>/dev/null || true)" != "not-found" ]]; then
  systemctl stop whetstone-forge.service
fi
if [[ "$(systemctl show whetstone-tools.service --property=LoadState --value 2>/dev/null || true)" != "not-found" ]]; then
  systemctl stop whetstone-tools.service
fi

if [[ -d /var/lib/whetstone-tools ]]; then
  cp -a /var/lib/whetstone-tools "${backup_root}/tools-state"
fi
if [[ -d /home/shankatsu/whetstone-forge/state ]]; then
  cp -a /home/shankatsu/whetstone-forge/state "${backup_root}/forge-state"
fi

install -m 0644 "${target}/deploy/whetstone-tools.service" "${unit_root}/whetstone-tools.service.new"
mv -f "${unit_root}/whetstone-tools.service.new" "${unit_root}/whetstone-tools.service"
install -m 0644 "${target}/deploy/whetstone-forge.service" "${unit_root}/whetstone-forge.service.new"
mv -f "${unit_root}/whetstone-forge.service.new" "${unit_root}/whetstone-forge.service"
install -d -m 0755 "$(dirname "${release_env}")"
printf 'WHETSTONE_BUILD_COMMIT=%s\n' "${commit}" > "${release_env}.new"
chmod 0644 "${release_env}.new"
mv -f "${release_env}.new" "${release_env}"

ln -s "${target}" "${current_link}.new-$$"
mv -Tf "${current_link}.new-$$" "${current_link}"
systemctl daemon-reload
systemctl enable whetstone-tools.service whetstone-forge.service
install -d -o shankatsu -g shankatsu -m 0700 /var/lib/whetstone-tools
receipt_key="/var/lib/whetstone-tools/receipt_signing_key"
receipt_trusted="/var/lib/whetstone-tools/receipt_trusted_keys"
install -d -o shankatsu -g shankatsu -m 0700 "$receipt_trusted"
if [[ ! -s "$receipt_key" ]]; then
  rm -f -- "$receipt_key.new" "$receipt_key.new.pub"
  runuser --user shankatsu -- ssh-keygen -q -t ed25519 -N '' -C 'whetstone-receipt' -f "$receipt_key.new"
  mv -f "$receipt_key.new" "$receipt_key"
  mv -f "$receipt_key.new.pub" "$receipt_key.pub"
elif [[ ! -s "$receipt_key.pub" ]]; then
  runuser --user shankatsu -- ssh-keygen -y -f "$receipt_key" > "$receipt_key.pub.new"
  mv -f "$receipt_key.pub.new" "$receipt_key.pub"
fi
chown shankatsu:shankatsu "$receipt_key" "$receipt_key.pub"
chmod 0600 "$receipt_key"
chmod 0644 "$receipt_key.pub"
receipt_key_hash="$(sha256sum "$receipt_key.pub" | awk '{print $1}')"
install -o shankatsu -g shankatsu -m 0644 "$receipt_key.pub" "$receipt_trusted/$receipt_key_hash.pub"

forge_status="/home/shankatsu/whetstone-forge/state/forge_release_status.json"
rm -f -- "${forge_status}"
systemctl start whetstone-forge.service
forge_restarts="$(systemctl show whetstone-forge.service --property=NRestarts --value)"
forge_started_pid=""
forge_ok=0
forge_stable=0
forge_last_json=""
for _ in $(seq 1 20); do
  if systemctl is-active --quiet whetstone-forge.service && [[ -s "${forge_status}" ]]; then
    forge_pid="$(systemctl show whetstone-forge.service --property=MainPID --value)"
    current_restarts="$(systemctl show whetstone-forge.service --property=NRestarts --value)"
    if [[ -z "${forge_started_pid}" ]]; then
      forge_started_pid="${forge_pid}"
    fi
    if [[ "${forge_pid}" != "${forge_started_pid}" || "${current_restarts}" != "${forge_restarts}" ]]; then
      printf 'error: forge restarted during activation (pid %s -> %s, restarts %s -> %s)\n' \
        "${forge_started_pid}" "${forge_pid}" "${forge_restarts}" "${current_restarts}" >&2
      false
    fi
    forge_last_json="$(cat "${forge_status}")"
    if printf '%s' "${forge_last_json}" | python3 "${state_validator}" forge \
      --commit "${commit}" --version "${target_version}" --pid "${forge_pid}" --quiet
    then
      forge_stable=$((forge_stable + 1))
      if (( forge_stable >= 5 )); then
        forge_ok=1
        break
      fi
    else
      forge_stable=0
    fi
  else
    forge_stable=0
  fi
  sleep 1
done
if (( ! forge_ok )); then
  printf 'error: forge release identity did not converge for %s\n' "${commit}" >&2
  if [[ -n "${forge_last_json}" ]]; then
    printf '%s' "${forge_last_json}" | python3 "${state_validator}" forge \
      --commit "${commit}" --version "${target_version}" --pid "${forge_pid}" || true
  fi
  false
fi

# Forge publishes/deduplicates its library before writing the status receipt.
# Starting tools afterward guarantees the hatchery warms exactly that revision.
systemctl start whetstone-tools.service
health_ok=0
health_json=""
health_deadline=$((SECONDS + 600))
while (( SECONDS < health_deadline )); do
  if health_json="$(curl --fail --silent --show-error --max-time 3 http://127.0.0.1:8988/api/health 2>/dev/null)"; then
    if printf '%s' "${health_json}" | python3 "${state_validator}" health \
      --commit "${commit}" --version "${target_version}" --quiet
    then
      health_ok=1
      break
    fi
  fi
  sleep 1
done
if (( ! health_ok )); then
  printf 'error: local health gate did not converge for %s\n' "${commit}" >&2
  if [[ -n "${health_json}" ]]; then
    printf '%s' "${health_json}" | python3 "${state_validator}" health \
      --commit "${commit}" --version "${target_version}" || true
  fi
  false
fi
systemctl is-enabled --quiet whetstone-tools.service
systemctl is-enabled --quiet whetstone-forge.service

install -m 0644 "${target}/deploy/nginx.conf" "${nginx_site}.new"
mv -f "${nginx_site}.new" "${nginx_site}"
nginx -t
systemctl reload nginx.service

rollback_needed=0
printf 'release=%s\nversion=%s\nprevious=%s\nbackup=%s\n' \
  "${commit}" "${target_version}" "${previous:-none}" "${backup_root}"
