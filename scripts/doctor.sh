#!/usr/bin/env bash
# Check that this machine is ready to build and run the benchmark, and say how to fix what is not.
# Usage: scripts/doctor.sh      (also: make doctor)
# Order: tools, the machine (WSL, disk), the network (the public data sources).
# Every failure comes with the next step to take.

bad=0
ok()   { echo "  ok    $1"; }
warn() { echo "  WARN  $1"; }
fail() { echo "  FAIL  $1"; bad=$((bad + 1)); }
hint() { echo "        -> $1"; }

# ---------------------------------------------------------------- tools

echo "Tools"
for t in bash make git curl python3; do
  command -v "$t" >/dev/null 2>&1 && ok "$t" || fail "$t not found"
done
if command -v python3 >/dev/null 2>&1; then
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null \
    && ok "python3 $(python3 -c 'import platform; print(platform.python_version())') (3.11+ needed)" \
    || fail "python3 is older than 3.11"
fi
if command -v uv >/dev/null 2>&1; then
  ok "uv"
else
  warn "uv not found (make setup will need it)"
  hint "install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# ---------------------------------------------------------------- the machine

echo "System"
if grep -qi "microsoft" /proc/version 2>/dev/null; then
  grep -qi "wsl2" /proc/version 2>/dev/null && ok "WSL2" \
    || warn "WSL1 detected: PyTorch and nemosis work, but file access is slow"
  case "$PWD" in
    /mnt/*) warn "repo is on the Windows drive ($PWD): file access is slow here"
            hint "keep the virtualenv in WSL's home (make setup does this); data/ on /mnt/c is fine" ;;
  esac
fi
echo "  info  $(uname -m), $(nproc) cpus, $(free -g 2>/dev/null | awk '/^Mem:/ {print $2}') GB RAM, no CUDA on this machine unless nvidia-smi works"
free_gb="$(df -Pk . 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}')"
if [ -n "$free_gb" ] && [ "$free_gb" -lt 10 ]; then
  fail "only ${free_gb} GB free here (the data and model cache need about 10 GB)"
  hint "free some space, or point the caches elsewhere"
else
  ok "${free_gb:-?} GB free"
fi

# ---------------------------------------------------------------- the network

echo "Network"
probe() {  # probe <name> <url>
  curl -fsS -o /dev/null --max-time 15 -I "$2" 2>/dev/null && ok "$1" \
    || { fail "cannot reach $1 ($2)"; hint "check the network or proxy; this data source is public and needs no key"; }
}
probe "AEMO MMSDM archive" "https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
probe "Hugging Face (model weights)" "https://huggingface.co/amazon/chronos-2/resolve/main/config.json"

echo
if [ "$bad" -eq 0 ]; then
  echo "Ready."
else
  echo "$bad problem(s) above. Fix them and run make doctor again."
  exit 1
fi
