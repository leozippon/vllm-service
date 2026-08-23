# Shared preamble for the deployment entry scripts. Resolves the active
# deployment profile and sources it. Sourced, never executed.

fail() {
    printf '%s: %s\n' "${0##*/}" "$*" >&2
    exit 1
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DEPLOYMENTS_DIR="$REPO_ROOT/config/deployments"
SELECTOR="$DEPLOYMENTS_DIR/active"

# The selector names the default deployment. VLLM_DEPLOYMENT overrides it so a
# trade-study run, or a future per-instance unit, can resolve a different profile
# without rewriting the selector -- and so a candidate configuration never has to
# be written over the production profile. It is deliberately absent from the
# unset list below: unlike the profile's own names it arrives from the
# environment by design.
if [[ -n "${VLLM_DEPLOYMENT:-}" ]]; then
    ACTIVE_DEPLOYMENT="$VLLM_DEPLOYMENT"
else
    [[ -r "$SELECTOR" ]] || fail "active deployment selector is not readable: $SELECTOR"
    ACTIVE_DEPLOYMENT=$(<"$SELECTOR")
fi
[[ "$ACTIVE_DEPLOYMENT" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || \
    fail "active deployment ID is empty or invalid: $ACTIVE_DEPLOYMENT"
PROFILE="$DEPLOYMENTS_DIR/$ACTIVE_DEPLOYMENT.conf"
[[ -r "$PROFILE" ]] || fail "deployment profile is not readable: $PROFILE"

# The profile is the authority for these names. Clearing them first keeps an
# inherited value from silently satisfying a required-value check. VLLM_API_KEY
# is deliberately absent: it legitimately arrives from .env through systemd.
unset VLLM_PYTHON VLLM_MODEL_PATH VLLM_SERVED_MODEL_NAME VLLM_PORT \
      VLLM_GPUS VLLM_CPU_AFFINITY VLLM_NUMA_NODES \
      VLLM_RUNTIME_ENV VLLM_ARGS

# Deployment profiles are trusted shell configuration maintained with this repository.
# shellcheck source=/dev/null
source "$PROFILE"

# Every entry script needs the port, so it is validated once here rather than in
# four places: the engine binds it, the gateway proxies to it and answers on the
# next one, and both health probes default from it. The upper bound is 65534 so
# that the gateway's port is a port too.
[[ "${VLLM_PORT:-}" =~ ^[1-9][0-9]*$ ]] && ((VLLM_PORT < 65535)) || \
    fail "VLLM_PORT must be a TCP port below 65535 in $PROFILE, got: ${VLLM_PORT:-}"
