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

[[ -r "$SELECTOR" ]] || fail "active deployment selector is not readable: $SELECTOR"
ACTIVE_DEPLOYMENT=$(<"$SELECTOR")
[[ "$ACTIVE_DEPLOYMENT" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || \
    fail "active deployment ID is empty or invalid: $ACTIVE_DEPLOYMENT"
PROFILE="$DEPLOYMENTS_DIR/$ACTIVE_DEPLOYMENT.conf"
[[ -r "$PROFILE" ]] || fail "deployment profile is not readable: $PROFILE"

# The profile is the authority for these names. Clearing them first keeps an
# inherited value from silently satisfying a required-value check. VLLM_API_KEY
# is deliberately absent: it legitimately arrives from .env through systemd.
unset VLLM_PYTHON VLLM_MODEL_PATH VLLM_SERVED_MODEL_NAME \
      VLLM_GPUS VLLM_CPU_AFFINITY VLLM_NUMA_NODES \
      VLLM_RUNTIME_ENV VLLM_ARGS

# Deployment profiles are trusted shell configuration maintained with this repository.
# shellcheck source=/dev/null
source "$PROFILE"
