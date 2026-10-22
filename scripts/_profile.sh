# Shared preamble for the entry scripts: the runtime interpreter, and loaders
# for model profiles and for the shared deployment. Sourced, never executed.

fail() {
    printf '%s: %s\n' "${0##*/}" "$*" >&2
    exit 1
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[1]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PROFILES_DIR="$REPO_ROOT/config/profiles"
DEPLOYMENTS_DIR="$REPO_ROOT/config/deployments"

# Every engine and the gateway run from the repository's own runtime. It is a
# fact of this checkout, not of a model or a deployment, so it is stated once.
VLLM_PYTHON="$REPO_ROOT/.venv/bin/python"

valid_name() {
    [[ "$1" =~ ^[a-z0-9][a-z0-9._-]*$ ]]
}

positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

# Engine options the launcher sets itself. Neither a profile's engine arguments
# nor a caller's extra arguments may restate one: each has exactly one source,
# and the API key must never reach a command line, because /proc/<pid>/cmdline
# is world-readable on a shared host.
refuse_launcher_options() {  # <where> <engine-arg>...
    local where=$1 token name
    shift
    for token in "$@"; do
        name=${token%%=*}
        name=${name//_/-}  # vLLM accepts --max_model_len for --max-model-len
        case $name in
            --model | --served-model-name | --host | --port | --api-key | \
            --runner | --convert | --tensor-parallel-size | -tp | \
            --max-model-len | --gpu-memory-utilization)
                fail "$where must not set $name: the launcher sets it" ;;
        esac
    done
}

# load_profile <name>: source config/profiles/<name>.conf and validate it.
# Profiles are trusted shell configuration maintained with this repository.
load_profile() {
    local name=$1 key declaration
    valid_name "$name" || fail "profile name is empty or invalid: $name"
    PROFILE="$PROFILES_DIR/$name.conf"
    [[ -r "$PROFILE" ]] || fail "profile is not readable: $PROFILE"

    # The profile is the authority for these names. Clearing them first keeps an
    # inherited value from silently satisfying a required-value check.
    unset VLLM_MODEL_PATH VLLM_SERVED_MODEL_NAME VLLM_TASK \
          VLLM_TENSOR_PARALLEL_SIZE VLLM_MAX_MODEL_LEN \
          VLLM_RELEASE_DATE VLLM_KNOWLEDGE_CUTOFF VLLM_NOTE VLLM_ARGS
    # shellcheck source=/dev/null
    source "$PROFILE"

    for key in VLLM_MODEL_PATH VLLM_SERVED_MODEL_NAME VLLM_TASK \
               VLLM_TENSOR_PARALLEL_SIZE VLLM_MAX_MODEL_LEN \
               VLLM_RELEASE_DATE VLLM_NOTE; do
        [[ -n "${!key:-}" ]] || fail "$key is required in $PROFILE"
    done
    [[ -d "$VLLM_MODEL_PATH" ]] || fail "VLLM_MODEL_PATH is not a directory: $VLLM_MODEL_PATH"
    [[ "$VLLM_TASK" == generate || "$VLLM_TASK" == embed ]] || \
        fail "VLLM_TASK must be generate or embed in $PROFILE, got: $VLLM_TASK"
    positive_integer "$VLLM_TENSOR_PARALLEL_SIZE" || \
        fail "VLLM_TENSOR_PARALLEL_SIZE must be a positive integer in $PROFILE"
    positive_integer "$VLLM_MAX_MODEL_LEN" || \
        fail "VLLM_MAX_MODEL_LEN must be a positive integer in $PROFILE"
    declaration=$(declare -p VLLM_ARGS 2>/dev/null) || fail "VLLM_ARGS is required in $PROFILE"
    [[ "$declaration" == "declare -a "* ]] || fail "VLLM_ARGS must be a Bash array in $PROFILE"
    refuse_launcher_options "VLLM_ARGS in $PROFILE" "${VLLM_ARGS[@]}"
}

# load_deployment: source the shared deployment's configuration. The selector
# config/deployments/active names it; VLLM_DEPLOYMENT in the environment
# overrides the selector, so a candidate can boot without rewriting it.
load_deployment() {
    local selector="$DEPLOYMENTS_DIR/active" id
    if [[ -n "${VLLM_DEPLOYMENT:-}" ]]; then
        id=$VLLM_DEPLOYMENT
    else
        [[ -r "$selector" ]] || fail "active deployment selector is not readable: $selector"
        id=$(<"$selector")
    fi
    valid_name "$id" || fail "active deployment ID is empty or invalid: $id"
    DEPLOYMENT="$DEPLOYMENTS_DIR/$id.conf"
    [[ -r "$DEPLOYMENT" ]] || fail "deployment is not readable: $DEPLOYMENT"

    unset VLLM_PROFILE VLLM_PORT VLLM_GPUS VLLM_CPU_CORES VLLM_CPU_AFFINITY \
          VLLM_NUMA_NODES VLLM_LAUNCH_ARGS VLLM_GATEWAY_REASONING_EFFORT_ALIAS
    # shellcheck source=/dev/null
    source "$DEPLOYMENT"

    # Every deployment script needs the port, so it is validated once here: the
    # engine binds it, the gateway proxies to it and answers on the next one, and
    # both health probes default from it. The upper bound is 65534 so that the
    # gateway's port is a port too.
    positive_integer "${VLLM_PORT:-}" && ((VLLM_PORT < 65535)) || \
        fail "VLLM_PORT must be a TCP port below 65535 in $DEPLOYMENT, got: ${VLLM_PORT:-}"
}
