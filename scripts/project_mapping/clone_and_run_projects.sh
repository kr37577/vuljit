#!/usr/bin/env bash
# Clone OSS-Fuzz C/C++ projects listed in c_cpp_vulnerability_summary.csv and run a command per project.


here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${here}/../.." && pwd)"
default_csv="${repo_root}/datasets/derived_artifacts/oss_fuzz_metadata/c_cpp_vulnerability_summary.csv"
default_clone_dir="${repo_root}/datasets/raw/cloned_c_cpp_projects"

usage() {
  cat <<EOF
Usage: $(basename "$0") [-c csv] [-d clone_dir] [-s since] [-u until] [-r runner] [-C "cmd ..."] [-- runner_args...]

  -c csv        Path to c_cpp_vulnerability_summary.csv (default: ${default_csv})
  -d clone_dir  Directory where repositories will be cloned (default: ${default_clone_dir})
  -r runner     Executable invoked with <project> <repo> followed by extra args.
  -C command    Command template executed per project (placeholders: {project}, {repo}, {runner}, {runner_args}).
  -s since      Optional since date appended as --since when using -r (ignored if --since already present).
  -u until      Optional until date appended as --until when using -r (ignored if --until already present).
  -h            Show this help.

If both -r and -C are provided, the template is executed with placeholders expanded.
Example:
  $(basename "$0") -r "/work/riku-ka/vuljit/scripts/metric_extraction/cluster.sh" \\
    -C "sbatch {runner} --since 20180101 --until 20251001 {project} {repo}" -- --force
EOF
}

csv_path="${default_csv}"
clone_base="${default_clone_dir}"
runner=""
command_template=""
since_arg=""
until_arg=""

shell_escape() {
  printf '%q' "$1"
}

join_args() {
  local out=""
  for arg in "$@"; do
    if [[ -n "${out}" ]]; then
      out+=" "
    fi
    out+="$(shell_escape "$arg")"
  done
  printf '%s' "$out"
}

contains_arg() {
  local needle="$1"
  shift
  for arg in "$@"; do
    if [[ "$arg" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

trim_cell() {
  local value="${1-}"
  value="${value%$'\r'}"
  value="${value#\"}"
  value="${value%\"}"
  printf '%s' "${value}"
}

while getopts ":c:d:r:C:s:u:h" opt; do
  case "${opt}" in
    c) csv_path="${OPTARG}" ;;
    d) clone_base="${OPTARG}" ;;
    r) runner="${OPTARG}" ;;
    C) command_template="${OPTARG}" ;;
    s) since_arg="${OPTARG}" ;;
    u) until_arg="${OPTARG}" ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "ERROR: Invalid option -${OPTARG}" >&2
      usage >&2
      exit 1
      ;;
    :)
      echo "ERROR: Option -${OPTARG} requires an argument." >&2
      usage >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))
runner_args=("$@")

if [[ -z "${runner}" && -z "${command_template}" ]]; then
  echo "ERROR: specify at least a runner (-r) or a command template (-C)." >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${csv_path}" ]]; then
  echo "ERROR: CSV not found: ${csv_path}" >&2
  exit 1
fi

mkdir -p "${clone_base}"

if [[ -n "${runner}" ]]; then
  if [[ -n "${since_arg}" ]] && ! contains_arg "--since" "${runner_args[@]}"; then
    runner_args=(--since "${since_arg}" "${runner_args[@]}")
  fi
  if [[ -n "${until_arg}" ]] && ! contains_arg "--until" "${runner_args[@]}"; then
    runner_args=(--until "${until_arg}" "${runner_args[@]}")
  fi
fi

while IFS=, read -r project language main_repo homepage primary_contact vulnerability_count extra_columns; do
  project="$(trim_cell "${project}")"
  main_repo="$(trim_cell "${main_repo}")"

  if [[ -z "${project}" || "${project}" == "project" ]]; then
    continue
  fi
  if [[ -z "${main_repo}" || "${main_repo}" == "main_repo" ]]; then
    continue
  fi

  repo_dir="${clone_base}/${project}"
  if [[ -d "${repo_dir}/.git" ]]; then
    echo "Updating ${project}..."
    git -C "${repo_dir}" fetch --all --prune >/dev/null
  else
    echo "Cloning ${project} from ${main_repo}..."
    git clone "${main_repo}" "${repo_dir}"
  fi

  if [[ -n "${command_template}" ]]; then
    escaped_project="$(shell_escape "${project}")"
    escaped_repo="$(shell_escape "${repo_dir}")"
    escaped_runner="$(shell_escape "${runner}")"
    escaped_runner_args="$(join_args "${runner_args[@]}")"

    cmd="${command_template}"
    cmd="${cmd//\{project\}/${escaped_project}}"
    cmd="${cmd//\{repo\}/${escaped_repo}}"
    cmd="${cmd//\{runner\}/${escaped_runner}}"
    cmd="${cmd//\{runner_args\}/${escaped_runner_args}}"

    echo "Executing command for ${project}: ${cmd}"
    bash -c "${cmd}"
  elif [[ -n "${runner}" ]]; then
    echo "Running ${runner} for ${project}..."
    "${runner}" "${project}" "${repo_dir}" "${runner_args[@]}" || {
      echo "WARNING: runner failed for ${project}" >&2
    }
  fi
done < <(tail -n +2 "${csv_path}")
