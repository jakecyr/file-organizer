#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="${UV:-uv}"
COMPLETION_DIR="${XDG_DATA_HOME:-"$HOME/.local/share"}/file-organizer/completions"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  cat >&2 <<'EOF'
uv is required to install file-organizer.

Install uv first:
  curl -LsSf https://astral.sh/uv/install.sh | sh
EOF
  exit 1
fi

"$UV_BIN" tool install --force --editable "$ROOT_DIR"

BIN_DIR="$("$UV_BIN" tool dir --bin)"
mkdir -p "$COMPLETION_DIR"

write_bash_completion() {
  local path="$COMPLETION_DIR/file-organizer.bash"
  cat >"$path" <<'EOF'
_file_organizer_complete() {
  local cur prev opts
  COMPREPLY=()
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"
  opts="--dry-run --yes --env-file --include-nested --profile-subfolders --profile-sample-size --include-hidden --allow-singleton-clusters --workers --unmatched-policy --help"

  case "$prev" in
    --unmatched-policy)
      COMPREPLY=($(compgen -W "other error leave" -- "$cur"))
      return 0
      ;;
    --env-file)
      COMPREPLY=($(compgen -f -- "$cur"))
      return 0
      ;;
    --workers)
      COMPREPLY=($(compgen -W "2 1 3 4 6 8" -- "$cur"))
      return 0
      ;;
    --profile-sample-size)
      COMPREPLY=($(compgen -W "5 0 10 25 50 100" -- "$cur"))
      return 0
      ;;
  esac

  if [[ "$cur" == -* ]]; then
    COMPREPLY=($(compgen -W "$opts" -- "$cur"))
  else
    COMPREPLY=($(compgen -d -- "$cur"))
  fi
}

complete -o default -o bashdefault -F _file_organizer_complete file-organizer
complete -o default -o bashdefault -F _file_organizer_complete fo
EOF

  local rc="$HOME/.bashrc"
  touch "$rc"
  if ! grep -q "file-organizer completion" "$rc"; then
    cat >>"$rc" <<EOF

# file-organizer completion
source "$path"
EOF
  fi
}

write_zsh_completion() {
  local zfunc_dir="$HOME/.zfunc"
  mkdir -p "$zfunc_dir"
  cat >"$zfunc_dir/_file-organizer" <<'EOF'
#compdef file-organizer

_arguments \
  '--dry-run[only print the organization plan]' \
  '--yes[move files without prompting for confirmation]' \
  '--env-file[load configuration from a .env file]:env file:_files' \
  '--include-nested[also consider files already inside subfolders]' \
  '--profile-subfolders[analyze existing subfolder contents as folder examples]' \
  '--profile-sample-size[files sampled per top-level folder; 0 means all]:sample size:(5 0 10 25 50 100)' \
  '--include-hidden[include hidden files and folders]' \
  '--allow-singleton-clusters[create folders for single unmatched files]' \
  '--workers[number of files to analyze in parallel]:workers:(2 1 3 4 6 8)' \
  '--unmatched-policy[unmatched file behavior]:policy:(other error leave)' \
  '--help[show help]' \
  '*:folder:_files -/'
EOF
  cat >"$zfunc_dir/_fo" <<'EOF'
#compdef fo

_arguments \
  '--dry-run[only print the organization plan]' \
  '--yes[move files without prompting for confirmation]' \
  '--env-file[load configuration from a .env file]:env file:_files' \
  '--include-nested[also consider files already inside subfolders]' \
  '--profile-subfolders[analyze existing subfolder contents as folder examples]' \
  '--profile-sample-size[files sampled per top-level folder; 0 means all]:sample size:(5 0 10 25 50 100)' \
  '--include-hidden[include hidden files and folders]' \
  '--allow-singleton-clusters[create folders for single unmatched files]' \
  '--workers[number of files to analyze in parallel]:workers:(2 1 3 4 6 8)' \
  '--unmatched-policy[unmatched file behavior]:policy:(other error leave)' \
  '--help[show help]' \
  '*:folder:_files -/'
EOF

  local rc="$HOME/.zshrc"
  touch "$rc"
  if ! grep -q "file-organizer completion" "$rc"; then
    cat >>"$rc" <<'EOF'

# file-organizer completion
fpath=("$HOME/.zfunc" $fpath)
autoload -Uz compinit
compinit
EOF
  fi
}

write_fish_completion() {
  local fish_dir="$HOME/.config/fish/completions"
  mkdir -p "$fish_dir"
  for command_name in file-organizer fo; do
    cat >"$fish_dir/$command_name.fish" <<EOF
complete -c $command_name -l dry-run -d 'Only print the organization plan'
complete -c $command_name -l yes -s y -d 'Move files without prompting for confirmation'
complete -c $command_name -l env-file -r -F -d 'Load configuration from a .env file'
complete -c $command_name -l include-nested -d 'Also consider files already inside subfolders'
complete -c $command_name -l profile-subfolders -d 'Analyze existing subfolder contents as folder examples'
complete -c $command_name -l profile-sample-size -xa '5 0 10 25 50 100' -d 'Files sampled per top-level folder; 0 means all'
complete -c $command_name -l include-hidden -d 'Include hidden files and folders'
complete -c $command_name -l allow-singleton-clusters -d 'Create folders for single unmatched files'
complete -c $command_name -l workers -xa '2 1 3 4 6 8' -d 'Number of files to analyze in parallel'
complete -c $command_name -l unmatched-policy -xa 'other error leave' -d 'Unmatched file behavior'
complete -c $command_name -l help -d 'Show help'
complete -c $command_name -a '(__fish_complete_directories)'
EOF
  done
}

install_completions() {
  case "$(basename "${SHELL:-}")" in
    bash)
      write_bash_completion
      echo "Installed bash completions for file-organizer and fo."
      ;;
    zsh)
      write_zsh_completion
      echo "Installed zsh completions for file-organizer and fo."
      ;;
    fish)
      write_fish_completion
      echo "Installed fish completions for file-organizer and fo."
      ;;
    *)
      write_bash_completion || true
      write_zsh_completion || true
      write_fish_completion || true
      echo "Wrote completion files for bash, zsh, and fish."
      ;;
  esac
}

install_completions || echo "Completion install skipped; run file-organizer --help for usage."

cat <<EOF
Installed file-organizer.

Commands:
  file-organizer
  fo

If your shell cannot find it, add this to your shell profile:
  export PATH="$BIN_DIR:\$PATH"

Restart your shell to load tab completion.
EOF
