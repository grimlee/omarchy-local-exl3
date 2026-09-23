#!/usr/bin/env bash
set -euo pipefail

src=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
plugin_dir="${XDG_CONFIG_HOME:-$HOME/.config}/omarchy/plugins/grimlee.local-exl3"
bin_dir="$HOME/.local/bin"
state_dir="${XDG_STATE_HOME:-$HOME/.local/state}/omarchy-local-exl3"

command -v omarchy >/dev/null || { echo "Omarchy CLI is required" >&2; exit 1; }
omarchy plugin validate "$src"
if [[ -e "$plugin_dir" && ! -f "$plugin_dir/manifest.json" ]]; then
  echo "Refusing to replace an unrelated directory: $plugin_dir" >&2
  exit 1
fi
if [[ -e "$plugin_dir/manifest.json" ]]; then
  current_id=$(jq -r .id "$plugin_dir/manifest.json")
  [[ "$current_id" == grimlee.local-exl3 ]] || { echo "Plugin directory belongs to $current_id" >&2; exit 1; }
fi
mkdir -p "$(dirname "$plugin_dir")" "$bin_dir" "$state_dir" "${XDG_CONFIG_HOME:-$HOME/.config}/omarchy-local-exl3"
chmod 700 "$state_dir"
stage=$(mktemp -d "$(dirname "$plugin_dir")/.local-exl3.XXXXXXXX")
trap 'rm -rf "$stage"' EXIT
cp -a "$src/manifest.json" "$src/bin" "$src/lib" "$src/recipes" "$src/ui" "$stage/"
chmod +x "$stage/bin/local-exl3"
omarchy plugin validate "$stage"
if [[ -e "$plugin_dir" ]]; then
  # Replace only files owned by this plugin; config, state, runtime and weights live elsewhere.
  rm -rf "$plugin_dir"
fi
mv "$stage" "$plugin_dir"
trap - EXIT
link="$bin_dir/local-exl3"
if [[ -e "$link" && ! -L "$link" ]]; then
  echo "Plugin installed, but $link exists and is not a symlink" >&2
  exit 1
fi
if [[ -L "$link" && "$(readlink "$link")" != "$plugin_dir/bin/local-exl3" ]]; then
  echo "Plugin installed, but $link belongs to another installation" >&2
  exit 1
fi
ln -sfn "$plugin_dir/bin/local-exl3" "$link"
omarchy-shell shell rescanPlugins >/dev/null
if ! omarchy plugin list --json | jq -e 'any(.[]; .id == "grimlee.local-exl3" and .enabled == true)' >/dev/null; then
  omarchy plugin enable grimlee.local-exl3 --section right
fi
echo "Installed Local EXL3: $plugin_dir"
echo "CLI: $link"
