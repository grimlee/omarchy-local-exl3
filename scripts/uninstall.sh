#!/usr/bin/env bash
set -euo pipefail

plugin_dir="${XDG_CONFIG_HOME:-$HOME/.config}/omarchy/plugins/grimlee.local-exl3"
link="$HOME/.local/bin/local-exl3"
if [[ -f "$plugin_dir/manifest.json" ]] && [[ "$(jq -r .id "$plugin_dir/manifest.json")" != grimlee.local-exl3 ]]; then
  echo "Refusing to remove unrelated plugin: $plugin_dir" >&2
  exit 1
fi
if [[ -x "$plugin_dir/bin/local-exl3" ]]; then
  "$plugin_dir/bin/local-exl3" stop
fi
if command -v omarchy >/dev/null && omarchy plugin list --json | jq -e 'any(.[]; .id == "grimlee.local-exl3" and .enabled == true)' >/dev/null; then
  omarchy plugin disable grimlee.local-exl3
fi
if [[ -d "$plugin_dir" ]]; then rm -rf "$plugin_dir"; fi
if [[ -L "$link" && "$(readlink "$link")" == "$plugin_dir/bin/local-exl3" ]]; then rm "$link"; fi
if command -v omarchy-shell >/dev/null; then omarchy-shell shell rescanPlugins >/dev/null; fi
echo "Removed Local EXL3 plugin. Model, runtime, Pi and logs were preserved."
