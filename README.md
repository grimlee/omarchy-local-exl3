# Omarchy Local EXL3

A small Omarchy bar launcher for the existing local ExLlamaV3 server. V1 supports the already downloaded Huihui Qwen3.8-27B EXL3 3.0bpw model on the tested NVIDIA RTX 4060 Ti 16GB. It does not download weights or build a runtime.

| Profile | KV | Context | Chunk | Stable MM | Vision cache | Recurrent PP checkpoint |
|---|---|---:|---:|---|---|---:|
| Balanced (default) | Q4/Q4 | 117,760 | 4,096 | On | Off | 32,768 |
| Max Context | Q3/Q3 | 150,016 | 4,096 | On | Off | 32,768 |

Stable MM Identity keeps repeated multimodal prefixes stable in long running Pi sessions. The launcher checks that the implementation exists before starting; merely setting the environment variable is insufficient. These presets are validated for the listed hardware and model, not universal GPU settings. The 180K allocation probe and Q4 at 150K are not supported presets.

## Install and use

Requires Omarchy Quattro, Python 3.11+, a working NVIDIA driver, the validated ExLlamaV3 1.5.0 checkout/venv with the Stable MM implementation, and the model files. No Docker or root is required.

```bash
git clone https://github.com/grimlee/omarchy-local-exl3.git
cd omarchy-local-exl3
./scripts/install.sh
local-exl3 doctor
local-exl3 start --profile balanced
local-exl3 status --json
local-exl3 stop
```

Click the **◆ Local EXL3** icon in the Omarchy top bar. Choose Balanced or Max Context while stopped, then press Start. Stop before changing profiles. Logs and Open Pi are available from the same panel. `local-exl3 restart --profile max-context` restarts with that profile from the CLI.

The API binds to `http://127.0.0.1:8881/v1` by default and reports model ID `exllama-v3`, matching the tested Pi provider. `local-exl3 open-pi` uses the existing Pi configuration without editing it. If Pi has no matching provider on a different machine, configure it separately.

Set `LOCAL_EXL3_MODEL_PATH` and `LOCAL_EXL3_RUNTIME_DIR` or place absolute paths in `~/.config/omarchy-local-exl3/config.toml` (`model_path`, `runtime_path`). Optional keys are `port` and `default_profile`. The runtime detector checks a bounded set of common directories, including the current Qwen3.8 eval checkout under `~/Documents/Codex`. Model detection checks `~/models`, `~/.local/share/models`, and the detected checkout's `models` directory. The UI can save a model path. Config files are not rewritten by install or uninstall.

State and log: `~/.local/state/omarchy-local-exl3/`. The CLI owns only the process it launched and validates its PID, process start time, command, model and port before stopping. An occupied port or foreign PID causes a refusal. On startup, the CLI waits for `/v1/models`; it returns only after that endpoint reports `exllama-v3`.

```bash
local-exl3 profiles
local-exl3 logs
local-exl3 doctor
./scripts/uninstall.sh
```

Uninstall leaves model weights, the ExLlamaV3 checkout, Pi settings and logs in place.

## Verification

CI checks recipe values, Python unit tests, shell syntax and ShellCheck, the Omarchy manifest, and an Arch Linux container. CI cannot exercise the RTX 4060 Ti, CUDA inference or Hyprland rendering. The local smoke procedure is in `docs/smoke.md`; it uses short text and image requests only and does not rerun long context benchmarks.
