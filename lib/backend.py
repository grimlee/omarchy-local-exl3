"""Local EXL3 lifecycle. No downloads, model edits, or runtime patches."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import tomllib
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes/qwen38-27b-exl3.json"
MODEL_ID = "exllama-v3"  # Matches the existing Pi provider on the tested host.
REQUIRED_PROFILES = {
    "balanced": (4, 4, 117760),
    "max-context": (3, 3, 150016),
}


class LauncherError(Exception):
    pass


def recipe():
    data = json.loads(RECIPE.read_text())
    model = data["model"]
    assert data["schema_version"] == 1
    assert data["default_profile"] == "balanced"
    assert set(data["profiles"]) == set(REQUIRED_PROFILES)
    assert model["runtime"] == "exllamav3" and model["api_model_id"] == MODEL_ID
    assert model["chunk_size"] == 4096 and model["page_size"] == 256
    assert model["recurrent_checkpoint"] == 32768
    assert model["vision"] == "auto" and model["draft_model"] == "mtp"
    for key, (k, v, context) in REQUIRED_PROFILES.items():
        profile = data["profiles"][key]
        assert (profile["kv_bits_k"], profile["kv_bits_v"], profile["context"]) == (k, v, context)
        assert context % 256 == 0
        assert profile["stable_mm"] is True
        assert profile["mm_cache"] is False and profile["prefix_diag"] is False
    return data


def paths():
    home = Path.home()
    config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / "omarchy-local-exl3/config.toml"
    state = Path(os.environ.get("XDG_STATE_HOME", home / ".local/state")) / "omarchy-local-exl3"
    return config, state


def config():
    file, _ = paths()
    data = tomllib.loads(file.read_text()) if file.exists() else {}
    allowed = {"model_path", "runtime_path", "port", "default_profile"}
    unknown = set(data) - allowed
    if unknown:
        raise LauncherError("Unknown config keys: " + ", ".join(sorted(unknown)))
    if not isinstance(data.get("port", 8881), int) or not 1024 <= data.get("port", 8881) <= 65535:
        raise LauncherError("port must be an integer from 1024 to 65535")
    if data.get("default_profile", "balanced") not in REQUIRED_PROFILES:
        raise LauncherError("Invalid default_profile")
    for name in ("model_path", "runtime_path"):
        if name in data and (not isinstance(data[name], str) or not Path(data[name]).is_absolute()):
            raise LauncherError(f"{name} must be an absolute path")
    return data


def candidates(kind):
    home = Path.home()
    if kind == "runtime":
        # Bounded search for this host's tested checkout; never scan the filesystem.
        yield from sorted((home / "Documents/Codex").glob("*/*/qwen3.8-27b-eval"), reverse=True)
        yield home / "src/qwen3.8-27b-eval"
    else:
        name = recipe()["model"]["directory_name"]
        yield home / "models" / name
        yield home / ".local/share/models" / name
        for runtime_dir in candidates("runtime"):
            yield runtime_dir / "models" / name


def resolve_path(kind, cfg):
    env_name = "LOCAL_EXL3_RUNTIME_DIR" if kind == "runtime" else "LOCAL_EXL3_MODEL_PATH"
    key = "runtime_path" if kind == "runtime" else "model_path"
    override = os.environ.get(env_name) or cfg.get(key)
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise LauncherError(f"{env_name}/{key} must be absolute")
        return path.resolve()
    for path in candidates(kind):
        if path.is_dir():
            return path.resolve()
    return None


def runtime_python(runtime):
    for name in (".venv-exl3-1.5.0", ".venv-exl3-1.5.0-sm89-attn"):
        path = runtime / name / "bin/python"
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise LauncherError("Validated ExLlamaV3 1.5.0 venv not found in runtime directory")


def verify_runtime(runtime):
    if not runtime or not runtime.is_dir():
        raise LauncherError("Runtime not found. Set LOCAL_EXL3_RUNTIME_DIR or runtime_path")
    server = runtime / "tools/serve_openai.py"
    registry = runtime / "experiments/mm_stable_id/registry.py"
    package = runtime / "experiments/mm_stable_id/__init__.py"
    if not server.is_file() or not registry.is_file() or not package.is_file():
        raise LauncherError("Stable MM runtime files missing; use the validated branch")
    source = server.read_text()
    stable = registry.read_text()
    if "EXL3_MM_STABLE_ID" not in source or "StableMMIdentityRegistry" not in source or "_stable_retag_embeddings" not in source or "class StableMMIdentityRegistry" not in stable or "from .registry import StableMMIdentityRegistry" not in package.read_text():
        raise LauncherError("Stable MM implementation not present in runtime")
    return runtime_python(runtime), server


def verify_model(model):
    if not model or not model.is_dir():
        raise LauncherError("Model not found. Set LOCAL_EXL3_MODEL_PATH or model_path")
    index = model / "model.safetensors.index.json"
    if not index.is_file() or not (model / "config.json").is_file():
        raise LauncherError("Model directory is incomplete")
    shards = json.loads(index.read_text()).get("weight_map", {}).values()
    if not shards or not all((model / file).is_file() for file in set(shards)):
        raise LauncherError("Model shard missing; no download will be started")


def gpu():
    if not shutil.which("nvidia-smi"):
        return {"name": None, "vram_used_mib": None, "vram_total_mib": None}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4, check=True)
        name, used, total = [v.strip() for v in result.stdout.splitlines()[0].split(",")]
        return {"name": name, "vram_used_mib": int(used), "vram_total_mib": int(total)}
    except (subprocess.SubprocessError, ValueError, IndexError, OSError):
        return {"name": None, "vram_used_mib": None, "vram_total_mib": None}


def command(runtime, python, server, model, profile, port):
    spec = recipe()
    p = spec["profiles"][profile]
    m = spec["model"]
    return [str(python), "-u", str(server), "--model", str(model), "--model_id", MODEL_ID,
            "--host", "127.0.0.1", "--port", str(port), "--cache_size", str(p["context"]),
            "--grid_size", str(m["grid_size_gb"]), "--vision", "auto", "--draft_model", "mtp",
            "--cache_quant", f"{p['kv_bits_k']},{p['kv_bits_v']}", "--chunk_size", "4096", "--ui", "off"]


def launch_env(runtime):
    env = os.environ.copy()
    # Override inherited process values. A flag alone cannot add missing source code.
    env.update(EXL3_MM_STABLE_ID="1", EXL3_MM_CACHE="0", EXL3_PREFIX_DIAG="0",
               EXL3_PREFIX_TRACE="0", EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP="32768",
               EXL3_CHUNK_SIZE="4096")
    env["PYTHONPATH"] = str(runtime) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _proc(pid):
    try:
        proc = Path(f"/proc/{int(pid)}")
        uid = proc.stat().st_uid
        fields = proc.joinpath("stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        start = fields[19]
        cmdline = proc.joinpath("cmdline").read_bytes().split(b"\0")
        return uid, start, [x.decode(errors="replace") for x in cmdline if x]
    except (OSError, ValueError):
        return None


def owned(state):
    pid = state.get("pid")
    proc = _proc(pid) if pid else None
    if not proc or proc[0] != os.getuid() or str(proc[1]) != str(state.get("start_ticks")):
        return False
    args = proc[2]
    return len(args) >= 4 and args[1:4] == ["-u", state.get("server"), "--model"] and state.get("model_path") in args and "--host" in args and args[args.index("--host") + 1] == "127.0.0.1" and "--port" in args and args[args.index("--port") + 1] == str(state.get("port"))


def read_state():
    _, directory = paths()
    file = directory / "state.json"
    try:
        return json.loads(file.read_text())
    except (OSError, ValueError):
        return {}


def write_state(state):
    _, directory = paths()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    file = directory / "state.json"
    temp = directory / "state.json.tmp"
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.chmod(0o600)
    temp.replace(file)


def clean_stale():
    state = read_state()
    if state and not owned(state) and not _proc(state.get("pid")):
        write_state({})
        return {}
    return state


def lock():
    _, directory = paths()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle = (directory / "lock").open("w")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def health(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as response:
            data = json.load(response)
        return any(m.get("id") == MODEL_ID for m in data.get("data", []))
    except (URLError, OSError, ValueError):
        return False


def port_free(port):
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def start(profile=None, timeout=300):
    cfg = config()
    profile = profile or cfg.get("default_profile", "balanced")
    if profile not in REQUIRED_PROFILES:
        raise LauncherError("Invalid profile: " + profile)
    with lock():
        state = clean_stale()
        if state:
            if owned(state):
                raise LauncherError("Already running with profile " + state["profile"])
            raise LauncherError("PID state belongs to another process; refusing to replace it")
        if not gpu()["name"]:
            raise LauncherError("NVIDIA GPU unavailable")
        runtime = resolve_path("runtime", cfg)
        python, server = verify_runtime(runtime)
        model = resolve_path("model", cfg)
        verify_model(model)
        port = cfg.get("port", 8881)
        if not port_free(port):
            raise LauncherError(f"Port {port} is occupied by another process; refusing to start")
        _, directory = paths()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_path = directory / "server.log"
        cmd = command(runtime, python, server, model, profile, port)
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(cmd, cwd=runtime, env=launch_env(runtime), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        proc = None
        for _ in range(20):
            proc = _proc(process.pid)
            if proc:
                break
            time.sleep(0.01)
        if not proc:
            raise LauncherError("Server exited before PID capture")
        state = {"pid": process.pid, "start_ticks": proc[1], "server": str(server), "model_path": str(model),
                 "runtime_path": str(runtime), "profile": profile, "port": port, "started_at": time.time()}
        write_state(state)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if health(port) and owned(state):
            print(f"Ready: {profile} at http://127.0.0.1:{port}/v1")
            return
        if process.poll() is not None:
            break
        time.sleep(2)
    if owned(state):
        os.kill(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            if owned(state):
                os.kill(process.pid, signal.SIGKILL)
    with lock():
        if read_state().get("pid") == process.pid:
            write_state({})
    tail = log_path.read_text(errors="replace").splitlines()[-50:]
    raise LauncherError("Startup failed or timed out. Last log lines:\n" + "\n".join(tail))


def stop():
    with lock():
        state = clean_stale()
        if not state:
            print("Stopped")
            return
        if not owned(state):
            raise LauncherError("PID state does not match this plugin server; refusing to stop")
        pid = state["pid"]
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 12
        while _proc(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if _proc(pid) and owned(state):
            os.kill(pid, signal.SIGKILL)
        write_state({})
    print("Stopped")


def status():
    spec, cfg, state = recipe(), config(), read_state()
    active = bool(state and owned(state))
    profile_name = state.get("profile") if active else cfg.get("default_profile", "balanced")
    p = spec["profiles"][profile_name]
    card = gpu()
    runtime = resolve_path("runtime", cfg)
    model = resolve_path("model", cfg)
    try:
        verify_runtime(runtime)
        stable_source = True
    except LauncherError:
        stable_source = False
    result = {"running": active, "pid": state.get("pid") if active else None,
              "model": spec["model"]["name"], "model_found": bool(model and model.is_dir()),
              "runtime_found": bool(runtime and runtime.is_dir()), "profile": profile_name,
              "endpoint": f"http://127.0.0.1:{state.get('port', cfg.get('port', 8881))}/v1",
              "context": p["context"], "kv_k_bits": p["kv_bits_k"], "kv_v_bits": p["kv_bits_v"],
              "stable_mm": bool(active and stable_source and p["stable_mm"]),
              "stable_mm_implementation": stable_source, "vision_cache": p["mm_cache"],
              "recurrent_checkpoint": spec["model"]["recurrent_checkpoint"],
              "gpu_name": card["name"], "vram_used_mib": card["vram_used_mib"],
              "vram_total_mib": card["vram_total_mib"],
              "health": health(state["port"]) if active else False}
    return result


def doctor():
    cfg = config()
    runtime = resolve_path("runtime", cfg)
    model = resolve_path("model", cfg)
    findings = {"runtime": str(runtime) if runtime else None, "model": str(model) if model else None,
                "gpu": gpu(), "port_free": port_free(cfg.get("port", 8881))}
    try:
        verify_runtime(runtime)
        findings["stable_mm_implementation"] = True
    except LauncherError as exc:
        findings["stable_mm_implementation"] = False
        findings["runtime_error"] = str(exc)
    try:
        verify_model(model)
        findings["model_complete"] = True
    except LauncherError as exc:
        findings["model_complete"] = False
        findings["model_error"] = str(exc)
    print(json.dumps(findings, indent=2))
    return 0 if findings["stable_mm_implementation"] and findings["model_complete"] and findings["gpu"]["name"] else 1


def configure_model(path):
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise LauncherError("Model path must be absolute")
    file, _ = paths()
    existing = config()
    existing["model_path"] = str(target)
    file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # This small schema has only scalar keys. JSON strings are valid TOML strings.
    lines = [f'{key} = {json.dumps(value)}' if isinstance(value, str) else f"{key} = {value}"
             for key, value in sorted(existing.items())]
    temp = file.with_suffix(".tmp")
    temp.write_text("\n".join(lines) + "\n")
    temp.chmod(0o600)
    temp.replace(file)
    print(f"Configured model path: {target}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="local-exl3")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("start", "restart"):
        command_parser = sub.add_parser(action)
        command_parser.add_argument("--profile", choices=REQUIRED_PROFILES)
    sub.add_parser("stop")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--json", action="store_true")
    sub.add_parser("profiles")
    sub.add_parser("logs").add_argument("--follow", action="store_true")
    sub.add_parser("doctor")
    sub.add_parser("open-pi")
    sub.add_parser("configure-model").add_argument("path")
    args = parser.parse_args(argv)
    try:
        if args.action == "start":
            start(args.profile)
        elif args.action == "restart":
            stop()
            start(args.profile)
        elif args.action == "stop":
            stop()
        elif args.action == "status":
            result = status()
            print(json.dumps(result) if args.json else f"{result['profile']}: {'ready' if result['health'] else 'stopped'} — {result['endpoint']}")
        elif args.action == "profiles":
            print(json.dumps(recipe()["profiles"], indent=2))
        elif args.action == "logs":
            log = paths()[1] / "server.log"
            if args.follow:
                os.execvp("tail", ["tail", "-n", "50", "-f", str(log)])
            print("\n".join(log.read_text(errors="replace").splitlines()[-50:]) if log.exists() else "No log yet")
        elif args.action == "doctor":
            return doctor()
        elif args.action == "open-pi":
            if not status()["health"]:
                raise LauncherError("Start the model before opening Pi")
            if not shutil.which("pi"):
                raise LauncherError("Pi is not installed")
            os.execvp("pi", ["pi", "--provider", "exllama-v3", "--model", MODEL_ID])
        elif args.action == "configure-model":
            configure_model(args.path)
        return 0
    except (LauncherError, AssertionError, OSError, ValueError) as exc:
        print(f"local-exl3: {exc}", file=sys.stderr)
        return 1
