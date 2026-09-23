import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("backend", ROOT / "lib/backend.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class RecipeTests(unittest.TestCase):
    def test_schema(self):
        self.assertEqual(backend.recipe()["schema_version"], 1)

    def test_exact_profiles(self):
        data = backend.recipe()
        self.assertEqual(data["default_profile"], "balanced")
        self.assertEqual(set(data["profiles"]), {"balanced", "max-context"})
        self.assertEqual([(p["kv_bits_k"], p["kv_bits_v"], p["context"])
                          for p in data["profiles"].values()], [(4, 4, 117760), (3, 3, 150016)])

    def test_safety_values(self):
        data = backend.recipe()
        self.assertEqual(data["model"]["chunk_size"], 4096)
        self.assertEqual(data["model"]["recurrent_checkpoint"], 32768)
        for profile in data["profiles"].values():
            self.assertTrue(profile["stable_mm"])
            self.assertFalse(profile["mm_cache"])
            self.assertFalse(profile["prefix_diag"])

    def test_modified_recipe_rejected(self):
        data = backend.recipe()
        data["profiles"]["balanced"]["context"] = 150016
        with mock.patch.object(backend, "RECIPE") as file:
            file.read_text.return_value = json.dumps(data)
            with self.assertRaises(AssertionError):
                backend.recipe()

    def test_command(self):
        cmd = backend.command(Path("/runtime"), Path("/runtime/python"), Path("/runtime/tools/serve_openai.py"),
                              Path("/model"), "max-context", 8881)
        self.assertEqual(cmd[cmd.index("--cache_size") + 1], "150016")
        self.assertEqual(cmd[cmd.index("--cache_quant") + 1], "3,3")
        self.assertEqual(cmd[cmd.index("--chunk_size") + 1], "4096")
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        self.assertEqual(cmd[cmd.index("--draft_model") + 1], "mtp")
        self.assertEqual(cmd[cmd.index("--model_id") + 1], "exllama-v3")

    def test_balanced_command(self):
        cmd = backend.command(Path("/r"), Path("/p"), Path("/s"), Path("/m"), "balanced", 8881)
        self.assertEqual(cmd[cmd.index("--cache_quant") + 1], "4,4")
        self.assertEqual(cmd[cmd.index("--cache_size") + 1], "117760")

    def test_environment_isolation(self):
        with mock.patch.dict(os.environ, {"EXL3_MM_STABLE_ID": "0", "EXL3_MM_CACHE": "1",
                                          "EXL3_PREFIX_DIAG": "1", "EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP": "4096"}):
            env = backend.launch_env(Path("/r"))
        self.assertEqual(env["EXL3_MM_STABLE_ID"], "1")
        self.assertEqual(env["EXL3_MM_CACHE"], "0")
        self.assertEqual(env["EXL3_PREFIX_DIAG"], "0")
        self.assertEqual(env["EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP"], "32768")
        self.assertEqual(env["EXL3_QC_STAGING"], "1")
        self.assertEqual(env["EXL3_SM89_PAGED_ATTN"], "1")
        self.assertEqual(env["TORCH_CUDA_ARCH_LIST"], "8.9")


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(os.environ, {"HOME": self.tmp.name, "XDG_CONFIG_HOME": self.tmp.name + "/config",
                                             "XDG_STATE_HOME": self.tmp.name + "/state"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_invalid_profile(self):
        with self.assertRaisesRegex(backend.LauncherError, "Invalid profile"):
            backend.start("180k")

    def test_missing_runtime(self):
        with mock.patch.object(backend, "gpu", return_value={"name": "GPU"}), \
             mock.patch.object(backend, "resolve_path", return_value=None):
            with self.assertRaisesRegex(backend.LauncherError, "Runtime not found"):
                backend.start()

    def test_missing_model(self):
        with mock.patch.object(backend, "gpu", return_value={"name": "GPU"}), \
             mock.patch.object(backend, "resolve_path", side_effect=[Path("/r"), None]), \
             mock.patch.object(backend, "verify_runtime", return_value=(Path("/p"), Path("/s"))):
            with self.assertRaisesRegex(backend.LauncherError, "Model not found"):
                backend.start()

    def test_occupied_port_never_launches(self):
        with mock.patch.object(backend, "gpu", return_value={"name": "GPU"}), \
             mock.patch.object(backend, "resolve_path", side_effect=[Path("/r"), Path("/m")]), \
             mock.patch.object(backend, "verify_runtime", return_value=(Path("/p"), Path("/s"))), \
             mock.patch.object(backend, "verify_model"), \
             mock.patch.object(backend, "port_free", return_value=False), \
             mock.patch.object(backend.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(backend.LauncherError, "occupied"):
                backend.start()
            popen.assert_not_called()

    def test_start_waits_for_health_and_records_own_pid(self):
        process = mock.Mock(pid=123)
        with mock.patch.object(backend, "gpu", return_value={"name": "GPU"}), \
             mock.patch.object(backend, "resolve_path", side_effect=[Path("/r"), Path("/m")]), \
             mock.patch.object(backend, "verify_runtime", return_value=(Path("/p"), Path("/s"))), \
             mock.patch.object(backend, "verify_model"), \
             mock.patch.object(backend, "port_free", return_value=True), \
             mock.patch.object(backend, "_proc", return_value=(os.getuid(), "7", [])), \
             mock.patch.object(backend, "owned", return_value=True), \
             mock.patch.object(backend, "health", return_value=True), \
             mock.patch.object(backend.subprocess, "Popen", return_value=process) as popen:
            backend.start("max-context")
        self.assertEqual(backend.read_state()["pid"], 123)
        self.assertEqual(backend.read_state()["profile"], "max-context")
        args, kwargs = popen.call_args
        self.assertEqual(args[0][args[0].index("--cache_quant") + 1], "3,3")
        self.assertEqual(kwargs["env"]["EXL3_MM_STABLE_ID"], "1")
        self.assertTrue(kwargs["start_new_session"])

    def test_foreign_pid_never_killed(self):
        with mock.patch.object(backend, "read_state", return_value={"pid": 123}), \
             mock.patch.object(backend, "_proc", return_value=(os.getuid(), "10", ["python", "foreign"])), \
             mock.patch.object(backend.os, "kill") as kill:
            with self.assertRaisesRegex(backend.LauncherError, "refusing to stop"):
                backend.stop()
            kill.assert_not_called()

    def test_stop_owned_pid(self):
        backend.write_state({"pid": 123, "profile": "balanced"})
        with mock.patch.object(backend, "owned", return_value=True), \
             mock.patch.object(backend, "_proc", return_value=None), \
             mock.patch.object(backend.os, "kill") as kill:
            backend.stop()
        kill.assert_called_once_with(123, backend.signal.SIGTERM)
        self.assertEqual(backend.read_state(), {})

    def test_stale_pid_cleaned(self):
        backend.write_state({"pid": 987654321})
        self.assertEqual(backend.clean_stale(), {})
        self.assertEqual(backend.read_state(), {})

    def test_owned_requires_exact_identity(self):
        state = {"pid": 123, "start_ticks": "7", "server": "/r/tools/serve_openai.py",
                 "model_path": "/m", "port": 8881}
        args = ["/p", "-u", "/r/tools/serve_openai.py", "--model", "/m", "--host", "127.0.0.1", "--port", "8881"]
        with mock.patch.object(backend, "_proc", return_value=(os.getuid(), "7", args)):
            self.assertTrue(backend.owned(state))
        with mock.patch.object(backend, "_proc", return_value=(os.getuid(), "8", args)):
            self.assertFalse(backend.owned(state))

    def test_status_json_shape(self):
        result = backend.status()
        self.assertEqual(result["profile"], "balanced")
        self.assertFalse(result["running"])
        self.assertFalse(result["stable_mm"])
        self.assertIn("gpu_name", result)
        self.assertIn("health", result)
        json.dumps(result)

    def test_config_override(self):
        file, _ = backend.paths()
        file.parent.mkdir(parents=True)
        file.write_text('port = 9999\ndefault_profile = "max-context"\n')
        self.assertEqual(backend.config()["port"], 9999)
        self.assertEqual(backend.status()["profile"], "max-context")

    def test_env_model_override(self):
        with mock.patch.dict(os.environ, {"LOCAL_EXL3_MODEL_PATH": "/my/model"}):
            self.assertEqual(backend.resolve_path("model", {}), Path("/my/model"))

    def test_configure_model_preserves_other_keys(self):
        file, _ = backend.paths()
        file.parent.mkdir(parents=True)
        file.write_text('port = 9999\n')
        backend.configure_model("/models/qwen")
        self.assertEqual(backend.config(), {"port": 9999, "model_path": "/models/qwen"})

class PackagingTests(unittest.TestCase):
    def test_manifest(self):
        data = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(data["entryPoints"]["barWidget"], "ui/Panel.qml")
        self.assertTrue((ROOT / data["entryPoints"]["barWidget"]).is_file())

    def test_qml_actions(self):
        qml = (ROOT / "ui/Panel.qml").read_text()
        for action in ("start", "stop", "restart", "status", "configure-model", "open-pi"):
            self.assertIn(action, qml)


if __name__ == "__main__":
    unittest.main()
