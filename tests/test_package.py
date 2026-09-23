import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        bin_dir = self.home / "mock-bin"
        bin_dir.mkdir()
        omarchy = bin_dir / "omarchy"
        omarchy.write_text("""#!/usr/bin/env bash
set -e
case "$1 $2" in
  'plugin validate') exit 0 ;;
  'plugin list')
    if [[ -f "$HOME/enabled" ]]; then
      printf '%s\\n' '[{"id":"grimlee.local-exl3","enabled":true}]'
    else
      printf '%s\\n' '[]'
    fi ;;
  'plugin enable') touch "$HOME/enabled" ;;
  'plugin disable') rm -f "$HOME/enabled" ;;
  *) exit 1 ;;
esac
""")
        omarchy.chmod(0o755)
        shell = bin_dir / "omarchy-shell"
        shell.write_text("#!/usr/bin/env bash\nexit 0\n")
        shell.chmod(0o755)
        self.env = os.environ | {"HOME": str(self.home), "XDG_CONFIG_HOME": str(self.home / "config"),
                                 "XDG_STATE_HOME": str(self.home / "state"), "PATH": str(bin_dir) + ":" + os.environ["PATH"]}
        self.model = self.home / "models/keep-model"
        self.runtime = self.home / "runtime/keep-runtime"
        self.model.mkdir(parents=True)
        self.runtime.mkdir(parents=True)
        self.pi = self.home / ".pi/agent/models.json"
        self.pi.parent.mkdir(parents=True)
        self.pi.write_text('{"keep":true}\n')

    def test_install_twice_and_uninstall_preserves_data(self):
        plugin = self.home / "config/omarchy/plugins/grimlee.local-exl3"
        cli = self.home / ".local/bin/local-exl3"
        for _ in range(2):
            subprocess.run(["bash", str(ROOT / "scripts/install.sh")], env=self.env, check=True, capture_output=True)
            self.assertTrue((plugin / "manifest.json").is_file())
            self.assertTrue(cli.is_symlink())
            self.assertTrue((self.home / "enabled").exists())
        subprocess.run(["bash", str(ROOT / "scripts/uninstall.sh")], env=self.env, check=True, capture_output=True)
        self.assertFalse(plugin.exists())
        self.assertFalse(cli.exists())
        self.assertFalse((self.home / "enabled").exists())
        self.assertTrue(self.model.is_dir())
        self.assertTrue(self.runtime.is_dir())
        self.assertEqual(self.pi.read_text(), '{"keep":true}\n')


if __name__ == "__main__":
    unittest.main()
