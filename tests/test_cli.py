import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from qmagent.cli import main


class CliTests(unittest.TestCase):
    def test_decode_backend_configuration(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--home", directory, "configure", "--decode-backend", "hf"]), 0)
            config = json.loads((Path(directory) / "config.json").read_text())
            self.assertEqual(config["decode_backend"], "hf")

    def test_prompt_profile_configuration(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            model = Path(directory) / "model"
            model.mkdir()
            self.assertEqual(main(["--home", directory, "configure", "--policy", "hf", "--model", str(model),
                                   "--prompt-profile", "minimal"]), 0)
            config = json.loads((Path(directory) / "config.json").read_text())
            self.assertEqual(config["prompt_profile"], "minimal")

    def test_old_and_new_batch_entrypoints(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            for prefix, name in (([], "old"), (["run"], "new")):
                self.assertEqual(main(prefix + ["--policy", "rule", "--episodes", "1", "--output-dir", str(root / name)]), 0)
                metrics = json.loads((root / name / "summary.json").read_text())
                self.assertEqual(metrics["statuses"]["accepted"], 1)

    def test_repeated_batch_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            root = Path(directory)
            (root / "keep.txt").write_text("keep")
            with self.assertRaises(SystemExit) as caught:
                main(["run", "--output-dir", str(root)])
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual((root / "keep.txt").read_text(), "keep")

    def test_doctor_and_sessions_do_not_create_home(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            home = Path(directory) / "not-created"
            self.assertEqual(main(["--home", str(home), "doctor"]), 0)
            self.assertEqual(main(["--home", str(home), "sessions"]), 0)
            self.assertFalse(home.exists())

    def test_profile_switch_clears_model_and_preserves_old_file(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            home = Path(directory)
            model = home / "model"
            model.mkdir()
            self.assertEqual(main(["--home", directory, "configure", "--policy", "hf", "--model", str(model),
                                   "--trust-remote-code"]), 0)
            self.assertEqual(main(["--home", directory, "configure", "--policy", "rule"]), 0)
            config = json.loads((home / "config.json").read_text())
            self.assertIsNone(config["model"])
            self.assertFalse(config["trust_remote_code"])
            self.assertTrue(model.exists())
