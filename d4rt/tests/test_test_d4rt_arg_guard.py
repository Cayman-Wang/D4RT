import subprocess
import sys
import unittest
from pathlib import Path


class TestTestD4RTArgGuard(unittest.TestCase):
    def test_num_queries_larger_than_n_fails_early(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "test_d4rt.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--test_data_path",
            "/tmp",
            "--ckpt",
            "/tmp/fake.ckpt",
            "--num_queries",
            "64",
            "--N",
            "32",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        self.assertNotEqual(result.returncode, 0)
        error_text = f"{result.stdout}\n{result.stderr}"
        self.assertIn("must be <= --N", error_text)


if __name__ == "__main__":
    unittest.main()
