import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class TestTrainD4RTTrainerConfig(unittest.TestCase):
    @staticmethod
    def _load_script_module():
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "train_d4rt.py"
        spec = importlib.util.spec_from_file_location("train_d4rt", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_trainer_receives_max_steps_and_resume_checkpoint(self):
        module = self._load_script_module()
        trainer_calls = {}

        class FakeTrainer:
            def __init__(self, **kwargs):
                trainer_calls["kwargs"] = kwargs

            def fit(self, model, datamodule=None, ckpt_path=None):
                trainer_calls["fit"] = {
                    "model": model,
                    "datamodule": datamodule,
                    "ckpt_path": ckpt_path,
                }

        fake_checkpoint = SimpleNamespace(name="checkpoint")
        fake_datamodule = SimpleNamespace(name="datamodule")
        fake_model = SimpleNamespace(name="model")

        with mock.patch.object(module, "PointOdysseyDataModule", return_value=fake_datamodule), \
             mock.patch.object(module, "D4RTTrainLit", return_value=fake_model), \
             mock.patch.object(module, "ModelCheckpoint", return_value=fake_checkpoint), \
             mock.patch.object(module.L, "Trainer", FakeTrainer), \
             mock.patch.object(
                 sys,
                 "argv",
                 [
                     "train_d4rt.py",
                     "--dataset_location",
                     "/tmp",
                     "--accelerator",
                     "cpu",
                     "--devices",
                     "1",
                     "--max_epochs",
                     "20",
                     "--warmup_steps",
                     "2500",
                     "--max_steps",
                     "12345",
                     "--checkpoint_every_n_train_steps",
                     "250",
                     "--resume_from_checkpoint",
                     "/tmp/resume.ckpt",
                 ],
             ):
            module.main()

        self.assertEqual(trainer_calls["kwargs"]["max_steps"], 12345)
        self.assertEqual(trainer_calls["kwargs"]["max_epochs"], 20)
        self.assertEqual(trainer_calls["kwargs"]["callbacks"], [fake_checkpoint])
        self.assertEqual(trainer_calls["fit"]["ckpt_path"], "/tmp/resume.ckpt")

    def test_max_steps_must_exceed_warmup_steps(self):
        module = self._load_script_module()

        with mock.patch.object(
            sys,
            "argv",
            [
                "train_d4rt.py",
                "--dataset_location",
                "/tmp",
                "--accelerator",
                "cpu",
                "--devices",
                "1",
                "--warmup_steps",
                "2500",
                "--max_steps",
                "2500",
            ],
        ):
            with self.assertRaisesRegex(
                ValueError, "max_steps must be greater than warmup_steps."
            ):
                module.main()


if __name__ == "__main__":
    unittest.main()
