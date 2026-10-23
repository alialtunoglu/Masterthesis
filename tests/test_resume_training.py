import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.recovery import RecoverySession, TrainingStopped, load_generation
from src.training.trainer import fit
from training.kd_config import load_kd_config, merge_kd_config
from training.kd_strategies import create_distillation_strategy
from training.kd_trainer import fit_kd
from training.multi_kd_config import load_multi_kd_config, merge_multi_kd_config
from training.multi_kd_strategies import create_multi_teacher_strategy
from training.multi_kd_trainer import fit_multi_teacher_kd


class ResumeTests(unittest.TestCase):
    def test_single_and_multi_kd_resume_interfaces_complete(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = TensorDataset(torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16,
                                 torch.tensor([0, 1] * 4))

            single_config = merge_kd_config(
                load_kd_config(next(Path("configs/knowledge_distillation").glob("**/logit_based/*.json"))),
                {"training": {"epochs": 2, "batch_size": 4, "num_workers": 0, "early_stopping_patience": 5}},
            )
            teacher = torch.nn.Linear(2, 2)
            student = torch.nn.Linear(2, 2)
            strategy = create_distillation_strategy("logit_based", single_config.distillation)
            optimizer = torch.optim.AdamW(student.parameters(), lr=.01)
            loader = DataLoader(data, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(4))
            stop = root / "single.stop"; stop.touch()
            first = RecoverySession(root / "single-first/recovery", {"protocol": "same"}, stop_file=stop)
            with self.assertRaises(TrainingStopped):
                fit_kd(student, teacher, strategy, loader, loader, loader, ["a", "b"], optimizer, None,
                       torch.device("cpu"), single_config, root / "single-first/best.pt", recovery=first)
            resumed_student = torch.nn.Linear(2, 2)
            resumed_strategy = create_distillation_strategy("logit_based", single_config.distillation)
            resumed_optimizer = torch.optim.AdamW(resumed_student.parameters(), lr=.01)
            resumed = RecoverySession(root / "single-second/recovery", {"protocol": "same"}, resume_from=first.last_generation)
            single_output = fit_kd(resumed_student, teacher, resumed_strategy, loader, loader, loader, ["a", "b"],
                                   resumed_optimizer, None, torch.device("cpu"), single_config,
                                   root / "single-second/best.pt", recovery=resumed)
            self.assertEqual([row["epoch"] for row in single_output["history"]], [1, 2])
            self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

            multi_config = merge_multi_kd_config(
                load_multi_kd_config(next(Path("configs/multi_teacher_knowledge_distillation").glob("**/logit_based/*.json"))),
                {"training": {"epochs": 2, "batch_size": 4, "num_workers": 0, "early_stopping_patience": 5}},
            )
            teachers = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
            names = [teacher.model_name for teacher in multi_config.teachers]
            weights = [1 / len(names)] * len(names)
            multi_student = torch.nn.Linear(2, 2)
            multi_strategy = create_multi_teacher_strategy("logit_based", multi_config.distillation, names, weights)
            multi_optimizer = torch.optim.AdamW(multi_student.parameters(), lr=.01)
            multi_loader = DataLoader(data, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(5))
            multi_stop = root / "multi.stop"; multi_stop.touch()
            multi_first = RecoverySession(root / "multi-first/recovery", {"protocol": "same"}, stop_file=multi_stop)
            with self.assertRaises(TrainingStopped):
                fit_multi_teacher_kd(multi_student, teachers, names, multi_strategy, multi_loader, multi_loader,
                                     multi_loader, ["a", "b"], multi_optimizer, None, torch.device("cpu"),
                                     multi_config, root / "multi-first/best.pt", recovery=multi_first)
            multi_resumed_student = torch.nn.Linear(2, 2)
            multi_resumed_strategy = create_multi_teacher_strategy("logit_based", multi_config.distillation, names, weights)
            multi_resumed_optimizer = torch.optim.AdamW(multi_resumed_student.parameters(), lr=.01)
            multi_resumed = RecoverySession(root / "multi-second/recovery", {"protocol": "same"}, resume_from=multi_first.last_generation)
            multi_output = fit_multi_teacher_kd(
                multi_resumed_student, teachers, names, multi_resumed_strategy, multi_loader, multi_loader, multi_loader,
                ["a", "b"], multi_resumed_optimizer, None, torch.device("cpu"), multi_config,
                root / "multi-second/best.pt", recovery=multi_resumed,
            )
            self.assertEqual([row["epoch"] for row in multi_output["history"]], [1, 2])
            self.assertTrue(all(parameter.grad is None for model in teachers for parameter in model.parameters()))

    def test_feature_adapter_and_optimizer_state_resume_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = TensorDataset(torch.ones(2, 2), torch.tensor([0, 1]))
            loader = DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(2))
            model = torch.nn.Linear(2, 2)
            adapter = torch.nn.Conv2d(2, 3, 1)
            optimizer = torch.optim.AdamW([*model.parameters(), *adapter.parameters()], lr=.01)
            best = root / "best.pt"
            torch.save({"student_state_dict": model.state_dict(), "strategy_state_dict": adapter.state_dict()}, best)
            source = RecoverySession(root / "source/recovery", {"protocol": "same"})
            source.save(1, {"history": [], "best_epoch": 1}, model, optimizer, None, None, adapter, loader, best)

            restored_model = torch.nn.Linear(2, 2)
            restored_adapter = torch.nn.Conv2d(2, 3, 1)
            restored_optimizer = torch.optim.AdamW([*restored_model.parameters(), *restored_adapter.parameters()], lr=.01)
            target = RecoverySession(root / "target/recovery", {"protocol": "same"}, resume_from=source.last_generation)
            target.restore(restored_model, restored_optimizer, None, None, restored_adapter, loader, root / "target/best.pt")
            for name, value in adapter.state_dict().items():
                self.assertTrue(torch.equal(value, restored_adapter.state_dict()[name]), name)
            self.assertEqual(len(restored_optimizer.param_groups[0]["params"]), 4)

    def test_interrupted_matches_uninterrupted_optimizer_and_model(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = TensorDataset(torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24,
                                 torch.tensor([0, 1] * 4))
            def run(folder, resume=None, stop=False):
                torch.manual_seed(10)
                model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 2))
                optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.8)
                loader = DataLoader(data, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(9))
                flag = root / (folder + '.stop')
                if stop:
                    flag.touch()
                session = RecoverySession(root / folder / 'recovery', {'protocol': 'same'},
                                          resume_from=resume, stop_file=flag)
                output = fit(model, loader, loader, loader, ['a', 'b'], optimizer, torch.device('cpu'),
                             epochs=2, scheduler=scheduler, checkpoint_path=root / folder / 'best.pt', recovery=session)
                return output, load_generation(session.last_generation)
            full, full_state = run('full')
            with self.assertRaises(TrainingStopped):
                run('first', stop=True)
            resumed, resumed_state = run('second', root / 'first/recovery/epoch-000001')
            for key, value in full_state['model'].items():
                self.assertTrue(torch.equal(value, resumed_state['model'][key]), key)
            self.assertEqual(full_state['scheduler'], resumed_state['scheduler'])
            self.assertEqual(full['best_epoch'], resumed['best_epoch'])
            self.assertEqual(len(resumed['history']), 2)
