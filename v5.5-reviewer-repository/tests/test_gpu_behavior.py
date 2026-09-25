import importlib.util
import types
import time
import unittest

import numpy as np

from gpu import consequence_task
from gpu.behavior_engine import BehaviorEngine, CHOICES, datasets, event_span, mapped, prompt, schedule
from gpu.train import pca_basis
from reproduce.behavior_rerun import compact


class BehaviorDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = datasets()

    def test_fixed_populations_and_replacement_exposure(self):
        self.assertEqual(len(self.data["training"]), 1000)
        self.assertEqual(len(self.data["discovery"]), 120)
        self.assertEqual(len(self.data["checking"]), 120)
        for seed in (101, 102, 103):
            self.assertEqual(sorted(schedule(seed, "original_1000")), list(range(1000)))
            draws = schedule(seed, "mapping_300")
            self.assertEqual(len(draws), 300)
            self.assertLess(len(set(draws)), 300)

    def test_complete_event_excludes_initial_state(self):
        for pair in self.data["training"]:
            for role in ("base", "source"):
                record = pair[role]
                a, z = event_span(record)
                event = prompt(record)[a:z]
                self.assertTrue(event)
                self.assertNotIn("Everyone initially sees", event)
                self.assertNotIn("Question:", event)
                self.assertIn(pair["source"]["query"]["object"], event)

    def test_distinct_mapping_forecasts(self):
        self.assertEqual([mapped(c, "m1") for c in CHOICES], list(CHOICES[1:] + CHOICES[:1]))
        self.assertEqual([mapped(mapped(c, "m3"), "m3") for c in CHOICES], list(CHOICES))
        for group in self.data["checking"]:
            for view in consequence_task.VIEWS:
                rendered = consequence_task.record(group, view)
                self.assertEqual(rendered["answer"], consequence_task.clean_answer(group, view))
                if view in ("other_agent", "irrelevant_object", "observed_overwrite"):
                    self.assertEqual(consequence_task.predict(group, view, "m3", "event_rewrite"), rendered["answer"])

    def test_randomized_pca_projection_is_rank16(self):
        data = np.random.default_rng(13).normal(size=(90, 32)).astype(np.float32)
        basis = pca_basis(data, 117)
        self.assertEqual(basis.shape, (16, 32))
        np.testing.assert_allclose(basis @ basis.T, np.eye(16), atol=1e-5)

    def test_fresh_result_keeps_candidate_tie_order_and_invalid_global(self):
        score = {"choice_order": list(reversed(CHOICES)), "scores": {c: -2 for c in CHOICES},
                 "choice_token_ids": dict(zip(CHOICES, range(6))), "prediction": "closet",
                 "global_token_id": 99, "global_prediction": None, "candidate_mass": 0.8}
        self.assertEqual(compact(score), ["closet", None, 0.8])


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch CPU tensor checks")
class PatchTensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.engine = object.__new__(BehaviorEngine)
        self.engine.torch = torch
        self.engine.width = 32

    def test_self_patch_and_full_source(self):
        torch = self.torch
        base = torch.randn(4, 32, dtype=torch.bfloat16)
        source = torch.randn(4, 32, dtype=torch.bfloat16)
        basis = torch.eye(32, dtype=torch.float32)[:16]
        self.assertTrue(torch.equal(self.engine.fixed(base, base, basis), base))
        self.assertTrue(torch.equal(self.engine.fixed(base, source), source))

    def test_projection_preserves_complement_and_connected_gradient(self):
        torch = self.torch
        base = torch.zeros(4, 32, dtype=torch.bfloat16)
        source = torch.ones(4, 32, dtype=torch.bfloat16)
        basis = torch.eye(32, dtype=torch.float32)[:16].clone().requires_grad_(True)
        result = self.engine.fixed(base, source, basis)
        self.assertTrue(torch.equal(result[:, :16], source[:, :16]))
        self.assertTrue(torch.equal(result[:, 16:], base[:, 16:]))
        result.float().sum().backward()
        self.assertIsNotNone(basis.grad)
        self.assertTrue(bool(torch.isfinite(basis.grad).all()))
        self.assertGreater(float(basis.grad.norm()), 0)

    def test_materialized_and_differentiable_model_paths_agree(self):
        torch = self.torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(hidden_size=32)
                self.embedding = torch.nn.Embedding(24, 32, dtype=torch.bfloat16)
                self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
                self.head = torch.nn.Linear(32, 24, bias=False, dtype=torch.bfloat16)

            def forward(self, input_ids, attention_mask, use_cache):
                hidden = self.embedding(input_ids)
                for layer in self.layers:
                    hidden = layer(hidden)
                hidden = hidden.cumsum(dim=1)
                return types.SimpleNamespace(logits=self.head(hidden))

        model = TinyModel()
        native = types.SimpleNamespace(torch=torch, np=np, tok=types.SimpleNamespace(pad_token_id=0), model=model,
                                       layers=model.layers, device=torch.device("cpu"), environment={}, deadline=time.monotonic()+60,
                                       native_call_limit=10)
        engine = BehaviorEngine(native, "mistral", 8)
        encoded = {"input_ids": [1, 2, 3, 4], "span": (1, 3), "choice_ids": [5, 6, 7, 8, 9, 10],
                   "choices": list(CHOICES), "text_sha256": "test"}
        original, base, prefix = engine.forward(encoded)
        source = base + torch.ones_like(base)
        basis = torch.eye(32)[:16].clone().requires_grad_(True)
        differentiable = engine.forward(encoded, basis=basis, source=source, expected=base, expected_prefix=prefix, gradients=True)
        fixed = engine.fixed(base, source, basis.detach())
        materialized = engine.forward(encoded, fixed=fixed, expected=base)[0]
        np.testing.assert_array_equal(differentiable.detach().numpy(), [materialized["scores"][c] for c in CHOICES])
        differentiable[0].backward()
        self.assertTrue(bool(torch.isfinite(basis.grad).all()))
        self.assertGreater(float(basis.grad.norm()), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        self.assertEqual(engine.forward(encoded)[0], original)
        engine.call_count = 10
        with self.assertRaisesRegex(ValueError, "model-call limit"):
            engine.forward(encoded)


if __name__ == "__main__":
    unittest.main()
