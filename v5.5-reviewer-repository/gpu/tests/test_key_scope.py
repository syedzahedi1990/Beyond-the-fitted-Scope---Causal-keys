pass                                                                           

                                                                                
                                                                           
                                                               
   
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gpu.runtime import key_engine as r


class Tensor:
    dtype = "bf16"
    requires_grad = False
    device = "cpu"

    def __init__(self, array):
        self.a = np.asarray(array, dtype=np.float32).copy()

    @property
    def shape(self):
        return self.a.shape

    def detach(self): return self
    def contiguous(self): return self
    def cpu(self): return self
    def clone(self): return Tensor(self.a)
    def to(self, device): return self
    def reshape(self, *shape): return Tensor(self.a.reshape(*shape))
    def view(self, dtype): return SimpleNamespace(a=self.a.view(dtype))
    def __getitem__(self, key): return Tensor(self.a[key])
    def __setitem__(self, key, value): self.a[key] = value.a if isinstance(value, Tensor) else value


class Module:
    def __init__(self, transform):
        self.transform = transform
        self.training = False
        self._forward_hooks, self._forward_pre_hooks = {}, {}
        self.next_handle = 0
        self.raw = self.applied = None
        self.hook_replacements = 0

    def forward(self, x): return self.transform(x)

    def __call__(self, x):
        out = self.forward(x)
        self.raw = out.clone()
        self.hook_replacements = 0
        for hook in list(self._forward_hooks.values()):
            replacement = hook(self, (x,), out)
            if replacement is not None:
                self.hook_replacements += 1
                out = replacement
        self.applied = out.clone()
        return out

    def register_forward_hook(self, hook):
        self.next_handle += 1
        key = self.next_handle
        self._forward_hooks[key] = hook
        return SimpleNamespace(remove=lambda: self._forward_hooks.pop(key))


class Block(Module):
    def __init__(self):
        super().__init__(None)
        self.prefix_leak = False
        self.self_attn = SimpleNamespace(q_proj=Module(lambda x: x.clone()),
            k_proj=Module(lambda x: Tensor(x.a[:, :, :4])),
            v_proj=Module(lambda x: Tensor(x.a[:, :, :4])), o_proj=Module(lambda x: x.clone()))
        self.last_pre_rope_k = self.last_rotated_k = None

    def forward(self, x):
        attn = self.self_attn
        q, k, v = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
        self.last_pre_rope_k = k.clone()

        def rope(array):
            shaped = array.reshape(1, 7, -1, 2)
            angles = np.arange(7, dtype=np.float32)[None, :, None] * np.float32(.31)
            out = shaped.copy()
            c, s = np.cos(angles), np.sin(angles)
            out[:, :, :, 0] = shaped[:, :, :, 0] * c - shaped[:, :, :, 1] * s
            out[:, :, :, 1] = shaped[:, :, :, 0] * s + shaped[:, :, :, 1] * c
            return out

        qq, kk = rope(q.a), np.repeat(rope(k.a), 2, axis=2)
        vv = np.repeat(v.a.reshape(1, 7, 2, 2), 2, axis=2)
        self.last_rotated_k = kk.copy()
        logits = np.einsum("bihd,bjhd->bhij", qq, kk) / np.float32(np.sqrt(2))
        logits = np.where(np.tri(7, dtype=bool)[None, None], logits, -1e20)
        weights = np.exp(logits - logits.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        message = np.einsum("bhij,bjhd->bihd", weights, vv).reshape(1, 7, 8)
        hidden = x.a + attn.o_proj(Tensor(message)).a
        out = Tensor(hidden + np.float32(.04) * np.tanh(hidden))
        if self.prefix_leak:
            out.a[0, 0, 0] += 1
        return out


class Engine:
    def __init__(self):
        self.torch = SimpleNamespace(bfloat16="bf16", uint16=np.uint16,
            is_tensor=lambda value: isinstance(value, Tensor), equal=lambda a, b: np.array_equal(a.a, b.a))
        self.layers = [Block() for _ in range(4)]
        self.model = SimpleNamespace(parameters=lambda: ())
        self.device, self.call_count = "cpu", 0
        self.last_output = self.last_result = None
        self.corrupt_return = False

    def check(self): pass
    def fixed(self, base, source, basis=None): return source.clone()

    @staticmethod
    def tensor_hash(tensor): return hashlib.sha256(tensor.a.tobytes()).hexdigest()

    def run(self, encoded, layer4_patch=None, replacements=None, capture_layers=(1,), expected_prefix=None,
            expected_layer4=None):
        assert expected_prefix == encoded["input_ids"][:4]
        self.call_count += 1
        ids = encoded["input_ids"] + [0] * (7 - len(encoded["input_ids"]))
        hidden = Tensor(np.array(ids, dtype=np.float32)[None, :, None] * np.float32(.08) *
                        np.linspace(.5, 1.2, 8, dtype=np.float32)[None, None, :])
        captures = {}
        for layer, block in enumerate(self.layers, 1):
            hidden = block(hidden)
            if layer == 1:
                self.last_prepatch_layer4 = hidden[0, 1:4]
                if expected_layer4 is not None:
                    assert self.torch.equal(self.last_prepatch_layer4, expected_layer4)
                if layer4_patch is not None:
                    hidden[0, 1:4] = layer4_patch
            if layer in capture_layers:
                captures[layer] = hidden[0, 1:4]
        self.last_output = hidden
        result = {"score": float(hidden.a[0, len(encoded["input_ids"]) - 1].sum()), "audit": {
            "call_id": self.call_count, "prefix_sha256": encoded["prefix_sha256"],
            "input_ids_sha256": encoded["input_ids_sha256"], "event_span": encoded["span"],
            "layer4_patch_sha256": None if layer4_patch is None else self.tensor_hash(layer4_patch),
            "replacement_sha256": {}}}
        if self.corrupt_return:
            result["audit"]["input_ids_sha256"] = "wrong"
        self.last_result = result
        return result, captures


def enc(value, suffix=5):
    ids = [1, value, 3, 4, suffix, 6]
    return {"input_ids": ids, "span": [1, 4], "event_value_position": 1,
            "prefix_sha256": r.digest(ids[:4]), "input_ids_sha256": r.digest(ids)}


class HookTests(unittest.TestCase):
    def setUp(self):
        small = replace(r.PROFILES["mistral"], width=8, padding=7, n_layers=4, first_layer=2,
                        first_exchange_layer=2, fit_layer=1, q_heads=4, kv_heads=2, head_dim=2)
        self.profile_patch = patch.dict(r.PROFILES, {"mistral": small})
        self.profile_patch.start()
        self.addCleanup(self.profile_patch.stop)
        p = patch.object(r, "validate_runtime", return_value={"synthetic": "native-source"})
        p.start()
        self.addCleanup(p.stop)
        self.engine = Engine()
        self.edge = r.KeyOnlyEngine(self.engine, "mistral")
        self.base_result, _, self.base = self.edge.capture(enc(2))
        self.base_output = self.engine.last_output.clone()
        _, _, self.donor = self.edge.capture(enc(7))
        self.groups = self.edge.profile.full_groups

    def clean(self):
        self.assertFalse(self.edge._active)
        self.assertTrue(all(not module._forward_hooks for module in self.edge.modules.values()))

    def test_both_self_arms_exact_and_native_returns_unmodified(self):
        for mode in ("none", "token"):
            result, captures, receipt = self.edge.run(enc(2), recipient=self.base, donor=self.base,
                                                     groups=self.groups, restoration=mode)
            self.assertEqual(result["score"], self.base_result["score"])
            self.assertIs(result, self.engine.last_result)
            self.assertIs(result, self.edge.last_native_result)
            self.assertIs(captures, self.edge.last_native_captures)
            np.testing.assert_array_equal(self.engine.last_output.a.view(np.uint32), self.base_output.a.view(np.uint32))
            self.assertEqual(receipt["hook_calls"], 15)
            self.clean()

    def test_none_allows_downstream_native_qv_response_and_has_no_o_replacements(self):
        _, _, receipt = self.edge.run(enc(2), recipient=self.base, donor=self.donor,
                                      groups=self.groups, restoration="none")
        self.assertEqual(receipt["guard_stop"], 1)
        self.assertFalse(receipt["changes"]["3"]["q"]["received_critical_matches_reference"])
        self.assertFalse(receipt["changes"]["3"]["v"]["received_critical_matches_reference"])
        for layer in self.edge.profile.layers:
            attn = self.engine.layers[layer - 1].self_attn
            for kind in ("q", "v", "o"):
                self.assertEqual(getattr(attn, kind + "_proj").hook_replacements, 0)
        np.testing.assert_array_equal(self.engine.last_output.a[:, :1], self.base_output.a[:, :1])
        self.assertFalse(np.array_equal(self.engine.last_output.a[:, 1:2], self.base_output.a[:, 1:2]))
        self.clean()

    def test_token_restores_native_critical_qv_but_allows_later_event_relay(self):
        _, _, receipt = self.edge.run(enc(2), recipient=self.base, donor=self.donor,
                                      groups=self.groups, restoration="token")
        self.assertEqual(receipt["guard_stop"], 2)
        for layer in self.edge.profile.layers:
            for kind in ("q", "v"):
                self.assertTrue(receipt["changes"][str(layer)][kind]["received_critical_matches_reference"])
                self.assertFalse(receipt["changes"][str(layer)][kind]["directly_modified"])
            self.assertEqual(self.engine.layers[layer - 1].self_attn.o_proj.hook_replacements, 1)
        np.testing.assert_array_equal(self.engine.last_output.a[:, :2], self.base_output.a[:, :2])
        self.assertFalse(np.array_equal(self.engine.last_output.a[:, 2:4], self.base_output.a[:, 2:4]))
        self.clean()

    def test_partial_group_scope_and_native_rope(self):
        _, _, receipt = self.edge.run(enc(2), recipient=self.base, donor=self.donor,
                                      groups={2: (0,)}, restoration="none")
        block = self.engine.layers[1]
        k = block.self_attn.k_proj
        np.testing.assert_array_equal(k.applied.a[0, 1, :2], self.donor.keys[2].a[0])
        np.testing.assert_array_equal(k.applied.a[0, 1, 2:], k.raw.a[0, 1, 2:])
        np.testing.assert_array_equal(k.applied.a[:, :1], k.raw.a[:, :1])
        np.testing.assert_array_equal(k.applied.a[:, 2:], k.raw.a[:, 2:])
        np.testing.assert_array_equal(block.last_pre_rope_k.a, k.applied.a)
        self.assertFalse(np.array_equal(block.last_rotated_k[0, 1, 0], self.donor.keys[2].a[0]))
        self.assertTrue(receipt["changes"]["2"]["k"]["unselected_groups_exact"])
        self.clean()

    def test_direct_reference_works_for_new_question_suffix(self):
        result, _, _ = self.edge.run(enc(2, 8), recipient=self.base, donor=self.donor,
                                     groups=self.groups, restoration="none")
        self.assertEqual(result["audit"]["input_ids_sha256"], enc(2, 8)["input_ids_sha256"])
        self.clean()

    def test_actual_patch_bound_and_passed_through_both_modes(self):
        _, original = self.engine.run(enc(2), expected_prefix=enc(2)["input_ids"][:4])
        patch_tensor = original[1].clone()
        patch_tensor.a[0] += np.float32(.25)
        _, _, patched = self.edge.capture(enc(2), layer4_patch=patch_tensor, expected_layer4=original[1])
        for mode in ("none", "token"):
            result, _, _ = self.edge.run(enc(2), recipient=patched, donor=self.donor, groups=self.groups,
                                         restoration=mode, layer4_patch=patch_tensor, expected_layer4=original[1])
            self.assertEqual(result["audit"]["layer4_patch_sha256"], self.engine.tensor_hash(patch_tensor))
        with self.assertRaisesRegex(ValueError, "Recipient prefix/patch"):
            self.edge.run(enc(2), recipient=patched, donor=self.donor, groups=self.groups, restoration="none")
        self.clean()

    def test_synthetic_key_donor_provenance_and_no_alias(self):
        keys = {layer: payload.clone() for layer, payload in self.donor.keys.items()}
        synthetic = self.edge.make_key_donor(self.base, keys, {"kind": "signed_coordinate_permutation", "seed": 1})
        before = synthetic.keys[2].a.copy()
        keys[2].a[:] = 0
        np.testing.assert_array_equal(synthetic.keys[2].a, before)
        self.assertEqual(synthetic.kind, "synthetic_keys")
        self.assertFalse(synthetic.output_prefixes)
        self.assertEqual(set(self.edge.reference_fingerprint(synthetic)["hashes"]),
                         {"2/k_payload", "3/k_payload", "4/k_payload"})
        self.edge.run(enc(2), recipient=self.base, donor=synthetic, groups=self.groups, restoration="none")
        self.clean()

    def test_scope_leak_rejected_with_cleanup(self):
        self.engine.layers[2].prefix_leak = True
        with self.assertRaisesRegex(ValueError, "preserved residual prefix"):
            self.edge.run(enc(2), recipient=self.base, donor=self.base, groups=self.groups, restoration="none")
        self.assertEqual(self.edge.last_receipt["status"], "FAILED")
        self.clean()

    def test_mutated_reference_rejected_before_forward(self):
        frontier = self.engine.call_count
        self.donor.keys[2].a[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Frozen donor key mutated"):
            self.edge.run(enc(2), recipient=self.base, donor=self.donor, groups=self.groups, restoration="none")
        self.assertEqual(self.engine.call_count, frontier)
        self.clean()

    def test_returned_capture_retained_when_post_return_audit_fails(self):
        self.engine.corrupt_return = True
        with self.assertRaisesRegex(ValueError, "Engine call/intervention binding"):
            self.edge.capture(enc(2))
        self.assertIs(self.edge.last_native_result, self.engine.last_result)
        self.assertEqual(set(self.edge.last_native_captures), {1})
        self.assertEqual(set(self.edge.last_native_reference.keys), {2, 3, 4})
        self.assertTrue(self.edge.last_receipt["native_return_retained"])
        self.clean()

    def test_changed_dispatch_and_hook_conflict_are_preflight_failures(self):
        frontier = self.engine.call_count
        block = self.engine.layers[1]
        original = block.self_attn.v_proj.forward
        block.self_attn.v_proj.forward = lambda x: x
        with self.assertRaisesRegex(ValueError, "dispatch/training"):
            self.edge.capture(enc(2))
        block.self_attn.v_proj.forward = original
        handle = block.self_attn.q_proj.register_forward_hook(lambda *args: None)
        with self.assertRaisesRegex(ValueError, "Preexisting hooks"):
            self.edge.capture(enc(2))
        handle.remove()
        self.assertEqual(self.engine.call_count, frontier)
        self.clean()


class ProfileTests(unittest.TestCase):
    def test_fixed_production_geometry_and_bias_profiles(self):
        for name, layers, width, q_heads, bias in (("mistral", 40, 5120, 32, False), ("qwen", 80, 8192, 64, True)):
            profile = r.PROFILES[name]
            self.assertEqual((profile.n_layers, profile.width, profile.q_heads, profile.qkv_bias), (layers, width, q_heads, bias))
            self.assertEqual(profile.layers, tuple(range(5, layers + 1)))
            self.assertEqual(profile.full_groups, {layer: tuple(range(8)) for layer in range(6, layers + 1)})
            self.assertEqual(profile.padding, 1024)

    @staticmethod
    def runtime_double(profile):
        attn = SimpleNamespace(training=False, layer_idx=0, head_dim=profile.head_dim,
            num_key_value_groups=profile.q_heads // profile.kv_heads, is_causal=True,
            attention_dropout=0, sliding_window=None)
        for name, shape in (("q_proj", (profile.q_heads * profile.head_dim, profile.width)),
                            ("k_proj", (profile.kv_heads * profile.head_dim, profile.width)),
                            ("v_proj", (profile.kv_heads * profile.head_dim, profile.width)),
                            ("o_proj", (profile.width, profile.q_heads * profile.head_dim))):
            weight = SimpleNamespace(shape=shape, dtype="bf16", device="cpu")
            bias = SimpleNamespace(shape=(shape[0],), dtype="bf16", device="cpu") if profile.qkv_bias and name != "o_proj" else None
            setattr(attn, name, SimpleNamespace(weight=weight, bias=bias))
        block = SimpleNamespace(training=False, hidden_size=profile.width, self_attn=attn)
        return SimpleNamespace(environment={"model": profile.model, "revision": profile.revision},
            layers=[block], device="cpu", torch=SimpleNamespace(bfloat16="bf16"),
            model=SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa", use_cache=False,
                                                        sliding_window=131072, use_sliding_window=False)))

    def test_runtime_validates_each_native_bias_geometry_and_effective_window(self):
        for name in ("mistral", "qwen"):
            profile = replace(r.PROFILES[name], n_layers=1)
            engine = self.runtime_double(profile)
            with patch.dict(sys.modules, {"transformers.integrations.sdpa_attention": SimpleNamespace()}), \
                 patch.object(r.inspect, "getsourcefile", return_value="/synthetic-native.py"), \
                 patch.object(r, "file_sha", side_effect=[profile.modeling_sha256, r.SDPA_SHA256]):
                self.assertEqual(r.validate_runtime(engine, profile)[profile.modeling_name], profile.modeling_sha256)
            engine.layers[0].self_attn.k_proj.bias = None if profile.qkv_bias else SimpleNamespace()
            with patch.dict(sys.modules, {"transformers.integrations.sdpa_attention": SimpleNamespace()}), \
                 patch.object(r.inspect, "getsourcefile", return_value="/synthetic-native.py"), \
                 patch.object(r, "file_sha", side_effect=[profile.modeling_sha256, r.SDPA_SHA256]):
                with self.assertRaisesRegex(ValueError, "projection bias"):
                    r.validate_runtime(engine, profile)

    def test_runtime_rejects_unpinned_native_source(self):
        profile = replace(r.PROFILES["qwen"], n_layers=1)
        engine = self.runtime_double(profile)
        with patch.dict(sys.modules, {"transformers.integrations.sdpa_attention": SimpleNamespace()}), \
             patch.object(r.inspect, "getsourcefile", return_value="/synthetic-native.py"), \
             patch.object(r, "file_sha", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "Pinned native source differs"):
                r.validate_runtime(engine, profile)


if __name__ == "__main__":
    unittest.main()
