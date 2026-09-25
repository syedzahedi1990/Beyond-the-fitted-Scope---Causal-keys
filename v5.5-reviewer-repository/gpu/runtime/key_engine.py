







from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    revision: str
    width: int
    n_layers: int
    q_heads: int
    qkv_bias: bool
    modeling_name: str
    modeling_sha256: str
    kv_heads: int = 8
    head_dim: int = 128
    padding: int = 1024
    first_layer: int = 5
    first_exchange_layer: int = 6
    fit_layer: int = 4

    @property
    def layers(self):
        return tuple(range(self.first_layer, self.n_layers + 1))

    @property
    def full_groups(self):
        return {layer: tuple(range(self.kv_heads))
                for layer in range(self.first_exchange_layer, self.n_layers + 1)}


PROFILES = {
    "mistral": ModelProfile("mistral", "mistralai/Mistral-Small-24B-Instruct-2501",
                            "9527884be6e5616bdd54de542f9ae13384489724", 5120, 40, 32, False,
                            "modeling_mistral.py", "94035ed16e1f206905625840ffdad9b7b094f043b90a4160c3c2d0b0cc33f55b"),
    "qwen": ModelProfile("qwen", "Qwen/Qwen2.5-72B-Instruct",
                         "495f39366efef23836d0cfae4fbe635880d2be31", 8192, 80, 64, True,
                         "modeling_qwen2.py", "99fa98c5676604cf6ef505892b70fda5c1c4cd835971459f38d61090cccab1e4"),
}
SDPA_SHA256 = "87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564"


def validate_runtime(engine, profile):
    pass                                                                             
    require(engine.environment["model"] == profile.model and
            engine.environment["revision"] == profile.revision, "Unexpected model/revision")
    require(len(engine.layers) == profile.n_layers and
            engine.model.config._attn_implementation == "sdpa" and
            engine.model.config.use_cache is False, "Unexpected native architecture/cache/backend")
    sdpa = sys.modules.get("transformers.integrations.sdpa_attention")
    require(sdpa is not None, "Native SDPA module is not loaded")
    paths = {profile.modeling_name: inspect.getsourcefile(type(engine.layers[0])),
             "sdpa_attention.py": inspect.getsourcefile(sdpa)}
    expected = {profile.modeling_name: profile.modeling_sha256,
                "sdpa_attention.py": SDPA_SHA256}
    for name, path in paths.items():
        require(path is not None and file_sha(path) == expected[name], "Pinned native source differs: " + name)
    for index, block in enumerate(engine.layers):
        attn = block.self_attn
        require(not block.training and not attn.training and block.hidden_size == profile.width and
                attn.layer_idx == index and attn.head_dim == profile.head_dim and
                attn.num_key_value_groups == profile.q_heads // profile.kv_heads and
                attn.is_causal and attn.attention_dropout == 0 and
                getattr(attn, "sliding_window", None) is None, "Native attention contract differs")
        for name, shape in (("q_proj", (profile.q_heads * profile.head_dim, profile.width)),
                            ("k_proj", (profile.kv_heads * profile.head_dim, profile.width)),
                            ("v_proj", (profile.kv_heads * profile.head_dim, profile.width)),
                            ("o_proj", (profile.width, profile.q_heads * profile.head_dim))):
            module = getattr(attn, name)
            require(tuple(module.weight.shape) == shape and module.weight.dtype == engine.torch.bfloat16 and
                    module.weight.device == engine.device, "Native projection contract differs")
            biased = profile.qkv_bias and name != "o_proj"
            require((module.bias is not None) == biased, "Native projection bias differs")
            if biased:
                require(tuple(module.bias.shape) == (shape[0],) and module.bias.dtype == engine.torch.bfloat16 and
                        module.bias.device == engine.device, "Native projection bias tensor differs")
    return expected


@dataclass(frozen=True)
class KeyReference:
    metadata: dict
    keys: dict
    output_prefixes: dict
    hashes: dict
    kind: str = "captured"
    provenance: dict = field(default_factory=dict)


def reference_fingerprint(reference):
    pass                                                                           
    return {"schema_version": 1, "kind": reference.kind, "metadata": reference.metadata,
            "provenance": reference.provenance,
            "hashes": {f"{layer}/{kind}": value for (layer, kind), value in sorted(reference.hashes.items())}}


class KeyOnlyEngine:
    pass                                                                    

                                                           
                                                                              
                                             
                                                                             
                                                                                 
       
    def __init__(self, engine, profile):
        require(profile in PROFILES, "Explicit supported model profile required")
        self.engine, self.torch, self.profile = engine, engine.torch, PROFILES[profile]
        self.source_identity = validate_runtime(engine, self.profile)
        self.adapter_provenance = {"source_sha256": file_sha(__file__),
                                   "key_site": "k_proj output after native bias and before native RoPE",
                                   "query_hook_returns_replacement": False,
                                   "value_hook_returns_replacement": False}
        self._active = False
        self.last_receipt = None
        self.last_native_result = self.last_native_captures = self.last_native_reference = None
        self.modules = {}
        for layer in self.profile.layers:
            block = engine.layers[layer - 1]
            self.modules[layer, "residual"] = block
            for kind in ("q", "k", "v", "o"):
                self.modules[layer, kind] = getattr(block.self_attn, kind + "_proj")
        require(len({id(module) for module in self.modules.values()}) == len(self.modules),
                "Shared native modules unsupported")
        self._dispatch = {key: self._function(module.forward) for key, module in self.modules.items()}
        self._engine_dispatch = {name: self._function(getattr(engine, name)) for name in ("run", "fixed")}

    @staticmethod
    def _function(function):
        return getattr(function, "__func__", function)

    def _native_guard(self):
        require(file_sha(__file__) == self.adapter_provenance["source_sha256"], "Adapter source changed")
        for key, module in self.modules.items():
            require(self._function(module.forward) is self._dispatch[key] and not module.training,
                    "Native dispatch/training changed")
        for name, function in self._engine_dispatch.items():
            require(self._function(getattr(self.engine, name)) is function, "Native Engine dispatch changed")
        return tuple((id(parameter), int(parameter._version)) for parameter in self.engine.model.parameters())

    def _hash(self, tensor):
        return self.engine.tensor_hash(tensor.detach().contiguous())

    def _clone_cpu(self, tensor):
        return tensor.detach().cpu().clone()

    def _equal_bytes(self, left, right):
        return self.torch.equal(left.contiguous().view(self.torch.uint16), right.contiguous().view(self.torch.uint16))

    def _tensor(self, tensor, shape):
        require(self.torch.is_tensor(tensor) and tuple(tensor.shape) == tuple(shape) and
                tensor.dtype == self.torch.bfloat16 and not tensor.requires_grad,
                "Unexpected BF16 tensor shape/dtype")

    def _identity(self, encoded, patch):
        ids, (a, z), p = encoded["input_ids"], encoded["span"], encoded["event_value_position"]
        require(type(p) is int and 0 <= a <= p < z < len(ids) < self.profile.padding,
                "Invalid event/value coordinates")
        require(digest(ids) == encoded["input_ids_sha256"] and digest(ids[:z]) == encoded["prefix_sha256"],
                "Encoded input hash differs")
        return {"schema_version": 1, "model_profile": self.profile.name, "prefix_ids": list(ids[:z]),
                "prefix_sha256": encoded["prefix_sha256"], "span": [a, z], "value_position": p,
                "token_restore_stop": p + 1, "layer4_patch_sha256": None if patch is None else self._hash(patch),
                "source_identity": self.source_identity, "adapter_provenance": self.adapter_provenance}

    reference_fingerprint = staticmethod(reference_fingerprint)

    def make_key_donor(self, recipient, keys, provenance):
        pass                                                                             
        require(isinstance(recipient, KeyReference) and recipient.kind == "captured" and
                isinstance(provenance, dict) and provenance, "Synthetic donor needs captured coordinate origin and provenance")
        require(set(keys) == set(self.profile.layers), "Synthetic keys must cover every captured layer")
        copied, hashes = {}, {}
        for layer in self.profile.layers:
            self._tensor(keys[layer], (self.profile.kv_heads, self.profile.head_dim))
            copied[layer] = self._clone_cpu(keys[layer])
            hashes[layer, "k_payload"] = self._hash(copied[layer])
        return KeyReference(dict(recipient.metadata), copied, {}, hashes, "synthetic_keys", dict(provenance))

    def capture(self, encoded, *, layer4_patch=None, expected_layer4=None):
        return self._execute(encoded, layer4_patch=layer4_patch, expected_layer4=expected_layer4,
                             recipient=None, donor=None, groups=None, restoration=None)

    def run(self, encoded, *, recipient, donor, groups, restoration, layer4_patch=None, expected_layer4=None):
        return self._execute(encoded, layer4_patch=layer4_patch, expected_layer4=expected_layer4,
                             recipient=recipient, donor=donor, groups=groups, restoration=restoration)

    def _execute(self, encoded, *, layer4_patch, expected_layer4, recipient, donor, groups, restoration):
        require(not self._active, "Overlapping key-only execution")
        self.last_native_result = self.last_native_captures = self.last_native_reference = None
        self.last_receipt = None
        identity = self._identity(encoded, layer4_patch)
        profile = self.profile
        p, z, n = identity["value_position"], identity["span"][1], len(encoded["input_ids"])
        capture = recipient is None
        selected = {}
        if capture:
            require(donor is None and groups is None and restoration is None, "Capture contains an intervention")
            stop = p + 1
        else:
            require(restoration in ("none", "token"), "Explicit none/token restoration required")
            require(isinstance(recipient, KeyReference) and recipient.kind == "captured" and
                    isinstance(donor, KeyReference) and donor.kind in ("captured", "synthetic_keys"),
                    "Captured recipient and bound key donor required")
            require(recipient.metadata == identity, "Recipient prefix/patch/source binding differs")
            dm = donor.metadata
            require(dm["span"] == identity["span"] and dm["value_position"] == p and
                    dm["source_identity"] == self.source_identity and dm["model_profile"] == profile.name and
                    dm["adapter_provenance"] == self.adapter_provenance, "Donor coordinates/native source differ")
            require(len(dm["prefix_ids"]) == z and
                    all(a == b for i, (a, b) in enumerate(zip(dm["prefix_ids"], identity["prefix_ids"])) if i != p),
                    "Donor changes matched prefix outside value token")
            require(donor.kind != "synthetic_keys" or bool(donor.provenance), "Synthetic donor lacks provenance")
            require(isinstance(groups, dict) and groups, "Explicit nonempty KV-group selection required")
            for layer, values in groups.items():
                values = tuple(values)
                require(type(layer) is int and profile.first_exchange_layer <= layer <= profile.n_layers and values and
                        all(type(group) is int and 0 <= group < profile.kv_heads for group in values) and
                        len(set(values)) == len(values), "Invalid KV-group selection")
                selected[layer] = tuple(sorted(values))
            stop = p if restoration == "none" else p + 1
            for layer in profile.layers:
                self._tensor(recipient.output_prefixes[layer], (p + 1, profile.width))
                require(self._hash(recipient.output_prefixes[layer]) == recipient.hashes[layer, "o_prefix_token"],
                        "Frozen recipient O prefix mutated")
            for layer in selected:
                self._tensor(donor.keys[layer], (profile.kv_heads, profile.head_dim))
                require(self._hash(donor.keys[layer]) == donor.hashes[layer, "k_payload"], "Frozen donor key mutated")
        reference_before = None if capture else (digest(reference_fingerprint(recipient)), digest(reference_fingerprint(donor)))
        before_parameters = self._native_guard()
        before_hooks = {key: tuple(module._forward_hooks) for key, module in self.modules.items()}
        require(not any(before_hooks.values()) and all(not getattr(module, "_forward_pre_hooks", {})
                                                      for module in self.modules.values()), "Preexisting hooks conflict with adapter")
        self._active = True
        handles, observed, keys, outputs, hashes, changes = [], [], {}, {}, {}, {}
        expected = [(layer, kind) for layer in profile.layers for kind in ("q", "k", "v", "o", "residual")]
        receipt = {"schema_version": 1, "status": "STARTED", "operation": "capture" if capture else "key_only",
                   "restoration": restoration, "guard_stop": None if capture else stop,
                   "restoration_stop": p + 1 if restoration == "token" else None,
                   "identity": identity, "groups": {str(layer): list(values) for layer, values in sorted(selected.items())},
                   "expected_native_call_id": self.engine.call_count + 1,
                   "query_directly_modified": False, "value_directly_modified": False,
                   "key_site": "post_bias_pre_RoPE", "changes": changes,
                   "none_mode_critical_QV_may_respond": restoration == "none"}
        self.last_receipt = receipt

        def hook(layer, kind):
            def apply(_module, _inputs, output):
                self.engine.check()
                observed.append((layer, kind))
                require(observed == expected[:len(observed)], "Native hook order/repetition differs")
                tensor = output[0] if isinstance(output, tuple) else output
                width = profile.width if kind in ("o", "residual") else (
                    profile.q_heads if kind == "q" else profile.kv_heads) * profile.head_dim
                self._tensor(tensor, (1, profile.padding, width))
                for label, end in (("before", p), ("token", p + 1)):
                    hashes[layer, kind + "_prefix_" + label] = self._hash(tensor[0, :end])
                hashes[layer, kind + "_critical"] = self._hash(tensor[0, p])
                if not capture:
                                                                             
                                                                                
                    guard_label = "before" if kind == "o" or restoration == "none" else "token"
                    require(hashes[layer, kind + "_prefix_" + guard_label] ==
                            recipient.hashes[layer, kind + "_prefix_" + guard_label],
                            "Native preserved " + kind + " prefix differs")
                    changes.setdefault(str(layer), {})[kind] = {
                        "directly_modified": False,
                        "received_critical_sha256": hashes[layer, kind + "_critical"],
                        "reference_critical_sha256": recipient.hashes[layer, kind + "_critical"],
                        "received_critical_matches_reference": hashes[layer, kind + "_critical"] == recipient.hashes[layer, kind + "_critical"]}
                if kind == "residual":
                    hashes[layer, "residual_later_event"] = self._hash(tensor[0, p + 1:z])
                    hashes[layer, "residual_valid_suffix"] = self._hash(tensor[0, z:n])
                    return None
                if kind in ("q", "v"):
                    return None                                                
                if kind == "k":
                    if capture:
                        keys[layer] = self._clone_cpu(tensor[0, p].reshape(profile.kv_heads, profile.head_dim))
                        hashes[layer, "k_payload"] = self._hash(keys[layer])
                        return None
                    if layer not in selected:
                        return None
                    original, mixed = tensor.clone(), tensor.clone()
                    for group in selected[layer]:
                        a, b = group * profile.head_dim, (group + 1) * profile.head_dim
                        mixed[0, p, a:b] = donor.keys[layer][group].to(self.engine.device)
                    require(self._equal_bytes(mixed[:, :p], original[:, :p]) and
                            self._equal_bytes(mixed[:, p + 1:], original[:, p + 1:]), "Noncritical K row changed")
                    for group in range(profile.kv_heads):
                        a, b = group * profile.head_dim, (group + 1) * profile.head_dim
                        wanted = donor.keys[layer][group].to(self.engine.device) if group in selected[layer] else original[0, p, a:b]
                        require(self._equal_bytes(mixed[0, p, a:b], wanted), "Selected/unselected K group differs")
                    require(self._equal_bytes(tensor, original), "Native K storage mutated")
                    changes[str(layer)]["k"].update({"directly_modified": True,
                        "donor_payload_sha256": donor.hashes[layer, "k_payload"],
                        "applied_row_sha256": self._hash(mixed[0, p]), "other_rows_exact": True,
                        "unselected_groups_exact": True, "native_storage_unchanged": True})
                    return mixed
                if capture:
                    outputs[layer] = self._clone_cpu(tensor[0, :p + 1])
                    return None
                if restoration == "none" or layer not in selected:
                    return None                                                
                original, restored = tensor.clone(), tensor.clone()
                restored[0, :p + 1] = recipient.output_prefixes[layer].to(self.engine.device)
                require(self._hash(restored[0, :p + 1]) == recipient.hashes[layer, "o_prefix_token"],
                        "Recipient O restoration differs")
                require(self._equal_bytes(restored[:, p + 1:], original[:, p + 1:]), "Retained O rows changed")
                require(self._equal_bytes(tensor, original), "Native O storage mutated")
                changes[str(layer)]["o"].update({"directly_modified": True, "restored_span": [0, p + 1],
                    "restored_prefix_sha256": recipient.hashes[layer, "o_prefix_token"],
                    "all_retained_rows_exact": True, "native_storage_unchanged": True})
                return restored
            return apply

        try:
            for layer, kind in expected:
                handles.append(self.modules[layer, kind].register_forward_hook(hook(layer, kind)))
            result, native_captures = self.engine.run(encoded, layer4_patch=layer4_patch,
                capture_layers=(profile.fit_layer,), expected_prefix=identity["prefix_ids"], expected_layer4=expected_layer4)
                                                                                 
            self.last_native_result, self.last_native_captures = result, native_captures
            if capture:
                self.last_native_reference = KeyReference(identity, keys, outputs, hashes)
            require(observed == expected, "Native hook sequence incomplete")
            audit = result["audit"]
            require(audit["call_id"] == receipt["expected_native_call_id"] and
                    audit["prefix_sha256"] == identity["prefix_sha256"] and
                    audit["input_ids_sha256"] == encoded["input_ids_sha256"] and
                    list(audit["event_span"]) == identity["span"] and
                    audit["layer4_patch_sha256"] == identity["layer4_patch_sha256"] and
                    audit["replacement_sha256"] == {}, "Engine call/intervention binding differs")
            require(self._native_guard() == before_parameters, "Model parameters changed")
            if not capture:
                require(reference_before == (digest(reference_fingerprint(recipient)), digest(reference_fingerprint(donor))),
                        "Reference metadata/hash identity changed")
                for layer in profile.layers:
                    require(self._hash(recipient.output_prefixes[layer]) == recipient.hashes[layer, "o_prefix_token"],
                            "Frozen recipient O prefix changed")
                for layer in selected:
                    require(self._hash(donor.keys[layer]) == donor.hashes[layer, "k_payload"], "Frozen donor key changed")
            receipt.update({"status": "COMPLETE", "native_call_id": audit["call_id"],
                            "guarded_layers": list(profile.layers), "hook_calls": len(observed),
                            "prefix_scope_exact": not capture,
                            "residual_prefix_sha256": {str(layer): hashes[layer, "residual_prefix_" +
                                ("before" if restoration == "none" else "token")] for layer in profile.layers},
                            "residual_later_event_sha256": {str(layer): hashes[layer, "residual_later_event"] for layer in profile.layers},
                            "residual_valid_suffix_sha256": {str(layer): hashes[layer, "residual_valid_suffix"] for layer in profile.layers}})
            if capture:
                return result, native_captures, self.last_native_reference
            return result, native_captures, receipt
        except BaseException as error:
            receipt.update({"status": "FAILED", "error_type": type(error).__name__, "error": str(error),
                            "native_call_frontier": self.engine.call_count, "observed_hook_calls": len(observed),
                            "native_return_retained": self.last_native_result is not None})
            raise
        finally:
            for handle in reversed(handles):
                handle.remove()
            self._active = False
            require(all(tuple(module._forward_hooks) == before_hooks[key] for key, module in self.modules.items()),
                    "Key-only hook cleanup failed")
