








from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

from .qwen_geometry import MODEL, REVISION, PADDING, WIDTH, RANK, N_LAYERS, MODEL_CONFIG_SHA256
CHOICES = ("box", "basket", "shelf", "drawer", "cabinet", "closet")
FORMATS = ("original", "answer_prefill")
DEFAULT_LAYERS = (4,)
EXPECTED_PYTHON = "3.12.14"
EXPECTED_PACKAGES = {
    "accelerate": "1.13.0", "huggingface_hub": "1.16.1",
    "numpy": "1.26.4", "regex": "2026.9.10", "safetensors": "0.7.0",
    "tokenizers": "0.22.2", "torch": "2.11.0+cu128",
    "transformers": "5.9.0", "Jinja2": "3.1.6", "MarkupSafe": "3.0.3",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def prompt(record):
    require(len(record["choices"]) == 6 and set(record["choices"]) == set(CHOICES),
            "Exactly the six location choices are required")
    return ("Read the story and answer the question.\n\nStory: " + record["story"] +
            "\nQuestion: " + record["query"]["text"] + "\nChoices: " +
            ", ".join(record["choices"]) + "\nAnswer with exactly one choice.\nAnswer:")


def encode_record(tokenizer, record, fmt="answer_prefill"):
    pass                                                                      

                                                                               
                                                                               
                                                           
       
    require(fmt in FORMATS, "Unknown response format")
    raw = prompt(record)
    wrapped = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": raw}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    require(wrapped.count(raw) == 1, "Native wrapper must preserve the raw prompt exactly once")
    start, stop = record["event_char_span"]
    require(type(start) is int and type(stop) is int and 0 <= start < stop <= len(raw),
            "Invalid explicit complete-event character span")
    value = record["event_value"]
    require(value in CHOICES, "Unknown critical-event value")
    marker = "to the " + value
    event = raw[start:stop]
    require(event.count(marker) == 1, "Critical event must contain exactly one location phrase")
    value_start = start + event.index(marker) + len("to the ")
    value_stop = value_start + len(value)
    if "event_value_char_span" in record:
        require(list(record["event_value_char_span"]) == [value_start, value_stop],
                "Explicit value-character metadata disagrees with the critical event")
    text = wrapped + ("Answer:" if fmt == "answer_prefill" else "")
    offset = wrapped.index(raw)
    tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = tokens["input_ids"], tokens["offset_mapping"]
    hits = [i for i, (a, b) in enumerate(offsets) if b > offset + start and a < offset + stop]
    value_hits = [i for i, (a, b) in enumerate(offsets)
                  if b > offset + value_start and a < offset + value_stop]
    require(hits and hits == list(range(hits[0], hits[-1] + 1)), "Noncontiguous or empty event span")
    require(len(value_hits) == 1 and value_hits[0] in hits,
            "Critical-event location must be exactly one token inside the event")
    require(len(ids) < PADDING, "No truncation allowed: input reaches fixed 1024 padding")
    span = (hits[0], hits[-1] + 1)
    require(tokenizer(wrapped, add_special_tokens=False)["input_ids"][:span[1]] == ids[:span[1]],
            "Answer prefill changed the event prefix")
    choice_ids = []
    for choice in record["choices"]:
        continuation = tokenizer(" " + choice, add_special_tokens=False)["input_ids"]
        require(len(continuation) == 1, "Each leading-space choice must be one token")
        require(tokenizer(text + " " + choice, add_special_tokens=False)["input_ids"] == ids + continuation,
                "Candidate append retokenized the prefix")
        choice_ids.append(continuation[0])
    require(len(set(choice_ids)) == 6, "Choice-token identities collide")
    return dict(input_ids=list(ids), span=span, choice_ids=choice_ids,
                choices=list(record["choices"]), format=fmt,
                event_value_position=value_hits[0],
                event_value_relative_position=value_hits[0] - span[0],
                event_last_position=span[1] - 1,
                event_last_relative_position=span[1] - span[0] - 1,
                prefix_sha256=digest(ids[:span[1]]),
                input_ids_sha256=digest(ids), record_sha256=digest(record),
                text_sha256=hashlib.sha256(text.encode()).hexdigest())


def layers_contract(capture_layers, replacements):
    layers = tuple(capture_layers)
    require(len(set(layers)) == len(layers), "Duplicate capture layer")
    for layer in (*layers, *replacements):
        require(type(layer) is int and 4 <= layer <= N_LAYERS,
                "Only decoder-output event hooks at layers 4 through 80 are allowed")
    return tuple(sorted(set(layers) | set(replacements) | {4}))


def validate_local_assets(model_path, model_receipt):
    pass                                                                           
    root = Path(model_path).resolve(strict=True)
    require(root.is_dir() and model_receipt is not None,
            "A local model directory requires an independently downloaded asset receipt")
    receipt = json.loads(Path(model_receipt).read_text())
    require(receipt.get("repo_id") == MODEL and receipt.get("revision") == REVISION,
            "Local asset receipt model/revision differs")
    files = receipt["files"]
    names = [item["relative_path"] for item in files]
    require(len(names) == len(set(names)) and
            {"config.json", "model.safetensors.index.json", "tokenizer_config.json"} <= set(names),
            "Incomplete or duplicate local model receipt")
    require(any(name in names for name in ("tokenizer.json", "tokenizer.model", "tekken.json")),
            "Tokenizer bytes absent from local receipt")
    for item in files:
        relative = Path(item["relative_path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                "Unsafe model receipt member")
        path = root / relative
        require(path.is_file() and
                path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"],
                "Local model bytes differ from their receipt")
    require(sha(root / "config.json") == MODEL_CONFIG_SHA256, "Pinned Qwen configuration bytes differ")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    require(shards and shards <= set(names), "Model shards missing from the receipt")
    by_name = {item["relative_path"]: item for item in files}
    require(all(by_name[name].get("published_lfs_sha256") == by_name[name]["sha256"]
                for name in shards), "Weight bytes lack a published revision digest")
    return str(root), sha(model_receipt)


class Engine:
    def __init__(self, model_path=None, deadline_seconds=14400, *, model_receipt=None,
                 native_call_limit=20000, journal_path=None):
        require(isinstance(deadline_seconds, (float, int)) and
                math.isfinite(deadline_seconds) and deadline_seconds > 0, "Invalid deadline")
        require(type(native_call_limit) is int and native_call_limit > 0, "Invalid native-call limit")
        if journal_path is not None:
            require(not Path(journal_path).exists() and not Path(journal_path).is_symlink(),
                    "Refuse to append to any existing native journal")
        self.deadline = time.monotonic() + deadline_seconds
        self.call_count, self.native_call_limit = 0, native_call_limit
        self.journal, self.last_prepatch_layer4 = None, None
        require(platform.python_version() == EXPECTED_PYTHON, "Pinned Python 3.12.14 required")
        versions = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
        require(versions == EXPECTED_PACKAGES, "Pinned numerical/tokenization package versions required")
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.np, self.torch = np, torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
                torch.cuda.is_bf16_supported(), "Exactly one BF16-capable CUDA GPU is required")
        gpu = torch.cuda.get_device_properties(0)
        require("B200" in gpu.name and gpu.total_memory >= 178 * 1024**3,
                "One B200 with at least 178 GiB physical GPU memory required")
        self.device = torch.device("cuda:0")
        source, kwargs, receipt_hash = MODEL, dict(revision=REVISION), None
        if model_path is not None:
            source, receipt_hash = validate_local_assets(model_path, model_receipt)
            kwargs = dict(local_files_only=True)
        else:
            require(model_receipt is None, "A model receipt without its model path is ambiguous")
        self.check()
        self.tok = AutoTokenizer.from_pretrained(source, **kwargs, use_fast=True,
                                                trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            source, **kwargs, trust_remote_code=False, torch_dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation="sdpa")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.config.use_cache = False
        self.layers = self.model.model.layers
        require((receipt_hash is not None or self.model.config._commit_hash == REVISION) and
                self.model.config.hidden_size == WIDTH and len(self.layers) == N_LAYERS and
                self.model.config.model_type == "qwen2",
                "Pinned model revision or architecture changed")
        require(not getattr(self.model, "is_quantized", False) and
                all(p.dtype == torch.bfloat16 and p.device == self.device and
                    not p.requires_grad and p.grad is None for p in self.model.parameters()),
                "Model must be frozen single-device unquantized BF16")
        require(self.model.config._attn_implementation == "sdpa", "Native SDPA required")
        self.environment = dict(model=MODEL, revision=REVISION, model_source=source,
                                local_asset_receipt_sha256=receipt_hash,
                                gpu=gpu.name, gpu_bytes=gpu.total_memory,
                                python=platform.python_version(), sys_version=sys.version,
                                packages=versions, cuda_runtime=torch.version.cuda,
                                padding=PADDING, attention="sdpa", gradients=False,
                                cross_model_replication=True, historical_qwen_bitwise_execution_not_claimed=True,
                                native_call_limit=native_call_limit,
                                deadline_seconds=deadline_seconds)
        self.check()
        if journal_path is not None:
            self.journal = Path(journal_path).open("x")
            self._journal(dict(event="engine_ready", environment=self.environment))

    def close(self):
        if self.journal is not None:
            self.journal.close()
            self.journal = None

    def _journal(self, record):
        if self.journal is not None:
            self.journal.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            self.journal.flush()
            os.fsync(self.journal.fileno())

    def check(self):
        require(time.monotonic() < self.deadline, "Inference stage deadline reached")

    def encode(self, record, fmt="answer_prefill"):
        return encode_record(self.tok, record, fmt)

    def tensor_hash(self, value):
        return hashlib.sha256(value.detach().contiguous().view(self.torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def _event(self, tensor, shape=None):
        require(len(tensor.shape) == 2 and tensor.shape[0] > 0 and tensor.shape[1] == WIDTH and
                tensor.dtype == self.torch.bfloat16 and not tensor.requires_grad and
                bool(self.torch.isfinite(tensor).all()), "Invalid detached BF16 complete-event tensor")
        require(shape is None or tuple(tensor.shape) == tuple(shape), "Complete event shapes differ")
        return tensor

    def fixed(self, base_event, source_event, basis=None):
        pass                                                                      

                                                                              
                                                                         
           
        self.check()
        torch = self.torch
        self._event(base_event)
        self._event(source_event, base_event.shape)
        with torch.inference_mode():
            base = base_event.to(self.device).clone()
            source = source_event.to(self.device)
            if basis is None:
                result = source.clone()
            else:
                require(tuple(basis.shape) == (RANK, WIDTH) and bool(torch.isfinite(basis).all()),
                        "Invalid rank-16 source basis")
                native_basis = basis.to(self.device, dtype=torch.bfloat16)
                delta = source.unsqueeze(0) - base.unsqueeze(0)
                result = (base.unsqueeze(0) + (delta @ native_basis.T) @ native_basis)[0]
            self._event(result, base.shape)
            result = result.detach().cpu().clone()
        self.check()
        return result

    def load_basis(self, path, expected_sha):
        require(sha(path) == expected_sha, "Saved basis hash changed")
        with self.np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) in ({"rank_16"}, {"rank_16", "raw_randomized_pca", "cpu_raw_parameter"}),
                    "Unexpected original/reconstructed Qwen basis array keys")
            array = archive["rank_16"]
        require(array.shape == (RANK, WIDTH) and array.dtype == self.np.float32 and
                self.np.isfinite(array).all(), "Invalid saved basis shape/dtype/values")
        require(self.np.abs(array.astype(float) @ array.astype(float).T - self.np.eye(RANK)).max() <= 1e-5,
                "Saved basis is not orthonormal")
        return self.torch.tensor(array, dtype=self.torch.float32)

    def natural(self, encoded, capture_layers=DEFAULT_LAYERS, **checks):
        return self.run(encoded, capture_layers=capture_layers, **checks)

    def run(self, encoded, layer4_patch=None, replacements=None, capture_layers=DEFAULT_LAYERS,
            *, expected_prefix=None, expected_layer4=None, expected_replacements=None, clone=False):
        pass                                                                      

                                                                           
                                                                           
                                                                              
                                                                             
                                                                               
                                                                           
                                                                            
                                                                            
           
        self.check()
        require(self.call_count < self.native_call_limit, "Native inference call limit reached")
        capture_layers = tuple(capture_layers)
        replacements = {} if replacements is None else dict(replacements)
        expected_replacements = {} if expected_replacements is None else dict(expected_replacements)
        if expected_replacements:
            require(set(expected_replacements) == set(replacements),
                    "Pre-replacement guards must cover exactly the replacement layers")
        active_layers = layers_contract(capture_layers, replacements)
        ids = encoded["input_ids"]
        a, z = encoded["span"]
        require(0 <= a < z <= len(ids) < PADDING, "Invalid token count or event span")
        prefix = ids[:z]
        require(digest(ids) == encoded["input_ids_sha256"] and
                digest(prefix) == encoded["prefix_sha256"], "Encoded token sequence changed")
        if expected_prefix is not None:
            require(prefix == list(expected_prefix), "Pre-continuation token prefix changed across branches")
        if expected_layer4 is not None:
            self._event(expected_layer4, (z - a, WIDTH))
        requested = dict(replacements)
        if layer4_patch is not None:
            self._event(layer4_patch, (z - a, WIDTH))
        for tensor in requested.values():
            self._event(tensor, (z - a, WIDTH))
        for tensor in expected_replacements.values():
            self._event(tensor, (z - a, WIDTH))
        torch = self.torch
        replacements = {layer: tensor.detach().to(self.device).clone() for layer, tensor in requested.items()}
        initial = None if layer4_patch is None else layer4_patch.detach().to(self.device).clone()
        hashes = {str(layer): self.tensor_hash(tensor) for layer, tensor in replacements.items()}
        patch_hash = None if initial is None else self.tensor_hash(initial)
        self.call_count += 1
        call_id, started = self.call_count, time.monotonic()
        self.last_prepatch_layer4 = None
        captures, capture_hashes, handles = {}, {}, []
        self._journal(dict(event="forward_started", call_id=call_id,
                           input_ids_sha256=encoded["input_ids_sha256"], prefix_sha256=digest(prefix),
                           event_span=[a, z], input_tokens=len(ids), clone=bool(clone),
                           layer4_patch_sha256=patch_hash, replacement_sha256=hashes,
                           expected_recipient_sha256={str(layer): self.tensor_hash(tensor)
                                                      for layer, tensor in expected_replacements.items()},
                           capture_layers=list(capture_layers)))

        def make_hook(layer):
            def hook(_module, _inputs, output):
                self.check()
                hidden = output[0] if isinstance(output, tuple) else output
                segment = hidden[0, a:z]
                self._event(segment, (z - a, WIDTH))
                if layer == 4:
                    self.last_prepatch_layer4 = segment.detach().cpu().clone()
                    if expected_layer4 is not None:
                        expected = expected_layer4.to(self.device)
                        require(torch.equal(segment, expected), "Layer-4 native prefix tensor changed")
                if clone or (layer == 4 and initial is not None) or layer in replacements:
                    hidden = hidden.clone()
                    if layer == 4 and initial is not None:
                        hidden[0, a:z] = initial
                    if layer in replacements:
                        if layer in expected_replacements:
                            require(torch.equal(hidden[0, a:z], expected_replacements[layer].to(self.device)),
                                    "Pre-replacement recipient event changed across branches")
                        hidden[0, a:z] = replacements[layer]
                if layer in capture_layers:
                    captured = hidden[0, a:z].detach().cpu().clone()
                    self._event(captured, (z - a, WIDTH))
                    captures[layer] = captured
                    capture_hashes[str(layer)] = self.tensor_hash(captured)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            return hook

        try:
            input_ids = torch.tensor([ids + [self.tok.pad_token_id] * (PADDING - len(ids))], device=self.device)
            mask = torch.tensor([[1] * len(ids) + [0] * (PADDING - len(ids))], device=self.device)
            with torch.inference_mode():
                for layer in active_layers:
                    handles.append(self.layers[layer - 1].register_forward_hook(make_hook(layer)))
                output = self.model(input_ids=input_ids, attention_mask=mask, use_cache=False)
                log_probs = torch.log_softmax(output.logits[0, len(ids) - 1].float(), dim=-1)
                require(bool(torch.isfinite(log_probs).all()), "Nonfinite full-vocabulary log probabilities")
                choices = encoded["choices"]
                choice_ids = encoded["choice_ids"]
                values = {choice: float(log_probs[token]) for choice, token in zip(choices, choice_ids)}
                top = int(log_probs.argmax())
                global_prediction = next((choice for choice, token in zip(choices, choice_ids) if token == top), None)
                result = dict(prediction=max(choices, key=lambda choice: values[choice]), scores=values,
                              candidate_mass=math.fsum(math.exp(value) for value in values.values()),
                              global_token_id=top, global_token=self.tok.decode([top]),
                              global_prediction=global_prediction, global_choice_valid=global_prediction is not None,
                              choice_token_ids=dict(zip(choices, choice_ids)),
                              top10_tokens=[dict(token_id=int(token), token=self.tok.decode([int(token)]),
                                                 log_prob=float(log_probs[token])) for token in log_probs.topk(10).indices])
            require(set(captures) == set(capture_layers), "Requested capture hooks did not all execute")
            require(self.last_prepatch_layer4 is not None, "Layer-4 prefix guard did not execute")
            require(hashes == {str(layer): self.tensor_hash(tensor) for layer, tensor in replacements.items()} and
                    patch_hash == (None if initial is None else self.tensor_hash(initial)),
                    "Materialized intervention changed during execution")
            self.check()
            result["audit"] = dict(call_id=call_id, prefix_sha256=digest(prefix),
                                   input_ids_sha256=encoded["input_ids_sha256"], event_span=[a, z],
                                   layer4_patch_sha256=patch_hash, replacement_sha256=hashes,
                                   post_replacement_capture_sha256=capture_hashes,
                                   native_layer4_sha256=self.tensor_hash(self.last_prepatch_layer4),
                                   exact_recipient_guard_layers=sorted(expected_replacements),
                                   seconds=time.monotonic() - started,
                                   cuda_max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   cuda_max_memory_reserved_bytes=torch.cuda.max_memory_reserved())
            self._journal(dict(event="forward_returned", call_id=call_id, result=result))
            return result, captures
        except BaseException as error:
            self._journal(dict(event="forward_failed", call_id=call_id,
                               error_type=type(error).__name__, error=str(error)))
            raise
        finally:
            for handle in reversed(handles):
                handle.remove()
