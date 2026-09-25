import gzip
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path


DATA = Path(__file__).resolve().parent / "behavior_data"
CHOICES = ("box", "basket", "shelf", "drawer", "cabinet", "closet")
MAPPINGS = {"f_star": (0, 1, 2, 3, 4, 5), "m1": (1, 2, 3, 4, 5, 0), "m3": (1, 0, 3, 2, 5, 4)}
FAMILIES = ("independent_template", "choice_order", "lexical_support", "distractor_structure", "joint_shift")
PREFIX = "Read the story and answer the question.\n\nStory: "
INITIAL = re.compile(r"(?:Everyone initially sees that the [A-Za-z0-9_ -]+ is in the [A-Za-z0-9_ -]+\."
                     r"|At first, everyone sees the [A-Za-z0-9_ -]+ at the [A-Za-z0-9_ -]+\.)\s*")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def datasets():
    with gzip.open(DATA / "datasets.json.gz", "rt") as stream:
        return json.load(stream)


def mapped(answer, objective):
    return CHOICES[MAPPINGS[objective][CHOICES.index(answer)]]


def schedule(seed, cohort):
    require(seed in (101, 102, 103), "Only the three published fit seeds are supported")
    rng = random.Random(seed + 16000)
    if cohort == "original_1000":
        result = list(range(1000))
        rng.shuffle(result)
        return result
    require(cohort == "mapping_300", "Unknown fitting cohort")
    return [rng.choice(range(1000)) for _ in range(300)]


def event_span(record):
    if "event_char_span" in record:
        start, stop = record["event_char_span"]
        require(isinstance(start, int) and isinstance(stop, int) and 0 <= start < stop, "Invalid explicit span")
        return start, stop
    story = record["story"]
    position = 0
    while match := INITIAL.match(story, position):
        position = match.end()
    require(0 < position < len(story), "Could not locate complete event after initial-state sentences")
    return len(PREFIX) + position, len(PREFIX) + len(story)


def prompt(record):
    require(len(record["choices"]) == 6 and set(record["choices"]) == set(CHOICES), "Location support changed")
    return (PREFIX + record["story"] + "\nQuestion: " + record["query"]["text"] +
            "\nChoices: " + ", ".join(record["choices"]) + "\nAnswer with exactly one choice.\nAnswer:")


def encode(tokenizer, record, interface, model, padding):
    require(interface in ("original", "answer_prefill"), "Unknown answer interface")
    raw = prompt(record)
    messages = [{"role": "user", "content": raw}]
    if model == "mistral":
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    wrapped = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    require(wrapped.count(raw) == 1, "Chat wrapper changed the story text")
    text = wrapped + ("Answer:" if interface == "answer_prefill" else "")
    start, stop = event_span(record)
    offset = wrapped.index(raw)
    tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = tokens["input_ids"]
    span = [i for i, (a, b) in enumerate(tokens["offset_mapping"]) if b > offset + start and a < offset + stop]
    require(span and span == list(range(span[0], span[-1] + 1)), "Noncontiguous event tokens")
    require(padding is None or len(ids) < padding, "Input exceeds fixed padding; truncation is forbidden")
    interval = (span[0], span[-1] + 1)
    require(tokenizer(wrapped, add_special_tokens=False)["input_ids"][:interval[1]] == ids[:interval[1]], "Prefill changed event prefix")
    choices = []
    for choice in record["choices"]:
        continuation = tokenizer(" " + choice, add_special_tokens=False)["input_ids"]
        require(len(continuation) == 1, "Candidate is not a single leading-space token")
        require(tokenizer(text + " " + choice, add_special_tokens=False)["input_ids"] == ids + continuation, "Appending a candidate changed the prefix")
        choices.append(continuation[0])
    require(len(set(choices)) == 6, "Candidate token identities collide")
    return {"input_ids": ids, "span": interval, "choice_ids": choices, "choices": record["choices"],
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def encode_pair(tokenizer, pair, interface, model, padding):
    require(pair["base"]["query"] == pair["source"]["query"], "Paired questions differ")
    require(pair["base"]["choices"] == pair["source"]["choices"], "Paired choices differ")
    base, source = [encode(tokenizer, pair[k], interface, model, padding) for k in ("base", "source")]
    require(base["span"] == source["span"], "Complete paired event spans differ")
    a, z = base["span"]
    require(base["input_ids"][:a] == source["input_ids"][:a] and base["input_ids"][z:] == source["input_ids"][z:], "Pair differs outside the event")
    return base, source


def same(left, right):
    difference = max(abs(left["scores"][c] - right["scores"][c]) for c in left["scores"])
    require(difference <= 1e-4 and left["global_token_id"] == right["global_token_id"] and
            left["prediction"] == right["prediction"], "Execution identity control failed")
    return difference


class BehaviorEngine:
    def __init__(self, native, model, padding):
        self.native = native
        self.model_name = model
        self.padding = padding
        for name in ("torch", "np", "tok", "model", "layers", "device", "environment"):
            setattr(self, name, getattr(native, name))
        self.width = self.model.config.hidden_size
        self.model.requires_grad_(False)
        self.model.eval()
        self.call_count = 0
        self.last_result = None
        self.journal = None

    def encode(self, record, interface):
        return encode(self.tok, record, interface, self.model_name, self.padding)

    def fixed(self, base, source, basis=None):
        require(base.shape == source.shape and base.ndim == 2 and base.shape[1] == self.width, "Unequal event tensors")
        require(base.dtype == source.dtype == self.torch.bfloat16, "Event tensors must be BF16")
        require(bool(self.torch.isfinite(base).all()) and bool(self.torch.isfinite(source).all()), "Nonfinite event")
        if basis is None:
            return source.clone()
        require(basis.shape == (16, self.width), "Rank or hidden width changed")
        effective = basis.to(device=base.device, dtype=base.dtype)
        delta = source.unsqueeze(0) - base.unsqueeze(0)
        projected = (delta @ effective.T) @ effective
        return (base.unsqueeze(0) + projected)[0]

    def tensor_hash(self, value):
        return hashlib.sha256(value.detach().contiguous().view(self.torch.uint8).cpu().numpy().tobytes()).hexdigest()

    def forward(self, encoded, fixed=None, basis=None, source=None, expected=None, expected_prefix=None, gradients=False, clone=False):
        require(time.monotonic() < self.native.deadline, "The fixed run deadline was reached")
        require(self.call_count < self.native.native_call_limit, "The fixed model-call limit was reached")
        torch = self.torch
        ids = encoded["input_ids"]
        a, z = encoded["span"]
        require(0 <= a < z <= len(ids), "Invalid complete-event span")
        require(fixed is None or source is None, "Choose a fixed or differentiable patch")
        require(expected_prefix is None or ids[:z] == expected_prefix, "Consumer changed the event prefix")
        length = self.padding or len(ids)
        require(len(ids) <= length, "Truncation is forbidden")
        input_ids = torch.tensor([ids + [self.tok.pad_token_id] * (length - len(ids))], device=self.device)
        mask = torch.tensor([[1] * len(ids) + [0] * (length - len(ids))], device=self.device)
        capture = {}
        self.call_count += 1
        if self.journal:
            self.journal({"event": "started", "call": self.call_count, "text_sha256": encoded["text_sha256"], "gradients": gradients})

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            event = hidden[0, a:z]
            require(event.shape == (z-a, self.width) and event.dtype == torch.bfloat16 and bool(torch.isfinite(event).all()), "Invalid hidden event")
            if expected is not None:
                require(event.shape == expected.shape and float((event.float() - expected.float()).abs().max()) <= 1e-4, "Event activation changed across branches")
            capture["event"] = event.detach().clone()
            if fixed is None and source is None and not clone:
                return output
            changed = hidden.clone()
            if source is not None:
                changed[:, a:z] = self.fixed(event, source, basis)
            elif fixed is not None:
                require(fixed.shape == event.shape and fixed.dtype == event.dtype and bool(torch.isfinite(fixed).all()), "Invalid materialized patch")
                changed[:, a:z] = fixed
            return (changed,) + output[1:] if isinstance(output, tuple) else changed

        handle = self.layers[3].register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(gradients):
                logits = self.model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits[0, len(ids)-1].float()
                log_probabilities = torch.log_softmax(logits, dim=-1)
                scores = log_probabilities[torch.tensor(encoded["choice_ids"], device=self.device)]
                require(bool(torch.isfinite(scores).all()), "Nonfinite candidate scores")
                top = int(log_probabilities.argmax())
                values = dict(zip(encoded["choices"], (float(x) for x in scores.detach())))
                selected = next((c for c, t in zip(encoded["choices"], encoded["choice_ids"]) if t == top), None)
                result = {"scores": values, "prediction": max(values, key=values.get), "global_token_id": top,
                          "global_prediction": selected, "global_choice_valid": selected is not None,
                          "choice_order": list(encoded["choices"]),
                          "candidate_mass": math.fsum(math.exp(v) for v in values.values()),
                          "choice_token_ids": dict(zip(encoded["choices"], encoded["choice_ids"]))}
                self.last_result = result
                if self.journal:
                    self.journal({"event": "returned", "call": self.call_count, "result": result})
                if gradients:
                    require(scores.requires_grad, "Intervention gradient is disconnected")
                    return scores
                return result, capture["event"], ids[:z]
        finally:
            handle.remove()
