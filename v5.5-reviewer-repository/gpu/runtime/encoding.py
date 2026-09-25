from __future__ import annotations
import hashlib
from .mistral_engine import CHOICES,FORMATS,PADDING,require,digest
COLORS=("red","blue","green","yellow","black","white")

def prompt(record):
    choices = record["choices"]
    require(len(choices) == 6 and (set(choices) == set(CHOICES) or set(choices) == set(COLORS)),
            "Exactly one of the six-location or six-color answer alphabets is required")
    return ("Read the story and answer the question.\n\nStory: " + record["story"] +
            "\nQuestion: " + record["query"]["text"] + "\nChoices: " +
            ", ".join(choices) + "\nAnswer with exactly one choice.\nAnswer:")

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

def qualify_alphabet(tokenizer):
    pass                                                                   
    ids = {}
    for word in (*CHOICES, *COLORS):
        encoded = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
        require(len(encoded) == 1 and type(encoded[0]) is int,
                "Every fixed leading-space location/color must be exactly one token")
        ids[word] = encoded[0]
    require(len(set(ids.values())) == 12, "Color and location answer token IDs must all be distinct")
    return {"schema_version": 1, "locations": list(CHOICES), "colors": list(COLORS),
            "location_token_ids": {w: ids[w] for w in CHOICES},
            "color_token_ids": {w: ids[w] for w in COLORS},
            "all_twelve_leading_space_choices_single_token": True,
            "answer_alphabets_token_disjoint": True,
            "model_calls": 0, "substitutions": [],
            "append_and_prefix_checks": "Performed by encode_record on every actual record"}
