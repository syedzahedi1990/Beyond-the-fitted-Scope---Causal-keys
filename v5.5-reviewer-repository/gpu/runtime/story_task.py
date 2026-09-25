from __future__ import annotations
import hashlib,json
from pathlib import Path

LOCATIONS = ("box", "basket", "shelf", "drawer", "cabinet", "closet")

PERM = dict(zip(LOCATIONS, ("basket", "box", "drawer", "shelf", "closet", "cabinet")))

VIEWS = ("direct", "world", "other_agent", "search", "observed_overwrite",
         "unobserved_overwrite", "irrelevant_object", "report")

ORDER_VIEWS = ("report_then_observe", "observe_then_report")

PREFIX = "Read the story and answer the question.\n\nStory: "

CORE_FIELDS = ("initial", "agent", "other", "object", "distractor", "distractor_location")

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def story_key(group):
    pass                                                                   

                                                                            
                                                                                
                                                                             
       
    return digest({key: group[key] for key in CORE_FIELDS})

def expected(group, view, event_value):
    pass                                                                          
    if event_value not in LOCATIONS:
        raise ValueError(f"Unknown event value: {event_value}")
    if view not in VIEWS + ORDER_VIEWS:
        raise ValueError(f"Unknown view: {view}")
    if view == "other_agent":
        return group["initial"]
    if view in ("observed_overwrite", "report_then_observe"):
        return group["new"]
    if view == "irrelevant_object":
        return group["distractor_location"]
    return event_value

def record(group, view, location=None):
    pass                                                                                         
    g = group
    a, b, o, d = (g[k] for k in ("agent", "other", "object", "distractor"))
    loc = g["base"] if location is None else location
    answer = expected(g, view, loc)
    initial = " ".join(f"Everyone initially sees that the {obj} is in the {where}."
                       for obj, where in sorted([(o, g["initial"]), (d, g["distractor_location"])]))
    event = f"{a} watches as the {o} is moved to the {loc}. {b} does not see this happen."
    story = initial + " " + event
    span = [len(PREFIX) + len(initial) + 1, len(PREFIX) + len(story)]
    value_start = span[0] + event.index("to the ") + len("to the ")
    value_span = [value_start, value_start + len(loc)]
    if view == "observed_overwrite":
        story += (f"\nNext, {a} watches as the {o} is moved to the {g['new']}. "
                  f"{b} does not see this later move. This is the final move of the {o}.")
    elif view == "unobserved_overwrite":
        story += (f"\nNext, {b} watches as the {o} is moved to the {g['new']}. "
                  f"{a} does not see this later move. This is the final move of the {o}.")
    elif view == "search":
        story += (f"\nAfter all events described above, {a} begins a new search for the {o}. "
                  f"{a} first checks the place where {a} currently believes the {o} is.")
    elif view == "report":
        story += (f"\nAfter the events above, {a} tells {b} the location where {a} currently believes the {o} is. "
                  f"{a} accurately communicates this current belief. "
                  f"{b} accepts the report and updates {b}'s belief accordingly. "
                  "No object moves during this conversation.")
    elif view in ORDER_VIEWS:
        report = (f"{a} tells {b} the location where {a} currently believes the {o} is. "
                  f"{a} accurately communicates this current belief. "
                  f"{b} accepts the report and updates {b}'s belief accordingly. "
                  "No object moves during this conversation.")
        observe = (f"{b} watches as the {o} is moved to the {g['new']}. "
                   f"{a} does not see this later move. This is the final move of the {o}.")
        first, second = (report, observe) if view == "report_then_observe" else (observe, report)
        story += f"\nNext, {first}\nAfter that, {second}"
    query = {
        "direct": f"Where does {a} believe the {o} is?",
        "world": f"Where is the {o} actually located?",
        "other_agent": f"Where does {b} believe the {o} is?",
        "search": f"Where does {a} check first in this new search?",
        "observed_overwrite": f"After both moves, where does {a} now believe the {o} is?",
        "unobserved_overwrite": f"After both moves, where does {a} now believe the {o} is?",
        "irrelevant_object": f"Where does {a} believe the {d} is?",
        "report": f"After this conversation, where does {b} now believe the {o} is?",
        "report_then_observe": f"After both later events, where does {b} now believe the {o} is?",
        "observe_then_report": f"After both later events, where does {b} now believe the {o} is?",
    }[view]
    return {"story": story, "query": {"text": query}, "choices": list(LOCATIONS),
            "event_char_span": span, "event_value": loc, "event_value_char_span": value_span,
            "answer": answer}

def prompt(rec):
    return (PREFIX + rec["story"] + "\nQuestion: " + rec["query"]["text"] +
            "\nChoices: " + ", ".join(rec["choices"]) + "\nAnswer with exactly one choice.\nAnswer:")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_record(core, view, value):
    require(view in VIEWS, 'Only the original eight views are allowed')
    adapted = dict(core, new=core['update'])
    return record(adapted, view, value)

def forecast(core, view, value):
    require(view in VIEWS, 'Only the original eight views are allowed')
    return expected(dict(core, new=core['update']), view, value)
