from __future__ import annotations
from . import story_task as _old
LOCATIONS,PERM,CORE_FIELDS=_old.LOCATIONS,_old.PERM,_old.CORE_FIELDS
PREFIX,prompt,story_key,digest,sha,require=_old.PREFIX,_old.prompt,_old.story_key,_old.digest,_old.sha,_old.require

COLORS = ("red", "blue", "green", "yellow", "black", "white")

PAIRS = {"observer": ("observer_table1", "observer_table2"),
         "nonobserver": ("nonobserver_table1", "nonobserver_table2")}

VIEWS = ("direct", "other_agent", *PAIRS["observer"], *PAIRS["nonobserver"])

COLOR_VIEWS = (*PAIRS["observer"], *PAIRS["nonobserver"])

ALPHABETS = {v: COLORS if v in COLOR_VIEWS else LOCATIONS for v in VIEWS}

TABLES = {"table1": dict(zip(LOCATIONS, COLORS)),
          "table2": dict(zip(LOCATIONS, COLORS[1:] + COLORS[:1]))}

INSTRUCTION = "Use this table to answer the question. Find the location where {queried_agent} believes the {object} is, then give the color assigned to that location."

QUERY = "Which color does the table assign to the location where {queried_agent} believes the {object} is?"

def forecast(core, view, value):
    require(view in VIEWS and value in LOCATIONS, "Unknown view or location-valued event")
    if view not in COLOR_VIEWS:
        return _old.forecast(core, view, value)
    queried, table = view.split("_")
    location = value if queried == "observer" else core["initial"]
    return TABLES[table][location]

def build_record(core, view, value):
    require(view in VIEWS and value in LOCATIONS, "Unknown view or location-valued event")
    if view not in COLOR_VIEWS:
        return _old.build_record(core, view, value)
    queried, table = view.split("_")
    agent = core["agent"] if queried == "observer" else core["other"]
    rec = _old.build_record(core, "direct", value)
    mapping = TABLES[table]
    rec["story"] += "\n" + INSTRUCTION.format(queried_agent=agent, object=core["object"])
    rec["story"] += "\n" + "\n".join(f"{location}: {mapping[location]}" for location in LOCATIONS)
    rec["query"]["text"] = QUERY.format(queried_agent=agent, object=core["object"])
    rec["choices"] = list(COLORS)
    rec["answer"] = forecast(core, view, value)
    return rec
