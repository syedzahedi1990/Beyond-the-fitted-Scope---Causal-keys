# Compact evidence schema

All files use JSON schema version 1 and are gzip-compressed with deterministic headers. File and uncompressed-payload hashes are in `SOURCE_PROVENANCE.json`.

- `main.json.gz`: `primary_records` gives the shared ordered 1,500-pair metadata; `cores` gives both shared 120-story panels. Each `models[model].primary[interface]` array has dimensions `[pair, primary_methods, score_fields]`. Each `models[model].consequences[split].scores` array has dimensions `[core, views, consequence_methods, score_fields]`. Score fields are explicitly ordered as candidate label, global label (or null), candidate mass. Method and view orders are included. The method arrays retain clean/source and all nine basis conditions; consequence arrays additionally retain natural transformed-event controls.
- `supporting.json.gz`: `breadth` retains twelve fits with IID/shifted case records; `rank` retains all case records in five historical rank panels; `belief` retains complete cores, views, controls, expected labels and three-field scores; `route` retains original core data and all natural row keys/scores. Symbolic belief/route labels are independently checked by the replay.
- `mediation.json.gz`: each row retains its original `component`, `story_key`, `uid`, `view`, `seed`, `clean` and `target` labels. Condition score triples are `[target_log_probability, clean_log_probability, global_prediction]`. The seven primary conditions and all three controls remain separate. `source_sha256` hashes the complete original row before scalar projection.

The reference results are under `expected/behavior.json`, outside the evidence directory. No reference counts are inputs to metric computation.
