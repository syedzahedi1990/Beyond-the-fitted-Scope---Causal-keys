import hashlib
import json
import os
from pathlib import Path
import numpy as np

from .runtime.load import require


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def append(handle, value):
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    handle.flush()
    os.fsync(handle.fileno())


def bits(engine, tensor):
    return tensor.detach().cpu().contiguous().view(engine.torch.uint16).numpy().copy()


def save_arrays(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    return {'path': path.name, 'sha256': sha(path), 'size_bytes': path.stat().st_size,
            'arrays': {key: {'shape': list(value.shape), 'dtype': value.dtype.str,
                             'raw_sha256': hashlib.sha256(value.tobytes()).hexdigest()}
                       for key, value in arrays.items()}}


def fingerprint(reference):
    return {'metadata': reference.metadata,
            'hashes': {str(layer) + '/' + kind: value for (layer, kind), value in sorted(reference.hashes.items())},
            'kind': getattr(reference, 'kind', 'relay_reference'),
            'provenance': getattr(reference, 'provenance', None)}


def save_reference(engine, directory, name, reference):
    arrays = {f'k_{layer}': bits(engine, value) for layer, value in reference.keys.items()}
    arrays.update({f'v_{layer}': bits(engine, value) for layer, value in getattr(reference, 'values', {}).items()})
    payload = save_arrays(directory / (name + '.npz'), arrays)
    write(directory / (name + '.json'), {'reference': fingerprint(reference), 'payload': payload,
          'omitted_output_prefix_arrays': [{'layer': layer, 'shape': list(value.shape),
                  'raw_sha256': engine.tensor_hash(value)} for layer, value in reference.output_prefixes.items()],
          'output_prefix_policy': 'Exact in-memory guards and hashes; O arrays not serialized. This compact run is not a standalone raw O-prefix audit.'})


def signature(result):
    return {key: value for key, value in result.items() if key != 'audit'}


def native_check(engine, result, captures, intent, expected):
    audit = result['audit']
    require(audit['call_id'] == intent['call_id'] == engine.call_count, 'Native call identity differs')
    require(all(audit[key] == intent[key] for key in ('input_ids_sha256', 'prefix_sha256', 'layer4_patch_sha256')),
            'Native input/patch binding differs')
    require(list(audit['event_span']) == intent['event_span'] and audit['replacement_sha256'] == {}
            and audit['exact_recipient_guard_layers'] == [] and set(captures) == {4}, 'Native scope differs')
    if intent['expected_native_layer4_sha256'] is not None:
        require(audit['native_layer4_sha256'] == intent['expected_native_layer4_sha256'], 'Base L4 bytes differ')
    actual = engine.tensor_hash(captures[4])
    require(audit['post_replacement_capture_sha256'] == {'4': actual}, 'Saved capture hash differs')
    if expected is not None:
        require(actual == engine.tensor_hash(expected), 'Fixed L4 patch differs across consumers')


def verify_journals(out, planned):
    count = 0
    with (out / 'rows.jsonl').open() as f, (out / 'INTENTS.jsonl').open() as g, (out / 'native_calls.jsonl').open() as h:
        rows, intents, native = ((json.loads(line) for line in handle) for handle in (f, g, h))
        require(next(native, {}).get('event') == 'engine_ready', 'Missing engine-ready event')
        for wanted in planned:
            row, intent, start, end = next(rows, None), next(intents, None), next(native, None), next(native, None)
            require(all(isinstance(x, dict) for x in (row, intent, start, end)), 'Missing returned row or native call')
            count += 1
            require(all(row[key] == value and intent[key] == value for key, value in wanted.items()), 'Scientific ledger differs')
            require(start['event'] == 'forward_started' and end['event'] == 'forward_returned'
                    and start['call_id'] == end['call_id'] == intent['call_id'] == row['call_id'] == count,
                    'Native ledger order differs')
            require(end['result'] == row['result'], 'Outer result differs from native return')
            require(all(start[key] == intent[key] == row['result']['audit'][key]
                    for key in ('input_ids_sha256', 'prefix_sha256', 'layer4_patch_sha256', 'event_span')),
                    'Native intended input differs')
            require(start['capture_layers'] == [4] and start['replacement_sha256'] == {}
                    and start['expected_recipient_sha256'] == {} and start['clone'] is False, 'Native operation scope differs')
        require(next(rows, None) is next(intents, None) is next(native, None) is None, 'Extra native rows')
    return count


def inventory(out):
    return {path.relative_to(out).as_posix(): {'sha256': sha(path), 'size_bytes': path.stat().st_size}
            for path in sorted(out.rglob('*')) if path.is_file() and path.name not in ('COMPLETE.json', 'FAILED.json')}
