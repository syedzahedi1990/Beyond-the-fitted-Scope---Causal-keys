import argparse
import copy
import gzip
import json
import math
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from reproduce.integrity import verify
from .component_evidence import append, bits, digest, fingerprint, inventory, native_check, save_arrays, save_reference, sha, signature, verify_journals, write
from .component_plan import SEEDS, VIEWS, mapped_core, plan, read_configuration
from .runtime import color_task
from .runtime.hybrid_refs import COORDINATE_FIELDS, assemble_hybrid
from .runtime.key_engine import KeyOnlyEngine
from .runtime.load import GPU_ROOT, load_basis, load_engine, load_relay, load_training_bases, require
from .runtime.null_control import make_null

ROOT = GPU_ROOT.parent


def orientations():
    with np.load(GPU_ROOT / 'component_data/orientations.npz', allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    require(values['layers'].tolist() == list(range(5, 81)), 'Frozen orientation layers differ')
    require(values['permutation'].shape == values['signs'].shape == (76, 8, 128), 'Orientation shape differs')
    require(np.array_equal(np.sort(values['permutation'], axis=-1), np.broadcast_to(np.arange(128), (76, 8, 128)))
            and np.isin(values['signs'], [-1, 1]).all(), 'Orientation is not a signed coordinate permutation')
    return values


def preflight(engine, config, cores, out):
    result = {}
    with (out / 'ENCODED_INPUTS.jsonl').open('x') as handle:
        for core in cores:
            for mapping in config['mappings']:
                current = mapped_core(core, mapping)
                reference = None
                for role, field in (('B', 'base'), ('S', 'source'), ('T', 'target')):
                    direct = None
                    for view in VIEWS:
                        record = color_task.build_record(current, view, current[field])
                        encoded = engine.encode(record)
                        prefix = encoded['input_ids'][:encoded['span'][1]]
                        position = encoded['event_value_position']
                        if direct is None:
                            direct = (list(encoded['span']), position, prefix)
                        require(direct == (list(encoded['span']), position, prefix), 'Consumer prefix changed')
                        if reference is None:
                            reference = direct
                        require(reference[:2] == direct[:2] and len(reference[2]) == len(prefix) and
                                all(a == b for i, (a, b) in enumerate(zip(reference[2], prefix)) if i != position),
                                'Natural counterfactual changed more than the critical location token')
                        alphabet = engine.alphabet_qualification['location_token_ids' if view == 'direct' else 'color_token_ids']
                        require(dict(zip(encoded['choices'], encoded['choice_ids'])) == alphabet, 'Encoded alphabet differs')
                        result[core['id'], mapping, role, view] = encoded
                        append(handle, {'story_id': core['id'], 'mapping': mapping, 'role': role,
                                        'view': view, 'record': record, 'encoded': encoded})
    return result


def reference_snapshot(engine, reference):
    return {'identity': copy.deepcopy(fingerprint(reference)),
            'arrays': {field: {str(layer): engine.tensor_hash(value) for layer, value in getattr(reference, field, {}).items()}
                       for field in ('keys', 'values', 'output_prefixes')}}


def fixed_null(engine, recipient, other, orientation, provenance):
    def make_reference(coordinate, keys, notes):
        metadata = {field: copy.deepcopy(coordinate.metadata[field]) for field in COORDINATE_FIELDS}
        metadata['synthetic_provenance'] = dict(notes, kind='signed_coordinate_permutation', natural_capture=False,
                                               coordinate_compatibility_reference_id=notes['recipient_reference_id'])
        return type(coordinate)(metadata=metadata, keys=keys, values={}, output_prefixes={},
                                hashes={(layer, 'k_payload'): engine.tensor_hash(value) for layer, value in keys.items()})
    wrapper = SimpleNamespace(engine=engine, profile=SimpleNamespace(layers=tuple(range(5, len(engine.layers) + 1))),
                              make_key_donor=make_reference)
    return make_null(wrapper, recipient, other, orientation, provenance)


def basis_inputs(config, training_dir=None):
    trained = load_training_bases(training_dir, config['model'], config['basis_cohort']) if training_dir else None
    result = {}
    for seed in SEEDS:
        for objective in ('pca', *config['mappings']):
            result[objective, seed] = trained[objective, seed] if trained else load_basis(config['model'], config['basis_cohort'], objective, seed)
    return result


class Session:
    def __init__(self, engine, edge, config, cores, encoded, bases, orientation, output):
        self.engine, self.edge, self.config, self.cores = engine, edge, config, cores
        self.encoded, self.bases, self.orientation, self.out = encoded, bases, orientation, output
        self.rows = (output / 'rows.jsonl').open('x')
        self.intents = (output / 'INTENTS.jsonl').open('x')
        self.count = 0

    def close(self):
        self.rows.close()
        self.intents.close()

    def call(self, identity, encoded, patch, expected_native, expected_capture, *, capture=False,
             recipient=None, donor=None, groups=None, source_ids=None):
        engine = self.engine
        intent = dict(identity, call_id=engine.call_count + 1,
                      input_ids_sha256=encoded['input_ids_sha256'], prefix_sha256=encoded['prefix_sha256'],
                      event_span=list(encoded['span']), layer4_patch_sha256=None if patch is None else engine.tensor_hash(patch),
                      expected_native_layer4_sha256=None if expected_native is None else engine.tensor_hash(expected_native),
                      expected_capture_sha256=None if expected_capture is None else engine.tensor_hash(expected_capture),
                      source_references=source_ids, groups=groups,
                      operation='capture' if capture else 'exchange' if recipient is not None else 'natural_forward',
                      donor_fingerprint=None if donor is None else fingerprint(donor),
                      recipient_fingerprint=None if recipient is None else fingerprint(recipient))
        append(self.intents, intent)
        returned = result = captures = None
        try:
            if capture:
                result, captures, returned = self.edge.capture(encoded, layer4_patch=patch, expected_layer4=expected_native)
                receipt = self.edge.last_receipt
            elif recipient is not None:
                kwargs = dict(recipient=recipient, donor=donor, groups=groups,
                              layer4_patch=patch, expected_layer4=expected_native)
                if self.config['kind'] == 'fixed_value':
                    kwargs.update(mode='kv', restore_cutoff='critical_token')
                else:
                    kwargs.update(restoration='none')
                result, captures, receipt = self.edge.run(encoded, **kwargs)
            else:
                result, captures = engine.run(encoded, layer4_patch=patch, capture_layers=(4,),
                    expected_prefix=encoded['input_ids'][:encoded['span'][1]], expected_layer4=expected_native)
                receipt = None
            row = dict(identity, call_id=engine.call_count, result=result, edge_receipt=receipt)
            append(self.rows, row)
            self.count += 1
            native_check(engine, result, captures, intent, expected_capture)
            return result, captures, returned
        except BaseException:
            failed = self.out / ('RETURNED_FAILURE_' + str(engine.call_count))
            failed.mkdir(exist_ok=False)
            last_result = result if result is not None else getattr(self.edge, 'last_native_result', None)
            last_captures = captures if captures is not None else getattr(self.edge, 'last_native_captures', None)
            last_reference = returned if returned is not None else getattr(self.edge, 'last_native_reference', None)
            if last_result is not None and last_result['audit']['call_id'] != engine.call_count:
                last_result = last_captures = last_reference = None
            write(failed / 'receipt.json', {'intent': intent, 'last_native_result': last_result,
                  'last_edge_receipt': self.edge.last_receipt,
                  'native_journal_retains_returns_even_if_outer_validation_failed': True})
            if last_captures:
                save_arrays(failed / 'captures.npz', {str(k): bits(engine, v) for k, v in last_captures.items()})
            if last_reference is not None:
                save_reference(engine, failed, 'reference', last_reference)
            if engine.last_prepatch_layer4 is not None:
                save_arrays(failed / 'native_layer4.npz', {'event': bits(engine, engine.last_prepatch_layer4)})
            raise

    def core(self, original, mapping):
        engine, edge, config = self.engine, self.edge, self.config
        core = mapped_core(original, mapping)
        directory = self.out / 'tensors' / core['id'] / mapping
        directory.mkdir(parents=True, exist_ok=False)
        references, snapshots, natural4, patches, anchors, nulls, hybrids = {}, {}, {}, {}, {}, {}, {}
        common = {'story_id': core['id'], 'mapping': mapping}
        ref_id = lambda endpoint, seed: core['id'] + '/' + mapping + '/' + endpoint + ('' if seed is None else '_ts' + str(seed))
        try:
            for identity in plan(config, [original]):
                if identity['mapping'] != mapping:
                    continue
                seed, view = identity['seed'], identity['view']
                if identity['stage'] == 'endpoint':
                    endpoint = identity['endpoint']
                    key = (endpoint, seed)
                    role = endpoint if endpoint in ('B', 'S', 'T') else 'B'
                    encoded = self.encoded[core['id'], mapping, role, view]
                    if endpoint in ('P', 'M') and key not in patches:
                        objective = 'pca' if endpoint == 'P' else mapping
                        patch = engine.fixed(natural4['B'], natural4['S'], self.bases[objective, seed])
                        patches[key] = patch
                        save_arrays(directory / ('patch_' + endpoint + '_ts' + str(seed) + '.npz'), {'event_bf16': bits(engine, patch)})
                    patch = patches.get(key)
                    expected_native = natural4.get(role)
                    expected_capture = patch if patch is not None else natural4.get(role)
                    result, captures, reference = self.call(identity, encoded, patch, expected_native, expected_capture,
                                                             capture=view == 'direct')
                    anchors[endpoint, seed, view] = result
                    if view == 'direct':
                        if endpoint in ('B', 'S', 'T'):
                            natural4[endpoint] = captures[4].clone()
                            save_arrays(directory / ('natural_' + endpoint + '.npz'), {'event_bf16': bits(engine, captures[4])})
                        references[key] = reference
                        snapshots[key] = reference_snapshot(engine, reference)
                        save_reference(engine, directory, 'reference_' + endpoint + ('' if seed is None else '_ts' + str(seed)), reference)
                else:
                    frame, arm = identity['frame'], identity['arm']
                    key = (frame, seed)
                    opposite = 'M' if frame == 'P' else 'P'
                    recipient = references[key]
                    if key not in nulls:
                        provenance = {'kind': 'signed_coordinate_permutation', 'natural_capture': False,
                            'seed': 2026091610, 'recipient_reference_id': ref_id(frame, seed),
                            'donor_reference_id': ref_id(opposite, seed),
                            'orientation_sha256': sha(GPU_ROOT / 'component_data/orientations.npz')}
                        maker = fixed_null if config['kind'] == 'fixed_value' else lambda e, r, d, o, p: make_null(edge, r, d, o, p)
                        null_reference, plans = maker(engine, recipient, references[opposite, seed], self.orientation, provenance)
                        nulls[key] = null_reference
                        save_arrays(directory / ('null_' + frame + '_ts' + str(seed) + '_fp32.npz'), plans)
                        save_reference(engine, directory, 'null_' + frame + '_ts' + str(seed), null_reference)
                    if config['kind'] == 'fixed_value':
                        hybrid_key = (frame, seed, arm)
                        if hybrid_key not in hybrids:
                            k_name, v_name = arm.split('__')
                            k_name, v_name = k_name[2:], v_name[2:]
                            k_seed = seed if k_name in ('P', 'M') else None
                            k_ref = nulls[key] if k_name == 'NULL' else references[k_name, k_seed]
                            kid = 'synthetic/' + ref_id(frame, seed) if k_name == 'NULL' else ref_id(k_name, k_seed)
                            vid = ref_id(v_name, None)
                            origin = {'kind': 'signed_component', 'reference_id': kid, 'component': 'k',
                                      'recipient_reference_id': ref_id(frame, seed)} if k_name == 'NULL' else {'kind': 'captured', 'reference_id': kid}
                            donor, notes = assemble_hybrid(engine, recipient, k_ref, references[v_name, None],
                                recipient_id=ref_id(frame, seed), key_source_id=kid, value_source_id=vid,
                                key_origin=origin, value_origin={'kind': 'captured', 'reference_id': vid})
                            hybrids[hybrid_key] = donor
                            name = 'hybrid_' + frame + '_ts' + str(seed) + '_' + arm
                            save_reference(engine, directory, name, donor)
                            write(directory / (name + '_provenance.json'), notes)
                        donor = hybrids[hybrid_key]
                        donor_id = 'hybrid/' + ref_id(frame, seed) + '/' + arm
                    else:
                        donor = recipient if arm == 'self' else references[opposite, seed] if arm == 'other' else nulls[key] if arm == 'null' else references['B', None]
                        donor_id = ref_id(frame if arm == 'self' else opposite, seed) if arm in ('self', 'other') else 'synthetic/' + ref_id(frame, seed) if arm == 'null' else ref_id('B', None)
                    mask = next(m for m in config['masks'] if m['name'] == identity['mask'])
                    groups = {int(layer): tuple(values) for layer, values in mask['groups'].items()}
                    result, _, _ = self.call(identity, self.encoded[core['id'], mapping, 'B', view], patches[key],
                        natural4['B'], patches[key], recipient=recipient, donor=donor, groups=groups,
                        source_ids={'recipient': ref_id(frame, seed), 'donor': donor_id})
                    if arm == 'self':
                        require(signature(result) == signature(anchors[frame, seed, view]), 'Native self signature differs')
            require(all(reference_snapshot(engine, reference) == snapshots[key] for key, reference in references.items()),
                    'Captured reference mutated across conditions')
            print(json.dumps({'stage': 'core_complete', **common, 'native_calls': self.count}), flush=True)
        finally:
            references.clear()
            hybrids.clear()
            nulls.clear()


def export_scalars(out, config, cores, alphabet):
    data_dir = out / 'data'
    data_dir.mkdir(exist_ok=False)
    counts = {'endpoint': 0, 'exchange': 0}
    indices = {core['id']: i for i, core in enumerate(cores)}
    data_path = data_dir / 'rows.jsonl.gz'
    with data_path.open('xb') as raw, gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as target:
        with (out / 'rows.jsonl').open() as source:
            for line in source:
                row = json.loads(line)
                view = VIEWS.index(row['view'])
                result = row['result']
                labels = color_task.LOCATIONS if view == 0 else color_task.COLORS
                item = [row['call_id'], indices[row['story_id']], row['mapping'], row['seed'], row.get('endpoint'),
                        row.get('mask'), row.get('frame'), row.get('arm'), view, result['global_token_id'],
                        [result['scores'][label] for label in labels], result['candidate_mass']]
                target.write((json.dumps(item, separators=(',', ':'), allow_nan=False) + '\n').encode())
                counts[row['stage']] += 1
    require(sum(counts.values()) == config['native_calls'], 'Scalar export census differs')
    metadata = {'schema_version': 1, 'family': config['kind'], 'model': config['model'],
                'study': config['model'] + '_' + config['study'], 'cores': cores, 'seeds': list(SEEDS),
                'alphabets': {'locations': alphabet['location_token_ids'], 'colors': alphabet['color_token_ids']},
                'counts': dict(counts, total=sum(counts.values())), 'config': config,
                'data': {'path': data_path.name, 'sha256': sha(data_path), 'size_bytes': data_path.stat().st_size},
                'provenance': {'kind': 'new_gpu_rerun', 'untouched_native_rows_sha256': sha(out / 'rows.jsonl'),
                               'historical_numerical_identity_claimed': False}}
    meta_path = data_dir / 'study.json'
    write(meta_path, metadata)
    write(data_dir / 'MANIFEST.json', {'schema_version': 1, 'studies': [
        {'path': meta_path.name, 'sha256': sha(meta_path), 'size_bytes': meta_path.stat().st_size}]})


def execute(args):
    verify()
    require(math.isfinite(args.deadline_seconds) and args.deadline_seconds > 0, 'Finite positive deadline required')
    config_path = GPU_ROOT / 'component_configs' / (args.model + '_' + args.study + '.json')
    config, cores = read_configuration(config_path, sha(config_path))
    initial_release_sha = sha(ROOT / 'RELEASE.json')
    basis_records = basis_inputs(config, args.basis_dir)
    orientation = orientations()
    if args.plan_only:
        print(json.dumps({'status': 'PLAN_VERIFIED', 'config': config, 'basis_files': len(basis_records), 'model_calls': 0}, indent=2))
        return
    require(args.output is not None and not args.output.exists(), 'A new output directory is required')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    engine = session = None
    started = time.monotonic()
    try:
        write(out / 'CONFIGURATION.json', config)
        write(out / 'RUN_SPEC.json', {'configuration_sha256': sha(config_path), 'release_sha256': initial_release_sha,
              'derived_implementation': True, 'historical_bit_identity_claimed': False,
              'original_dataset': config['dataset'], 'dataset_sha256': config['dataset_sha256'],
              'basis_provenance': [item[1] for item in basis_records.values()],
              'orientation_sha256': sha(GPU_ROOT / 'component_data/orientations.npz'),
              'deadline_seconds': args.deadline_seconds, 'native_call_limit': config['native_calls'],
              'scientific_gate': None, 'all_conditions_retained': True})
        snapshot = out / 'source_snapshot'
        for path in [*sorted((GPU_ROOT / 'runtime').glob('*.py')), GPU_ROOT / 'components.py',
                     GPU_ROOT / 'component_plan.py', GPU_ROOT / 'component_evidence.py']:
            target = snapshot / path.relative_to(GPU_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.copyfile(ROOT / 'RELEASE.json', out / 'RELEASE_INPUT.json')
        shutil.copyfile(GPU_ROOT / config['dataset'], out / 'INPUT_DATA.json')
        for (objective, seed), (array, item) in basis_records.items():
            save_arrays(out / 'bases' / (objective + '_ts' + str(seed) + '.npz'), {'rank_16': array})
        engine = load_engine(args.model, model_path=args.model_path, model_receipt=args.model_receipt,
                             native_call_limit=config['native_calls'], journal_path=out / 'native_calls.jsonl',
                             deadline_seconds=args.deadline_seconds)
        write(out / 'ENVIRONMENT.json', engine.environment)
        write(out / 'ALPHABET_QUALIFICATION.json', engine.alphabet_qualification)
        encoded = preflight(engine, config, cores, out)
        edge = load_relay(engine, args.model)[0] if config['kind'] == 'fixed_value' else KeyOnlyEngine(engine, args.model)
        bases = {key: engine.torch.from_numpy(array.copy()) for key, (array, _) in basis_records.items()}
        session = Session(engine, edge, config, cores, encoded, bases, orientation, out)
        for core in cores:
            for mapping in config['mappings']:
                session.core(core, mapping)
        session.close()
        session = None
        require(engine.call_count == config['native_calls'], 'Incomplete finite native ledger')
        engine.close()
        require(verify_journals(out, plan(config, cores)) == config['native_calls'], 'Incomplete returned-call ledger')
        verify()
        require(sha(ROOT / 'RELEASE.json') == initial_release_sha, 'Release manifest changed during run')
        export_scalars(out, config, cores, engine.alphabet_qualification)
        write(out / 'COMPLETE.json', {'status': 'COMPLETE', 'stage': 'component_experiment_complete',
              'model': args.model, 'study': args.study, 'calls': config['native_calls'],
              'configuration_sha256': sha(out / 'CONFIGURATION.json'), 'all_conditions_retained': True,
              'scientific_gate': None, 'seconds': time.monotonic() - started,
              'artifacts': inventory(out), 'raw_scope': 'K/V payloads, L4 events and raw predictions retained; O-prefix arrays omitted with hashes and live guards retained.'})
    except BaseException as error:
        write(out / 'FAILED.json', {'status': 'FAILED', 'error_type': type(error).__name__, 'error': str(error),
              'native_calls_started': None if engine is None else engine.call_count,
              'returned_rows': None if session is None else session.count, 'artifacts': inventory(out)})
        raise
    finally:
        if session is not None:
            session.close()
        if engine is not None:
            engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('mistral', 'qwen'), required=True)
    parser.add_argument('--study', choices=('native-full', 'native-localized', 'fixed-value', 'mapping'), required=True)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--model-receipt', type=Path)
    parser.add_argument('--basis-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline-seconds', type=float, default=172800)
    parser.add_argument('--plan-only', action='store_true')
    execute(parser.parse_args())


if __name__ == '__main__':
    main()
