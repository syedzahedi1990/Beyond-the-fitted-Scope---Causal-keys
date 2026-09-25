import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gpu import components as c
from gpu.component_evidence import append, bits, sha, verify_journals, write
from gpu.component_plan import VIEWS, SEEDS, plan, read_configuration
from gpu.runtime.load import GPU_ROOT, load_basis, load_training_bases
from gpu.runtime import color_task
from gpu.runtime.key_engine import KeyReference
from gpu.runtime.mistral_relay import RelayReference
from reproduce.mechanism import load_evidence

try:
    import torch
except ImportError:
    torch = None


class PlanTests(unittest.TestCase):
    def test_all_fixed_ledgers_and_bundled_arrays(self):
        for path in sorted((GPU_ROOT / 'component_configs').glob('*.json')):
            config, cores = read_configuration(path, sha(path))
            rows = list(plan(config, cores))
            self.assertEqual(len(rows), config['native_calls'])
            self.assertEqual(sum(r['stage'] == 'endpoint' for r in rows), len(cores) * 45 * len(config['mappings']))
            self.assertEqual({r['seed'] for r in rows}, {None, 101, 102, 103})
        registry = json.loads((GPU_ROOT / 'component_data/BASES.json').read_text())
        self.assertEqual(len(registry['bases']), 36)
        for item in registry['bases']:
            array, descriptor = load_basis(item['model'], item['cohort'], item['objective'], item['seed'])
            self.assertEqual(hashlib.sha256(array.tobytes()).hexdigest(), descriptor['tensor_sha256'])

    def test_six_cycle_and_pair_swap_have_distinct_targets(self):
        config, cores = read_configuration(GPU_ROOT / 'component_configs/qwen_mapping.json', sha(GPU_ROOT / 'component_configs/qwen_mapping.json'))
        differences = sum(c.mapped_core(core, 'm1')['target'] != c.mapped_core(core, 'm3')['target'] for core in cores)
        self.assertEqual(differences, 54)
        for mapping in ('m1', 'm3'):
            triples = [c.mapped_core(core, mapping) for core in cores]
            self.assertEqual(sum(x['base'] == x['source'] for x in triples), 18)
            self.assertEqual(sum(x['base'] == x['target'] for x in triples), 18)

    def test_real_orientation_is_groupwise_and_reproducible(self):
        orientation = c.orientations()
        self.assertEqual(orientation['layers'].tolist(), list(range(5, 81)))
        self.assertEqual(orientation['permutation'].shape, (76, 8, 128))
        self.assertTrue(np.array_equal(orientation['permutation'], c.orientations()['permutation']))

    def test_new_training_inventory_binds_all_nine_effective_bases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bases, artifacts = [], {}
            for objective in ('pca', 'm1', 'm3'):
                for seed in SEEDS:
                    array, descriptor = load_basis('mistral', 'mapping_300', objective, seed)
                    relative = 'bases/' + objective + '_ts' + str(seed) + '.npz'
                    target = root / relative
                    target.parent.mkdir(exist_ok=True)
                    np.savez(target, rank_16=array)
                    artifacts[relative] = {'sha256': sha(target), 'size_bytes': target.stat().st_size}
                    bases.append({'objective': objective, 'seed': seed, 'file': relative, 'sha256': sha(target),
                                  'tensor_sha256': descriptor['tensor_sha256']})
            marker = {'status': 'COMPLETE', 'specification': {'model': 'mistral', 'cohort': 'mapping_300',
                      'seeds': list(SEEDS), 'rank': 16, 'block': 4}, 'gradient_qualification_records': 18,
                      'bases': bases, 'artifacts': artifacts}
            write(root / 'COMPLETE.json', marker)
            self.assertEqual(len(load_training_bases(root, 'mistral', 'mapping_300')), 9)
            with self.assertRaisesRegex(ValueError, 'model/cohort'):
                load_training_bases(root, 'qwen', 'mapping_300')
            path = root / bases[-1]['file']
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                load_training_bases(root, 'mistral', 'mapping_300')


class FakeEngine:
    def __init__(self, out):
        self.torch, self.device, self.layers = torch, 'cpu', [None] * 40
        self.call_count = 0
        self.journal = (out / 'native_calls.jsonl').open('x')
        append(self.journal, {'event': 'engine_ready'})
        self.alphabet_qualification = {
            'location_token_ids': dict(zip(color_task.LOCATIONS, range(100, 106))),
            'color_token_ids': dict(zip(color_task.COLORS, range(200, 206)))}
        self.last_prepatch_layer4 = None

    def close(self):
        self.journal.close()

    def encode(self, record):
        view = record['query']['text']
        ids = [10, 20, 100 + color_task.LOCATIONS.index(record['event_value']), 30, 40 + len(view)]
        choices = list(record['choices'])
        alphabet = self.alphabet_qualification['location_token_ids' if choices == list(color_task.LOCATIONS) else 'color_token_ids']
        return {'input_ids': ids, 'span': (1, 4), 'event_value_position': 2, 'choices': choices,
                'choice_ids': [alphabet[x] for x in choices], 'input_ids_sha256': c.digest(ids),
                'prefix_sha256': c.digest(ids[:4])}

    @staticmethod
    def tensor_hash(value):
        return hashlib.sha256(value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def fixed(self, base, source, basis):
        return (base.float() + (source.float() - base.float()) * basis[0, 0]).to(torch.bfloat16)

    def run(self, encoded, layer4_patch=None, capture_layers=(4,), expected_prefix=None, expected_layer4=None):
        self.call_count += 1
        base = torch.full((3, 16), float(encoded['input_ids'][2]), dtype=torch.bfloat16)
        self.last_prepatch_layer4 = base.clone()
        if expected_layer4 is not None:
            c.require(torch.equal(base, expected_layer4), 'Fake native prefix mismatch')
        actual = base if layer4_patch is None else layer4_patch.clone()
        scope = {'call_id': self.call_count, 'input_ids_sha256': encoded['input_ids_sha256'],
                 'prefix_sha256': encoded['prefix_sha256'], 'event_span': list(encoded['span']),
                 'layer4_patch_sha256': None if layer4_patch is None else self.tensor_hash(layer4_patch),
                 'replacement_sha256': {}}
        append(self.journal, dict(scope, event='forward_started', capture_layers=[4], clone=False, expected_recipient_sha256={}))
        result = {'global_token_id': 999, 'global_prediction': None, 'scores': {x: -float(i + 2) for i, x in enumerate(encoded['choices'])},
                  'candidate_mass': .2, 'audit': dict(scope, native_layer4_sha256=self.tensor_hash(base),
                   post_replacement_capture_sha256={'4': self.tensor_hash(actual)}, exact_recipient_guard_layers=[])}
        append(self.journal, {'event': 'forward_returned', 'call_id': self.call_count, 'result': result})
        return result, {4: actual}


class FakeEdge:
    def __init__(self, engine, fixed=False, corrupt_self=False):
        self.engine, self.fixed, self.corrupt_self = engine, fixed, corrupt_self
        self.profile = SimpleNamespace(layers=tuple(range(5, 41)))
        self.last_receipt = None

    def capture(self, encoded, layer4_patch=None, expected_layer4=None):
        result, captures = self.engine.run(encoded, layer4_patch=layer4_patch, expected_layer4=expected_layer4)
        metadata = {'schema_version': 1, 'span': [1, 4], 'value_position': 2,
                    'prefix_ids': encoded['input_ids'][:4], 'source_identity': {'fake': 1},
                    'adapter_provenance': {'fake': True}}
        scalar = float(captures[4][0, 0])
        keys = {layer: torch.full((8, 128), scalar + layer, dtype=torch.bfloat16) for layer in self.profile.layers}
        values = {layer: torch.full((8, 128), scalar - layer, dtype=torch.bfloat16) for layer in self.profile.layers}
        hashes = {(layer, 'k_payload'): self.engine.tensor_hash(v) for layer, v in keys.items()}
        outputs = {layer: torch.ones((4, 16), dtype=torch.bfloat16) for layer in self.profile.layers}
        self.last_receipt = {'status': 'COMPLETE', 'mode': 'capture'}
        if self.fixed:
            hashes.update({(layer, 'v_payload'): self.engine.tensor_hash(v) for layer, v in values.items()})
            ref = RelayReference(metadata, keys, values, outputs, hashes)
        else:
            ref = KeyReference(metadata, keys, outputs, hashes, 'captured', {})
        return result, captures, ref

    def make_key_donor(self, recipient, keys, provenance):
        return KeyReference(copy.deepcopy(recipient.metadata), {k: v.clone() for k, v in keys.items()}, {},
                            {(layer, 'k_payload'): self.engine.tensor_hash(v) for layer, v in keys.items()}, 'synthetic_keys', provenance)

    def run(self, encoded, *, recipient, donor, groups, layer4_patch, expected_layer4, **kwargs):
        if self.fixed:
            c.require(kwargs == {'mode': 'kv', 'restore_cutoff': 'critical_token'}, 'Fixed-value scope mismatch')
        else:
            c.require(kwargs == {'restoration': 'none'}, 'Native path scope mismatch')
        c.require(groups and all(tuple(g) == tuple(range(8)) for g in groups.values()), 'Selection differs')
        result, captures = self.engine.run(encoded, layer4_patch=layer4_patch, expected_layer4=expected_layer4)
        if self.corrupt_self and donor is recipient:
            result = dict(result, global_token_id=998)
        self.last_receipt = {'status': 'COMPLETE', 'null_is_explicit': getattr(donor, 'kind', None) == 'synthetic_keys'}
        return result, captures, self.last_receipt


@unittest.skipIf(torch is None, 'CPU Torch optional; numerical GPU execution is not exercised')
class TensorLedgerTests(unittest.TestCase):
    def execute_fixture(self, study, corrupt_self=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        out = Path(temporary.name)
        config, cores = read_configuration(GPU_ROOT / ('component_configs/mistral_' + study + '.json'),
                                          sha(GPU_ROOT / ('component_configs/mistral_' + study + '.json')))
        cores = cores[:1]
        config = dict(config, n_cores=1)
        config['native_calls'] = len(list(plan(config, cores)))
        engine = FakeEngine(out)
        self.addCleanup(lambda: engine.close() if not engine.journal.closed else None)
        edge = FakeEdge(engine, fixed=study == 'fixed-value', corrupt_self=corrupt_self)
        encoded = c.preflight(engine, config, cores, out)
        bases = {(o, s): torch.full((16, 16), .4 if o == 'pca' else .7) for s in SEEDS for o in ('pca', *config['mappings'])}
        session = c.Session(engine, edge, config, cores, encoded, bases, c.orientations(), out)
        self.addCleanup(session.close)
        for core in cores:
            for mapping in config['mappings']:
                session.core(core, mapping)
        session.close()
        engine.close()
        return out, config, cores, engine

    def test_native_and_mapping_invalid_outputs_never_filter_cases(self):
        for study, expected in (('native-full', 165), ('native-localized', 525), ('mapping', 330)):
            out, config, cores, engine = self.execute_fixture(study)
            self.assertEqual(engine.call_count, expected)
            self.assertEqual(verify_journals(out, plan(config, cores)), expected)
            c.export_scalars(out, config, cores, engine.alphabet_qualification)
            entry = json.loads((out / 'data/MANIFEST.json').read_text())['studies'][0]
            panel = load_evidence(out / 'data', entry)
            self.assertTrue(all((item['tokens'] == 999).all() for item in panel.exchanges.values()))

    def test_fixed_value_all_six_sources_replay_with_compact_schema(self):
        out, config, cores, engine = self.execute_fixture('fixed-value')
        self.assertEqual(engine.call_count, 225)
        self.assertEqual(verify_journals(out, plan(config, cores)), 225)
        c.export_scalars(out, config, cores, engine.alphabet_qualification)
        entry = json.loads((out / 'data/MANIFEST.json').read_text())['studies'][0]
        panel = load_evidence(out / 'data', entry)
        self.assertEqual(len(panel.exchanges), 12)
        descriptor = next((out / 'tensors').rglob('hybrid_P_ts101_K_NULL__V_T.json'))
        synthetic = json.loads(descriptor.read_text())
        self.assertEqual(synthetic['omitted_output_prefix_arrays'], [])
        self.assertFalse(synthetic['reference']['metadata']['synthetic_provenance']['natural_capture'])

    def test_self_mismatch_rejected_after_return_saved(self):
        with self.assertRaisesRegex(ValueError, 'self signature'):
            self.execute_fixture('native-full', corrupt_self=True)

    def test_return_or_scope_corruption_rejected(self):
        out, config, cores, engine = self.execute_fixture('native-full')
        path = out / 'rows.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]['result']['global_token_id'] = 777
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'differs from native'):
            verify_journals(out, plan(config, cores))

    def test_null_plan_single_bf16_rounding_and_exact_sign_permutation(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        engine = FakeEngine(Path(temporary.name))
        self.addCleanup(engine.close)
        edge = FakeEdge(engine, fixed=True)
        layers = range(5, 41)
        rng = np.random.default_rng(101)
        def ref():
            keys = {l: torch.from_numpy(rng.normal(size=(8, 128)).astype(np.float32)).to(torch.bfloat16) for l in layers}
            return RelayReference({field: {} for field in c.COORDINATE_FIELDS}, keys, {}, {}, {})
        recipient, other = ref(), ref()
        before = {l: engine.tensor_hash(t) for l, t in recipient.keys.items()}
        orientation = c.orientations()
        null, plans = c.fixed_null(engine, recipient, other, orientation, {'seed': 2026091610, 'recipient_reference_id': 'P'})
        for l in layers:
            r, d = recipient.keys[l].float().numpy(), other.keys[l].float().numpy()
            delta = np.subtract(d, r, dtype=np.float32)
            expected = np.add(r, np.take_along_axis(delta, orientation['permutation'][l - 5], -1) * orientation['signs'][l - 5], dtype=np.float32)
            self.assertTrue(np.array_equal(plans['key_' + str(l)], expected))
            self.assertTrue(torch.equal(null.keys[l], torch.from_numpy(expected).to(torch.bfloat16)))
            self.assertEqual(engine.tensor_hash(recipient.keys[l]), before[l])
        self.assertEqual(null.values, {})
        self.assertEqual(null.output_prefixes, {})

    def test_post_return_failure_retains_relay_capture_without_last_native_attributes(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            engine = FakeEngine(out)
            edge = FakeEdge(engine, fixed=True)
            session = c.Session(engine, edge, {'kind': 'fixed_value'}, [], {}, {}, {}, out)
            self.addCleanup(session.close)
            self.addCleanup(engine.close)
            config, cores = read_configuration(GPU_ROOT / 'component_configs/mistral_fixed-value.json', sha(GPU_ROOT / 'component_configs/mistral_fixed-value.json'))
            record = color_task.build_record(cores[0], 'direct', cores[0]['base'])
            encoded = engine.encode(record)
            identity = next(plan(config, cores[:1]))
            with patch.object(c, 'native_check', side_effect=ValueError('post-return mismatch')):
                with self.assertRaisesRegex(ValueError, 'post-return mismatch'):
                    session.call(identity, encoded, None, None, None, capture=True)
            failure = out / 'RETURNED_FAILURE_1'
            self.assertTrue((failure / 'reference.npz').is_file())
            self.assertTrue((failure / 'captures.npz').is_file())
            self.assertEqual(json.loads((failure / 'receipt.json').read_text())['last_native_result']['global_token_id'], 999)


if __name__ == '__main__':
    unittest.main()
