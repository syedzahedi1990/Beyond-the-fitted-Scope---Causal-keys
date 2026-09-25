import copy
import json
import unittest
from types import SimpleNamespace

import numpy as np

from reproduce import mechanism as m


def synthetic():
    cores = [{'id': str(i), 'source': 'box', 'target': 'basket', 'base': base, 'initial': 'closet'} for i, base in enumerate(('shelf', 'box', 'basket'))]
    alphabets = {'locations': dict(zip(m.LOCATIONS, range(10, 16))), 'colors': dict(zip(m.COLORS, range(20, 26)))}
    rows = []
    def add(c, seed, ep, frame, arm, v, role):
        label = m.semantic(cores[c], 'm3', role, v)
        labels = m.LOCATIONS if v == 0 else m.COLORS
        token = alphabets['locations' if v == 0 else 'colors'][label]
        scores = [-10.] * 6
        scores[labels.index(label)] = -.5
        rows.append([len(rows) + 1, c, 'm3', seed, ep, None if ep else 'full', frame, arm, v, token, scores, .8])
    for c in range(3):
        for ep, role in [('B', 'base'), ('S', 'source'), ('T', 'target')]:
            for v in range(5):
                add(c, None, ep, None, None, v, role)
        for seed in m.SEEDS:
            for ep, role in [('P', 'source'), ('M', 'target')]:
                for v in range(5):
                    add(c, seed, ep, None, None, v, role)
        for seed in m.SEEDS:
            for frame in ('P', 'M'):
                for arm in m.ARMS:
                    role = ('target' if frame == 'P' else 'source') if arm == 'other' else 'base' if arm == 'base' else ('source' if frame == 'P' else 'target')
                    for v in range(5):
                        add(c, seed, None, frame, arm, v, role)
    meta = {'study': 'synthetic', 'family': 'native', 'model': 'mistral', 'cores': cores, 'alphabets': alphabets, 'counts': {'endpoint': 135, 'exchange': 360, 'total': 495}, 'config': {'masks': [{'name': 'full', 'frames': ['P', 'M']}]}}
    return m.Evidence(meta, rows), meta, rows


class MechanismTests(unittest.TestCase):
    def test_independent_oracle_and_nonobserver(self):
        core = {'source': 'box', 'target': 'basket', 'base': 'shelf', 'initial': 'closet'}
        self.assertEqual([m.semantic(core, 'm3', 'target', v) for v in range(5)], ['basket', 'blue', 'green', 'white', 'red'])
        self.assertEqual(m.semantic(core, 'm3', 'source', 3), m.semantic(core, 'm3', 'target', 3))

    def test_all_endpoint_control_rows_required(self):
        _, meta, rows = synthetic()
        with self.assertRaisesRegex(ValueError, 'census'):
            m.Evidence(meta, rows[:-1])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            m.Evidence(meta, rows + [rows[0]])

    def test_self_control_and_actual_error_copy(self):
        e, _, _ = synthetic()
        item = copy.deepcopy(e.exchanges['m3', 'full', 'P', 'other'])
        e.endpoints['m3', 'M']['tokens'][0, 0, 0] = 999
        item['tokens'][0, 0, 0] = 999
        got = m.metrics(e, 'm3', 'P', item)
        self.assertEqual(got['view/direct/destination_id'][0, 0], 1.)
        self.assertEqual(got['view/direct/aligned_oracle'][0, 0], 0.)
        self.assertEqual(got['view/direct/invalid'][0, 0], 1.)
        self.assertEqual(got['view/direct/donor_error_copied'][0, 0], 1.)
        self.assertTrue(np.isnan(got['view/direct/endpoint_margin'][0, 0]))
        e.endpoints['m3', 'P']['tokens'][0, 0, 0] = 999
        got = m.metrics(e, 'm3', 'P', item)
        self.assertEqual(got['view/direct/endpoint_margin'][0, 0], 0.)

    def test_preservation_is_not_recipient_oracle(self):
        e, _, _ = synthetic()
        item = copy.deepcopy(e.exchanges['m3', 'full', 'P', 'other'])
        e.endpoints['m3', 'P']['tokens'][0, :, 3:5] = 999
        item['tokens'][0, :, 3:5] = 999
        got = m.metrics(e, 'm3', 'P', item)
        np.testing.assert_array_equal(got['nonobserver_pair/recipient_id_joint'][0], 1)
        np.testing.assert_array_equal(got['nonobserver_pair/base_oracle_joint'][0], 0)
        np.testing.assert_array_equal(got['full_pattern/destination_and_preservation'][0], 0)

    def test_six_cycle_inverse_is_not_twice(self):
        e, _, _ = synthetic()
        original = e.endpoints['m3', 'P']
        forward = m.recode(e, original, 'm1', 2)
        inverse = m.recode(e, forward, 'm1', 2, inverse=True)
        twice = m.recode(e, forward, 'm1', 2)
        np.testing.assert_array_equal(inverse['tokens'], original['tokens'])
        np.testing.assert_array_equal(inverse['scores'], original['scores'])
        self.assertFalse(np.array_equal(twice['tokens'], original['tokens']))

    def test_whole_core_fixed_fit_average(self):
        e, _, _ = synthetic()
        values = np.array([[1., 0., .5], [0., .5, 1.], [.25, 1., 0.]])
        bs = m.Bootstrap(e, 20000)
        got = bs.summarize({'affected3/destination_id': values}, 'm3', intervals=True)['all3']['affected3/destination_id']
        draws = np.random.default_rng(2026092201).integers(0, 3, size=(20000, 3), dtype=np.int64)
        manual = values.mean(axis=1)[draws].mean(axis=1)
        np.testing.assert_allclose(got['ci95'], np.quantile(manual, [.025, .975]), rtol=0, atol=1e-15)
        self.assertEqual(got['mean'], values.mean(axis=1).mean())

    def test_unavailable_margin_does_not_filter_or_leak_populations(self):
        e, _, _ = synthetic()
        bs = m.Bootstrap(e, 100)
        values = np.array([[1., 1., 1.], [np.nan, 0., 0.], [0., 0., 0.]])
        got = bs.summarize({'affected3/endpoint_margin': values}, 'm3', intervals=True)
        self.assertIsNone(got['all3']['affected3/endpoint_margin']['mean'])
        self.assertEqual(got['all_distinct']['affected3/endpoint_margin']['ci95'], [1., 1.])
        self.assertEqual(got['base_equals_source']['affected3/endpoint_margin']['available_core_fit_values'], 2)

    def test_stratified_full_panel_weights(self):
        e, _, _ = synthetic()
        e.family = 'fixed_value'
        bs = m.Bootstrap(e, 100)
        expected = sum(bs.weights['m3', p] / 3 for p in ('distinct', 'base_equals_source', 'base_equals_target'))
        np.testing.assert_array_equal(bs.weights['m3', 'full72'], expected)
        np.testing.assert_allclose(expected.sum(axis=1), 1.)

    def test_published_points_and_condition_counts(self):
        expected = m.read(m.ROOT / 'expected/mechanism.json')
        manifest = m.read(m.ROOT / 'data/mechanism/MANIFEST.json')
        for entry in manifest['studies']:
            e = m.load_evidence(m.ROOT / 'data/mechanism', entry)
            wanted = expected['studies'][e.meta['study']]
            bs = m.Bootstrap(e, 1)
            cache = {}
            for check in wanted['checks']:
                mapping = check['mapping']
                mask, frame, condition = check['condition'].split('/')
                key = mapping, mask, frame, condition
                if key not in cache:
                    cache[key] = m.metrics(e, mapping, frame, e.exchanges[key])
                ix = bs.indices[mapping, check['population']]
                array = cache[key][check['metric']][ix]
                self.assertAlmostEqual(array.mean(axis=1).mean(), check['mean'], places=12)
                np.testing.assert_allclose(array.sum(axis=0) * m.units(check['metric']), check['counts'], rtol=0, atol=2e-12)
            if e.family == 'fixed_value':
                bound = m.color_function_bound(e, 'm3', bs.indices['m3', 'distinct'])
                self.assertEqual(bound['source_location_counts'], {x: 8 for x in m.LOCATIONS})
                self.assertEqual(list(bound['actual_P_source_oracle_pairs_by_fit'].values()), [48, 48, 48])
                self.assertEqual(bound['source_oracle_maximum_pairs'], 24)
                self.assertEqual(list(bound['actual_P_maximum_pairs_by_fit'].values()), [24, 24, 24])


if __name__ == '__main__':
    unittest.main()
