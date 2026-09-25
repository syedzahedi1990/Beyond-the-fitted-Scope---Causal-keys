import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ('direct', 'observer_table1', 'observer_table2', 'nonobserver_table1', 'nonobserver_table2')
LOCATIONS = ('box', 'basket', 'shelf', 'drawer', 'cabinet', 'closet')
COLORS = ('red', 'blue', 'green', 'yellow', 'black', 'white')
SEEDS = (101, 102, 103)
PERMUTATIONS = {'m1': (1, 2, 3, 4, 5, 0), 'm3': (1, 0, 3, 2, 5, 4)}
GROUPS = {'affected3': (0, 1, 2), 'observer_pair': (1, 2), 'nonobserver_pair': (3, 4), 'all5': (0, 1, 2, 3, 4)}
ARMS = ('self', 'other', 'null', 'base')
FIXED = ('K_P__V_T', 'K_M__V_T', 'K_S__V_T', 'K_T__V_T', 'K_NULL__V_T', 'K_B__V_B')
CI_METRICS = ('affected3/destination_id', 'affected3/aligned_oracle', 'affected3/endpoint_margin', 'affected3/aligned_oracle_margin', 'observer_pair/aligned_oracle_joint', 'observer_pair/destination_id_joint', 'nonobserver_pair/recipient_id_joint', 'nonobserver_pair/base_oracle_joint', 'full_pattern/destination_and_preservation', 'full_pattern/aligned_oracle_and_preservation', 'affected3/donor_error_copied', 'affected3/donor_error_repaired', 'affected3/donor_correct_damaged')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def semantic(core, mapping, role, view):
    value = core['targets'][mapping] if role == 'target' and 'targets' in core else core[role]
    if view == 0:
        return value
    value = value if view < 3 else core['initial']
    return COLORS[(LOCATIONS.index(value) + (view % 2 == 0)) % 6]


def stratum(core, mapping):
    target = core['targets'][mapping] if 'targets' in core else core['target']
    return 'base_equals_source' if core['base'] == core['source'] else 'base_equals_target' if core['base'] == target else 'all_distinct'


class Evidence:
    def __init__(self, metadata, rows):
        self.meta = metadata
        self.cores = metadata['cores']
        self.family = metadata['family']
        self.n = len(self.cores)
        self.maps = ('m1', 'm3') if self.family == 'mapping' else ('m3',)
        self.alphabet = np.array([[metadata['alphabets']['locations' if v == 0 else 'colors'][x] for x in (LOCATIONS if v == 0 else COLORS)] for v in range(5)], dtype=np.int64)
        require(len(set(self.alphabet[0]) | set(self.alphabet[1])) == 12, 'Alphabets are not disjoint')
        self.endpoints = {}
        self.exchanges = {}
        self.mask_frames = [(m['name'], f) for m in metadata['config']['masks'] for f in m['frames']] if self.family == 'native' else [('full', 'P'), ('full', 'M')]
        seen = set()
        counts = {'endpoint': 0, 'exchange': 0}
        for r in rows:
            call, core, mapping, seed, ep, mask, frame, condition, view, token, scores, mass = r
            require(0 <= core < self.n and mapping in self.maps and view in range(5), 'Invalid row coordinates')
            require(type(token) is int and token >= 0 and len(scores) == 6 and all(np.isfinite(scores)), 'Invalid saved result')
            require(seed in (None, *SEEDS), 'Unexpected fit')
            stream = 'endpoint' if ep else 'exchange'
            call_key = (stream if self.family == 'fixed_value' else 'combined', call)
            require(call_key not in seen, 'Duplicate original call ID')
            seen.add(call_key)
            fit = SEEDS.index(seed) if seed is not None else 0
            if ep:
                require(ep in ('B', 'S', 'T', 'P', 'M') and (seed is None) == (ep in ('B', 'S', 'T')), 'Endpoint/fit mismatch')
                key = (mapping, ep)
                dest = self.endpoints
                nf = 1 if seed is None else 3
            else:
                require(seed is not None and (mask, frame) in self.mask_frames and condition in (FIXED if self.family == 'fixed_value' else ARMS), 'Unexpected scientific condition')
                key = (mapping, mask, frame, condition)
                dest = self.exchanges
                nf = 3
            if key not in dest:
                dest[key] = {'tokens': np.full((self.n, nf, 5), -1, dtype=np.int64), 'scores': np.full((self.n, nf, 5, 6), np.nan), 'mass': np.full((self.n, nf, 5), np.nan)}
            item = dest[key]
            require(item['tokens'][core, fit, view] == -1, 'Duplicate semantic row')
            item['tokens'][core, fit, view] = token
            item['scores'][core, fit, view] = scores
            item['mass'][core, fit, view] = np.nan if mass is None else mass
            counts[stream] += 1
        require(counts == {k: metadata['counts'][k] for k in counts}, 'Return census differs')
        expected = {(m, ep) for m in self.maps for ep in ('B', 'S', 'T', 'P', 'M')}
        require(set(self.endpoints) == expected, 'Missing endpoint')
        conditions = FIXED if self.family == 'fixed_value' else ARMS
        expected = {(m, mask, f, a) for m in self.maps for mask, f in self.mask_frames for a in conditions}
        require(set(self.exchanges) == expected, 'Missing condition')
        for item in (*self.endpoints.values(), *self.exchanges.values()):
            require(np.all(item['tokens'] >= 0) and np.isfinite(item['scores']).all(), 'Dropped core, fit, or view')
        if self.family != 'fixed_value':
            for m, mask, f in [(m, mask, f) for m in self.maps for mask, f in self.mask_frames]:
                a, b = self.exchanges[m, mask, f, 'self'], self.endpoints[m, f]
                require(np.array_equal(a['tokens'], b['tokens']) and np.array_equal(a['scores'], b['scores']) and np.array_equal(a['mass'], b['mass']), 'Native self result changed')
        for c in self.cores:
            for m in self.maps:
                target = c['targets'][m] if 'targets' in c else c['target']
                require(target == LOCATIONS[PERMUTATIONS[m][LOCATIONS.index(c['source'])]], 'Symbolic mapping differs')

    def oracle_indices(self, mapping, role):
        return np.array([[(LOCATIONS if v == 0 else COLORS).index(semantic(c, mapping, role, v)) for v in range(5)] for c in self.cores])

    def oracle_tokens(self, mapping, role):
        ix = self.oracle_indices(mapping, role)
        return self.alphabet[np.arange(5), ix][:, None, :]


def metrics(evidence, mapping, frame, item):
    t, scores = item['tokens'], item['scores']
    recipient = evidence.endpoints[mapping, frame]['tokens']
    dest = evidence.endpoints[mapping, 'M' if frame == 'P' else 'P']['tokens']
    aligned = 'target' if frame == 'P' else 'source'
    oracle = {r: evidence.oracle_tokens(mapping, r) for r in ('base', 'source', 'target')}
    valid = np.any(t[..., None] == evidence.alphabet[None, None, :, :], axis=-1)
    values = {'destination_id': (t == dest).astype(float), 'recipient_id': (t == recipient).astype(float), 'aligned_oracle': (t == oracle[aligned]).astype(float), 'invalid': (~valid).astype(float), 'candidate_mass': item['mass']}
    for role, ep in [('base', 'B'), ('source', 'S'), ('target', 'T')]:
        values[role + '_oracle'] = (t == oracle[role]).astype(float)
        values[role + '_id'] = (t == evidence.endpoints[mapping, ep]['tokens']).astype(float)
        ix = evidence.oracle_indices(mapping, role)[:, None, :, None]
        values[role + '_oracle_score'] = np.take_along_axis(scores, np.broadcast_to(ix, (*t.shape, 1)), axis=-1)[..., 0]
    values['aligned_oracle_margin'] = (values['target_oracle_score'] - values['source_oracle_score']) * (1 if frame == 'P' else -1)
    selected = []
    for target in (dest, recipient):
        matches = target[..., None] == evidence.alphabet[None, None, :, :]
        score = np.sum(np.where(matches, scores, 0), axis=-1)
        selected.append(np.where(matches.any(axis=-1), score, np.nan))
    values['endpoint_margin'] = np.where(dest == recipient, 0., selected[0] - selected[1])
    correct = dest == oracle[aligned]
    fidelity, right = t == dest, t == oracle[aligned]
    values.update({'donor_correct': correct.astype(float), 'donor_error': (~correct).astype(float),
                   'donor_correct_retained': (correct & right).astype(float), 'donor_correct_damaged': (correct & ~right).astype(float),
                   'donor_error_copied': (~correct & fidelity).astype(float), 'donor_error_repaired': (~correct & right).astype(float),
                   'donor_error_other': (~correct & ~fidelity & ~right).astype(float)})
    result = {}
    for v, name in enumerate(VIEWS):
        for metric, array in values.items():
            result['view/' + name + '/' + metric] = array[:, :, v]
    for group, ix in GROUPS.items():
        for metric, array in values.items():
            result[group + '/' + metric] = np.mean(array[:, :, ix], axis=2)
        for metric in ('destination_id', 'recipient_id', 'aligned_oracle', 'base_id', 'base_oracle', 'source_oracle', 'target_oracle'):
            result[group + '/' + metric + '_joint'] = np.all(values[metric][:, :, ix], axis=2).astype(float)
    preserve = np.all(values['base_oracle'][:, :, 3:5], axis=2)
    result['full_pattern/destination_and_preservation'] = (np.all(fidelity[:, :, :3], axis=2) & preserve).astype(float)
    result['full_pattern/aligned_oracle_and_preservation'] = (np.all(right[:, :, :3], axis=2) & preserve).astype(float)
    dc, op, fp = np.all(correct[:, :, 1:3], axis=2), np.all(right[:, :, 1:3], axis=2), np.all(fidelity[:, :, 1:3], axis=2)
    for key, a in {'donor_correct_joint': dc, 'donor_error_joint': ~dc, 'donor_correct_retained_joint': dc & op, 'donor_correct_damaged_joint': dc & ~op, 'donor_error_copied_joint': ~dc & fp, 'donor_error_repaired_joint': ~dc & op, 'donor_error_other_joint': ~dc & ~fp & ~op}.items():
        result['observer_pair/' + key] = a.astype(float)
    direct_matches = t[:, :, 0, None] == evidence.alphabet[0]
    direct_index = direct_matches.argmax(axis=-1)
    inferred = direct_matches.any(axis=-1) & (t[:, :, 1] == evidence.alphabet[1][direct_index]) & (t[:, :, 2] == evidence.alphabet[2][(direct_index + 1) % 6])
    result['computation/direct_inferred_observer_pair'] = inferred.astype(float)
    result['computation/direct_inferred_pair_and_preservation'] = (inferred & preserve).astype(float)
    return result


def units(metric):
    if metric.endswith(('margin', 'score', 'candidate_mass')):
        return None
    if metric.endswith('_joint') or metric.startswith(('view/', 'full_pattern/', 'computation/')):
        return 1
    return len(GROUPS[metric.split('/')[0]])


class Bootstrap:
    def __init__(self, evidence, replicates=20000):
        self.e = evidence
        self.replicates = replicates
        self.indices = {}
        self.weights = {}
        self.receipt = {}
        seed = {'native': 2026092201, 'mapping': 2026092301, 'fixed_value': 2026091635}[evidence.family]
        rng = np.random.default_rng(seed)
        if evidence.family == 'fixed_value':
            labels = ('distinct', 'base_equals_source', 'base_equals_target')
        else:
            labels = (f'all{evidence.n}', 'all_distinct', 'base_equals_source', 'base_equals_target') + (('mapping_disagreement',) if evidence.family == 'mapping' else ())
        for population in labels:
            for mapping in evidence.maps:
                self.indices[mapping, population] = [i for i, c in enumerate(evidence.cores) if population == f'all{evidence.n}' or (population == 'mapping_disagreement' and c['mapping_disagreement']) or stratum(c, mapping) == ('all_distinct' if population == 'distinct' else population)]
            n = len(self.indices[evidence.maps[0], population])
            require(all(len(self.indices[m, population]) == n for m in evidence.maps), 'Unpaired population size')
            draws = rng.integers(0, n, size=(replicates, n), dtype=np.int64) if n else np.empty((replicates, 0), dtype=np.int64)
            weight = np.zeros((replicates, evidence.n))
            if n:
                local = np.zeros((replicates, n))
                np.add.at(local, (np.arange(replicates)[:, None], draws), 1.)
                local /= n
            for mapping in evidence.maps:
                w = weight.copy()
                if n:
                    w[:, self.indices[mapping, population]] = local
                self.weights[mapping, population] = w
            self.receipt[population] = {'n_cores': n, 'draws_sha256': hashlib.sha256(draws.tobytes()).hexdigest(), 'core_ids': {m: [evidence.cores[i]['id'] for i in self.indices[m, population]] for m in evidence.maps}}
        self.labels = labels
        if evidence.family == 'fixed_value':
            self.labels += ('full72',)
            for mapping in evidence.maps:
                self.indices[mapping, 'full72'] = list(range(evidence.n))
                self.weights[mapping, 'full72'] = sum(self.weights[mapping, p] * len(self.indices[mapping, p]) / evidence.n for p in labels)
        self.description = {'seed': seed, 'replicates': replicates, 'method': 'paired whole-core percentile, fixed-fit average before resampling', 'draw_order': list(labels), 'stratified_primary': evidence.family == 'fixed_value', 'draws': self.receipt}

    def summarize(self, arrays, mapping, seeds=SEEDS, intervals=False, populations=None):
        keys = list(arrays)
        stack = np.stack([arrays[k] for k in keys], axis=2)
        require(stack.shape[:2] == (self.e.n, len(seeds)), 'Core/fit dimensions differ')
        out = {}
        for pop in populations or self.labels:
            ix = self.indices[mapping, pop]
            data = stack[ix]
            mean = data.mean(axis=1)
            features = np.concatenate([stack.mean(axis=1)] + [stack[:, i] for i in range(len(seeds))], axis=1)
            ci = {}
            if intervals and ix:
                selected = [j for j, k in enumerate(keys) if k in CI_METRICS]
                col = [j + i * len(keys) for i in range(len(seeds) + 1) for j in selected]
                for start in range(0, len(col), 48):
                    batch = col[start:start + 48]
                    quantiles = np.quantile(self.weights[mapping, pop][:, ix] @ features[ix][:, batch], [.025, .975], axis=0, method='linear')
                    ci.update({j: quantiles[:, i] for i, j in enumerate(batch)})
            popout = {}
            for j, key in enumerate(keys):
                unit = units(key)
                available = bool(len(ix) and np.isfinite(data[:, :, j]).all())
                fit = []
                for f, seed in enumerate(seeds):
                    value = data[:, f, j]
                    ok = bool(len(ix) and np.isfinite(value).all())
                    q = (f + 1) * len(keys) + j
                    fit.append({'seed': seed, 'mean': float(value.mean()) if ok else None, 'count': float(value.sum() * unit) if ok and unit else None,
                                'denominator': len(ix) * unit if unit else None, 'available_cores': int(np.isfinite(value).sum()),
                                'ci95': [float(x) for x in ci[q]] if q in ci and ok else None})
                popout[key] = {'mean': float(mean[:, j].mean()) if available else None, 'ci95': [float(x) for x in ci[j]] if j in ci and available else None,
                               'n_cores': len(ix), 'fit_count': len(seeds), 'available_core_fit_values': int(np.isfinite(data[:, :, j]).sum()),
                               'units_per_core': unit, 'per_fit': fit}
            out[pop] = popout
        return out


def recode(evidence, item, mapping, table, inverse=False):
    perm = PERMUTATIONS[mapping]
    if inverse:
        perm = tuple(perm.index(i) for i in range(6))
    offset = table - 1
    color = tuple((perm[(i - offset) % 6] + offset) % 6 for i in range(6))
    t, scores = item['tokens'].copy(), item['scores'].copy()
    for v in range(5):
        p = perm if v == 0 else color
        old = item['tokens'][:, :, v]
        for i, to in enumerate(p):
            t[:, :, v][old == evidence.alphabet[v, i]] = evidence.alphabet[v, to]
            scores[:, :, v, to] = item['scores'][:, :, v, i]
    return {'tokens': t, 'scores': scores, 'mass': item['mass'].copy()}


def endpoint_metrics(evidence, mapping, endpoint):
    item = evidence.endpoints[mapping, endpoint]
    role = {'B': 'base', 'S': 'source', 'T': 'target', 'P': 'source', 'M': 'target'}[endpoint]
    out = {}
    for r in ('base', 'source', 'target'):
        good = (item['tokens'] == evidence.oracle_tokens(mapping, r)).astype(float)
        for v, name in enumerate(VIEWS):
            out['view/' + name + '/' + r + '_oracle'] = good[:, :, v]
        for group, ix in GROUPS.items():
            out[group + '/' + r + '_oracle'] = good[:, :, ix].mean(axis=2)
            out[group + '/' + r + '_oracle_joint'] = good[:, :, ix].all(axis=2).astype(float)
        if r == role:
            out['full_pattern/own_oracle'] = good.all(axis=2).astype(float)
    invalid = ~np.any(item['tokens'][..., None] == evidence.alphabet[None, None, :, :], axis=-1)
    for v, name in enumerate(VIEWS):
        out['view/' + name + '/invalid'] = invalid[:, :, v].astype(float)
    out['all5/invalid'] = invalid.mean(axis=2)
    return out


def color_function_bound(evidence, mapping, indices):
    assignments = np.indices((6,) * 6).reshape(6, -1).T
    wanted = evidence.oracle_indices(mapping, 'target')[indices, 1:3]
    source = evidence.oracle_indices(mapping, 'source')[indices, 1:3]
    candidate = evidence.endpoints[mapping, 'P']['tokens'][indices, :, 1:3]
    match = candidate[..., None] == evidence.alphabet[None, None, 1:3, :]
    inferred = match.argmax(axis=-1)
    valid = match.any(axis=-1).all(axis=-1)
    def maximum(inputs, good):
        scores = ((assignments[:, inputs[:, 0]] == wanted[None, :, 0]) & (assignments[:, inputs[:, 1]] == wanted[None, :, 1]) & good[None, :]).sum(axis=1)
        return int(scores.max())
    return {'n_cores': len(indices), 'source_location_counts': {label: sum(evidence.cores[i]['source'] == label for i in indices) for label in LOCATIONS},
            'actual_P_source_oracle_pairs_by_fit': {str(seed): int(((inferred[:, f] == source).all(axis=1) & valid[:, f]).sum()) for f, seed in enumerate(SEEDS)},
            'source_oracle_maximum_pairs': maximum(source, np.ones(len(indices), dtype=bool)),
            'actual_P_maximum_pairs_by_fit': {str(seed): maximum(inferred[:, f], valid[:, f]) for f, seed in enumerate(SEEDS)},
            'function_family': 'One arbitrary function from the six color labels to six colors, identical across both observer tables; no query/table access. Actual invalid P outputs cannot be repaired by this six-label function.',
            'interpretation': 'This is a finite empirical ceiling for the specified input panel/function family, not a universal bound for arbitrary symbolic programs.'}


def analyze(evidence, replicates=20000):
    bs = Bootstrap(evidence, replicates)
    output = {'schema_version': 1, 'study': evidence.meta['study'], 'family': evidence.family, 'model': evidence.meta['model'], 'counts': evidence.meta['counts'], 'bootstrap': bs.description, 'mappings': {}}
    contrasts_by_mapping = {}
    for mapping in evidence.maps:
        conditions, contrasts, recoders, vectors = {}, {}, {}, {}
        for mask, frame in evidence.mask_frames:
            arms = FIXED if evidence.family == 'fixed_value' else ARMS
            for arm in arms:
                name = '/'.join((mask, frame, arm))
                vector = metrics(evidence, mapping, frame, evidence.exchanges[mapping, mask, frame, arm])
                vectors[name] = vector
                conditions[name] = bs.summarize(vector, mapping)
            changed = ('K_M__V_T' if frame == 'P' else 'K_P__V_T') if evidence.family == 'fixed_value' else 'other'
            controls = [('self', 'K_P__V_T' if frame == 'P' else 'K_M__V_T'), ('null', 'K_NULL__V_T')] if evidence.family == 'fixed_value' else [('self', 'self'), ('null', 'null')]
            for label, control in controls:
                name = '/'.join((mask, frame, 'other_minus_' + label))
                a, b = vectors['/'.join((mask, frame, changed))], vectors['/'.join((mask, frame, control))]
                delta = {k: a[k] - b[k] for k in a}
                contrasts_by_mapping[mapping, name] = delta
                contrasts[name] = bs.summarize(delta, mapping, intervals=True)
        for frame in ('P', 'M'):
            for table in (1, 2):
                result = recode(evidence, evidence.endpoints[mapping, frame], mapping, table, inverse=frame == 'M')
                recoders[f'{frame}/table{table}'] = bs.summarize(metrics(evidence, mapping, frame, result), mapping)
        endpoints = {ep: bs.summarize(endpoint_metrics(evidence, mapping, ep), mapping, seeds=(None,) if ep in ('B', 'S', 'T') else SEEDS) for ep in ('B', 'S', 'T', 'P', 'M')}
        primary_population = 'distinct' if evidence.family == 'fixed_value' else f'all{evidence.n}'
        primary = {name: obj[primary_population]['affected3/destination_id'] for name, obj in contrasts.items() if name.startswith('full/')}
        output['mappings'][mapping] = {'conditions': conditions, 'contrasts': contrasts, 'endpoints': endpoints, 'output_recoders': recoders, 'primary': primary, 'primary_population': primary_population}
        if evidence.family == 'fixed_value':
            output['mappings'][mapping]['table_independent_color_function_bound'] = color_function_bound(evidence, mapping, bs.indices[mapping, 'distinct'])
    if evidence.family == 'mapping':
        output['paired_mapping_effect_differences_m1_minus_m3'] = {}
        for name in output['mappings']['m1']['contrasts']:
            a, b = contrasts_by_mapping['m1', name], contrasts_by_mapping['m3', name]
            output['paired_mapping_effect_differences_m1_minus_m3'][name] = bs.summarize({k: a[k] - b[k] for k in a}, 'm1', intervals=True, populations=('all108', 'mapping_disagreement'))
    output['interpretation'] = {'fidelity': 'Agreement with the actual opposite P/M global token ID, including errors and invalid IDs.', 'correctness': 'Independent symbolic location and color-table oracle.', 'preservation': 'Actual recipient agreement and base-oracle correctness are separate fields.', 'fixed_value_self_label': 'Same-key hybrid with fixed natural V_T; not a bare endpoint or native self identity.', 'intervals': 'Pointwise 95%; no multiplicity correction or efficacy gate; fixed trained fits, not training-seed population inference.', 'missing_scores': 'A missing actual-endpoint score makes the population/fit margin unavailable; no available-case filtering. Identical IDs cancel to zero.', 'raw_scope': 'Saved-prediction statistics only. No model execution or activation/tensor audit.'}
    return output


def load_evidence(data_dir, entry):
    path = data_dir / entry['path']
    require(path.parent == data_dir and sha(path) == entry['sha256'] and path.stat().st_size == entry['size_bytes'], 'Metadata digest differs')
    meta = read(path)
    file = data_dir / meta['data']['path']
    require(file.parent == data_dir and sha(file) == meta['data']['sha256'] and file.stat().st_size == meta['data']['size_bytes'], 'Evidence digest differs')
    with gzip.open(file, 'rt') as f:
        return Evidence(meta, (json.loads(line) for line in f))


def csv_tables(output, reports):
    with (output / 'mechanism_primary.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['study', 'mapping', 'population', 'contrast', 'mean_percent_points', 'lower_percent_points', 'upper_percent_points', 'fit101_percent_points', 'fit102_percent_points', 'fit103_percent_points'])
        for report in reports:
            for mapping, data in report['mappings'].items():
                for name, value in data['primary'].items():
                    w.writerow([report['study'], mapping, data['primary_population'], name, 100 * value['mean'], *[100 * x for x in value['ci95']], *[100 * x['mean'] for x in value['per_fit']]])
    with (output / 'mechanism_conditions.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['study', 'mapping', 'section', 'condition', 'population', 'metric', 'mean', 'fit', 'count', 'denominator', 'fit_mean', 'available_cores'])
        for report in reports:
            for mapping, data in report['mappings'].items():
                for section in ('conditions', 'endpoints', 'output_recoders', 'contrasts'):
                    for condition, populations in data[section].items():
                        for population, values in populations.items():
                            for metric, value in values.items():
                                for fit in value['per_fit']:
                                    w.writerow([report['study'], mapping, section, condition, population, metric, value['mean'], fit['seed'], fit['count'], fit['denominator'], fit['mean'], fit['available_cores']])


def validate_expected(report, expected):
    wanted = expected['studies'][report['study']]
    differences = []
    comparisons = 0
    def compare(actual, target, path):
        nonlocal comparisons
        comparisons += 1
        require((actual is None) == (target is None), 'Availability differs: ' + path)
        if actual is not None:
            error = abs(actual - target)
            require(error <= 2e-12, 'Published value differs: ' + path)
            if error:
                differences.append({'path': path, 'absolute_error': error})
    for path, value in wanted['primary'].items():
        mapping, name = path.split('/', 1)
        got = report['mappings'][mapping]['primary'][name]
        compare(got['mean'], value['mean'], path + '/mean')
        for i in range(2):
            compare(got['ci95'][i], value['ci95'][i], path + '/ci95/' + str(i))
        for i in range(3):
            compare(got['per_fit'][i]['mean'], value['per_fit'][i], path + '/fit/' + str(i))
    for check in wanted['checks']:
        path = '/'.join(check[k] for k in ('mapping', 'condition', 'population', 'metric'))
        got = report['mappings'][check['mapping']]['conditions'][check['condition']][check['population']][check['metric']]
        compare(got['mean'], check['mean'], path + '/mean')
        for i, count in enumerate(check['counts']):
            compare(got['per_fit'][i]['count'], count, path + '/count/' + str(i))
    return {'status': 'PASS', 'comparisons': comparisons, 'nonidentical_float_values': len(differences), 'max_absolute_error': max((x['absolute_error'] for x in differences), default=0.), 'absolute_tolerance': 2e-12, 'scope': 'Main pooled/per-fit estimates and intervals, all selected condition/count checks in expected/mechanism.json; not bitwise equality of historical reports.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', action='append')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/mechanism')
    parser.add_argument('--new-evidence', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'Output directory already exists')
    args.output.mkdir(parents=True)
    data_dir = args.data_dir.resolve()
    manifest = read(data_dir / 'MANIFEST.json')
    require(not args.new_evidence or data_dir != (ROOT / 'data/mechanism').resolve(), 'New-evidence mode requires a separate data directory')
    expected = None if args.new_evidence else read(ROOT / 'expected/mechanism.json')
    reports = []
    for entry in manifest['studies']:
        name = Path(entry['path']).stem
        if args.study and name not in args.study:
            continue
        evidence = load_evidence(data_dir, entry)
        report = analyze(evidence)
        report['validation'] = validate_expected(report, expected) if expected is not None else {'status': 'NEW_EVIDENCE', 'published_values_not_required': True}
        report['input_metadata_sha256'] = entry['sha256']
        report['implementation_sha256'] = sha(__file__)
        report['numpy_version'] = np.__version__
        write(args.output / (name + '.json'), report)
        reports.append(report)
        print(name + ': ' + str(evidence.meta['counts']['total']) + ' saved returns, all conditions retained', flush=True)
    require(bool(reports), 'No selected studies')
    csv_tables(args.output, reports)
    write(args.output / 'mechanism_summary.json', {'studies': {r['study']: {'counts': r['counts'], 'validation': r['validation'], 'primary': {m: x['primary'] for m, x in r['mappings'].items()}} for r in reports}, 'bootstrap_replicates': 20000, 'data_manifest_sha256': sha(data_dir / 'MANIFEST.json')})


if __name__ == '__main__':
    main()
