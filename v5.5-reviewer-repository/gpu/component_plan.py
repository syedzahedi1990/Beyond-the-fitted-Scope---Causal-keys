import hashlib
import json
from pathlib import Path

from .runtime.load import GPU_ROOT, require
from .runtime import color_task

VIEWS = ('direct', 'observer_table1', 'observer_table2', 'nonobserver_table1', 'nonobserver_table2')
SEEDS = (101, 102, 103)
PERMUTATIONS = {'m1': (1, 2, 3, 4, 5, 0), 'm3': (1, 0, 3, 2, 5, 4)}
ARMS = ('self', 'other', 'null', 'base')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapped_core(core, mapping):
    target = color_task.LOCATIONS[PERMUTATIONS[mapping][color_task.LOCATIONS.index(core['source'])]]
    if 'targets' in core:
        require(core['targets'][mapping] == target, 'Declared mapping target differs')
    elif mapping == 'm3':
        require(core['target'] == target, 'Declared pair-swap target differs')
    return dict(core, target=target)


def read_configuration(path, expected_sha):
    require(sha(path) == expected_sha, 'Configuration hash mismatch')
    config = json.loads(Path(path).read_text())
    require(config['schema_version'] == 1 and config['model'] in ('mistral', 'qwen'), 'Model/config schema')
    require(config['kind'] in ('native', 'mapping', 'fixed_value'), 'Unknown experiment kind')
    require(config['fit_seeds'] == list(SEEDS) and config['views'] == list(VIEWS), 'Fixed seeds/views differ')
    require(config['mappings'] == (['m1', 'm3'] if config['kind'] == 'mapping' else ['m3']), 'Mapping grid differs')
    require(config['basis_cohort'] == ('mapping_300' if config['kind'] == 'mapping' else 'original_1000'), 'Basis cohort differs')
    rel = Path(config['dataset'])
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe dataset')
    data_path = GPU_ROOT / rel
    require(sha(data_path) == config['dataset_sha256'], 'Dataset bytes differ')
    data = json.loads(data_path.read_text())
    cores = data['stories']
    expected_n = {'native': 120, 'mapping': 108, 'fixed_value': 72}[config['kind']]
    require(len(cores) == len({c['id'] for c in cores}) == config['n_cores'] == expected_n, 'Full panel required')
    for c in cores:
        for mapping in config['mappings']:
            g = mapped_core(c, mapping)
            for role in ('base', 'source', 'target'):
                direct = color_task.build_record(g, 'direct', g[role])
                end = direct['event_char_span'][1]
                for view in VIEWS:
                    record = color_task.build_record(g, view, g[role])
                    require(record['event_char_span'] == direct['event_char_span']
                            and color_task.prompt(record)[:end] == color_task.prompt(direct)[:end],
                            'Consumer changed critical prefix')
    maximum = 40 if config['model'] == 'mistral' else 80
    for mask in config['masks']:
        require(mask['frames'] and set(mask['frames']) <= {'P', 'M'}, 'Invalid mask frames')
        require(mask['groups'], 'Empty mask')
        for layer, groups in mask['groups'].items():
            require(6 <= int(layer) <= maximum and groups == list(range(8)), 'Invalid full-group block mask')
    require(len(list(plan(config, cores))) == config['native_calls'], 'Finite ledger count differs')
    return config, cores


def plan(config, cores):
    for core in cores:
        for mapping in config['mappings']:
            common = {'story_id': core['id'], 'mapping': mapping}
            for endpoint in ('B', 'S', 'T'):
                for view in VIEWS:
                    yield dict(common, stage='endpoint', endpoint=endpoint, seed=None, view=view)
            for seed in SEEDS:
                for endpoint in ('P', 'M'):
                    for view in VIEWS:
                        yield dict(common, stage='endpoint', endpoint=endpoint, seed=seed, view=view)
            for seed in SEEDS:
                for mask in config['masks']:
                    for frame in mask['frames']:
                        arms = ('K_P__V_T', 'K_M__V_T', 'K_S__V_T', 'K_T__V_T', 'K_NULL__V_T', 'K_B__V_B') if config['kind'] == 'fixed_value' else ARMS
                        for arm in arms:
                            for view in VIEWS:
                                yield dict(common, stage='exchange', seed=seed, mask=mask['name'],
                                           frame=frame, arm=arm, view=view)
