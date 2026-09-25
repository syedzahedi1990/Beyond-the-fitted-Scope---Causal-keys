import hashlib
import importlib
import json
from pathlib import Path

from .encoding import encode_record, qualify_alphabet

GPU_ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_engine(model, *, model_path, model_receipt=None, native_call_limit,
                journal_path, deadline_seconds):
    require(model in ('mistral', 'qwen'), 'Unknown model profile')
    native = importlib.import_module('gpu.runtime.' + model + '_engine')
    class ReaderEngine(native.Engine):
        def encode(self, record, fmt='answer_prefill'):
            return encode_record(self.tok, record, fmt)
    receipt = model_receipt or (GPU_ROOT / 'component_data' / (model + '_assets.json') if model_path is not None else None)
    engine = ReaderEngine(model_path=model_path, model_receipt=receipt,
                          native_call_limit=native_call_limit, journal_path=journal_path,
                          deadline_seconds=deadline_seconds)
    engine.alphabet_qualification = qualify_alphabet(engine.tok)
    return engine


def basis_descriptor(model, cohort, objective, seed):
    registry = json.loads((GPU_ROOT / 'component_data/BASES.json').read_text())
    found = [x for x in registry['bases'] if
             (x['model'], x['cohort'], x['objective'], x['seed']) == (model, cohort, objective, seed)]
    require(len(found) == 1, 'Basis absent or ambiguous')
    return found[0]


def load_basis(model, cohort, objective, seed):
    import numpy as np
    item = basis_descriptor(model, cohort, objective, seed)
    relative = Path(item['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe basis path')
    path = GPU_ROOT / relative
    require(not path.is_symlink() and sha(path) == item['sha256'], 'Saved basis file differs')
    with np.load(path, allow_pickle=False) as archive:
        array = archive[item['array_key']].copy()
    require(array.dtype == np.float32 and list(array.shape) == item['shape']
            and np.isfinite(array).all(), 'Saved effective basis shape or dtype differs')
    require(hashlib.sha256(array.tobytes()).hexdigest() == item['tensor_sha256'],
            'Saved effective basis bytes differ')
    require(np.max(np.abs(array.astype(float) @ array.astype(float).T - np.eye(16))) <= 1e-5,
            'Saved basis is not orthonormal')
    return array, dict(item)


def load_relay(engine, model):
    module = importlib.import_module('gpu.runtime.' + model + '_relay')
    return module.RelayEdgeEngine(engine), module.RelayReference


def load_training_bases(directory, model, cohort):
    import numpy as np
    directory = Path(directory).resolve()
    require(not (directory / 'FAILED.json').exists(), 'Training run failed')
    marker_path = directory / 'COMPLETE.json'
    marker_sha = sha(marker_path)
    marker = json.loads(marker_path.read_text())
    spec = marker['specification']
    require(marker['status'] == 'COMPLETE' and spec['model'] == model and spec['cohort'] == cohort,
            'Training model/cohort differs')
    require(spec['seeds'] == [101, 102, 103] and spec['rank'] == 16 and spec['block'] == 4,
            'Training design differs')
    require(marker['gradient_qualification_records'] == 18, 'Training qualification missing')
    require(marker['artifacts'], 'Training inventory missing')
    for name, item in marker['artifacts'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe training inventory')
        path = directory / relative
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == item['size_bytes']
                and sha(path) == item['sha256'], 'Training artifact changed: ' + name)
    objectives = ('pca', 'f_star', 'm3') if cohort == 'original_1000' else ('pca', 'm1', 'm3')
    require(len(marker['bases']) == 9 and {(x['objective'], x['seed']) for x in marker['bases']} ==
            {(o, s) for o in objectives for s in (101, 102, 103)}, 'All nine paired bases required')
    result = {}
    for item in marker['bases']:
        require(item['file'] in marker['artifacts'] and marker['artifacts'][item['file']]['sha256'] == item['sha256'],
                'Basis is not bound by training inventory')
        with np.load(directory / item['file'], allow_pickle=False) as archive:
            require(archive.files == ['rank_16'], 'Unexpected trained basis arrays')
            array = archive['rank_16'].copy()
        require(array.dtype == np.float32 and array.shape == (16, 5120 if model == 'mistral' else 8192)
                and np.isfinite(array).all() and hashlib.sha256(array.tobytes()).hexdigest() == item['tensor_sha256'],
                'Trained effective basis changed')
        require(np.max(np.abs(array.astype(float) @ array.astype(float).T - np.eye(16))) <= 1e-5,
                'Trained basis is not orthonormal')
        result[item['objective'], item['seed']] = (array, dict(item, training_complete_sha256=marker_sha,
               pca_identity='exact_saved_this_rerun', source_kind='new_training_run'))
    require(sha(marker_path) == marker_sha, 'Training closure changed while reading')
    return result
