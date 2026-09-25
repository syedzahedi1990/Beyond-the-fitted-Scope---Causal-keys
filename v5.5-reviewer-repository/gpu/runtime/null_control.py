pass                                                                            
import numpy as np

def make_null(edge, recipient, other, orientation, provenance):
    engine = edge.engine
    plans, keys, norms = {}, {}, {}
    coordinates = {int(layer): i for i, layer in enumerate(orientation['layers'])}
    for layer in edge.profile.layers:
        i = coordinates[layer]
        r = recipient.keys[layer].float().numpy()
        d = other.keys[layer].float().numpy()
        delta = np.subtract(d, r, dtype=np.float32)
        rotated = np.multiply(np.take_along_axis(delta, orientation['permutation'][i], axis=-1),
                              orientation['signs'][i], dtype=np.float32)
        value = np.add(r, rotated, dtype=np.float32)
        keys[layer] = engine.torch.from_numpy(value.copy()).to(dtype=engine.torch.bfloat16)
        plans[f'key_{layer}'] = value
        actual = keys[layer].float().numpy().astype(np.float64)
        norm = lambda x: np.sqrt(np.sum(np.asarray(x, dtype=np.float64)**2, axis=-1)).tolist()
        norms[str(layer)] = {'original': norm(delta), 'rotated': norm(rotated),
                             'requested': norm(value.astype(np.float64) - r.astype(np.float64)),
                             'actual': norm(actual - r.astype(np.float64)),
                             'rounding': norm(actual - value.astype(np.float64))}
    reference = edge.make_key_donor(recipient, keys, {**provenance, 'norms_by_layer': norms})
    return reference, plans
