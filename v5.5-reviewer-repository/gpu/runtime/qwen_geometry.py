pass                                                                                  
MODEL = "Qwen/Qwen2.5-72B-Instruct"
REVISION = "495f39366efef23836d0cfae4fbe635880d2be31"
FIT_LAYER, WIDTH, PADDING, RANK = 4, 8192, 1024, 16
N_LAYERS, FIRST_LAYER = 80, 5
Q_HEADS, KV_HEADS, HEAD_DIM = 64, 8, 128
LAYERS = tuple(range(FIRST_LAYER, N_LAYERS + 1))
FULL = {layer: tuple(range(KV_HEADS)) for layer in range(FIT_LAYER + 2, N_LAYERS + 1)}
NATIVE_SOURCE_HASHES = {
    "modeling_qwen2.py": "99fa98c5676604cf6ef505892b70fda5c1c4cd835971459f38d61090cccab1e4",
    "sdpa_attention.py": "87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564",
}
MODEL_CONFIG_SHA256 = "14ca217334fe0fd10148413592d68c99eeb33431ed89c1afa130fee560be2a29"
