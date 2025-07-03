import numpy as np

def fuse_features(audio_feat, text_feat):
    return np.concatenate([audio_feat, text_feat[0]])
