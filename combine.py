from fastai import *
from fastai.text import *
import torch

vocab_size = 1000
embedding_size = 3
n_hid = 70
n_layers = 3
lm = RNNCore(vocab_size, embedding_size, n_hid, n_layers, 0)
lm2 = RNNCore(vocab_size, embedding_size, n_hid, n_layers, 0)


def extract_embedding_layer(rnn_core):
    return rnn_core.encoder


class CombinedLM(nn.Module):

    def _combine_embedding(self, fwd, bwd):
        combined_embedding = extract_embedding_layer(self.core)
        fwd_embedding = extract_embedding_layer(fwd)
        bwd_embedding = extract_embedding_layer(bwd)
        combined_embedding.weight.data.set_(torch.cat((fwd_embedding.weight.data, bwd_embedding.weight.data), dim=1))

    def __init__(self, fwd: RNNCore, bwd: RNNCore):
        super().__init__()
        self.core = RNNCore(vocab_size, fwd.emb_sz + bwd.emb_sz,
                            fwd.n_hid + bwd.n_hid,
                            fwd.n_layers, 0)
        self._combine_embedding(fwd, bwd)


combined_lm = CombinedLM(lm, lm2)
