from fastai import *
from fastai.text import *
import torch

vocab_size = 1000


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
                            fwd.n_layers, 0, bidir=True)
        self._combine_embedding(fwd, bwd)
        self._combine_rnns(fwd, bwd)

    def _combine_rnns(self, fwd, bwd):
        combined_rnn_0 = self.core.rnns[0]
        fwd_rnn_0 = fwd.rnns[0]
        bwd_rnn_0 = bwd.rnns[0]
        fwd_weights = fwd_rnn_0.module.weight_ih_l0.data
        bwd_weights = bwd_rnn_0.module.weight_ih_l0.data
        combined_rnn_0.module.weight_ih_l0.data.set_(
            torch.cat((
                fwd_weights,
                torch.zeros(fwd_weights.size())
            ), dim =1)
        )
        combined_rnn_0.module.weight_ih_l0_reverse.data.set_(
            torch.cat((
                torch.zeros(bwd_weights.size()),
                bwd_weights
            ), dim=1)
        )

    def forward(self, input):
        self.core.reset()
        return self.core.forward(input)

    def reset(self):
        self.core.reset()
