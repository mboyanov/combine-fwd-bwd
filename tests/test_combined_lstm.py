from fastai import *
from fastai.text import *
import torch
import pytest
from combine import CombinedLM

vocab_size = 1000
embedding_size = 3
n_hid = 70
n_layers = 1


@pytest.fixture
def fwd():
    return RNNCore(vocab_size, embedding_size, n_hid, n_layers, 0)


@pytest.fixture
def bwd():
    return RNNCore(vocab_size, embedding_size, n_hid, n_layers, 0)


def test_should_combine_embeddings(fwd, bwd):
    # GIVEN
    combined = CombinedLM(fwd, bwd)
    embedding = combined.core.encoder
    inp = torch.LongTensor([1])
    # WHEN
    combined_embedding = embedding(inp)
    # THEN
    np.testing.assert_array_equal(combined_embedding.detach().numpy()[:, :embedding_size],
                                  fwd.encoder(inp).detach().numpy())
    np.testing.assert_array_equal(combined_embedding.detach().numpy()[:, embedding_size:],
                                  bwd.encoder(inp).detach().numpy())


def test_should_combine_rnns(fwd, bwd):
    # GIVEN
    combined = CombinedLM(fwd, bwd)
    inp = torch.LongTensor([[1, 2, 3]]).t()
    # WHEN
    combined.train(False)
    output = combined(inp)
    # THEN
    assert output is not None
