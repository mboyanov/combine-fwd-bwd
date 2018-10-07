from fastai import *
from fastai.text import *
import torch
import pytest
from combine import CombinedLM

vocab_size = 1000
embedding_size = 3
n_hid = 70
n_layers = 5


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
    reversed_input = torch.LongTensor([[3, 2, 1]]).t()
    # WHEN
    fwd.reset()
    fwd.train(False)
    fwd_outputs = fwd(inp)[1]

    bwd.reset()
    bwd.train(False)
    bwd_outputs = bwd(reversed_input)[1]

    combined.train(False)
    output = combined(inp)[1]

    # THEN
    assert output is not None

    for i in range(n_layers - 1):
        np.testing.assert_array_almost_equal(output[i].detach().numpy()[:, :, :n_hid], fwd_outputs[i].detach().numpy())
        bwd_expected = reversed(bwd_outputs[i].detach()).numpy()
        np.testing.assert_array_almost_equal(output[i].detach()[:, :, n_hid:], bwd_expected)

    np.testing.assert_array_almost_equal(output[-1].detach().numpy()[:, :, :embedding_size],
                                  fwd_outputs[-1].detach().numpy())
    bwd_expected = reversed(bwd_outputs[-1].detach()).numpy()
    np.testing.assert_array_almost_equal(output[-1].detach().numpy()[:, :, embedding_size:],
                                  bwd_expected)
