import pytest
import torch

# The tests use tiny tensors, for which a large intra-op thread pool only adds contention.
torch.set_num_threads(min(torch.get_num_threads(), 4))


@pytest.fixture
def float64():
    """Exact identities are checked in float64; the cluster tables take the default dtype."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)
