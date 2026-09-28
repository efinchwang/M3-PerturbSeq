import torch
import torch.nn.functional as F


def test_query_labels_have_zero_direct_condition_gradient():
    torch.manual_seed(1201)
    logits = torch.randn(8, 2, requires_grad=True)
    is_reference = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    y0 = torch.tensor([0, 1, 0, 1, 0, 0, 0, 0])
    y1 = torch.tensor([0, 1, 0, 1, 1, 1, 1, 1])

    def loss(y):
        ce = F.cross_entropy(logits, y, reduction="none")
        return (ce * is_reference).sum() / (is_reference.sum() + 1e-8)

    l0 = loss(y0)
    l1 = loss(y1)
    assert torch.equal(l0, l1)

    grad = torch.autograd.grad(l0, logits)[0]
    assert grad[:4].abs().sum().item() > 0
    assert grad[4:].abs().sum().item() == 0
