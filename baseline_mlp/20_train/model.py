"""Baseline MLP over a single 63-float hand frame.

The architecture follows the baseline MLP of Carbo & Nalisnick (EMNLP 2025,
arXiv:2509.18309): two Linear -> BatchNorm -> ReLU -> Dropout blocks on the
flattened 21x3 hand landmarks, then one linear classification head per
phonological feature. The authors have not released code; this module is
written from the paper's description.

The serialized layout is a compatibility contract: released checkpoints
store the trunk as `network.{0,1,4,5}` (Sequential positions of the two
Linear/BatchNorm pairs; ReLU/Dropout at 2,3,6,7 hold no parameters) and the
heads as `heads.<i>` / `classifier.<i>` + `handshape_classifier`. Keep the
attribute names and Sequential ordering exactly as they are.

Two wrappers, one per head set:

  MultiHeadMLP  N independent linear heads on the shared 256-d trunk. Used
                for the 16-head run (SL-GCN/SHuBERT Table-9 head list) and,
                with N=6, for the 6-head run.

  HierarchicalMLP  the 6-head run's optional variant: 5 attribute heads
                whose concatenated logits feed a handshape classifier.
"""

import torch
import torch.nn as nn


def _trunk(input_dim=63, hidden_dim=256, dropout=0.3):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class MultiHeadMLP(nn.Module):
    """Shared trunk + one linear head per feature.

    Args:
        num_classes: list of class counts, one per head, in feature order.
        input_dim/hidden_dim/dropout: trunk hyperparameters.
    """

    def __init__(self, num_classes, input_dim=63, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.network = _trunk(input_dim, hidden_dim, dropout)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, int(n)) for n in num_classes]
        )
        self.model_type = "multi"
        self.num_classes = [int(n) for n in num_classes]

    def forward(self, x):
        x = self.network(x)
        return [head(x) for head in self.heads]


class HierarchicalMLP(nn.Module):
    """Shared trunk + attribute heads -> concat -> handshape head.

    num_classes is [handshape, *attributes] (6 entries for our head set):
    num_classes[0] sizes the handshape classifier, the rest the subheads
    whose concatenated logits it reads.
    """

    def __init__(self, num_classes, input_dim=63, hidden_dim=256, dropout=0.3):
        num_classes = [int(n) for n in num_classes]
        if len(num_classes) < 2:
            raise ValueError("need [handshape, *attributes]")
        super().__init__()
        self.network = _trunk(input_dim, hidden_dim, dropout)
        self.handshape_num = num_classes[0]
        sub = num_classes[1:]
        self.classifier = nn.ModuleList(
            [nn.Linear(hidden_dim, n) for n in sub]
        )
        self.handshape_classifier = nn.Linear(sum(sub), self.handshape_num)
        self.model_type = "hierarchical"
        self.num_classes = num_classes

    def forward(self, x):
        x = self.network(x)
        sub_logits = [head(x) for head in self.classifier]
        handshape_logits = self.handshape_classifier(
            torch.cat(sub_logits, dim=1)
        )
        return [handshape_logits] + sub_logits


def build_model(head_set, num_classes, arch="multi", **kwargs):
    if arch == "hierarchical":
        return HierarchicalMLP(num_classes, **kwargs)
    return MultiHeadMLP(num_classes, **kwargs)


# --------------------------------------------------------------- self-test
def _self_test():
    torch.manual_seed(0)

    # The checkpoint compatibility contract, spelled out: any change to the
    # trunk's Sequential ordering shows up here before it corrupts a load.
    trunk_keys = [
        "network.0.weight", "network.0.bias",
        "network.1.weight", "network.1.bias",
        "network.1.running_mean", "network.1.running_var",
        "network.1.num_batches_tracked",
        "network.4.weight", "network.4.bias",
        "network.5.weight", "network.5.bias",
        "network.5.running_mean", "network.5.running_var",
        "network.5.num_batches_tracked",
    ]

    counts = [9, 7, 3, 2, 3]
    m = MultiHeadMLP(counts)
    expected = trunk_keys + [
        f"heads.{i}.{p}" for i in range(len(counts)) for p in ("weight", "bias")
    ]
    assert sorted(m.state_dict()) == sorted(expected), "state_dict layout drifted"
    sd = m.state_dict()
    assert sd["network.0.weight"].shape == (256, 63)
    assert sd["network.4.weight"].shape == (256, 256)
    outs = m.eval()(torch.randn(8, 63))
    assert [o.shape[1] for o in outs] == counts
    print(f"multi OK ({len(outs)} heads, "
          f"trunk {sum(p.numel() for p in m.network.parameters())} params)")

    h = HierarchicalMLP([37, 9, 7, 3, 2, 3])
    expected_h = trunk_keys + [
        f"classifier.{i}.{p}" for i in range(5) for p in ("weight", "bias")
    ] + ["handshape_classifier.weight", "handshape_classifier.bias"]
    assert sorted(h.state_dict()) == sorted(expected_h), "state_dict layout drifted"
    outs = h.eval()(torch.randn(4, 63))
    assert [o.shape[1] for o in outs] == [37, 9, 7, 3, 2, 3], [o.shape for o in outs]
    print("hierarchical 6-head OK:", [tuple(o.shape) for o in outs])

    m16 = MultiHeadMLP([5, 37, 37, 2, 3, 6, 2, 8, 2, 12, 2, 8, 3, 3, 56, 58])
    outs16 = m16.eval()(torch.randn(4, 63))
    assert len(outs16) == 16
    print("16-head OK, params", sum(p.numel() for p in m16.parameters()))


if __name__ == "__main__":
    _self_test()
