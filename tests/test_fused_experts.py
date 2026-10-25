# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import ModuleList

from heretic.config import Settings as HereticSettings
from heretic.modifier import Context
from heretic.modifiers.abliteration import (
    Abliteration,
    Parameters,
    WeightDistribution,
)
from heretic.modifiers.abliteration import (
    Settings as AbliterationSettings,
)


class FakeExperts(nn.Module):
    down_proj: Tensor

    def __init__(self, down_proj: Tensor) -> None:
        super().__init__()
        self.down_proj = nn.Parameter(down_proj)


class FakeMLP(nn.Module):
    experts: FakeExperts

    def __init__(self, experts: FakeExperts) -> None:
        super().__init__()
        self.experts = experts


class FakeLayer(nn.Module):
    mlp: FakeMLP

    def __init__(self, experts: FakeExperts) -> None:
        super().__init__()
        self.mlp = FakeMLP(experts)


class FakeModel:
    def __init__(self, layers: ModuleList) -> None:
        self.layers = layers

    def get_layers(self) -> ModuleList:
        return self.layers


class FakeContext:
    def __init__(self, model: FakeModel) -> None:
        self.model = model

    def get_model(self) -> FakeModel:
        return self.model


def make_abliteration() -> Abliteration:
    return Abliteration(
        heretic_settings=HereticSettings.model_construct(),
        settings=AbliterationSettings(),
    )


def bake_parameters(max_weight_position: float) -> Parameters:
    return Parameters(
        direction_index=None,
        weight_distributions={
            "mlp.down_proj": WeightDistribution(
                max_weight=0.8,
                max_weight_position=max_weight_position,
                min_weight=0.8,
                min_weight_distance=1.0,
            )
        },
    )


class FusedExpertBakeTests(unittest.TestCase):
    def test_bakes_every_fused_expert(self) -> None:
        torch.manual_seed(2)
        num_experts, out_features, in_features = 3, 16, 8
        original = 0.05 * torch.randn(num_experts, out_features, in_features)
        direction = torch.randn(out_features)
        direction /= direction.norm()

        layer = FakeLayer(FakeExperts(original.clone()))
        model = FakeModel(ModuleList([layer]))

        modifier = make_abliteration()
        modifier.residual_directions = torch.stack(
            [torch.zeros(out_features), direction]
        )

        modifier.bake(
            cast(Context, FakeContext(model)),
            None,
            bake_parameters(max_weight_position=0.0),
        )

        projected = torch.einsum("o,eoi->ei", direction, original)
        expected = original - 0.8 * torch.einsum("o,ei->eoi", direction, projected)
        self.assertTrue(
            torch.allclose(layer.mlp.experts.down_proj, expected, atol=1e-5)
        )

    def test_skips_layers_with_zero_weight(self) -> None:
        torch.manual_seed(3)
        original = torch.randn(2, 4, 3)
        layer = FakeLayer(FakeExperts(original.clone()))
        model = FakeModel(ModuleList([layer]))

        modifier = make_abliteration()
        modifier.residual_directions = torch.stack([torch.zeros(4), torch.ones(4)])

        modifier.bake(
            cast(Context, FakeContext(model)),
            None,
            bake_parameters(max_weight_position=10.0),
        )

        self.assertTrue(torch.equal(layer.mlp.experts.down_proj, original))


if __name__ == "__main__":
    unittest.main()
