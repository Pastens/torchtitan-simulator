# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.dataloader import BaseDataLoader


class SyntheticTokenDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        vocab_size: int = 2048
        seed: int = 42

    def __init__(
        self,
        config: Config,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: Any,
        seq_len: int,
        local_batch_size: int,
    ) -> None:
        del dp_world_size, dp_rank, tokenizer
        self.config = config
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(config.seed)

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        while True:
            tokens = torch.randint(
                low=0,
                high=self.config.vocab_size,
                size=(self.local_batch_size, self.seq_len),
                dtype=torch.long,
                generator=self.generator,
            )
            labels = tokens.clone()
            yield {"input": tokens}, labels

    def state_dict(self) -> dict[str, Any]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "generator_state" in state_dict:
            self.generator.set_state(state_dict["generator_state"])
