# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedCELoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.llama3.config_registry import model_registry

from ..trainer import SimulationConfig, SimulationTrainer


def llama3_sim_debugmodel() -> SimulationTrainer.Config:
    """
    CPU-friendly simulation config side-loaded via:
      --module simulator.llama3 --config llama3_sim_debugmodel
    """
    return SimulationTrainer.Config(
        loss=ChunkedCELoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=512,
            steps=1,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        parallelism=ParallelismConfig(
            pipeline_parallel_schedule="Interleaved1F1B",
        ),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=1000,
            last_save_model_only=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(
            mode="selective",
        ),
        simulation=SimulationConfig(
            output_dir="./simulator_output",
            output_formats=["json", "dot", "chrome_trace", "text"],
            capture_joint_fx=False,
        ),
    )
