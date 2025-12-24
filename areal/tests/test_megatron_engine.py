import os
import time
from importlib.metadata import version as get_version
from typing import Any

import pytest
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    MegatronEngineConfig,
    OptimizerConfig,
    TrainEngineConfig,
    MicroBatchSpec,
)
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta
from areal.engine.megatron_engine import MegatronEngine
from areal.platforms import current_platform
from areal.utils import logging
from areal.utils.device import log_gpu_stats
from areal.utils.tree_training import packed_tree_gather_logprobs
from areal.utils.functional import gather_logprobs

logger = logging.getLogger("MegatronEngine Test")

VOCAB_SIZE = 100
MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def mock_input(
    batch_size=5,
    min_seqlen=10,
    max_seqlen=20,
    device=current_platform.device_type,
) -> dict[str, Any]:
    """Create mock padded input data (same format for huggingface) for testing.
    Returns a dict with input_ids, attention_mask, and position_ids.
    """
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen, (batch_size,), dtype=torch.int, device=device
    )
    max_seqlen = int(max(seqlens))
    input_ids = torch.randint(
        0, VOCAB_SIZE, (batch_size, max_seqlen), dtype=torch.long, device=device
    )
    attn_mask = torch.zeros((batch_size, max_seqlen), dtype=torch.bool, device=device)

    attn_mask[
        torch.arange(0, max_seqlen, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attn_mask, pad_token_id)

    return dict(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )

@pytest.fixture(scope="module")
def mock_tree_input(
    batch_size=5,
    tree_tokens=30,
    total_tokens=60,
    device=current_platform.device_type,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if total_tokens < tree_tokens:
        raise ValueError("total_tokens must be >= tree_tokens")
    if total_tokens < batch_size:
        raise ValueError("total_tokens must be >= batch_size to allocate at least one token per sequence")

    device = device if isinstance(device, torch.device) else torch.device(device)
    lengths = [tree_tokens]
    remaining_tokens = total_tokens - tree_tokens
    remaining_slots = batch_size - 1

    if remaining_slots:
        if remaining_tokens < remaining_slots:
            raise ValueError("Not enough tokens available for the requested batch size")
        for index in range(remaining_slots):
            slots_left = remaining_slots - index - 1
            max_assignable = min(tree_tokens, remaining_tokens - slots_left)
            share = max(1, min(max_assignable, remaining_tokens // (slots_left + 1)))
            lengths.append(share)
            remaining_tokens -= share
        if remaining_tokens != 0:
            lengths[-1] += remaining_tokens
            remaining_tokens = 0
    else:
        if total_tokens != tree_tokens:
            raise ValueError("total_tokens must equal tree_tokens when batch_size is 1")

    lengths = [int(l) for l in lengths]
    if sum(lengths) != total_tokens:
        raise RuntimeError("Token length allocation mismatch")

    base_tokens = torch.arange(1, tree_tokens + 1, dtype=torch.long, device=device)
    max_len = max(lengths)
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    sequences = []
    for idx, length in enumerate(lengths):
        seq_tokens = base_tokens[:length]
        input_ids[idx, :length] = seq_tokens
        attention_mask[idx, :length] = True
        sequences.append(seq_tokens.tolist())

    def _count_unique_nodes(seqs: list[list[int]]) -> int:
        root: dict[int, dict] = {}
        count = 0
        for seq in seqs:
            node = root
            for token in seq:
                if token not in node:
                    node[token] = {}
                    count += 1
                node = node[token]
        return count

    unique_nodes = _count_unique_nodes(sequences)
    if unique_nodes != tree_tokens:
        raise RuntimeError(
            f"Constructed tree has {unique_nodes} tokens, expected {tree_tokens}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def mock_loss_fn(logits: torch.Tensor, input_data: dict) -> torch.Tensor:
    """Mock loss function for testing."""
    return torch.mean(logits)


# Cannot use a "module" scope since process groups can only be initialized once.
@pytest.fixture
def engine():
    logger.info(f"megatron.core version={get_version('megatron.core')}")
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7777",
        }
    )
    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            use_deterministic_algorithms=True,
        ),
    )
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine = MegatronEngine(config)
    engine.create_process_group(alloc_mode.train)
    engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    logger.info(f"mcore GPTModel initialized: {engine.model}")
    log_gpu_stats("initialize")
    try:
        yield engine
    finally:
        engine.destroy()
        assert not dist.is_initialized()


def test_simple_forward(engine, mock_input):
    engine.eval()
    result = engine.forward(mock_input)
    logger.info(f"Forward done, result: {result}")


def test_simple_train(engine, mock_input):
    engine.train()
    train_result = engine.train_batch(
        mock_input,
        loss_fn=mock_loss_fn,
        loss_weight_fn=lambda x: torch.tensor(1.0, device=engine.device),
    )
    engine.step_lr_scheduler()
    logger.info(f"Train done, result={train_result}")


def test_tree_training_forward(engine, mock_tree_input):
    for k, v in mock_tree_input.items():
        print(f"mock_tree_input[{k}].shape={v.shape}, dtype={v.dtype} v=\n{v}")

    def calc_logprobs_tree_training(logits, input_data):
        input_ids = input_data["input_ids"]
        sequence_ids = input_data["sequence_ids"]
        seq_id_to_tree_indices = input_data["seq_id_to_tree_indices"]
        logprobs = packed_tree_gather_logprobs(
            logits, input_ids, sequence_ids, seq_id_to_tree_indices, 1.0
        )
        return logprobs

    def calc_logprobs(logits, input_data):
        labels = input_data.get(
            "rolled_input_ids",
            torch.roll(input_data["input_ids"], shifts=-1, dims=-1),
        )
        logprobs = gather_logprobs(logits, labels, 1.0)
        return logprobs

    engine.eval()
    logprob_baseline =  engine.forward(
        input_=mock_tree_input,
        post_hook=calc_logprobs,
        aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
    )
    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            enable_tree_training=True, 
            use_deterministic_algorithms=True
        ),
    )
    tree_engine = MegatronEngine(config)
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    tree_engine.create_process_group(alloc_mode.train)
    tree_engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    tree_engine.eval()
    logprob_tree = tree_engine.forward(
        input_=mock_tree_input,
        post_hook=calc_logprobs_tree_training,
        aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
    )
    
    print(f"logprob_baseline={logprob_baseline}")
    print(f"logprob_tree={logprob_tree}")
    # print where logprob baseline and logprob_tree are zeros
    print(f"logprob_baseline == 0 at positions: {(logprob_baseline == 0).nonzero(as_tuple=True)}")
    print(f"logprob_tree == 0 at positions: {(logprob_tree == 0).nonzero(as_tuple=True)}")

    # print where logprob_baseline and logprob_tree differ
    diff_positions = (logprob_baseline - logprob_tree).abs() > 1e-6
    print(f"Positions where logprob_baseline and logprob_tree differ: {diff_positions.nonzero(as_tuple=True)}")
    print(f"diff = {logprob_baseline - logprob_tree}")
    assert torch.allclose(logprob_baseline, logprob_tree, atol=1e-6)


def _collect_gradients(engine) -> dict[str, torch.Tensor]:
    """Collect gradients from Megatron engine.
    
    In Megatron, gradients are stored in param.main_grad (gradient buffer),
    not param.grad. This function collects gradients from the correct location.
    """
    grads = {}
    for model in engine.model:
        for name, param in model.named_parameters():
            # Megatron stores gradients in main_grad attribute
            if hasattr(param, 'main_grad') and param.main_grad is not None:
                grads[name] = param.main_grad.clone()
            elif param.grad is not None:
                grads[name] = param.grad.clone()
    return grads


def _collect_parameters(engine) -> dict[str, torch.Tensor]:
    """Collect parameters from Megatron engine."""
    params = {}
    for model in engine.model:
        for name, param in model.named_parameters():
            params[name] = param.data.clone()
    return params


def _check_nan_params(params: dict[str, torch.Tensor], label: str) -> list[str]:
    """Check for NaN values in parameters and return list of affected param names."""
    nan_params = []
    for name, param in params.items():
        if torch.isnan(param).any():
            nan_count = torch.isnan(param).sum().item()
            total_count = param.numel()
            nan_params.append(name)
            print(f"  {name}: {nan_count}/{total_count} NaN values")
    if nan_params:
        print(f"\n⚠ NaN parameters in {label} ({len(nan_params)}):")
    return nan_params


def test_tree_training_forward_backward(mock_tree_input):
    """Test that tree training produces correct gradients for every weight in the model.
    
    This test compares gradients computed via:
    1. Baseline: Standard forward-backward pass with regular batched computation
    2. Tree training: Forward-backward pass with tree-structured deduplication
    
    Both should produce identical gradients for all model parameters.
    """
    for k, v in mock_tree_input.items():
        print(f"mock_tree_input[{k}].shape={v.shape}, dtype={v.dtype} v=\n{v}")

    def loss_fn_tree_training(logits, input_data):
        """Loss function for tree training that uses packed tree gather."""
        input_ids = input_data["input_ids"]
        sequence_ids = input_data["sequence_ids"]
        seq_id_to_tree_indices = input_data["seq_id_to_tree_indices"]
        logprobs = packed_tree_gather_logprobs(
            logits, input_ids, sequence_ids, seq_id_to_tree_indices, 1.0
        )
        # Sum of log probs as loss (for gradient comparison)
        return logprobs.sum()

    def loss_fn_baseline(logits, input_data):
        """Standard loss function using regular gather."""
        labels = input_data.get(
            "rolled_input_ids",
            torch.roll(input_data["input_ids"], shifts=-1, dims=-1),
        )
        logprobs = gather_logprobs(logits, labels, 1.0)
        # Sum of log probs as loss (for gradient comparison)
        return logprobs.sum()

    # ========== Setup baseline engine ==========
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7778",
        }
    )
    baseline_config = TrainEngineConfig(
        experiment_name="test_baseline",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            use_deterministic_algorithms=True,
        ),
    )
    alloc_mode = AllocationMode.from_str("d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    
    baseline_engine = MegatronEngine(baseline_config)
    baseline_engine.create_process_group(alloc_mode.train)
    baseline_engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    baseline_engine.train()
    
    # Run baseline forward-backward
    _ = baseline_engine.train_batch(
        mock_tree_input,
        loss_fn=loss_fn_baseline,
        loss_weight_fn=lambda x: torch.tensor(1.0, device=baseline_engine.device),
    )
    
    # Collect baseline gradients and updated parameters
    baseline_grads = _collect_gradients(baseline_engine)
    baseline_params = _collect_parameters(baseline_engine)
    
    logger.info(f"Collected {len(baseline_grads)} gradients from baseline engine")
    logger.info(f"Collected {len(baseline_params)} parameters from baseline engine")
    baseline_engine.destroy()
    
    # ========== Setup tree training engine ==========
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "7779",
        }
    )
    tree_config = TrainEngineConfig(
        experiment_name="test_tree",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=1024),
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(
            enable_tree_training=True,
            use_deterministic_algorithms=True,
        ),
    )
    
    tree_engine = MegatronEngine(tree_config)
    tree_engine.create_process_group(alloc_mode.train)
    tree_engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc_mode.train)
    tree_engine.train()
    
    # Run tree training forward-backward
    _ = tree_engine.train_batch(
        mock_tree_input,
        loss_fn=loss_fn_tree_training,
        loss_weight_fn=lambda x: torch.tensor(1.0, device=tree_engine.device),
    )
    
    # Collect tree training gradients and updated parameters
    tree_grads = _collect_gradients(tree_engine)
    tree_params = _collect_parameters(tree_engine)
    
    logger.info(f"Collected {len(tree_grads)} gradients from tree training engine")
    logger.info(f"Collected {len(tree_params)} parameters from tree training engine")
    tree_engine.destroy()
    
    # ========== Compare gradients ==========
    baseline_keys = set(baseline_grads.keys())
    tree_keys = set(tree_grads.keys())
    
    # Check for missing keys
    only_in_baseline = baseline_keys - tree_keys
    only_in_tree = tree_keys - baseline_keys
    
    if only_in_baseline:
        logger.warning(f"Gradients only in baseline: {only_in_baseline}")
    if only_in_tree:
        logger.warning(f"Gradients only in tree training: {only_in_tree}")
    
    common_keys = baseline_keys & tree_keys
    logger.info(f"Comparing {len(common_keys)} common gradient tensors")
    
    for k, v in baseline_grads.items():
        print(f"baseline_grads[{k}].shape={v.shape}, dtype={v.dtype} mean(v)={v.float().mean().item():.6e}")
    for k, v in tree_grads.items():
        print(f"tree_grads[{k}].shape={v.shape}, dtype={v.dtype} mean(v)={v.float().mean().item():.6e}")

    # Check for NaN gradients
    nan_in_baseline = []
    nan_in_tree = []
    for name in sorted(common_keys):
        if torch.isnan(baseline_grads[name]).any():
            nan_in_baseline.append(name)
        if torch.isnan(tree_grads[name]).any():
            nan_in_tree.append(name)
    
    if nan_in_baseline:
        print(f"\n⚠ NaN gradients in BASELINE ({len(nan_in_baseline)}):")
        for name in nan_in_baseline:
            nan_count = torch.isnan(baseline_grads[name]).sum().item()
            total_count = baseline_grads[name].numel()
            print(f"  {name}: {nan_count}/{total_count} NaN values")
    
    if nan_in_tree:
        print(f"\n⚠ NaN gradients in TREE TRAINING ({len(nan_in_tree)}):")
        for name in nan_in_tree:
            nan_count = torch.isnan(tree_grads[name]).sum().item()
            total_count = tree_grads[name].numel()
            print(f"  {name}: {nan_count}/{total_count} NaN values")
    
    # Check for NaN in updated parameters
    nan_params_baseline = _check_nan_params(baseline_params, "BASELINE PARAMS")
    nan_params_tree = _check_nan_params(tree_params, "TREE TRAINING PARAMS")
    
    mismatched_params = []
    max_diff_overall = 0.0
    
    for name in sorted(common_keys):
        baseline_grad = baseline_grads[name]
        tree_grad = tree_grads[name]
        
        if baseline_grad.shape != tree_grad.shape:
            mismatched_params.append((name, f"shape mismatch: {baseline_grad.shape} vs {tree_grad.shape}"))
            continue
        
        diff = (baseline_grad - tree_grad).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        max_diff_overall = max(max_diff_overall, max_diff)
        
        if max_diff > 1e-5:
            mismatched_params.append((name, f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}"))
            print(f"Gradient mismatch for {name}:")
            print(f"  Shape: {baseline_grad.shape}")
            print(f"  Baseline grad mean: {baseline_grad.float().mean().item():.6e}")
            print(f"  Tree grad mean: {tree_grad.float().mean().item():.6e}")
            print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    
    print(f"\n{'='*60}")
    print("GRADIENT COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  Total baseline gradients: {len(baseline_keys)}")
    print(f"  Total tree gradients: {len(tree_keys)}")
    print(f"  Common gradients: {len(common_keys)}")
    print(f"  NaN in baseline grads: {len(nan_in_baseline)}")
    print(f"  NaN in tree training grads: {len(nan_in_tree)}")
    print(f"  NaN in baseline params: {len(nan_params_baseline)}")
    print(f"  NaN in tree training params: {len(nan_params_tree)}")
    print(f"  Mismatched gradients: {len(mismatched_params)}")
    print(f"  Max diff overall: {max_diff_overall:.6e}")
    
    if mismatched_params:
        print(f"\nMismatched parameters:")
        for name, reason in mismatched_params:
            print(f"  {name}: {reason}")
    
    # Assert no mismatches
    assert len(only_in_baseline) == 0, f"Gradients missing in tree training: {only_in_baseline}"
    assert len(only_in_tree) == 0, f"Gradients missing in baseline: {only_in_tree}"
    assert len(nan_in_baseline) == 0, f"NaN gradients in baseline: {nan_in_baseline}"
    assert len(nan_in_tree) == 0, f"NaN gradients in tree training: {nan_in_tree}"
    assert len(nan_params_baseline) == 0, f"NaN parameters in baseline: {nan_params_baseline}"
    assert len(nan_params_tree) == 0, f"NaN parameters in tree training: {nan_params_tree}"
    assert len(mismatched_params) == 0, f"Gradient mismatches found: {mismatched_params}"
    
    print("\n✓ All gradients match between baseline and tree training!")
    print("✓ No NaN values in updated parameters!")


@torch.no_grad()
def test_hf_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("hf_engine_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="hf",
        tokenizer=tokenizer,
        with_optim=False,
        base_model_path=None,
    )

    old = engine.forward(input_=mock_input)
    start = time.perf_counter()
    engine.save(save_load_meta)
    logger.info(f"Save done, time cost: {time.perf_counter() - start:.4f} seconds.")
    for name, param in engine.model.named_parameters():
        param.zero_()

    start = time.perf_counter()
    engine.load(save_load_meta)
    logger.info(f"Load done, time cost: {time.perf_counter() - start:.4f} seconds.")
    new = engine.forward(input_=mock_input)
    assert torch.allclose(old, new)


@torch.no_grad()
@pytest.mark.slow
def test_dcp_save_load_weights(tmp_path_factory, engine, mock_input):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    path = tmp_path_factory.mktemp("megatron_engine_dcp_test")
    save_load_meta = SaveLoadMeta(
        path=path,
        weight_format="dcp",
        tokenizer=tokenizer,
        with_optim=True,
        base_model_path=None,
    )

    old = engine.forward(input_=mock_input)
    start = time.perf_counter()
    engine.save(save_load_meta)
    logger.info(f"Save done, time cost: {time.perf_counter() - start:.4f} seconds.")
    for name, param in engine.model.named_parameters():
        param.zero_()

    start = time.perf_counter()
    engine.load(save_load_meta)
    logger.info(f"Load done, time cost: {time.perf_counter() - start:.4f} seconds.")
    new = engine.forward(input_=mock_input)
    assert torch.allclose(old, new)
