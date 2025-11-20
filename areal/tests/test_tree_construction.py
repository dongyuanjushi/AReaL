import torch
import torch.nn.functional as F
import pytest

from areal.utils.tree_training import (
    CompressedTokenNode,
    greedy_build_tree,
    build_tree_input,
    packed_tree_gather_logprobs,
    amend_packed_tree_position_ids,
)


def _build_data(sequences):
    max_len = max(len(seq) for seq in sequences)
    input_ids = []
    attention_mask = []
    for seq in sequences:
        padded = seq + [0] * (max_len - len(seq))
        mask = [1] * len(seq) + [0] * (max_len - len(seq))
        input_ids.append(padded)
        attention_mask.append(mask)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def test_greedy_build_tree_packs_sequences_under_capacity():
    sequences = [[1, 2, 3], [1, 2, 4], [9], [10, 11, 12, 13], [5, 6]]
    data = _build_data(sequences)

    roots, node_counts, _ = greedy_build_tree(data, max_tokens_per_tree=5)

    assert len(roots) == 3
    assert node_counts == [5, 4, 2]

    # Validate that each tree respects the capacity constraint
    for count in node_counts:
        assert count <= 5

    # Ensure every root is a CompressedTokenNode and contains at least one sequence
    for root in roots:
        assert isinstance(root, CompressedTokenNode)
        assert root.children


def test_greedy_build_tree_raises_on_sequence_exceeding_capacity():
    sequences = [[1, 2, 3, 4, 5, 6]]
    data = _build_data(sequences)

    with pytest.raises(ValueError):
        greedy_build_tree(data, max_tokens_per_tree=5)


def test_build_tree_input_constructs_tree_packed_batches():
    sequences = [[1, 2, 3], [1, 2, 4], [1, 5]]
    data = _build_data(sequences)

    roots, node_counts, packed = build_tree_input(data, max_tokens_per_tree=5)

    assert len(roots) == 1
    assert node_counts == [5]
    assert len(packed) == 1

    tree_batch = packed[0]
    expected_keys = {
        "input_ids",
        "attention_mask",
        "sequence_ids",
        "seq_id_to_tree_indices",
        "tree_endpoints_to_seq_info",
        "cu_seqlens",
    }
    assert expected_keys.issubset(tree_batch.keys())
    assert tree_batch["input_ids"].device == data["input_ids"].device
    assert tree_batch["attention_mask"].device == data["attention_mask"].device

    expected_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    assert torch.equal(tree_batch["input_ids"], expected_ids)

    expected_mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ],
        dtype=torch.long,
    )
    assert torch.equal(tree_batch["attention_mask"], expected_mask)

    assert tree_batch["sequence_ids"] == [0, 1, 2]
    expected_cu_seqlens = torch.tensor([0, 3, 6, 8], dtype=torch.int32)
    assert torch.equal(tree_batch["cu_seqlens"], expected_cu_seqlens)

    # Verify that per-sequence tree indices reconstruct the original token sequences.
    reconstructed = {}
    for seq_id in tree_batch["sequence_ids"]:
        segments = tree_batch["seq_id_to_tree_indices"][seq_id]
        tokens = []
        for start, end in segments:
            assert start <= end
            tokens.extend(tree_batch["input_ids"][start : end + 1].tolist())
        reconstructed[seq_id] = tokens

    assert reconstructed[0] == [1, 2, 3]
    assert reconstructed[1] == [1, 2, 4]
    assert reconstructed[2] == [1, 5]

    expected_segments = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)}
    assert set(tree_batch["tree_endpoints_to_seq_info"].keys()) == expected_segments
    for start, end in expected_segments:
        seq_id, seq_pos = tree_batch["tree_endpoints_to_seq_info"][(start, end)]
        assert 0 <= seq_id < len(sequences)
        assert 0 <= seq_pos < len(sequences[seq_id])
        length = end - start + 1
        assert seq_pos + length <= len(sequences[seq_id])


def test_build_tree_input_packs_additional_tensor_fields():
    sequences = [[1, 2, 3], [1, 2, 4], [1, 5]]
    data = _build_data(sequences)

    extra_tensor = torch.tensor(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ]
    )
    data["extra"] = extra_tensor
    data["non_pack_tensor"] = torch.tensor([1.0, 2.0, 3.0])
    data["meta"] = {"note": "keep"}

    _, _, packed = build_tree_input(data, max_tokens_per_tree=5)
    tree_batch = packed[0]

    expected_extra = torch.tensor([10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0])
    assert torch.allclose(tree_batch["extra"], expected_extra)

    assert torch.equal(tree_batch["non_pack_tensor"], data["non_pack_tensor"])
    assert tree_batch["meta"] == data["meta"]


def test_packed_tree_gather_logprobs_matches_reference():
    sequences = [
        [2, 3, 4],
        [2, 3, 5, 6],
        [2, 7],
        [8, 9],
    ]
    data = _build_data(sequences)

    roots, node_counts, tree_infos = greedy_build_tree(data, max_tokens_per_tree=12)

    assert len(roots) == 1

    tree_info = tree_infos[0]
    num_tree_tokens = node_counts[0]
    input_ids = torch.empty(num_tree_tokens, dtype=torch.long)
    for (tree_start, tree_end), (seq_id, seq_offset) in tree_info["tree_endpoints_to_seq_info"].items():
        slice_tokens = sequences[seq_id][seq_offset : seq_offset + (tree_end - tree_start)]
        input_ids[tree_start:tree_end] = torch.tensor(slice_tokens, dtype=torch.long)

    vocab_size = 20
    torch.manual_seed(0)
    logits = torch.randn(num_tree_tokens, vocab_size, dtype=torch.float)

    temperature = 0.7
    expected_logprobs = []
    expected_entropies = []
    for seq_id in tree_info["sequence_ids"]:
        indices = tree_info["seq_id_to_tree_indices"][seq_id]
        seq_token_segments = [input_ids[start:end+1] for start, end in indices]
        seq_logits_segments = [logits[start:end+1] for start, end in indices]
        seq_tokens = torch.cat(seq_token_segments, dim=0)
        seq_logits = torch.cat(seq_logits_segments, dim=0)
        log_probs = F.log_softmax(seq_logits / temperature, dim=-1)
        seq_logprobs = log_probs.gather(dim=-1, index=seq_tokens.unsqueeze(-1)).squeeze(-1)
        seq_entropies = -(log_probs.exp() * log_probs).sum(dim=-1)
        expected_logprobs.append(seq_logprobs)
        expected_entropies.append(seq_entropies)

    expected_logprobs = torch.cat(expected_logprobs, dim=0)
    expected_entropies = torch.cat(expected_entropies, dim=0)

    flattened = packed_tree_gather_logprobs(
        logits,
        input_ids,
        tree_info["sequence_ids"],
        tree_info["seq_id_to_tree_indices"],
        temperature=temperature,
        calculate_entropy=False,
    )

    assert torch.allclose(flattened, expected_logprobs, atol=1e-6)

    flattened_with_entropy, entropies = packed_tree_gather_logprobs(
        logits,
        input_ids,
        tree_info["sequence_ids"],
        tree_info["seq_id_to_tree_indices"],
        temperature=temperature,
        calculate_entropy=True,
    )

    assert torch.allclose(flattened_with_entropy, expected_logprobs, atol=1e-6)
    assert torch.allclose(entropies, expected_entropies, atol=1e-6)

def test_amend_packed_tree_position_ids_assigns_depth():
    sequences = [[1, 2, 3], [1, 2, 4], [1, 5]]
    data = _build_data(sequences)

    _, _, packed = build_tree_input(data, max_tokens_per_tree=5)
    tree_batch = packed[0]

    updated = amend_packed_tree_position_ids({
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in tree_batch.items()
    })

    expected = torch.tensor(
        [0, 1, 2, 2, 1], dtype=torch.long, device=tree_batch["attention_mask"].device
    )

    assert "position_ids" in updated
    assert torch.equal(updated["position_ids"], expected)
