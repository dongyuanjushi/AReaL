import torch
import torch.nn.functional as F
import pytest

from areal.utils.tree_training import (
    CompressedTokenNode,
    simple_build_tree,
    greedy_build_tree,
    build_tree_input,
    unpack_tree_output_logits_into_sequences,
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


def test_simple_build_tree_structure_and_counts():
    sequences = [[1, 2, 3], [1, 2, 4], [2], []]
    data = _build_data(sequences)

    root, total_nodes = simple_build_tree(data)

    assert isinstance(root, CompressedTokenNode)
    # Unique token nodes excluding the dummy root: 1, 2 (child of 1), 3, 4, and 2 (direct child of root)
    assert total_nodes == 5

    assert root.terminates_here
    assert set(root.children.keys()) == {1, 2}

    node1 = root.children[1]
    assert isinstance(node1, CompressedTokenNode)
    assert node1.tokens == [1, 2]
    assert node1.end_flags == [False, False]

    node2_direct = root.children[2]
    assert node2_direct.tokens == [2]
    assert node2_direct.end_flags == [True]

    node1_children_keys = set(node1.children.keys())
    assert node1_children_keys == {3, 4}

    node1_child3 = node1.children[3]
    assert node1_child3.tokens == [3]
    assert node1_child3.end_flags == [True]

    node1_child4 = node1.children[4]
    assert node1_child4.tokens == [4]
    assert node1_child4.end_flags == [True]


def test_greedy_build_tree_packs_sequences_under_capacity():
    sequences = [[1, 2, 3], [1, 2, 4], [9], [10, 11, 12, 13], [5, 6]]
    data = _build_data(sequences)

    roots, node_counts = greedy_build_tree(data, max_tokens_per_tree=5)

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
    tree_batch = packed[0]

    assert set(tree_batch.keys()) == {"input_ids", "attention_mask", "sequence_indices"}
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

    expected_sequence_indices = [[0, 1, 2], [0, 1, 3], [0, 4]]
    assert tree_batch["sequence_indices"] == expected_sequence_indices


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


def test_build_tree_input_packed_tensor_fields_with_empty_sequence():
    sequences = [[]]
    data = _build_data(sequences)

    extra_tensor = torch.zeros_like(data["input_ids"], dtype=torch.float32)
    data["extra"] = extra_tensor

    _, _, packed = build_tree_input(data, max_tokens_per_tree=2)
    tree_batch = packed[0]

    assert tree_batch["extra"].numel() == 0
    assert tree_batch["extra"].dtype == torch.float32


def test_build_tree_input_sequence_indices_with_multiple_trees():
    sequences = [[1, 2], [3, 4], [1, 5], [6]]
    data = _build_data(sequences)

    roots, node_counts, packed = build_tree_input(data, max_tokens_per_tree=3)

    assert len(roots) == 2
    assert sorted(node_counts) == [3, 3]

    reconstructed = []
    for tree_batch in packed:
        assert "sequence_indices" in tree_batch
        tokens = tree_batch["input_ids"].tolist()
        for indices in tree_batch["sequence_indices"]:
            assert all(0 <= idx < len(tokens) for idx in indices)
            reconstructed.append([tokens[idx] for idx in indices])

    expected_sequences = [seq for seq in sequences if seq]
    assert sorted(reconstructed) == sorted(expected_sequences)


def test_unpack_tree_logits_restores_flat_sequences():
    sequences = [[1, 2, 3], [1, 2, 4], [1, 5]]
    data = _build_data(sequences)

    _, _, packed = build_tree_input(data, max_tokens_per_tree=5)
    tree_batch = packed[0]

    vocab = 3
    logits = torch.arange(len(tree_batch["input_ids"]) * vocab, dtype=torch.float32).view(
        len(tree_batch["input_ids"]), vocab
    )

    unpacked = unpack_tree_output_logits_into_sequences(logits, tree_batch)

    expected_length = sum(len(indices) for indices in tree_batch["sequence_indices"])
    assert unpacked.shape == (expected_length, vocab)
    assert unpacked.device == logits.device

    offset = 0
    for indices in tree_batch["sequence_indices"]:
        for token_idx in indices:
            assert torch.equal(unpacked[offset], logits[token_idx])
            offset += 1
    assert offset == expected_length


def test_unpack_tree_logits_handles_empty_sequences():
    sequences = [[]]
    data = _build_data(sequences)

    _, _, packed = build_tree_input(data, max_tokens_per_tree=3)

    assert len(packed) == 1
    empty_tree = packed[0]
    assert empty_tree["sequence_indices"] == [[]]

    logits = torch.empty((0, 7), dtype=torch.float32)
    unpacked = unpack_tree_output_logits_into_sequences(logits, empty_tree)

    assert unpacked.shape == (0, 7)
    assert unpacked.device == logits.device


def test_packed_tree_gather_logprobs_matches_reference():
    sequences = [[1, 2, 3], [1, 2, 4], [1, 5]]
    data = _build_data(sequences)

    _, _, packed = build_tree_input(data, max_tokens_per_tree=5)
    tree_batch = packed[0]

    vocab = 6
    temperature = 0.7
    logits = torch.randn(len(tree_batch["input_ids"]), vocab, dtype=torch.float32)

    gathered = packed_tree_gather_logprobs(
        logits,
        tree_batch["input_ids"],
        tree_batch["sequence_indices"],
        temperature=temperature,
    )

    expected_values = []
    for indices in tree_batch["sequence_indices"]:
        for idx in indices:
            log_probs = F.log_softmax(logits[idx] / temperature, dim=-1)
            expected_values.append(log_probs[tree_batch["input_ids"][idx]])

    expected = (
        torch.stack(expected_values)
        if expected_values
        else logits.new_empty(0)
    )

    assert torch.allclose(gathered, expected, atol=1e-6)


def test_packed_tree_gather_logprobs_handles_empty_sequences():
    sequences = [[]]
    data = _build_data(sequences)

    _, _, packed = build_tree_input(data, max_tokens_per_tree=3)
    tree_batch = packed[0]

    logits = torch.empty((0, 5), dtype=torch.float32)
    gathered = packed_tree_gather_logprobs(
        logits,
        tree_batch["input_ids"],
        tree_batch["sequence_indices"],
    )

    assert gathered.shape == (0,)


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


def test_amend_packed_tree_position_ids_handles_empty_tree():
    sequences = [[]]
    data = _build_data(sequences)

    _, _, packed = build_tree_input(data, max_tokens_per_tree=3)
    tree_batch = packed[0]

    updated = amend_packed_tree_position_ids({
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in tree_batch.items()
    })

    assert "position_ids" in updated
    assert updated["position_ids"].dtype == torch.long
    assert updated["position_ids"].numel() == 0

