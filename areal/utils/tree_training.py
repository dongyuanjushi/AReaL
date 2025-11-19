from typing import Any

import torch

from areal.utils import logging
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.data import pad_and_stack_tensors_along_first_dim
from areal.utils.perf_tracer import trace_perf, trace_scope


logger = logging.getLogger("Tree Training")

############################## Token Tree Construction ##############################

class TokenNode:
    def __init__(self, token_id: int):
        self.token_id = token_id
        self.children = {}
        self.is_end_of_sequence = False


class CompressedTokenNode:
    def __init__(
        self,
        tokens: list[int] | None = None,
        end_flags: list[bool] | None = None,
        *,
        terminates_here: bool = False,
    ):
        self.tokens = tokens or []
        if end_flags is not None:
            if len(end_flags) != len(self.tokens):
                raise ValueError("end_flags must match the number of tokens")
            self.end_flags = end_flags
        else:
            self.end_flags = [False] * len(self.tokens)
        self.terminates_here = terminates_here
        self.children: dict[int, "CompressedTokenNode"] = {}
        self.token_indices: list[int] = [-1] * len(self.tokens)

@trace_perf("tree_training._compress_token_tree")
def _compress_token_tree(root: TokenNode) -> CompressedTokenNode:
    def _compress_from(node: TokenNode) -> CompressedTokenNode:
        tokens: list[int] = []
        end_flags: list[bool] = []
        current = node
        while True:
            tokens.append(current.token_id)
            end_flags.append(current.is_end_of_sequence)
            if len(current.children) != 1:
                break
            # Only one child; continue along the chain without branching.
            (_, next_child) = next(
                iter(sorted(current.children.items(), key=lambda item: item[0]))
            )
            current = next_child

        compressed = CompressedTokenNode(tokens, end_flags)
        if current.children:
            compressed.children = {
                token: _compress_from(child)
                for token, child in sorted(current.children.items(), key=lambda item: item[0])
            }
        return compressed

    compressed_root = CompressedTokenNode(terminates_here=root.is_end_of_sequence)
    if root.children:
        compressed_root.children = {
            token: _compress_from(child)
            for token, child in sorted(root.children.items(), key=lambda item: item[0])
        }
    return compressed_root

@trace_perf("tree_training._to_sequence_list")
def _to_sequence_list(data: dict[str, Any]):
    assert "input_ids" in data, "Input data must contain 'input_ids'"
    assert "attention_mask" in data, "Input data must contain 'attention_mask'"
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    
    sequences = []
    for ids, mask in zip(input_ids, attention_mask):
        seq = ids[mask.bool()].tolist()
        sequences.append(seq)
    return sequences


def _compress_for_visualization(node: CompressedTokenNode) -> tuple[int, list[tuple[int, list]]]:
    span = len(node.tokens)
    children = [
        _compress_for_visualization(child)
        for _, child in sorted(node.children.items(), key=lambda item: item[0])
    ]
    return span, children


def _render_compressed_tree(root: CompressedTokenNode) -> str:
    compressed_roots = [
        _compress_for_visualization(child)
        for _, child in sorted(root.children.items(), key=lambda item: item[0])
    ]

    if not compressed_roots:
        return "[empty]"

    lines: list[str] = []

    def _render(node: tuple[int, list], prefix: str, is_last: bool):
        span, children = node
        connector = "\\- " if is_last else "+- "
        lines.append(f"{prefix}{connector}{span}")
        child_prefix = prefix + ("    " if is_last else "|   ")
        for index, child in enumerate(children):
            _render(child, child_prefix, index == len(children) - 1)

    for idx, node in enumerate(compressed_roots):
        _render(node, "", idx == len(compressed_roots) - 1)

    return "\n".join(lines)


def _count_additional_nodes(root: TokenNode, sequence: list[int]) -> int:
    """Return how many new nodes are required to insert ``sequence`` into ``root``."""

    current = root
    for index, token in enumerate(sequence):
        child = current.children.get(token)
        if child is None:
            # All remaining tokens will require new nodes once we branch.
            return len(sequence) - index
        current = child

    return 0


def _insert_sequence(root: TokenNode, sequence: list[int]) -> None:
    """Insert ``sequence`` into ``root`` (mutates tree in-place)."""

    current = root
    for token in sequence:
        if token not in current.children:
            current.children[token] = TokenNode(token)
        current = current.children[token]

    current.is_end_of_sequence = True


def simple_build_tree(data: dict[str, Any], visualize: bool = False):
    """Build token trees from a list of token sequences.

    Each node in the tree represents a token. Each edge represents the transition
    from one token to the next. The root node is a dummy node with token_id=-1.
    Each leaf node corresponds to a complete sequence.

    Args:
        data (dict[str, Any]): Dictionary containing ``input_ids`` and ``attention_mask``
            tensors describing the batch of sequences to insert into the tree.
        visualize (bool): When ``True`` also returns a plain-text visualization
            where each node displays the number of consecutive tokens without
            branching.

    Returns:
        tuple[CompressedTokenNode, int] | tuple[CompressedTokenNode, int, str]:
            Root node (first real tokens) of the constructed tree, total
            number of concrete nodes (excluding the dummy root), and optionally
            the visualization string when ``visualize`` is ``True``.
    """

    normalized_sequences = _to_sequence_list(data)

    root = TokenNode(-1)
    total_nodes = 0  # Do not count the dummy root node.

    for seq in normalized_sequences:
        current = root

        if not seq:
            current.is_end_of_sequence = True
            continue

        for token in seq:
            if token not in current.children:
                current.children[token] = TokenNode(token)
                total_nodes += 1
            current = current.children[token]

        current.is_end_of_sequence = True

    compressed_root = _compress_token_tree(root)

    if visualize:
        visualization = _render_compressed_tree(compressed_root)
        logger.info("Token Tree Visualization:\n%s", visualization)
    return compressed_root, total_nodes

@trace_perf("tree_training.greedy_build_tree")
def greedy_build_tree(data: dict[str, Any], max_tokens_per_tree: int, visualize: bool = False):
    """Build token trees from a list of token sequences using a greedy packing strategy.
    The number of tokens in each tree will not exceed ``max_tokens_per_tree``.
    Args:
        data (dict[str, Any]): Dictionary containing ``input_ids`` and ``attention_mask``
            tensors describing the batch of sequences to insert into the tree.
        max_tokens_per_tree (int): Maximum number of tokens allowed in each tree.
        visualize (bool): When ``True`` also returns a plain-text visualization
            where each node displays the number of consecutive tokens without
            branching.
    """
    if max_tokens_per_tree <= 0:
        raise ValueError("max_tokens_per_tree must be a positive integer")

    normalized_sequences = _to_sequence_list(data)

    forests: list[dict[str, Any]] = []

    for seq in normalized_sequences:
        inserted = False

        for tree in forests:
            additional = _count_additional_nodes(tree["root"], seq)
            if tree["nodes"] + additional <= max_tokens_per_tree:
                _insert_sequence(tree["root"], seq)
                tree["nodes"] += additional
                inserted = True
                break

        if inserted:
            continue

        additional = len(seq)
        if additional > max_tokens_per_tree:
            raise ValueError(
                "Sequence length exceeds max_tokens_per_tree; adjust the limit or split sequences."
            )

        new_root = TokenNode(-1)
        _insert_sequence(new_root, seq)
        forests.append({"root": new_root, "nodes": additional})

    roots = [tree["root"] for tree in forests]
    node_counts = [tree["nodes"] for tree in forests]

    compressed_roots = [_compress_token_tree(root) for root in roots]

    if not visualize:
        return compressed_roots, node_counts

    for idx, compressed_root in enumerate(compressed_roots):
        visualization = _render_compressed_tree(compressed_root)
        logger.info("Token Tree %d Visualization:\n%s", idx, visualization)

    return compressed_roots, node_counts

@trace_perf("tree_training._flatten_tree_tokens")
def _flatten_tree_tokens(root: CompressedTokenNode) -> tuple[list[int], list[list[int]]]:
    """Collect tokens and ancestor indices via iterative traversal of compressed nodes."""

    tokens: list[int] = []
    ancestor_indices: list[list[int]] = []

    stack: list[tuple[CompressedTokenNode, list[int]]] = [(root, [])]

    while stack:
        node, path = stack.pop()
        local_path = path
        for pos, token in enumerate(node.tokens):
            current_index = len(tokens)
            tokens.append(token)
            node.token_indices[pos] = current_index
            current_path = local_path + [current_index]
            ancestor_indices.append(current_path)
            local_path = current_path

        for _, child in sorted(node.children.items(), key=lambda item: item[0], reverse=True):
            stack.append((child, local_path))

    return tokens, ancestor_indices

@trace_perf("tree_training.build_tree_input")
def build_tree_input(data: dict[str, Any], max_tokens_per_tree: int):
    """ First construct token trees from input data, then convert input data into tree-packed format.
    The return value should be a list of dictionaries, each contains input_ids, attention_mask, and sequence_indices for a packed tree structure.
    The input id should be a flattened list of token ids in the tree structure with pre-ordered traversal.
    The attention mask represents the causal relationship between tokens in the token tree, in which entries are set to true when 
    two tokens are in the same sequence and follows causal relationship (lower triangular causal mask).
    The sequence_indices entry contains, for every original sequence packed into the tree, the indices of the tokens belonging to that
    sequence in the tree order.

    If there are other fields in the input data, check if their shape is identical to the original `input_ids`.
    If yes, they will be packed into the output dictionary in flatten full-sequence manner following `sequence_indices`.
    Otherwise, keep them unchanged in each output dictionary.

    Returns:
        tuple[list[CompressedTokenNode], list[int], list[dict[str, Any]]]:
            ``roots`` of the packed token trees, token counts per tree, and tree-packed inputs with per-sequence indices.
    """
    roots, node_counts = greedy_build_tree(data, max_tokens_per_tree=max_tokens_per_tree)
    packed_trees: list[dict[str, Any]] = []

    input_template: torch.Tensor = data["input_ids"]
    mask_template: torch.Tensor = data["attention_mask"]

    packable_keys = [
        key
        for key, value in data.items()
        if key not in {"input_ids", "attention_mask"}
        and torch.is_tensor(value)
        and value.shape == input_template.shape
    ]
    packable_key_set = set(packable_keys)

    tree_infos: list[dict[str, Any]] = []
    for idx, (root, node_count) in enumerate(zip(roots, node_counts)):
        tokens, ancestor_indices = _flatten_tree_tokens(root)
        if len(tokens) != node_count:
            raise RuntimeError(
                "Flattened token count does not match node count for tree "
                f"{idx}: {len(tokens)} != {node_count}."
            )
        info_entry: dict[str, Any] = {
            "root": root,
            "tokens": tokens,
            "ancestor_indices": ancestor_indices,
            "sequence_indices": [],
            "sequence_ids": [],
        }
        if packable_keys:
            info_entry["packed_fields"] = {key: [] for key in packable_keys}
        tree_infos.append(info_entry)

    sequences = _to_sequence_list(data)

    def _match_sequence(root: CompressedTokenNode, sequence: list[int]) -> list[int] | None:
        if not sequence:
            return [] if root.terminates_here else None

        node = root
        position = -1
        indices: list[int] = []

        for token in sequence:
            if position + 1 < len(node.tokens) and node.tokens[position + 1] == token:
                position += 1
            else:
                child = node.children.get(token)
                if child is None or not child.tokens or child.tokens[0] != token:
                    return None
                node = child
                position = 0

            token_index = node.token_indices[position]
            if token_index < 0:
                raise RuntimeError("Token indices not initialized for compressed tree traversal.")
            indices.append(token_index)

        final_flag = node.end_flags[position] if node.tokens else node.terminates_here
        if not final_flag:
            return None
        return indices

    def _locate_tree_with_indices(sequence: list[int]) -> tuple[int, list[int]]:
        for tree_idx, info in enumerate(tree_infos):
            indices = _match_sequence(info["root"], sequence)
            if indices is not None:
                return tree_idx, indices
        raise ValueError("Sequence not found in any constructed tree.")

    for seq_idx, sequence in enumerate(sequences):
        tree_idx, sequence_indices = _locate_tree_with_indices(sequence)
        info = tree_infos[tree_idx]
        mask_row = mask_template[seq_idx].bool()
        value_slices = (
            {key: data[key][seq_idx][mask_row] for key in packable_keys}
            if packable_keys
            else None
        )
        info["sequence_indices"].append(sequence_indices)
        info["sequence_ids"].append(seq_idx)
        if value_slices is not None:
            for key, slice_value in value_slices.items():
                info["packed_fields"][key].append(slice_value)

    remaining_keys = set(data.keys()) - {"input_ids", "attention_mask"} - packable_key_set

    for info in tree_infos:
        tokens = info["tokens"]
        ancestor_indices = info["ancestor_indices"]
        token_tensor = torch.tensor(
            tokens,
            dtype=input_template.dtype,
            device=input_template.device,
        )

        mask_tensor = mask_template.new_zeros((len(tokens), len(tokens)))
        for row, cols in enumerate(ancestor_indices):
            if cols:
                mask_tensor[row, cols] = 1

        tree_entry: dict[str, Any] = {
            "input_ids": token_tensor,
            "attention_mask": mask_tensor,
            "sequence_indices": info["sequence_indices"],
            "sequence_ids": info["sequence_ids"],
        }

        if packable_keys:
            # Flatten per-token fields so they align with the order implied by sequence_indices.
            for key in packable_keys:
                sequences_values = info["packed_fields"][key]
                value_template = data[key]
                if sequences_values:
                    tree_entry[key] = torch.cat(sequences_values, dim=0)
                else:
                    tree_entry[key] = value_template.new_empty((0,))

        for key in remaining_keys:
            tree_entry[key] = data[key]

        packed_trees.append(tree_entry)

    return roots, node_counts, packed_trees

@trace_perf("tree_training.recover_packed_tensor_list")
def recover_packed_tensor_list(
    tensor_list: list[torch.Tensor], 
    sequence_indices_list: list[list[int]], 
    sequence_ids_list: list[int]
) -> torch.Tensor:
    """ TODO: Refactor this to be compatible with old forward/train_batch impl, too messy.
    
    Recover the original per-sequence tensor from a list of packed tree tensors.
    Args:
        tensor_list: List of packed tree tensors, each of shape (num_total_tokens, ...).
        sequence_indices_list: List of per-tree sequence indices.
        sequence_ids_list: List of original sequence IDs corresponding to each sequence in sequence_indices_list.
    
    Returns:
        Tensor of shape (batch_size, max_seq_len, ...) containing the recovered per-sequence data.
    """
    seq_lens = [
        [len(indices) for indices in sequence_indices] 
        for sequence_indices in sequence_indices_list
    ]
    seq_lens = [length for sublist in seq_lens for length in sublist]
    seq_ids = [_id for sublist in sequence_ids_list for _id in sublist]
    full_tensor = torch.cat(tensor_list, dim=0)
    assert len(seq_lens) == len(seq_ids), "Mismatch in number of sequences and sequence IDs."
    assert full_tensor.shape[0] == sum(seq_lens), "Mismatch in total tokens and sum of sequence lengths."
    
    tensors = []
    cursor = 0
    for length in seq_lens:
        seq_tensor = full_tensor[cursor:cursor + length]
        tensors.append(seq_tensor)
        cursor += length
    recovered = pad_and_stack_tensors_along_first_dim(tensors)
    return recovered

def amend_packed_tree_position_ids(input_: dict[str, Any]) -> torch.Tensor:
    """Generate position ids for packed tree inputs.

    Args:
        input_: Dictionary containing 'input_ids' and 'attention_mask' for the packed tree.
    
    Returns:
        A new dictionary containing all entries from ``input_`` plus 'position_ids'.
    """
    assert "input_ids" in input_, "Input must contain 'input_ids'"
    assert "attention_mask" in input_, "Input must contain 'attention_mask'"
    input_ids = input_["input_ids"]
    attention_mask = input_["attention_mask"]
    if input_ids.ndim != 1:
        raise ValueError("Packed tree 'input_ids' must be a 1D tensor.")
    if attention_mask.ndim != 2 or attention_mask.shape[0] != attention_mask.shape[1]:
        raise ValueError("Packed tree attention_mask must be a square matrix.")
    if attention_mask.shape[0] != input_ids.shape[0]:
        raise ValueError("Packed tree attention_mask must align with input_ids length.")

    if attention_mask.shape[0] == 0:
        position_ids = torch.empty(0, dtype=torch.long, device=attention_mask.device)
    else:
        ancestor_counts = attention_mask.bool().sum(dim=-1, dtype=torch.long)
        position_ids = torch.clamp_min(ancestor_counts - 1, 0)

    input_["position_ids"] = position_ids
    return input_


@trace_perf("tree_training.unpack_tree_output_logits_into_sequences")
def unpack_tree_output_logits_into_sequences(
    logits: torch.Tensor,
    input_data: dict[str, Any],
) -> torch.Tensor:
    """Unpack tree-packed logits using ``sequence_indices`` into flattened sequence order."""

    if "sequence_indices" not in input_data:
        raise ValueError("input_data must contain 'sequence_indices' produced by build_tree_input().")

    sequence_indices = input_data["sequence_indices"]
    if not isinstance(sequence_indices, list):
        raise TypeError("input_data['sequence_indices'] must be a list of index lists.")

    num_tree_tokens = logits.shape[0]
    trailing_shape = logits.shape[1:]

    total_positions = sum(len(indices) for indices in sequence_indices)
    if total_positions == 0:
        return logits.new_empty((0, *trailing_shape))

    flattened = logits.new_empty((total_positions, *trailing_shape))
    cursor = 0
    for indices in sequence_indices:
        if not isinstance(indices, list):
            raise TypeError("Entries in 'sequence_indices' must be lists of token indices.")
        for token_idx in indices:
            if token_idx < 0 or token_idx >= num_tree_tokens:
                raise IndexError(
                    f"Token index {token_idx} in sequence_indices is out of bounds for logits of size {num_tree_tokens}."
                )
            flattened[cursor] = logits[token_idx]
            cursor += 1

    if cursor != total_positions:
        raise RuntimeError(
            f"Unexpected number of filled positions: expected {total_positions}, got {cursor}."
        )

    return flattened

@trace_perf("tree_training.packed_tree_gather_logprobs")
def packed_tree_gather_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    sequence_indices: list[list[int]],
    temperature: float = 1.0,
    calculate_entropy: bool = False,
) -> torch.Tensor:
    """Gather log probabilities for sequences represented by tree-packed logits.

    Args:
        logits: Tensor of shape ``(num_tree_tokens, vocab_size)`` from the packed tree forward.
        input_ids: Tensor of shape ``(num_tree_tokens,)`` representing tree-packed tokens.
        sequence_indices: For each original sequence, the indices of tokens in the tree order.
        temperature: Optional temperature scaling for logprob computation (default 1.0).

    Returns:
        If ``calculate_entropy`` is ``False``, returns a tensor of shape ``(num_total_tokens,)`` 
        containing log probabilities in flattened sequence order aligning with ``sequence_indices``.
        If ``calculate_entropy`` is ``True``, returns two tensors of shape ``(num_total_tokens,)``,
        the first containing log probabilities and the second containing entropies.
    """

    if input_ids.ndim != 1:
        raise ValueError("input_ids must be a 1D tensor of tree tokens.")
    if logits.ndim != 2 or logits.shape[0] != input_ids.shape[0]:
        raise ValueError("logits must be 2D with first dim equal to len(input_ids).")

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    total_tokens = sum(len(indices) for indices in sequence_indices)
    if total_tokens == 0:
        return logits.new_empty(0)

    flattened_logprobs = logits.new_empty(total_tokens)
    flattened_entropies = logits.new_empty(total_tokens) if calculate_entropy else None
    cursor = 0

    for indices in sequence_indices:
        if not isinstance(indices, list):
            raise TypeError("sequence_indices must contain lists of token indices.")
        if not indices:
            continue

        tree_tokens = input_ids[indices]
        tree_logits = logits[indices]
        if calculate_entropy:
            seq_logprobs, seq_entropies = gather_logprobs_entropy(
                tree_logits, tree_tokens, temperature
            )
            flattened_logprobs[cursor : cursor + len(indices)] = seq_logprobs
            flattened_entropies[cursor : cursor + len(indices)] = seq_entropies
        else:
            seq_logprobs = gather_logprobs(tree_logits, tree_tokens, temperature)
            flattened_logprobs[cursor : cursor + len(indices)] = seq_logprobs
        cursor += len(indices)

    if cursor != total_tokens:
        raise RuntimeError(
            f"Logprob write cursor {cursor} does not match total token count {total_tokens}."
        )

    if calculate_entropy:
        return flattened_logprobs, flattened_entropies
    else:
        return flattened_logprobs


############################## Model Initialization ##############################

import inspect
import warnings

from mbridge.core.llm_bridge import LLMBridge
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

# Copied from megatron core to support arbitrary attention mask for tree training.
def get_gpt_layer_with_tree_attention_transformer_engine_spec(
    num_experts: int | None = None,
    moe_grouped_gemm: bool | None = False,
    qk_layernorm: bool | None = False,
    multi_latent_attention: bool | None = False,
    fp8: str | None = None,  # pylint: disable=unused-argument
    moe_use_legacy_grouped_gemm: bool | None = False,
    qk_l2_norm: bool | None = False,
    use_te_op_fuser: bool | None = False,
    use_kitchen: bool = False,
) -> ModuleSpec:
    """Use this spec to use lower-level Transformer Engine modules (required for fp8 training).

    Args:
        num_experts (int, optional): Number of experts. Defaults to None.
        moe_grouped_gemm (bool, optional): To use Grouped GEMM. Defaults to False.
        qk_layernorm (bool, optional): To use layernorm for queries/keys. Defaults to False.
        fp8 (str, optional): Deprecated. For temporary Nemo compatibility.
        moe_use_legacy_grouped_gemm (bool, optional): Force use the legacy GroupedMLP.
                                                      Defaults to False.
        qk_l2_norm (bool, optional): To use l2 norm for queries/keys. Defaults to False.
        use_te_op_fuser (bool, optional): Use Transformer Engine's operation-based API, which may
                                          enable certain operation fusions. Defaults to False.

    Returns:
        ModuleSpec: Module specification with TE modules

    """
    if fp8 is not None:
        warnings.warn(
            'The fp8 argument in "get_gpt_layer_with_transformer_engine_spec" has been deprecated'
            " and will be removed soon. Please update your code accordingly."
        )

    if use_kitchen:
        raise RuntimeError(
            "Currently tree attention is only supported with Transformer Engine backend."
        )
    backend = TESpecProvider()

    mlp = get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
        use_te_op_fuser=use_te_op_fuser,
    )

    if multi_latent_attention:
        raise RuntimeError("Tree attention for multi-latent attention is not supported yet.")
    qk_norm = backend.layer_norm(for_qk=True)
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.arbitrary},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_layer_norm_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                    k_layernorm=(
                        L2Norm if qk_l2_norm else (qk_norm if qk_layernorm else IdentityOp)
                    ),
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "mlp.0.weight": "mlp.linear_fc1.layer_norm_weight",
                "mlp.0.bias": "mlp.linear_fc1.layer_norm_bias",
                "mlp.1.basic_ops.0.weight": "mlp.linear_fc1.weight",
                "mlp.1.basic_ops.1.bias": "mlp.linear_fc1.bias",
                "mlp.3.basic_ops.0.weight": "mlp.linear_fc2.weight",
                "mlp.3.basic_ops.1.bias": "mlp.linear_fc2.bias",
            },
        ),
    )


# Copied from megatron core to support arbitrary attention mask for tree training.
def get_te_tree_gpt_decoder_block_spec(
    config: TransformerConfig,
    normalization: str | None = None,
    qk_l2_norm: bool | None = False,
    vp_stage: int | None = None,
) -> TransformerBlockSubmodules:
    if not HAVE_TE:
        raise RuntimeError(
            "Currently tree attention is only supported with Transformer Engine backend, which is not installed"
        )
    layer_norm_impl = TENorm
    dense_layer_spec = get_gpt_layer_with_tree_attention_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
        use_kitchen=config.use_kitchen,
    )
    # Following contents are copied from get_gpt_decoder_block_spec from megatron
    moe_layer_spec = get_gpt_layer_with_tree_attention_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        qk_l2_norm=qk_l2_norm,
        use_kitchen=config.use_kitchen,
    )

    # Parse config.moe_layer_freq to determine the pattern of expert/dense layers.
    # 0 stands for dense layers, 1 stands for expert layers.
    # For integer N: Creates a pattern with one expert layer every N layers.
    # For string pattern: Evaluates the str directly (e.g. "[1,0,1]" for alternating expert/dense).
    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
        assert len(moe_layer_pattern) == config.num_layers, (
            f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
            f"expected {config.num_layers}, "
            f"current moe layer pattern: {config.moe_layer_freq}"
        )
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Slice the layer specs to only include the layers that are built in this pipeline stage.
    # Note: MCore layer_number starts at 1
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs, layer_norm=layer_norm_impl
    )

    return block_spec


def _get_transformer_layer_spec_with_tree_attention(self, vp_stage: int | None = None):
    """ Copied from mbridge, overwrite method `LLMBridge._get_transformer_layer_spec` 
    to substitute megatron default decoder spec into the one that supports arbitrary 
    attention mask for tree training.
    
    Gets the transformer layer specification.

    Creates and returns a specification for the transformer layers based on
    the current configuration.

    Returns:
        TransformerLayerSpec: Specification for transformer layers

    Raises:
        AssertionError: If normalization is not RMSNorm
    """
    logger.info("Using Tree Attention GPT Decoder Block Spec")
    assert (
        self.config.normalization == "RMSNorm"
    ), "only RMSNorm is supported for now"
    # check if get_gpt_decoder_block_spec has vp_stage parameter
    sig = inspect.signature(get_gpt_decoder_block_spec)
    self.has_vp_stage = "vp_stage" in sig.parameters  # for mcore 0.12 compatibility
    extra_args = {}
    if self.has_vp_stage:
        extra_args["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(
        self.config, use_transformer_engine=True, **extra_args
    )
    return transformer_layer_spec

def patch_bridge_for_tree_training():
    """ Patch LLMBridge to support tree training with arbitrary attention mask.
    """
    LLMBridge._get_transformer_layer_spec = _get_transformer_layer_spec_with_tree_attention


############################## Model Forward ##############################

@trace_perf("tree_training.model_with_tree_attention_forward")
def model_with_tree_attention_forward(model, tree_input: dict[str, torch.Tensor]):
    """ Patch LLMBridge.model_forward to support tree training with arbitrary attention mask.
    """
    input_ids = tree_input["input_ids"]
    attention_mask = tree_input["attention_mask"]
    position_ids = tree_input["position_ids"]
    
    # Transformer Engine expects True where values should be masked out.
    attention_mask = (~attention_mask).unsqueeze(0).unsqueeze(0)
    # Add batch dimension for input_ids and position_ids
    input_ids = input_ids.unsqueeze(0)
    position_ids = position_ids.unsqueeze(0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    output = output.squeeze(0)  # Remove batch dimension
    return output

