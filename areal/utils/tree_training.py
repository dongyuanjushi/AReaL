from typing import Any
from collections import defaultdict

import torch
import torch.distributed as dist

from areal.utils import logging
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.data import pad_and_stack_tensors_along_first_dim
from areal.utils.perf_tracer import trace_perf, trace_scope


logger = logging.getLogger("Tree Training")

############################## Token Tree Construction ##############################

class TokenNode:
    def __init__(self, tree_id: int, token_id: int, node_id: int):
        self.tree_id = tree_id
        self.node_id = node_id
        self.token_id = token_id
        self.children = {}
        self.is_end_of_sequence = False
        self.sequence_ids = []
        self.tree_nodes = [] # only available in root node
    
    def add_sequence(self, sequence_id: int):
        self.sequence_ids.append(sequence_id)


class CompressedTokenNode:
    def __init__(
        self,
        tree_id: int,
        start_node_id: int,
        end_node_id: int,
        tokens: list[int] | None = None,
        end_flags: list[bool] | None = None,
        *,
        terminates_here: bool = False,
    ):
        self.tree_id = tree_id
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.tokens = tokens or []
        if end_flags is not None:
            if len(end_flags) != len(self.tokens):
                raise ValueError("end_flags must match the number of tokens")
            self.end_flags = end_flags
        else:
            self.end_flags = [False] * len(self.tokens)
        self.terminates_here = terminates_here
        self.children: dict[int, "CompressedTokenNode"] = {}
        self.ancestors: list["CompressedTokenNode"] = []
        self.sequence_ids = []
        self.tree_nodes = []
    
    def set_sequence_ids(self, sequence_ids: list[int]):
        self.sequence_ids = sequence_ids.copy()


@trace_perf("tree_training._compress_token_tree")
def _compress_token_tree(root: TokenNode) -> CompressedTokenNode:
    compressed_root = CompressedTokenNode(root.tree_id, -1, -1, terminates_here=root.is_end_of_sequence)
    
    def _compress_from(node: TokenNode, ancestors: list[CompressedTokenNode]) -> CompressedTokenNode:
        tokens: list[int] = []
        end_flags: list[bool] = []
        current = node
        while True:
            tokens.append(current.token_id)
            end_flags.append(current.is_end_of_sequence)
            if len(current.children) != 1 or current.is_end_of_sequence:
                break
            # Only one child; continue along the chain without branching.
            (_, next_child) = next(
                iter(sorted(current.children.items(), key=lambda item: item[0]))
            )
            if current.sequence_ids != next_child.sequence_ids:
                raise ValueError(
                    "Sequence IDs do not match along compression path."
                    f" Current IDs: {current.sequence_ids}, Next IDs: {next_child.sequence_ids}"
                )
            if next_child.node_id != current.node_id + 1:
                raise ValueError(
                    "Node IDs are not consecutive along compression path."
                )
            current = next_child

        compressed = CompressedTokenNode(root.tree_id, node.node_id, current.node_id, tokens, end_flags)
        compressed.ancestors = ancestors.copy()
        compressed.set_sequence_ids(current.sequence_ids)
        compressed_root.tree_nodes.append(compressed)
        if current.children:
            compressed.children = {
                token: _compress_from(child, ancestors + [compressed])
                for token, child in sorted(current.children.items(), key=lambda item: item[0])
            }
        return compressed
    
    if root.children:
        compressed_root.children = {
            token: _compress_from(child, [compressed_root])
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


def _insert_sequence(root: TokenNode, sequence: list[int], tree_id: int, sequence_id: int) -> None:
    """Insert ``sequence`` into ``root`` (mutates tree in-place)."""
    current = root
    for token in sequence:
        if token not in current.children:
            num_tokens = len(root.tree_nodes)
            current.children[token] = TokenNode(tree_id, token, num_tokens)
            root.tree_nodes.append(current.children[token])        
        current.children[token].add_sequence(sequence_id)
        current = current.children[token]
    current.is_end_of_sequence = True

@trace_perf("tree_training.parse_tree_infos")
def parse_tree_infos(roots: list[CompressedTokenNode]) -> list[dict[str, Any]]:
    """Parse tree infos from compressed token trees."""
    tree_infos: list[dict[str, Any]] = []
    for root in roots:
        sequence_indices = set()
        seq_id_to_tree_indices = {}
        tree_endpoints_to_seq_info = {}
        positions = defaultdict(int)

        for node in root.tree_nodes:
            for seq_id in node.sequence_ids:
                sequence_indices.add(seq_id)
                if seq_id not in seq_id_to_tree_indices:
                    seq_id_to_tree_indices[seq_id] = []
                seq_id_to_tree_indices[seq_id].append((node.start_node_id, node.end_node_id))
                if (node.start_node_id, node.end_node_id) not in tree_endpoints_to_seq_info:
                    tree_endpoints_to_seq_info[(node.start_node_id, node.end_node_id)] = (seq_id, positions[seq_id])
                positions[seq_id] += node.end_node_id - node.start_node_id + 1

        tree_info = {
            "tree_id": root.tree_id,
            "sequence_ids": sorted(list(sequence_indices)),
            "seq_id_to_tree_indices": seq_id_to_tree_indices,
            "tree_endpoints_to_seq_info": tree_endpoints_to_seq_info,
        }
        tree_infos.append(tree_info)
    return tree_infos


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

    for seq_id, seq in enumerate(normalized_sequences):
        inserted = False

        for tree_id, tree in enumerate(forests):
            additional = _count_additional_nodes(tree["root"], seq)
            if tree["nodes"] + additional <= max_tokens_per_tree:
                _insert_sequence(tree["root"], seq, tree_id, seq_id)
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

        new_tree_id = len(forests)
        new_root = TokenNode(new_tree_id, -1, -1)
        _insert_sequence(new_root, seq, new_tree_id, seq_id)
        forests.append({"root": new_root, "nodes": additional})

    roots = [tree["root"] for tree in forests]
    node_counts = [tree["nodes"] for tree in forests]

    compressed_roots = [_compress_token_tree(root) for root in roots]
    tree_infos = parse_tree_infos(compressed_roots)

    if visualize:
        for idx, compressed_root in enumerate(compressed_roots):
            visualization = _render_compressed_tree(compressed_root)
            logger.info("Token Tree %d Visualization:\n%s", idx, visualization)

    return compressed_roots, node_counts, tree_infos

@trace_perf("tree_training.build_tree_input")
def build_tree_input(data: dict[str, Any], max_tokens_per_tree: int):
    """ First construct token trees from input data, then convert input data into tree-packed format.
    The return value should be a list of dictionaries, each contains input_ids, attention_mask, and tree infos for a packed tree structure.
    The input id should be a flattened list of token ids in the tree structure with pre-ordered traversal.
    The attention mask represents the causal relationship between tokens in the token tree, in which entries are set to true when 
    two tokens are in the same sequence and follows causal relationship (lower triangular causal mask).

    Returns:
        tuple[list[CompressedTokenNode], list[int], list[dict[str, Any]]]:
            ``roots`` of the packed token trees, token counts per tree, and tree-packed inputs with per-sequence indices.
    """
    roots, num_tree_tokens_list, tree_infos = greedy_build_tree(data, max_tokens_per_tree=max_tokens_per_tree)
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
    non_packable_keys = set(data.keys()) - packable_key_set - {"input_ids", "attention_mask"}
    seq_lens = data["attention_mask"].sum(dim=1, dtype=torch.int32)  
    count = 0
    for root, num_tree_tokens, tree_info in zip(roots, num_tree_tokens_list, tree_infos):
        sequence_ids = tree_info["sequence_ids"]
        seq_id_to_tree_indices = tree_info["seq_id_to_tree_indices"]
        tree_endpoints_to_seq_info = tree_info["tree_endpoints_to_seq_info"]

        with trace_scope("tree_training.build_tree_input.pack_input_ids"):
            input_ids: list[int] = torch.empty((num_tree_tokens,), dtype=input_template.dtype, device=input_template.device)
            for (tree_start, tree_end), (seq_id, seq_start) in tree_endpoints_to_seq_info.items():
                input_ids[tree_start:tree_end + 1] = input_template[seq_id][seq_start:seq_start + (tree_end - tree_start + 1)]
        
        with trace_scope("tree_training.build_tree_input.build_attention_mask"):
            mask_tensor = mask_template.new_zeros((num_tree_tokens, num_tree_tokens))
            tril_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

            for seq_id in sequence_ids:
                segments = seq_id_to_tree_indices.get(seq_id, [])
                if not segments:
                    continue

                seq_position_chunks = [
                    torch.arange(start, end + 1, device=mask_tensor.device)
                    for start, end in segments
                    if end >= start
                ]
                if not seq_position_chunks:
                    continue

                seq_positions = torch.cat(seq_position_chunks, dim=0)
                seq_len = seq_positions.numel()
                if seq_len == 0:
                    continue

                rows_cols = tril_cache.get(seq_len)
                if rows_cols is None:
                    rows_cols = torch.tril_indices(seq_len, seq_len, device=mask_tensor.device)
                    tril_cache[seq_len] = rows_cols
                rows, cols = rows_cols
                mask_tensor[seq_positions[rows], seq_positions[cols]] = True
        
        lens = [seq_lens[seq_id].item() for seq_id in sequence_ids]
        cu_seqlens = torch.cumsum(torch.tensor([0] + lens, dtype=torch.int32), dim=0)
        packed_tree = {
            "input_ids": input_ids,
            "attention_mask": mask_tensor,
            "sequence_ids": sequence_ids,
            "seq_id_to_tree_indices": seq_id_to_tree_indices,
            "tree_endpoints_to_seq_info": tree_endpoints_to_seq_info,
            "cu_seqlens": cu_seqlens,
        }
        
        with trace_scope("tree_training.build_tree_input.pack_others"):
            for packable_key in packable_key_set:
                packable_value = data[packable_key]
                packed_value = torch.empty((sum(lens), *packable_value.shape[2:]), dtype=packable_value.dtype, device=packable_value.device)
                cursor = 0
                for length, seq_id in zip(lens, sequence_ids):
                    packed_value[cursor: cursor + length] = packable_value[seq_id][:length]
                    cursor += length
                packed_tree[packable_key] = packed_value

        for non_packable_key in non_packable_keys:
            packed_tree[non_packable_key] = data[non_packable_key]

        packed_trees.append(packed_tree)
        count += 1
    return roots, num_tree_tokens_list, packed_trees

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

def get_seq_lens(
    sequence_ids: list[int],
    seq_id_to_tree_indices: dict[int, list[int]],
) -> list[int]:
    """Get sequence lengths from tree indices mapping.

    Args:
        sequence_ids (list[int]): List of sequence IDs.
        seq_id_to_tree_indices (dict[int, list[int]]): Mapping from sequence ID to list of (start, end) indices in the tree.

    Returns:
        list[int]: List of sequence lengths corresponding to ``sequence_ids``.
    """
    sequence_lens = []
    for seq_id in sequence_ids:
        if seq_id not in seq_id_to_tree_indices:
            raise ValueError(f"Sequence ID {seq_id} not found in seq_id_to_tree_indices.")
        indices = seq_id_to_tree_indices[seq_id]
        length = sum(end - start + 1 for start, end in indices)
        sequence_lens.append(length)
    return sequence_lens

@trace_perf("tree_training.packed_tree_gather_logprobs")
def packed_tree_gather_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    sequence_ids: list[int],
    seq_id_to_tree_indices: dict[int, list[int]],
    temperature: float = 1.0,
    calculate_entropy: bool = False,
) -> torch.Tensor:
    """Gather log probabilities for sequences represented by tree-packed logits.
    """

    if input_ids.ndim != 1:
        raise ValueError("input_ids must be a 1D tensor of tree tokens.")
    if logits.ndim != 2 or logits.shape[0] != input_ids.shape[0]:
        raise ValueError("logits must be 2D with first dim equal to len(input_ids).")

    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    seq_lens = get_seq_lens(sequence_ids, seq_id_to_tree_indices)
    total_tokens = sum(seq_lens)
    flattened_logprobs = logits.new_empty(total_tokens)
    flattened_entropies = logits.new_empty(total_tokens) if calculate_entropy else None
    cursor = 0

    for seq_len, seq_id in zip(seq_lens, sequence_ids):
        if seq_id not in seq_id_to_tree_indices:
            raise ValueError(f"Sequence ID {seq_id} not found in seq_id_to_tree_indices.")
        tree_indices_list = seq_id_to_tree_indices[seq_id]
        
        tree_token_segments = []
        tree_logits_segments = []
        for start, end in tree_indices_list:
            tree_token_segments.append(input_ids[start:end+1])
            tree_logits_segments.append(logits[start:end+1])
        tree_tokens = torch.cat(tree_token_segments, dim=0)
        labels = torch.roll(tree_tokens, shifts=-1, dims=-1)
        tree_logits = torch.cat(tree_logits_segments, dim=0)

        if calculate_entropy:
            seq_logprobs, seq_entropies = gather_logprobs_entropy(
                tree_logits, labels, temperature
            )
            flattened_logprobs[cursor : cursor + seq_len] = seq_logprobs
            flattened_entropies[cursor : cursor + seq_len] = seq_entropies
        else:
            seq_logprobs = gather_logprobs(tree_logits, labels, temperature)
            flattened_logprobs[cursor : cursor + seq_len] = seq_logprobs
        cursor += seq_len

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
    sig = inspect.signature(get_te_tree_gpt_decoder_block_spec)
    self.has_vp_stage = "vp_stage" in sig.parameters  # for mcore 0.12 compatibility
    extra_args = {}
    if self.has_vp_stage:
        extra_args["vp_stage"] = vp_stage
    transformer_layer_spec = get_te_tree_gpt_decoder_block_spec(
        self.config, **extra_args
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

