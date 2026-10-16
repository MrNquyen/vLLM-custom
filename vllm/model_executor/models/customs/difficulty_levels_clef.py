# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HUB's Clef decision model (``src/models/decision_model/clef.py``) on vLLM.

The Qwen3.5 backbone is vLLM's ``Qwen3_5ForConditionalGeneration``, so the
checkpoint's LoRA must be merged into it first. ``ClefDecisionHead`` is HUB's
head reading one prompt at a time instead of a padded batch. It keeps HUB's
parameter names, so ``joint_head.safetensors`` loads as HUB saved it, and
``joint_head_config.json`` may use HUB's names (``embed_dim``, ``num_heads``)
or Clef-Flash's (``width``, ``heads``).

Requests carry their schema as for Clef-Flash; see ``clef_joint_schema``.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from ..clef_joint_schema import JointSchemaPooler, SchemaIndices, segment_mean
from ..interfaces_base import default_pooling_type
from ..qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from ..qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor

HEAD_FILES = ("joint_head.safetensors", "joint_head_config.json")
HEAD_CONFIG_ALIASES = {"width": "embed_dim", "heads": "num_heads"}
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
MAX_LOGIT_SCALE = math.log(100.0)


class EvidenceRoutingLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, feedforward: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(embed_dim)
        # Identity takes the place of HUB's dropout, keeping its weight indices.
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(feedforward, embed_dim),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Separate key and value tensors, like HUB, keep attention's unpacked
        # input projections.
        routed, _ = self.attention(
            self.query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + routed
        return queries + self.feedforward(self.feedforward_norm(queries))


class ClefDecisionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        embed_dim: int,
        routing_layers: int,
        layers: int,
        num_heads: int,
        feedforward: int,
    ) -> None:
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, embed_dim, bias=False)
        self.question_projection = nn.Linear(hidden_size, embed_dim, bias=False)
        self.option_question_projection = nn.Linear(
            hidden_size, embed_dim, bias=False
        )
        self.global_projection = nn.Linear(hidden_size, embed_dim, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, embed_dim, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, embed_dim, bias=False)
        self.type_embedding = nn.Embedding(len(QUESTION_TYPES), embed_dim)
        self.evidence_layers = nn.ModuleList(
            EvidenceRoutingLayer(embed_dim, num_heads, feedforward)
            for _ in range(routing_layers)
        )
        self.option_summary_norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=feedforward,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.field_norm = nn.LayerNorm(embed_dim)
        self.option_norm = nn.LayerNorm(embed_dim)
        self.residual_scorer = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(embed_dim, 1),
        )
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        lexical: torch.Tensor,
        schema: SchemaIndices,
    ) -> torch.Tensor:
        """HUB's ``ClefDecisionHead.forward`` for a single unpadded prompt.

        Args:
            hidden_states: The prompt's ``[L, hidden]`` final hidden states.
            lexical: Each option's ``[O, hidden]`` mean output embedding.
            schema: Where the prompt's questions and options are.

        Returns:
            The ``[O]`` option logits, in ``schema`` order.
        """
        num_questions = schema.question_types.shape[0]
        option_question = schema.option_question
        hidden_states = self.hidden_norm(hidden_states)
        memory = self.memory_projection(hidden_states).unsqueeze(0)
        global_embed = hidden_states[-1]

        # 1. Mean over each question / option span.
        means = segment_mean(
            hidden_states[schema.positions], schema.span_ids, schema.span_lengths
        )
        question_embed, options_embed = means[:num_questions], means[num_questions:]

        # 2. Option queries routed over the memory.
        routed_options = (
            self.option_context_projection(options_embed)
            + self.option_lexical_projection(lexical)
            + self.option_question_projection(question_embed)[option_question]
        ).unsqueeze(0)
        for layer in self.evidence_layers:
            routed_options = layer(routed_options, memory)
        routed_options = routed_options[0]

        # 3. Fields: each question's summary of its routed options, laid out
        # as a [Q, max_options] grid so padded options can be masked.
        base_fields = self.question_projection(question_embed)
        embed_dim = routed_options.shape[-1]
        grid = routed_options.new_zeros(schema.option_mask.numel(), embed_dim)
        grid[schema.option_slots] = routed_options
        grid = grid.view(*schema.option_mask.shape, embed_dim)
        routing_scores = torch.einsum("qke,qe->qk", grid, base_fields) / math.sqrt(
            embed_dim
        )
        routing_scores = routing_scores.masked_fill(
            ~schema.option_mask, torch.finfo(routing_scores.dtype).min
        )
        option_summaries = torch.einsum(
            "qk,qke->qe", routing_scores.softmax(dim=-1), grid
        )
        fields = (
            base_fields
            + self.option_summary_norm(option_summaries)
            + self.global_projection(global_embed).unsqueeze(0)
            + self.type_embedding(schema.question_types)
        ).unsqueeze(0)
        for layer in self.layers:
            fields = layer(fields, memory)
        fields = self.field_norm(fields[0])[option_question]

        # 4. Logits = lexical prior + gated (cosine + residual) joint score.
        anchor = F.normalize(question_embed + global_embed, dim=-1)[option_question]
        prior_scale = self.prior_logit_scale.clamp(max=MAX_LOGIT_SCALE).exp()
        prior = prior_scale * (F.normalize(lexical, dim=-1) * anchor).sum(dim=-1)

        options = self.option_norm(routed_options)
        cosine = F.cosine_similarity(fields, options, dim=-1)
        features = torch.cat(
            [fields, options, fields * options, torch.abs(fields - options)], dim=-1
        )
        residual = self.residual_scorer(features).squeeze(-1)
        joint_scale = self.joint_logit_scale.clamp(max=MAX_LOGIT_SCALE).exp()
        joint = joint_scale * cosine + residual
        return prior + torch.sigmoid(self.residual_gate) * joint


def load_head_config(model_config: ModelConfig) -> dict[str, Any]:
    """The checkpoint's head config, under ``ClefDecisionHead``'s names."""
    head_config = getattr(model_config.hf_config, "joint_head_config", None)
    if head_config is None:
        head_config = get_hf_file_to_dict(
            HEAD_FILES[1], model_config.model, model_config.revision
        )
    if head_config is None:
        raise ValueError(f"{model_config.model} has no {HEAD_FILES[1]}")
    head_config = {
        HEAD_CONFIG_ALIASES.get(key, key): value for key, value in head_config.items()
    }
    # Dropout only applies in training.
    head_config.pop("dropout", None)
    hidden_size = model_config.hf_text_config.hidden_size
    if head_config["hidden_size"] != hidden_size:
        raise ValueError(
            f"joint head hidden_size={head_config['hidden_size']}, "
            f"backbone hidden_size={hidden_size}"
        )
    return head_config


@default_pooling_type(tok_pooling_type="ALL")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class DifficultyLevelsClefForJointSchema(Qwen3_5ForConditionalGeneration):
    is_pooling_model = True
    # Only the backbone's shards; secondary_weights loads the head's file.
    allow_patterns_overrides = ["model*.safetensors"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        model_config = vllm_config.model_config
        self.pooler = JointSchemaPooler(
            ClefDecisionHead(**load_head_config(model_config)),
            self._embed_lexical,
        )
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=model_config.model,
                revision=model_config.revision,
                prefix="pooler.head.",
                allow_patterns_overrides=[HEAD_FILES[0]],
            )
        ]

    def _embed_lexical(self, token_ids: torch.Tensor) -> torch.Tensor:
        # HUB embeds options with the output embeddings. ParallelLMHead
        # forbids lookups, so use the vocab-parallel embedding's.
        return VocabParallelEmbedding.forward(self.language_model.lm_head, token_ids)
