from __future__ import annotations

import torch
from torch import nn

from .gpa import GatedPoolingAttention
from .multimodal import MultiModalProjector
from .router import QuestionGuidedTokenRouter


class EfficientVLMForAD(nn.Module):
    def __init__(
        self,
        *,
        text_model: nn.Module,
        visual_adapter: nn.Module | None = None,
        visual_fusion: nn.Module | None = None,
        vision_dim: int,
        d_model: int,
        seq_len: int,
        gpa_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.text_model = text_model
        self.visual_adapter = visual_adapter
        self.visual_fusion = visual_fusion
        self.seq_len = seq_len
        self.gpa = GatedPoolingAttention(seq_len=seq_len, input_dim=vision_dim, hidden_size=gpa_hidden_size)
        projector_seq_len = visual_fusion.top_k if isinstance(visual_fusion, QuestionGuidedTokenRouter) else seq_len
        self.projector = MultiModalProjector(input_dim=vision_dim, d_model=d_model, seq_len=projector_seq_len)
        self.modal_embeddings = nn.Embedding(2, d_model)
        side = int(seq_len**0.5)
        self.row_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.col_embeddings = nn.Embedding(side, vision_dim) if side * side == seq_len else None
        self.last_gpa_weights: torch.Tensor | None = None

    @property
    def visual_token_len(self) -> int:
        return self.projector.seq_len

    def add_spatial_embeddings(self, visual_features: torch.Tensor) -> torch.Tensor:
        if self.row_embeddings is None or self.col_embeddings is None:
            return visual_features
        side = self.row_embeddings.num_embeddings
        rows = torch.arange(side, device=visual_features.device).repeat_interleave(side)
        cols = torch.arange(side, device=visual_features.device).repeat(side)
        spatial = self.row_embeddings(rows) + self.col_embeddings(cols)
        return visual_features + spatial.view(1, 1, self.seq_len, -1)

    def prepare_multimodal_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor | None = None,
        target_optimizer_steps: int | None = None,
        blind_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Prepare multimodal embeddings.
        If blind_mode is True or visual_features is None:
            Visual tokens are masked out or zeroed, enabling pure text (blind) inference.
        """
        text_tokens = self.text_model.get_input_embeddings()(input_ids)
        batch_size = input_ids.shape[0]
        debug_info = {}

        if blind_mode or visual_features is None:
            # Blind branch (L branch in understanding_lp): No vision input
            text_tokens = text_tokens + self.modal_embeddings(torch.zeros_like(input_ids))
            inputs_embeds = text_tokens
            combined_mask = attention_mask
            return inputs_embeds, combined_mask, debug_info

        # Multimodal branch (VL branch)
        if self.visual_adapter is not None:
            visual_features = self.visual_adapter(visual_features, text_tokens, attention_mask)
        visual_features = self.add_spatial_embeddings(visual_features)

        if self.visual_fusion is not None:
            fused = self.visual_fusion(
                visual_features,
                text_tokens,
                attention_mask,
                target_optimizer_steps=target_optimizer_steps,
            )
            if hasattr(self.visual_fusion, "last_debug"):
                debug_info = self.visual_fusion.last_debug
        else:
            fused, weights = self.gpa(visual_features)
            self.last_gpa_weights = weights
            debug_info["gpa_weights"] = weights

        visual_tokens = self.projector(fused)
        visual_tokens = visual_tokens + self.modal_embeddings(
            torch.ones((visual_tokens.shape[0], visual_tokens.shape[1]), dtype=torch.long, device=visual_tokens.device)
        )
        text_tokens = text_tokens + self.modal_embeddings(torch.zeros_like(input_ids))

        inputs_embeds = torch.cat([visual_tokens, text_tokens], dim=1)
        visual_mask = torch.ones(
            (attention_mask.shape[0], self.visual_token_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        return inputs_embeds, combined_mask, debug_info

    def forward_from_features(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        target_optimizer_steps: int | None = None,
        output_hidden_states: bool = False,
        blind_mode: bool = False,
    ):
        inputs_embeds, combined_mask, _ = self.prepare_multimodal_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            target_optimizer_steps=target_optimizer_steps,
            blind_mode=blind_mode,
        )

        kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": combined_mask,
            "labels": labels,
        }
        if labels is None:
            # T5 requires decoder_input_ids when labels is not provided
            decoder_start_token_id = getattr(self.text_model.config, "decoder_start_token_id", 0)
            if decoder_start_token_id is None:
                decoder_start_token_id = 0
            kwargs["decoder_input_ids"] = torch.full(
                (inputs_embeds.shape[0], 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        if output_hidden_states:
            kwargs["output_hidden_states"] = True

        output = self.text_model(**kwargs)
        if isinstance(self.visual_fusion, QuestionGuidedTokenRouter) and hasattr(self.visual_fusion, "last_debug"):
            if "diversity_loss_weighted" in self.visual_fusion.last_debug and getattr(output, "loss", None) is not None:
                output.diversity_loss_raw = self.visual_fusion.last_debug.get("diversity_loss_raw")
                output.diversity_loss_weighted = self.visual_fusion.last_debug.get("diversity_loss_weighted")
                output.loss = output.loss + output.diversity_loss_weighted
        return output

    def get_encoder_hidden_states(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor | None = None,
        blind_mode: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """
        Extract hidden states directly from T5 Encoder across all layers.
        Returns a tuple of tensors (one per layer, from layer 0 to L).
        """
        inputs_embeds, combined_mask, _ = self.prepare_multimodal_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            blind_mode=blind_mode,
        )
        encoder_outputs = self.text_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return encoder_outputs.hidden_states

    def extract_last_token_representations(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor | None = None,
        blind_mode: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Extract exact last prompt token representations h_l across all layers.
        Uses dynamic attention_mask length indexing to prevent corrupting embeddings with <pad> tokens in batches.
        Optionally extracts layer-wise visual attention weights if output_attentions=True.

        Returns:
            hidden_states: Tensor [B, num_layers, D]
            visual_attentions: Tensor [B, num_transformer_layers] or None
        """
        inputs_embeds, combined_mask, _ = self.prepare_multimodal_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            blind_mode=blind_mode,
        )

        encoder_outputs = self.text_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
        )

        batch_size = input_ids.shape[0]
        device = inputs_embeds.device

        # Compute exact last prompt token index per sample
        valid_text_lengths = attention_mask.sum(dim=1).long()  # [B]
        if not blind_mode and visual_features is not None:
            last_token_indices = (self.visual_token_len + valid_text_lengths - 1).long()
        else:
            last_token_indices = (valid_text_lengths - 1).long()

        batch_indices = torch.arange(batch_size, device=device)

        # Extract last token vector for every layer: layer_tensor[b, last_idx, :] -> [B, D]
        last_tokens_per_layer = []
        for layer_tensor in encoder_outputs.hidden_states:
            selected = layer_tensor[batch_indices, last_token_indices, :]
            last_tokens_per_layer.append(selected.detach().cpu().to(torch.float32))

        final_hidden_states = torch.stack(last_tokens_per_layer, dim=1)  # [B, num_layers, D]

        final_visual_attentions = None
        if output_attentions and encoder_outputs.attentions and not blind_mode and visual_features is not None:
            layer_attn_weights = []
            k_vis = self.visual_token_len
            for layer_attn in encoder_outputs.attentions:
                if layer_attn is not None:
                    # layer_attn: [B, num_heads, seq_len, seq_len]
                    # Query token attention to visual tokens: [B, num_heads, K]
                    query_to_vis = layer_attn[batch_indices, :, last_token_indices, :k_vis]
                    vis_weight = query_to_vis.sum(dim=-1).mean(dim=1)  # sum over visual tokens, mean over heads -> [B]
                    layer_attn_weights.append(vis_weight.detach().cpu().to(torch.float32))
            if layer_attn_weights:
                final_visual_attentions = torch.stack(layer_attn_weights, dim=1)  # [B, num_transformer_layers]

        return final_hidden_states, final_visual_attentions

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        target_optimizer_steps: int | None = None,
        output_hidden_states: bool = False,
        blind_mode: bool = False,
    ):
        return self.forward_from_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            labels=labels,
            target_optimizer_steps=target_optimizer_steps,
            output_hidden_states=output_hidden_states,
            blind_mode=blind_mode,
        )

    def generate_from_features(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_features: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        num_beams: int = 1,
        early_stopping: bool = False,
        length_penalty: float = 1.0,
        blind_mode: bool = False,
    ) -> torch.Tensor:
        was_training = False
        if self.visual_fusion is not None:
            was_training = self.visual_fusion.training
            self.visual_fusion.eval()

        try:
            inputs_embeds, combined_mask, _ = self.prepare_multimodal_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_features=visual_features,
                blind_mode=blind_mode,
            )
        finally:
            if self.visual_fusion is not None:
                self.visual_fusion.train(was_training)

        return self.text_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=early_stopping,
            length_penalty=length_penalty,
        )
