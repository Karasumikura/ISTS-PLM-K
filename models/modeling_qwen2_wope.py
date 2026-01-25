from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
import torch
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import BaseModelOutputWithPast

class Qwen2Model_wope(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # [MODIFIED] Disable internal RoPE by forcing position_ids to zero
        # This assumes that the input embeddings (inputs_embeds) already have 
        # Time-RoPE or other positional encodings applied.
        if position_ids is None:
            if input_ids is not None:
                shape = input_ids.shape
                device = input_ids.device
            elif inputs_embeds is not None:
                shape = inputs_embeds.shape[:2]
                device = inputs_embeds.device
            else:
                raise ValueError("Both input_ids and inputs_embeds are None")
            
            # Create a zero-filled position_ids tensor
            # Since Qwen2RotaryEmbedding uses these ids to calculate cos/sin, 
            # all zeros will result in the same rotation (effectively none relative to each other)
            # or a constant rotation for all tokens, preserving the relative info from TimeRoPE.
            position_ids = torch.zeros(shape, dtype=torch.long, device=device)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
