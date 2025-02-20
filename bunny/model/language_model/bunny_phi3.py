from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from .phi3 import Phi3Model, Phi3Config, Phi3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..bunny_arch import BunnyMetaModel, BunnyMetaForCausalLM


class BunnyPhi3Config(Phi3Config):
    model_type = "bunny-phi3"


class BunnyPhi3Model(BunnyMetaModel, Phi3Model):
    config_class = BunnyPhi3Config

    def __init__(self, config: Phi3Config):
        super(BunnyPhi3Model, self).__init__(config)


class BunnyPhi3ForCausalLM(Phi3ForCausalLM, BunnyMetaForCausalLM):
    config_class = BunnyPhi3Config

    def __init__(self, config):
        super(Phi3ForCausalLM, self).__init__(config)
        self.model = BunnyPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            '/zhaobai46d/share/bunny/models/Phi-3-mini-128k-instruct',
            cache_dir=None,
            model_max_length=3000,
            padding_side="right",
            use_fast=True,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            windows: Optional[list] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            windows=windows
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        images = kwargs.pop("images", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("bunny-phi3", BunnyPhi3Config)
AutoModelForCausalLM.register(BunnyPhi3Config, BunnyPhi3ForCausalLM)
