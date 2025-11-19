from typing import List, Optional, Tuple, Union
from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from model.m2t_arch import M2TMetaModel, M2TMetaForCausalLM



class M2TConfig(LlamaConfig):
    model_type = "m2t_llama"

class M2TLlamaModel(M2TMetaModel, LlamaModel):
    config_class = M2TConfig

    def __init__(self, config: LlamaConfig):
        super(M2TLlamaModel, self).__init__(config)


class M2TLlamaForCausalLM(LlamaForCausalLM, M2TMetaForCausalLM):
    config_class = M2TConfig

    def __init__(self, config):
        # here could print config for debug
        super(LlamaForCausalLM, self).__init__(config)
        self.model = M2TLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size # vocab_size; hidden_size: 5120
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # laster layer of transformer, mapping hidden_size feature to vocab_size

        # Initialize weights and apply final processing
        # A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        # modules properly initialized (such as weight initialization).
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
            motion: Optional[torch.FloatTensor] = None,
            cache_position = None,
            motion_type = None, # "joint" or "humanml"
            # image_sizes: Optional[List[List[int]]] = None, # this might be not useful in our case, since we don't need to padding images
            # but motion length may be necessary. and we still don't have it in data part.
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            # for tcn, but at last we don't use it, it has some bugs
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, motion, motion_type=motion_type)

            # in this case, input_ids is None, 
            # inputs_embeds is the embedding of tokens + motion_embedding.


        return super().forward(
            input_ids=input_ids, # should be None
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self, 
        inputs: Optional[torch.Tensor] = None, # input_ids
        images: Optional[torch.Tensor] = None, # motions
        image_sizes: Optional[torch.Tensor] = None, 
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "input_embeds" in kwargs:
            raise NotImplementedError("input_embeds not supported")
  
        if images is not None:
            print(len(images))
            print("images.shape: ", images[0].shape)
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, dtype=torch.float32
            )

        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        print("input_embeds.shape: ", inputs_embeds.shape)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        motion = kwargs.pop("motion", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if motion is not None:
            inputs["motion"] = motion
        return inputs

AutoConfig.register("m2t_llama", M2TConfig)
AutoModelForCausalLM.register(M2TConfig, M2TLlamaForCausalLM)