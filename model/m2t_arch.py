from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from model.builder.mm_projector import build_vision_projector, build_vision_tower
from utils.utils import IGNORE_INDEX, MOTION_TOKEN_INDEX, DEFAULT_MOTION_PATH_TOKEN, DEFAULT_MOTION_START_TOKEN, DEFAULT_MOTION_END_TOKEN


def get_position_embedding(L, D, dtype=torch.float32):
    """
    sinusoidal position embedding
    """
    position = torch.arange(L, dtype=dtype).unsqueeze(1) # L,1
    div_term = torch.exp(torch.arange(0, D, 2).float() * -(torch.log(torch.tensor(10000.0)) / D))
    pos_embedding = torch.zeros(L, D)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)

    return pos_embedding

class M2TMetaModel:
    def __init__(self, config):
        super(M2TMetaModel, self).__init__(config)
        # TODO: here can still add the vision_tower for motion encoder, just same as vision encoder
        
        if hasattr(config, "mm_vision_tower"): # vision tower is a 2-layer mlp
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_vision_projector(config) # map vision feature to space
    
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        # mm_projector = getattr(self, 'mm_projector', None)
        # if type(mm_projector) is list:
        #     mm_projector = mm_projector[0]
        return vision_tower
    
    def initialize_motion_modules(self, args, fsdp=None):
        # initialize encoders and projector layers for motion encoder
        vision_tower = args.vision_tower # mlp or attn
        pretrain_mm = args.pretrain_mm # include vision_tower and mm_projector
        mm_patch_merge_type = args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None: # yes, None.
            vision_tower = build_vision_tower(args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_state_dict(pretrain_mm, strict=False)
        
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(args, 'mm_projector_type', 'linear')
        # self.config.mm_hidden_size = vision_tower.hidden_size
        # HERE: vision_tower is Sequential, which has no hidden_size. therefore we use mm_hidden_size
        self.config.mm_hidden_size = args.mm_hidden_size # 1024
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.pretrain_mm = pretrain_mm
        # NO mm_projector in pretrain stage. also NO projector in lora finetune stage...

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type: # None
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
        if pretrain_mm is not None: 
            mm_projector_weights = torch.load(pretrain_mm, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            self.vision_tower.load_state_dict(get_w(mm_projector_weights, 'vision_tower'), strict=False)

class M2TMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def embed_motion(self, motions):
        motion_embeddings = self.get_model().get_vision_tower()(motions) # first vision_tower is a 2-layer mlp
        motion_embeddings = self.get_model().mm_projector(motion_embeddings) # map motion feature to space
        return motion_embeddings
    
    def motion_to_patch(self, motions, W=16, S=8):
        motion_patches = []
        for motion in motions:
            # do patches
            l = motion.shape[0]
            motion = motion.reshape(l, -1)
            patches = []
            for i in range(0, len(motion) - W + 1, S):
                patch = motion[i:i+W].reshape(1, -1)
                patches.append(patch)
            patches = torch.cat(patches, dim=0) # N_patches,66*W
            motion_patches.append(patches)
        return motion_patches
    
    def pad_motion(self, x, pad_L, pad_value=0):
        return torch.cat((x, torch.full((pad_L - x.shape[0], x.shape[1]), pad_value, dtype=x.dtype, device=x.device)))

    # TODO
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask,
                                             past_key_values, labels, motions, dtype=torch.bfloat16, motion_type="joint"
        ):
        vision_tower = self.get_vision_tower()

        if vision_tower is None or motions is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # input_ids: [B, L] -> [8, L]
        # motions: List[Tensor] -> len: 8; each: [N_len, 22, 3]
        B = len(motions) # FIXME defination of B,W,S should be outside, use the args configuration.
        W = 16 # window_size
        S = 8 # slide size

        if motions[0].shape[-1] == 263: # 263 is the joint number of humanml
            W = 8
            S = 8

        # 1. motions should be prepared as batch data
        # 2. then embed the motions to get the feature: [B,L,4096]
        # x should be [1, 66]
        # motions_patches = self.motion_to_patch(motions, W, S) if not motions[0].shape[-1] == 263 else motions
        motions_patches = self.motion_to_patch(motions, W, S)
        p_len = [len(x) for x in motions_patches]
        max_p = max(p_len)

        motion_patch_mask = torch.tensor([[1] * len(x) + [0] * (max_p - len(x)) for x in motions_patches])
        motion_patch_mask = motion_patch_mask.bool()

        motions_patches = torch.stack([self.pad_motion(x, max_p) for x in motions_patches]) # B, N_patches, 66*W 
        motions_patches = motions_patches.to(dtype=dtype)
        motions_features = self.embed_motion(motions_patches)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = attention_mask.bool()
        
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # remove the padding using attention_mask
        _input_ids = input_ids
        
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_m_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # basically, in each data there is just one motion. num_motions=1
            num_motions = (cur_input_ids == MOTION_TOKEN_INDEX).sum()

            if num_motions == 0: # NO
                cur_motion_features = motions_features[cur_m_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_motion_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_m_idx += 1
                continue
            
            m_token_indices = [-1] + torch.where(cur_input_ids == MOTION_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

            # delete MOTION_TOKEN_INDEX
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(m_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[m_token_indices[i]+1:m_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[m_token_indices[i]+1:m_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0) # get_embedding, concat, then split and recover to original shape
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_motions + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_motions:
                    cur_motion_features = motions_features[cur_m_idx][motion_patch_mask[cur_m_idx]]
                    cur_m_idx += 1
                    cur_new_input_embeds.append(cur_motion_features)
                    cur_new_labels.append(torch.full((cur_motion_features.shape[0], ), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)


        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None) # 2048
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_vision_tokenizer(self, args, tokenizer):
        if args.mm_use_start_end: # False
            num_new_tokens = tokenizer.add_tokens([DEFAULT_MOTION_START_TOKEN, DEFAULT_MOTION_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif args.mm_use_patch_tokens: # False
            if args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
