from typing import Optional, Tuple, Union
from functools import partial
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast, LlamaModel, Unpack, FlashAttentionKwargs
from transformers.cache_utils import DynamicCache, Cache


def get_allowed_tokens(stoi, vocab_size):
    index_tokens = [stoi['none']]
    for ind_token in range(0, 1000): # 1000 is an arbitrary, but sufficiently large number for the number of tokens we have.
        try:
            index_tokens.append(stoi[str(ind_token)])
        except:
            pass
    index_tokens = torch.tensor(index_tokens)

    allow_tokens = torch.zeros(vocab_size, dtype=torch.bool)
    allow_tokens[index_tokens] = True

    all_allowed_tokens = torch.ones(vocab_size, dtype=torch.bool)
    return allow_tokens, all_allowed_tokens


def get_allowed_tokens_with_unexplored_children(stoi, vocab_size, len_unexplored_children):
    index_tokens = []

    if len_unexplored_children == 0:
        index_tokens.append(stoi['none'])
    else:
        for ind_token in range(0, len_unexplored_children):
            try:
                index_tokens.append(stoi[str(ind_token)])
            except:
                pass
    index_tokens = torch.tensor(index_tokens)

    allow_tokens = torch.zeros(vocab_size, dtype=torch.bool)
    allow_tokens[index_tokens] = True

    return allow_tokens


class MyLlamaConfig(LlamaConfig):
    pass


class MyLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        targets: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            print(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        B, L = hidden_states.size()[:2]
        D = self.config.hidden_size // self.config.num_attention_heads
        position_embeddings = torch.ones(B, L, D, device=hidden_states.device), torch.zeros(B, L, D, device=hidden_states.device)

        if self.config.rope_theta != torch.inf:
            rotary_dim = int(self.config.partial_rotary_factor * D)
            cos, sin = self.rotary_emb(hidden_states, position_ids)

            position_embeddings[0][:, :, :rotary_dim] = cos
            position_embeddings[1][:, :, :rotary_dim] = sin

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
#                    decoder_layer.__call__,
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        #return output if return_dict else output.to_tuple()
        #the above is the original return, I modified it to the following
        #logits = hidden_states
        logits = self.lm_head(hidden_states)
        
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            # this loss might be not applicable, needs experiments to check
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = logits[:, [-1], :]
            loss = None

        return logits, loss

    def distribution(self, idx, temperature=1.0, allowed_tokens=None, top_k=None):
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        assert len(logits.shape) == 2

        if allowed_tokens is not None:
            assert allowed_tokens.shape[0] == logits.shape[1]
            logits = logits.masked_fill(~allowed_tokens, float('-inf'))

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')

        dist = torch.distributions.Categorical(logits=logits)
        return dist

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, allowed_tokens=None, top_k=None):
        for _ in range(0, max_new_tokens):
            dist = self.distribution(idx, temperature=temperature, allowed_tokens=allowed_tokens, top_k=top_k)
            next_token = dist.sample()

            idx_next = torch.unsqueeze(next_token, 0)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

if __name__ == '__main__':
    model_config = MyLlamaConfig(
        vocab_size=102, # follow Nayoung but we need to change this
        hidden_size=384, # follow Nayoung
        intermediate_size=1536, # follow Nayoung
        num_attention_heads=6, # follow Nayoung
        num_hidden_layers=6, # follow Nayoungs
        max_position_embeddings=1024, # follow Nayoungs
        # _attn_implementation='flash_attention_2' if train_args.bf16 else 'sdpa',
        _attn_implementation='sdpa',
        rope_theta=torch.inf,
        partial_rotary_factor=1.0,
        use_lpe=False,
        attention_dropout=0.0
    )
    #model = MyLlamaModel(model_config).cuda()
    model = MyLlamaModel(model_config)

    # Dummy input
    batch_size = 2
    seq_length = 16
    vocab_size = model_config.vocab_size

    # Randomly generate some input_ids
    #input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).long().cuda()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).long()
    print("Input shape:", input_ids.shape)

    # Generate random targets (same shape as input_ids)
    #targets = torch.randint(0, vocab_size, (batch_size, seq_length)).long().cuda()
    targets = torch.randint(0, vocab_size, (batch_size, seq_length)).long()
    print("Target shape:", targets.shape)

    # Run forward pass
    #with torch.no_grad():
    logits, loss = model(input_ids=input_ids, targets=targets)

    loss.backward()

    # Print output shape
    # outputs.logits has shape (batch_size, seq_length, vocab_size)
    print("Output shape:", logits.shape)
    print("loss:", loss)