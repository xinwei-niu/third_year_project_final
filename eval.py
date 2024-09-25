import torch
import json
from dataclasses import dataclass, field
import wandb
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import math
from functools import partial
import json
import os
from datetime import datetime
from tqdm import tqdm
from collections import namedtuple
from config import MambaConfig
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import pandas as pd


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx,  **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

# Mamba implementation from paper Mamba: Linear-Time Sequence Modeling with Selective State Spaces by Gu A. and Dao T.
# Source: https://github.com/state-spaces/mamba
class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        d_time = 16
        self.add_bos = True
        self.vocab_size = vocab_size
        self.div_term = torch.exp(torch.arange(0, d_time, 2) * -(math.log(10000.0) / d_time)).reshape(1, 1, -1)
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.inten_linear = nn.Linear(d_model+d_time, vocab_size, device=device)
        self.softplus = nn.Softplus()
        # self.inner_linear = nn.Linear(d_model+d_time, d_model, device=device)
        self.eps = torch.finfo(torch.float32).eps
        self.d_time = 16
        self.d_model = d_model
        self.output_linear = nn.Linear(d_model+d_time, d_model, bias=False, **factory_kwargs)
        self.div_term = torch.exp(torch.arange(0, self.d_time, 2) * -(math.log(10000.0) / self.d_time)).reshape(1, 1, -1)
        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model+d_time,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model+d_time, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, type_seq, time_seq, temp_seq = [], inference_params=None):
        
        hidden_states = torch.tanh(self.embedding(type_seq.cuda()))
        
        
        tem_enc = self.compute_temporal_embedding(time_seq.cuda())
        hidden_states = torch.cat((hidden_states, tem_enc), dim=-1)
        seqlen = hidden_states.size(1)
        seqlen_ = temp_seq.size(1)
        init_cur_layer_ = torch.zeros((temp_seq.size(0), temp_seq.size(1), self.d_model)).to(
                "cuda")
        # if temp_seq != []:
        #     init_cur_layer_ = torch.zeros((temp_seq.size(0), temp_seq.size(1), self.d_model)).to(
        #     "cuda")

            
        #     temp_seq = torch.cat((init_cur_layer_, temp_seq.cuda()), dim=-1)

        #     hidden_states = torch.cat((hidden_states, temp_seq.cuda()), dim=1)
        residual = None
        _cur_layer_ = init_cur_layer_
        _temp_seq = torch.cat((_cur_layer_, temp_seq.cuda()), dim=-1)
            
        hidden_states = torch.cat((hidden_states, _temp_seq.cuda()), dim=1)
        for layer in self.layers:
            # if temp_seq != []:
                


            # _temp_seq = torch.cat((_cur_layer_, temp_seq.cuda()), dim=-1)
            
            # hidden_states = torch.cat((hidden_states, _temp_seq.cuda()), dim=1)
            
            # hidden_states, residual = layer(
            #     hidden_states, residual, inference_params=inference_params
            # )

            # _cur_layer_ = self.inner_linear(hidden_states)[:, seqlen:, :]
            # if layer == self.layers[-1]:
            #     break
            # hidden_states = self.inner_linear(hidden_states)


            # _cur_layer_ = torch.tanh(_cur_layer_) + _cur_layer_

            # hidden_states = torch.cat([hidden_states[:, :seqlen, :],tem_enc], dim=-1)
            # else:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # # output = self.output_linear(hidden_states)
        # if temp_seq != []:
        #     # print(hidden_states.shape)
        #     # print(type_seq.size(1))
            
        #     return hidden_states[:, :temp_seq.size(1), :]
            
        # else:
        # hidden_states = self.inner_linear(hidden_states)
        hidden_states = hidden_states[:,:seqlen_, :]

        return hidden_states
    

    def compute_temporal_embedding(self, time):
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.d_time).to(time.device)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time.device)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)
        # pe = pe * non_pad_mask.unsqueeze(-1)
        return pe
    
    def compute_loglik(self, time_seq, event_seq):

        type_mask = torch.zeros([*event_seq.size(), self.vocab_size], device="cuda:0")
        for i in range(self.vocab_size):
            type_mask[:, :, i] = event_seq == i

        # 1. compute event-loglik
        enc_out = self.forward(event_seq[:, :-1], time_seq[:, :-1], self.compute_temporal_embedding(time_seq[:, 1:]))
        enc_inten = self.softplus(self.inten_linear(enc_out))
        # original: 1->1, 2->2
        # event_lambdas = torch.sum(enc_inten * type_mask, dim=2) + self.eps
        # now: 1->2, 2->3
        event_lambdas = torch.sum(enc_inten * type_mask[:, 1:], dim=2) + self.eps
        # in case event_lambdas == 0

        # event_lambdas.masked_fill_(~batch_non_pad_mask, 1.0)
        # event_lambdas.masked_fill_(~batch_non_pad_mask[:, 1:], 1.0)

        event_ll = torch.log(event_lambdas)
        res_enc_inten = enc_inten

        num_samples = 100
        # 2.1 sample times
        # 2.2 compute intensities at sampled times
        # due to GPU memory limitation, we may not be able to compute all intensities at all sampled times,
        # step gives the batch size w.r.t how many sampled times we should process at each batch
        step = 20

        diff_time = (time_seq[:, 1:] - time_seq[:, :-1]) #* batch_non_pad_mask[:, 1:]
        temp_time = diff_time.unsqueeze(0) * \
                    torch.rand([num_samples, *diff_time.size()], device=time_seq.device)
        temp_time += time_seq[:, :-1].unsqueeze(0)


        all_lambda, samp_times = self._compute_intensities_fast(event_seq[:, :-1], time_seq[:, :-1],
                                                    temp_time, step)
        all_lambda = all_lambda.sum(dim=-1)
        
        diffs = torch.sum(torch.square((torch.sub(time_seq[:, 1:], samp_times)))).squeeze()
        total = time_seq[:, 1:].shape[1]
        # mse = torch.torch.mean(diffs, dim=-1).squeeze(0)
        # print(math.sqrt(mse))

        # 2.3 compute the empirical expectation of the summation
        all_lambda = all_lambda.sum(dim=0) / num_samples
        non_event_ll = all_lambda * diff_time.cuda()

        # return enc_inten to compute accuracy
        return event_ll, non_event_ll, res_enc_inten, diffs, total
    


    def _compute_intensities_fast(self, event_seq, time_seq, temp_time, step=20):
        # fast version, can only use in log-likelihood computation
        # assume we will sample the same number of times in each interval of the event_seqs
        all_lambda = []
        batch_size = event_seq.size(0)
        seq_len = event_seq.size(1)
        num_samples = temp_time.size(0)
        temp_time_ = None
        for i in range(0, num_samples, step):
            _extra_time = temp_time[i: i + step, :, :]
            _step = _extra_time.size(0)
            _extra_time = _extra_time.reshape(_step * batch_size, -1)
            if temp_time_ is None:
                temp_time_ = torch.mean(_extra_time, dim=0).unsqueeze(0)
            else:
                temp_time_ = torch.cat([temp_time_, torch.mean(_extra_time, dim=0).unsqueeze(0)], dim=0)
            _types = event_seq.expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _times = time_seq.expand(_step, -1, -1).reshape(_step * batch_size, -1)
            # print(_extra_time.shape)
            # print(_times.shape)

            # print(self.compute_temporal_embedding(_extra_time).shape)
            _enc_output = self.forward(_types, _times,self.compute_temporal_embedding(_extra_time)
                                       )
            
            all_lambda.append(self.softplus(self.inten_linear(_enc_output)).reshape(_step, batch_size, seq_len, -1))
        
        samp_times = torch.mean(temp_time_,dim=0)

        

        all_lambda = torch.cat(all_lambda, dim=0)
        return all_lambda, samp_times
    

    def compute_intensities_at_sampled_times(self, event_seq, time_seq, sampled_times):
        # Assumption: all the sampled times are distributed [time_seq[...,-1], next_event_time]
        # used for thinning algorithm
        
        num_batches = event_seq.size(0)
        seq_len = event_seq.size(1)
        assert num_batches == 1, "Currently, no support for batch mode (what is a good way to do batching in thinning?)"
        if num_batches == 1 and num_batches < sampled_times.size(0):
            _sample_size = sampled_times.size(0)
            # multiple sampled_times
            event_seq = event_seq.expand(_sample_size, -1, -1).reshape(_sample_size, -1)
            time_seq = time_seq.expand(_sample_size, -1, -1).reshape(_sample_size, -1)
            num_batches = event_seq.size(0)

        # print(sampled_times.size(0))
        num_samples = sampled_times.size(1)

        # 1. prepare input embeddings for "history"
        tem_enc = self.compute_temporal_embedding(time_seq)
        enc_input = torch.tanh(self.embedding(event_seq))

        enc_input = torch.cat([enc_input, tem_enc], dim=-1)
        tem_layer_ = self.compute_temporal_embedding(sampled_times)

       
        cur_layer_ = self.forward(event_seq, time_seq, temp_seq=tem_layer_)

        sampled_intensities = self.softplus(self.inten_linear(cur_layer_))

        return sampled_intensities


class MambaTPPModel(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        self.d_model = d_model
        d_time = 16
        self.eps = torch.finfo(torch.float32).eps
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = field(default_factory=dict)
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.type_head = nn.Linear(d_model+d_time, vocab_size, bias=False, **factory_kwargs)
        # self.time_head = nn.Linear(d_model, 1, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
    #     self.tie_weights()

    # def tie_weights(self):
    #     self.type_head.weight = self.backbone.embedding.weight 

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, type_seq, time_seq, inference_params=None, num_last_tokens=0):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        
        hidden_states = self.backbone(type_seq, time_seq, inference_params=inference_params)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]


        type_logits = self.type_head(hidden_states)
        # time_pred = self.time_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=type_logits)
    

    def compute_loglik(self, time_seq, event_seq):
        return self.backbone.compute_loglik(time_seq, event_seq)
    

    def compute_intensities_at_sampled_times(self, event_seq, time_seq, sampled_times):
        return self.backbone.compute_intensities_at_sampled_times(event_seq, time_seq, sampled_times)


    
    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        # config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(d_model= 32,
                            n_layer= 2,
                            vocab_size= 3,
                            ssm_cfg = {},
                            rms_norm=True,
                            residual_in_fp32= True,
                            fused_add_norm=True,
                            pad_vocab_size_multiple = 1)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # # Save the configuration of the model
        # config_path = os.path.join(save_directory, 'config.json')
        # print(self.config.__dict__)
        # with open(config_path, 'w') as f:
        #     json.dump(self.config.__dict__, f)




class TrainDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.lowest_age = 18
        self.highest_age = 91
        self.time_data = []
        self.type_data = []
        df = pd.read_pickle(file_path)

        seqs = df["train"]
        for i in seqs:
            batch_type_seq = np.array([x['type_event'] for x in i][:-1] , dtype=np.int64)
            #self.time_data.append(([i["icd_code"] for i in json.loads(line)["occurance"]]))
            batch_time_seq = np.array([x['time_since_start'] for x in i][:-1], dtype=np.float32)
            batch_type_true = np.array([x['type_event'] for x in i][1:], dtype=np.int64)
            batch_time_true = np.array([x['time_since_start'] for x in i][1:], dtype=np.float32)
            batch_time_delta = np.array([x['time_since_last_event'] for x in i][:-1], dtype=np.float32)
            event_seq = np.array([x['type_event'] for x in i] , dtype=np.int64)
            time_seq = batch_time_true = np.array([x['time_since_start'] for x in i], dtype=np.float32)
            self.data.append([batch_type_seq,batch_time_seq, batch_time_delta, {"time":batch_time_true, "type":batch_type_true}, event_seq, time_seq])
            # print(self.data)
                


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class DevDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.lowest_age = 18
        self.highest_age = 91
        self.time_data = []
        self.type_data = []
        df = pd.read_pickle(file_path)

        seqs = df["dev"]
        for i in seqs:
            batch_type_seq = np.array([x['type_event'] for x in i][:-1] , dtype=np.int64)
            #self.time_data.append(([i["icd_code"] for i in json.loads(line)["occurance"]]))
            batch_time_seq = np.array([x['time_since_start'] for x in i][:-1], dtype=np.float32)
            batch_type_true = np.array([x['type_event'] for x in i][1:], dtype=np.int64)
            batch_time_true = np.array([x['time_since_start'] for x in i][1:], dtype=np.float32)
            batch_time_delta = np.array([x['time_since_last_event'] for x in i][:-1], dtype=np.float32)
            event_seq = np.array([x['type_event'] for x in i] , dtype=np.int64)
            time_seq = batch_time_true = np.array([x['time_since_start'] for x in i], dtype=np.float32)
            self.data.append([batch_type_seq,batch_time_seq, batch_time_delta, {"time":batch_time_true, "type":batch_type_true}, event_seq, time_seq])
            # print(self.data)
                


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    

class TestDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        self.lowest_age = 18
        self.highest_age = 91
        self.time_data = []
        self.type_data = []
        df = pd.read_pickle(file_path)

        seqs = df["test"]
        for i in seqs:
            batch_type_seq = np.array([x['type_event'] for x in i][:-1] , dtype=np.int64)
            #self.time_data.append(([i["icd_code"] for i in json.loads(line)["occurance"]]))
            batch_time_seq = np.array([x['time_since_start'] for x in i][:-1], dtype=np.float32)
            batch_time_delta = np.array([x['time_since_last_event'] for x in i][:-1], dtype=np.float32)
            batch_type_true = np.array([x['type_event'] for x in i][1:] , dtype=np.int64)
            batch_time_true = np.array([x['time_since_start'] for x in i][1:], dtype=np.float32)
            event_seq = np.array([x['type_event'] for x in i] , dtype=np.int64)
            time_seq = batch_time_true = np.array([x['time_since_start'] for x in i], dtype=np.float32)
            self.data.append([batch_type_seq,batch_time_seq, batch_time_delta, {"time":batch_time_true, "type":batch_type_true}, event_seq, time_seq])

            # print(self.data)
                


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == 0:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


if __name__ == '__main__':
    model = MambaTPPModel.from_pretrained("ptr_models", device="cuda")

    dev_metric = 0
    table = []
    eval_table = []
    test_dataset = TestDataset('test.pkl')
    dev_dataset = DevDataset('dev.pkl')
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, pin_memory=True, collate_fn=None)
    dev_dataloader = DataLoader(dev_dataset, 1, shuffle=False, pin_memory=True, collate_fn=None)
    with torch.no_grad():
        
        model.eval()

        train_loss = 0
        total_log_like = 0
        total_acc = 0
        total_event_ll, total_non_event_ll = 0, 0
        num_tokens = 0
        pad_idx = 75
        num_events = 0
        total_diff = 0
        total = 0
        for batchidx, sample in tqdm(enumerate(dev_dataloader)):
            if len(sample[0].squeeze(1).shape) == 1:
                sample[0] = sample[0].unsqueeze(0)
                sample[1] = sample[1].unsqueeze(0) 
            # print(sample[0].squeeze(1).shape)

            # pred_logits= model.compute_loglik(sample[-1].squeeze(1), sample[-2].squeeze(1))
    


            
            

            all_logs = []
            all_logs_token = []
            all_type_ll_token = []
            all_time_ll_token = []


            time_seq, event_seq = sample[-1].squeeze(1), sample[-2].squeeze(1)

            event_ll, non_event_ll, enc_inten, diffs, total_ = model.compute_loglik(time_seq=time_seq, event_seq=event_seq)
            total_diff += diffs
            total += total_

            _batch_loss = event_ll.sum(dim=-1) - non_event_ll.sum(dim=-1)
            _loss = -torch.sum(_batch_loss)

            total_log_like += -_loss.item()
            total_event_ll += event_ll.sum().item()
            total_non_event_ll += non_event_ll.sum().item()
            type_lls = event_ll - torch.log(enc_inten.sum(dim=-1) + model.eps)
            time_lls = event_ll - non_event_ll - type_lls
            # print(torch.argmax(enc_inten, dim=-1))
            # print(enc_inten)
            enc_inten = enc_inten.detach().cpu()
            event_seq = event_seq[:,1:]
            # print(event_seq)
            total_acc += ((torch.argmax(enc_inten, dim=-1) == event_seq.detach().cpu()).sum())
            num_tokens += event_seq.ne(pad_idx).sum().item()
            num_events += (event_seq < pad_idx).sum().item()
            all_logs_token.extend([(x, 1.0) for x in (event_ll - non_event_ll).tolist()])
            all_type_ll_token.extend([(x, 1.0) for x in type_lls.tolist()])
            all_time_ll_token.extend([(x, 1.0) for x in time_lls.tolist()])
            all_logs.extend([(x, y) for x, y in zip(_batch_loss.tolist(), event_seq.ne(pad_idx).sum(dim=-1).tolist())])
        print(total_log_like, total_acc / num_tokens, total_event_ll/num_tokens, total_non_event_ll/num_tokens)
        print(f"rmse: {math.sqrt(total_diff/total):.4f}")
        if (-(total_event_ll/num_tokens) + total_non_event_ll/num_tokens) < dev_metric:
            dev_metric = -(total_event_ll/num_tokens) + total_non_event_ll/num_tokens


        model.eval()

        train_loss = 0
        total_log_like = 0
        total_acc = 0
        total_event_ll, total_non_event_ll = 0, 0
        num_tokens = 0
        total_diff = 0
        total = 0
        for batchidx, sample in tqdm(enumerate(test_dataloader)):
            if len(sample[0].squeeze(1).shape) == 1:
                sample[0] = sample[0].unsqueeze(0)
                sample[1] = sample[1].unsqueeze(0) 
            # print(sample[0].squeeze(1).shape)

            # pred_logits= model.compute_loglik(sample[-1].squeeze(1), sample[-2].squeeze(1))
    


            
            

            all_logs = []
            all_logs_token = []
            all_type_ll_token = []
            all_time_ll_token = []


            time_seq, event_seq = sample[-1].squeeze(1), sample[-2].squeeze(1)

            event_ll, non_event_ll, enc_inten, diffs, total_ = model.compute_loglik(time_seq=time_seq, event_seq=event_seq)
            total_diff += diffs
            total += total_

            _batch_loss = event_ll.sum(dim=-1) - non_event_ll.sum(dim=-1)
            _loss = -torch.sum(_batch_loss)

            total_log_like += -_loss.item()
            total_event_ll += event_ll.sum().item()
            total_non_event_ll += non_event_ll.sum().item()
            type_lls = event_ll - torch.log(enc_inten.sum(dim=-1) + model.eps)
            time_lls = event_ll - non_event_ll - type_lls
            # print(torch.argmax(enc_inten, dim=-1))
            # print(enc_inten)
            enc_inten = enc_inten.detach().cpu()
            event_seq = event_seq[:,1:]
            # print(event_seq)
            total_acc += ((torch.argmax(enc_inten, dim=-1) == event_seq.detach().cpu()).sum())
            num_tokens += event_seq.ne(pad_idx).sum().item()
            num_events += (event_seq < pad_idx).sum().item()
            all_logs_token.extend([(x, 1.0) for x in (event_ll - non_event_ll).tolist()])
            all_type_ll_token.extend([(x, 1.0) for x in type_lls.tolist()])
            all_time_ll_token.extend([(x, 1.0) for x in time_lls.tolist()])
            all_logs.extend([(x, y) for x, y in zip(_batch_loss.tolist(), event_seq.ne(pad_idx).sum(dim=-1).tolist())])
        print(total_log_like, total_acc / num_tokens, total_event_ll/num_tokens, total_non_event_ll/num_tokens)
        print(f"rmse: {math.sqrt(total_diff/total):.4f}")
    # for batch
    












