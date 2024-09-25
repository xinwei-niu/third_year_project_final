import torch
import json
from dataclasses import dataclass, field

import torch
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
import pandas as pd
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


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

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=75, **factory_kwargs)

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
                    d_model,
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
            d_model, eps=norm_epsilon, **factory_kwargs
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

    def forward(self, type_seq, time_seq, inference_params=None):

        hidden_states = torch.tanh(self.embedding(type_seq.cuda()))

        residual = None
        for layer in self.layers:
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
        return hidden_states
    


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
        self.type_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        self.time_head = nn.Linear(d_model, 1, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        self.type_head.weight = self.backbone.embedding.weight 

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
        time_pred = self.time_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=type_logits), time_pred

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        # config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(d_model= 32,
    n_layer= 10,
    vocab_size= 75,
    ssm_cfg = {},
    rms_norm=True,
    residual_in_fp32= True,
    fused_add_norm=True,
    pad_vocab_size_multiple = 8)
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




class JsonlDataset(Dataset):
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
            batch_time_seq = np.array([])
            batch_type_true = np.array([x['type_event'] for x in i][1:] , dtype=np.int64)
            batch_time_true =  np.array([])
            self.data.append([batch_type_seq,batch_time_seq, {"time":batch_time_true, "type":batch_type_true}])
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
            if token == 75:
                m[i].append(0)
            else:
                m[i].append(1)
    return m



if __name__ == '__main__':
    pad_index = 75

    # Load the dataset
    epoch = 10
    torch.random.manual_seed(114514)
    dataset = JsonlDataset('test.pkl')


    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    

    train_dataloader = DataLoader(train_dataset, 24, shuffle=True, pin_memory=True, collate_fn=None)
    
    model = MambaTPPModel.from_pretrained("pretrained", device="cuda:0")

    
    # val_dataloader = DataLoader(val_dataset, 24, shuffle=True, pin_memory=True, collate_fn=None)
    test_dataloader = DataLoader(dataset, 1, shuffle=True, pin_memory=True, collate_fn=None)
    

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        
        for batchidx, sample in tqdm(enumerate(test_dataloader)):
            model.eval()
            # print(sample[0].squeeze(1).shape)
            type_loss = nn.NLLLoss()
            time_loss = nn.MSELoss()
            print(sample[0].squeeze(1).shape)
            if len(sample[0].squeeze(1).shape) == 1:
                continue
            pred_logits, time_pred = model(sample[0].squeeze(1), sample[1].squeeze(1))
            pred_logits = pred_logits.logits.contiguous()
            time_pred = time_pred.contiguous()
            true_time = sample[2]["time"].squeeze(1).contiguous().cuda()

            true_type = sample[2]["type"].squeeze(1).contiguous().cuda()


            # print(pred_logits.view(-1, pred_logits.size(-1)).shape)
            # print(true_type.view(-1).shape)
            mask = binaryMatrix(sample[0].squeeze(1))
            
            mask = torch.BoolTensor(mask)
            print(mask)
            log_prob = nn.LogSoftmax(dim=1)
            print(np.argmax(pred_logits.view(-1, pred_logits.size(-1)).detach().cpu().numpy(), axis=1))
            type_loss_output = type_loss(log_prob(pred_logits.view(-1, pred_logits.size(-1))), true_type.view(-1)).masked_select(mask.cuda()).mean()
            mask = true_type.view(-1) != 75
            
            total += true_type.view(-1)[mask].size(0)
            correct+= (np.argmax(pred_logits.view(-1, pred_logits.size(-1)).detach().cpu().numpy(), axis=1)[mask.cpu()] == true_type.view(-1).detach().cpu().numpy()[mask.cpu()]).sum().item()
            
            # time_loss_output = math.sqrt(time_loss(time_pred.squeeze(2), true_time))
            print(true_type.view(-1))
            loss = type_loss_output #+ time_loss_output
            # print(time_loss_output)
            test_loss += loss.item()

            

        print(test_loss/len(test_dataloader))
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
    # for batch
    












