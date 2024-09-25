from dataclasses import dataclass, field


@dataclass
class MambaConfig:
    d_model: int = 32
    n_layer: int = 2
    vocab_size: int = 3
    ssm_cfg: dict = None
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 2

