from dataclasses import dataclass
@dataclass
class model_config:
    
    vocab_size :int = 50432
    n_emb:int = 384
    n_heads: int = 6 #Number of Heads for Multi Head Attention
    n_bias : bool = True
    dropout : float = 0.3
    n_layers : int = 6

    
    block_size : int = 512
    batch_size: int = 4
    num_workers: int = 2


@dataclass
class train_config:
    ...
    