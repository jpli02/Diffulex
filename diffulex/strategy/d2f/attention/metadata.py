import torch

from dataclasses import dataclass

from diffulex.attention.metadata import AttnMetaDataBase


@dataclass
class D2FAttnMetaData(AttnMetaDataBase):
    seq_lens: list[int] = None
    seq_lens_ts: torch.Tensor | None = None
    d2f_pp: bool = False
    block_mask: list[torch.Tensor] | None = None
    

D2F_ATTN_METADATA = D2FAttnMetaData()

def fetch_d2f_attn_metadata() -> D2FAttnMetaData:
    return D2F_ATTN_METADATA

def set_d2f_attn_metadata() -> None:
    # TODO
    global D2F_ATTN_METADATA
    D2F_ATTN_METADATA = D2FAttnMetaData()

def reset_d2f_attn_metadata() -> None:
    global D2F_ATTN_METADATA
    D2F_ATTN_METADATA = D2FAttnMetaData()