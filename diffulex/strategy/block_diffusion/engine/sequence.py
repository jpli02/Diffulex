from __future__ import annotations

import torch

from enum import Enum, auto
from dataclasses import dataclass

from diffulex.config import Config
from diffulex.sampling_params import SamplingParams
from diffulex.engine.sequence import AutoSequence, SequenceBase


class BDDiffusionBlockStatus(Enum):
    ACTIVE = auto()
    TO_CACHE = auto()
    IN_CACHE = auto()


@dataclass
class BDDiffusionBlock:
    block_id: int = 0
    status: BDDiffusionBlockStatus = BDDiffusionBlockStatus.ACTIVE

    global_start_id: int = 0
    global_end_id: int | None = None
    cursor: int = 0

    mask_token_id: int = 151666
    size: int = 32
    is_prompt: bool = False
    
    seq: "BDSequence" | None = None

    def __post_init__(self) -> None:
        self.global_end_id = self.global_start_id + self.size

    def __getitem__(self, key: int) -> int:
        return self.seq[self.global_start_id + key]  # type: ignore[index]

    def __len__(self) -> int:
        return self.size
    
    @property
    def token_ids(self) -> list[int]:
        return self.seq.token_ids[self.global_start_id: self.global_end_id]
    
    @property
    def has_mask_token(self) -> bool:
        return any(token == self.mask_token_id for token in self.token_ids)
    
    @property
    def is_active(self) -> bool:
        return self.status == BDDiffusionBlockStatus.ACTIVE
    
    @property
    def is_to_cache(self) -> bool:
        return self.status == BDDiffusionBlockStatus.TO_CACHE
    
    @property
    def is_in_cache(self) -> bool:
        return self.status == BDDiffusionBlockStatus.IN_CACHE
    
    @property
    def available_to_cache(self) -> bool:
        return not self.has_mask_token and self.is_active
    
    @property
    def available_in_cache(self) -> bool:
        return self.is_to_cache
    
    @property
    def available_to_add_new_block(self) -> bool:
        return self.is_in_cache
    

@AutoSequence.register("block_diffusion", is_default=True)
class BDSequence(SequenceBase):
    """Sequence implementation tailored for diffusion-based decoding."""

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        config: Config | None = None,
    ):
        super().__init__(token_ids, sampling_params)
        if config is None:
            raise ValueError("BDSequence requires a Config instance.")
        
        self.config = config
        self.diffusion_blocks: list[BDDiffusionBlock] = []
        self.diffusion_block_size = config.diffusion_block_size
        self.mask_token_id = config.mask_token_id
        
    @property
    def prefix_len_with_padding(self) -> int:
        return self.prefix_len + self.pad_prefix_len
    
    @property
    def diffusion_block_status(self) -> list[BDDiffusionBlockStatus]:
        return [block.status for block in self.diffusion_blocks]
    
    @property
    def num_prefix_blocks(self) -> int:
        return (self.prefix_len + self.block_size - 1) // self.block_size
    
    @property
    def prefix_last_block_num_tokens(self) -> int:
        return self.prefix_len - (self.num_prefix_blocks - 1) * self.block_size
    
    @property
    def active_block_token_ids(self) -> list[int]:
        return self.diffusion_blocks[-1].token_ids
    
    @property
    def num_blocks_in_active_diffusion_block(self) -> int:
        return self.diffusion_block_size // self.block_size
    
    def diffusion_decoding_inputs(self) -> tuple[list[int], list[int], int]:
        return (
            self.active_block_token_ids,
            list(range(self.num_tokens - self.diffusion_block_size, self.num_tokens)),
            self.num_tokens - self.diffusion_block_size,
        )
    
    def extend_mask_tokens(self, extend_len: int) -> None:
        self.token_ids.extend([self.mask_token_id] * extend_len)
          
    def init_diffusion_blocks(self) -> None:
        """Initialize diffusion blocks: prefix blocks are TO_CACHE, last block with mask tokens is ACTIVE."""
        self.prefix_len = len(self.token_ids)
        block_size = self.diffusion_block_size
        
        # Calculate prefix blocks and padding
        num_prefix_blocks = self.prefix_len // block_size
        self.pad_prefix_len = block_size - (self.prefix_len % block_size)
        if self.prefix_len % block_size == 0:
            self.pad_prefix_len = 0
        
        # Add mask tokens for the last prefix block
        self.extend_mask_tokens(self.pad_prefix_len)
        
        # Calculate total blocks needed
        total_num_blocks = num_prefix_blocks if self.pad_prefix_len == 0 else num_prefix_blocks + 1
        
        # Create all blocks
        current_pos = 0
        for block_id in range(total_num_blocks):
            # Determine block status
            block_tokens = self.token_ids[current_pos:current_pos + block_size]
            has_mask_token = any(token == self.mask_token_id for token in block_tokens)
            is_last_prefix_block = (block_id == num_prefix_blocks)
            
            if block_id < num_prefix_blocks:
                status = BDDiffusionBlockStatus.TO_CACHE
            elif is_last_prefix_block:
                status = BDDiffusionBlockStatus.ACTIVE if has_mask_token else BDDiffusionBlockStatus.TO_CACHE
            else:
                status = BDDiffusionBlockStatus.TO_CACHE
            
            block = BDDiffusionBlock(
                block_id=block_id,
                status=status,
                global_start_id=current_pos,
                size=block_size,
                mask_token_id=self.mask_token_id,
                is_prompt=(block_id <= num_prefix_blocks),
                seq=self,
            )
            self.diffusion_blocks.append(block)
            current_pos += block_size
    
    def next_diffusion_step(self) -> None:
        """Append new diffusion block if needed."""
        if self.diffusion_blocks[-1].available_to_add_new_block:
            self.extend_mask_tokens(self.diffusion_block_size)
            self.diffusion_blocks.append(
                BDDiffusionBlock(
                    block_id=len(self.diffusion_blocks),
                    status=BDDiffusionBlockStatus.ACTIVE,
                    global_start_id=self.num_tokens,
                    size=self.diffusion_block_size,
                    mask_token_id=self.mask_token_id,
                    is_prompt=False,
                    seq=self,
                )
            )