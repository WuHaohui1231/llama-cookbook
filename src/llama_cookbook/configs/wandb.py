# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class wandb_config:
    project: str = 'finetune-llama3-tatdqa' # wandb project name
    entity: Optional[str] = None # wandb entity name
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = "FT on TATDQA dataset with pretrained fin_lora_pt_31l-30"
    mode: Optional[str] = None
