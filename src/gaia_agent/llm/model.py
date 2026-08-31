from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class LLMModel():
    provider:str
    model:str
    max_tokens:int    
    temperature:float

