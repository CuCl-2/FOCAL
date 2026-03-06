"""
FOCAL loss: semantic-similarity-guided soft sigmoid loss.

Replaces the hard binary labels in SigLIP with a continuous semantic
similarity matrix S derived from the tag text embeddings.

Loss formula (per the paper):
    L = (1/K^2) * sum_{k,l} [
          S_{k,l}       * log(1 + exp(-x_{k,l}))
        + (1 - S_{k,l}) * log(1 + exp( x_{k,l}))
    ]
where:
    x_{k,l} = t * sim(e_k, z_l)          — temperature-scaled ECG-text similarity
    S_{k,l} = (1 + sim(z_k, z_l)) / 2    — semantic similarity in [0, 1]
    t        = exp(log_temperature)        — learnable temperature
"""

import torch
import torch.nn.functional as F


def focal_loss(tag_ecg_embs: torch.Tensor,
               tag_text_embs: torch.Tensor,
               log_temperature: torch.Tensor) -> torch.Tensor:
    """
    Semantic-guided soft sigmoid contrastive loss.

    Args:
        tag_ecg_embs   : (K, d) normalized tag-specific ECG representations
        tag_text_embs  : (K, d) normalized tag text embeddings
        log_temperature: scalar — learnable log temperature parameter

    Returns:
        loss: scalar
    """
    t = log_temperature.exp()

    # Temperature-scaled ECG → text similarity matrix
    # x_{k,l} = t * sim(e_k, z_l)
    x = t * tag_ecg_embs @ tag_text_embs.t()   # (K, K)

    # Semantic similarity matrix (text ↔ text), mapped to [0, 1]
    # S_{k,l} = (1 + cos_sim(z_k, z_l)) / 2
    with torch.no_grad():
        S = (1.0 + tag_text_embs @ tag_text_embs.t()) / 2.0   # (K, K)

    # Soft sigmoid BCE loss:
    #   S * log(1 + exp(-x)) + (1-S) * log(1 + exp(x))
    #   = S * softplus(-x)   + (1-S) * softplus(x)
    loss = (S * F.softplus(-x) + (1.0 - S) * F.softplus(x)).mean()
    return loss
