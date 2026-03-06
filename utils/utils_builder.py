"""
FOCAL: Fine-grained Optimal transport Contrastive ALignment for ECG-report pairs.

Architecture:
  1. SpatialTemporalViT  — ECG patch encoder (returns all N patch embeddings)
  2. BioClinicalBERT     — tag-level text encoder (each tag encoded independently)
  3. Semi-UOT            — tag-specific ECG representation via semi-unbalanced OT
  4. Soft Sigmoid Loss   — semantic-similarity-guided contrastive loss

Zero-shot interface (compatible with zeroshot_val.py):
  - _tokenize(text)
  - get_text_emb(input_ids, attention_mask)  → pooler output (no grad)
  - proj_t  — text projection head (nn.Module)
  - ext_ecg_emb(ecg)  → normalized global ECG embedding (no grad)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

from vit1d import (spatial_temporal_vit_tiny, spatial_temporal_vit_small,
                   spatial_temporal_vit_middle, spatial_temporal_vit_base)


_ECG_MODEL_FACTORY = {
    'vit_tiny':   spatial_temporal_vit_tiny,
    'vit_small':  spatial_temporal_vit_small,
    'vit_middle': spatial_temporal_vit_middle,
    'vit_base':   spatial_temporal_vit_base,
}


class FOCAL(nn.Module):
    def __init__(self, network_config: dict):
        super().__init__()

        self.proj_out = network_config['projection_head']['projection_size']
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.num_leads = network_config['num_leads']

        # ── ECG encoder ─────────────────────────────────────────────────────
        ecg_model_name = network_config['ecg_model']
        assert ecg_model_name in _ECG_MODEL_FACTORY, \
            f"Unsupported ECG model '{ecg_model_name}'. Choose from {list(_ECG_MODEL_FACTORY)}"

        patch_size = network_config.get('patch_size_per_lead', 500)
        seq_len    = network_config.get('seq_len', 5000)
        ecg_model  = _ECG_MODEL_FACTORY[ecg_model_name](
            num_leads=self.num_leads, seq_len=seq_len, patch_size=patch_size)
        self.ecg_encoder = ecg_model
        encoder_dim = ecg_model.width

        # Linear projection: ECG patch embeddings → shared latent space
        self.proj_ecg = nn.Linear(encoder_dim, self.proj_out, bias=False)

        # ── Text encoder ─────────────────────────────────────────────────────
        text_model_url = network_config['text_model']
        self.lm_model = AutoModel.from_pretrained(
            text_model_url, trust_remote_code=True, revision='main')
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model_url, trust_remote_code=True, revision='main')

        # Text projection head: BERT pooler (768) → shared latent space
        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )

        # ── Learnable temperature ─────────────────────────────────────────────
        # Initialized to log(10) per the paper
        self.log_temperature = nn.Parameter(torch.ones([]) * math.log(10))

        # ── Semi-UOT hyperparameters ──────────────────────────────────────────
        self.uot_eps    = network_config.get('uot_eps', 0.1)
        self.uot_tau    = network_config.get('uot_tau', 1.0)
        self.uot_n_iter = network_config.get('uot_n_iter', 30)

    # ── Tokenisation / text helpers ──────────────────────────────────────────

    def _tokenize(self, text):
        return self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            add_special_tokens=True,
            truncation=True,
            max_length=64,
            padding='max_length',
            return_tensors='pt',
        )

    @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        """Return BERT pooler output (used by zero-shot eval)."""
        return self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output

    # ── ECG helper (zero-shot eval) ──────────────────────────────────────────

    @torch.no_grad()
    def ext_ecg_emb(self, ecg: torch.Tensor) -> torch.Tensor:
        """
        Return normalized global ECG embedding for zero-shot classification.
        Averages projected patch embeddings.
        """
        patch_embs = self.ecg_encoder.forward_patches(ecg)   # (B, N, d_enc)
        global_emb = patch_embs.mean(dim=1)                   # (B, d_enc)
        proj_emb   = self.proj_ecg(global_emb)                # (B, d_proj)
        return normalize(proj_emb, dim=-1)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _split_reports(self, reports):
        """
        Split each report string by comma into individual tags.
        Returns:
            all_tags  : flat list of tag strings (total K)
            tag_counts: list of m_i (number of tags per report)
        """
        all_tags   = []
        tag_counts = []
        for report in reports:
            tags = [t.strip() for t in report.split(',') if t.strip()]
            if not tags:
                tags = [report.strip() or 'normal sinus rhythm']
            all_tags.extend(tags)
            tag_counts.append(len(tags))
        return all_tags, tag_counts

    def _encode_tags(self, tags: list) -> torch.Tensor:
        """
        Encode a flat list of tag strings.
        Returns: (K, d_proj) normalized tag embeddings.
        """
        device = next(self.parameters()).device
        tok    = self._tokenize(tags)
        input_ids      = tok.input_ids.to(device)
        attention_mask = tok.attention_mask.to(device)
        text_emb  = self.lm_model(input_ids=input_ids,
                                  attention_mask=attention_mask).pooler_output
        proj_emb  = self.proj_t(text_emb)
        return normalize(proj_emb, dim=-1)

    def _semi_uot(self, tag_embs: torch.Tensor,
                  patch_embs: torch.Tensor) -> torch.Tensor:
        """
        Semi-Unbalanced Optimal Transport alignment.

        The tag side has a strict marginal constraint (every tag is fully
        assigned), while the ECG-patch side is relaxed via a KL penalty.
        Solved with the log-domain unbalanced Sinkhorn algorithm.

        Args:
            tag_embs   : (m, d) — normalized tag embeddings
            patch_embs : (N, d) — normalized ECG patch embeddings
        Returns:
            e          : (m, d) — normalized tag-specific ECG representations
        """
        m = tag_embs.shape[0]
        N = patch_embs.shape[0]
        device = tag_embs.device
        eps = self.uot_eps
        tau = self.uot_tau

        # Cost matrix: cosine distance ∈ [0, 2]
        C = 1.0 - tag_embs @ patch_embs.t()   # (m, N)

        # Run Sinkhorn in fp32 for numerical stability
        C = C.float()
        log_K = -C / eps                                        # (m, N)
        log_a = torch.full((m,), -math.log(m), device=device)  # log(1/m)
        log_b = torch.full((N,), -math.log(N), device=device)  # log(1/N)

        log_v = torch.zeros(N, device=device)
        log_u = torch.zeros(m, device=device)

        for _ in range(self.uot_n_iter):
            # Strict update for tag side (balanced)
            log_Kv = torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)  # (m,)
            log_u  = log_a - log_Kv

            # Soft update for ECG patch side (KL penalty)
            log_Ktu = torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)  # (N,)
            log_v   = (tau / (tau + eps)) * (log_b - log_Ktu)

        # Transport plan: T = diag(u) @ K @ diag(v)
        log_T = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)  # (m, N)
        T = torch.exp(log_T).to(tag_embs.dtype)                   # (m, N)

        # Normalize rows to sum to 1:  T_hat = m * T  (row sums = 1/m → 1)
        T_hat = T * m   # (m, N)

        # Tag-specific ECG representation: weighted sum over patches
        e = T_hat @ patch_embs   # (m, d)
        return normalize(e, dim=-1)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, ecg: torch.Tensor, reports: list):
        """
        Args:
            ecg     : (B, num_leads, seq_len)
            reports : list of B report strings (comma-separated tags)
        Returns:
            dict with:
              'tag_ecg_embs'  : (K, d) tag-specific ECG representations
              'tag_text_embs' : (K, d) tag text embeddings
              'tag_counts'    : list of m_i per sample
        """
        # ── 1. ECG patch embeddings ────────────────────────────────────────
        patch_embs = self.ecg_encoder.forward_patches(ecg)   # (B, N, d_enc)
        patch_embs = self.proj_ecg(patch_embs)               # (B, N, d_proj)
        patch_embs = normalize(patch_embs, dim=-1)           # (B, N, d_proj)

        # ── 2. Tag-level text embeddings ──────────────────────────────────
        all_tags, tag_counts = self._split_reports(reports)
        tag_text_embs = self._encode_tags(all_tags)          # (K, d_proj)

        # ── 3. Semi-UOT per ECG ───────────────────────────────────────────
        tag_ecg_list = []
        tag_offset   = 0
        for i, m_i in enumerate(tag_counts):
            z_i = tag_text_embs[tag_offset: tag_offset + m_i]  # (m_i, d)
            h_i = patch_embs[i]                                 # (N, d)
            e_i = self._semi_uot(z_i, h_i)                     # (m_i, d)
            tag_ecg_list.append(e_i)
            tag_offset += m_i

        tag_ecg_embs = torch.cat(tag_ecg_list, dim=0)   # (K, d)

        return {
            'tag_ecg_embs':  tag_ecg_embs,    # (K, d)
            'tag_text_embs': tag_text_embs,   # (K, d)
            'tag_counts':    tag_counts,       # [m_1, ..., m_B]
        }
