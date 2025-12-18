import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


########################################################
# Wayformer-style Decoder (learnable queries) + Gaussian Head
########################################################

class MultiModalAnchors(nn.Module):
    """
    Learnable K anchors (queries). Optionally condition anchors by adding a
    pooled context vector (LayerNorm'ed) to each anchor.

    Args:
        hidden_dim: transformer model dim D
        n_pred:     number of modes (K)
        mode_emb:   "none"  -> anchors are unconditional
                    "add"   -> anchors += LN(context)
        scale:      initial scale multiplier for anchors
    """
    def __init__(self, hidden_dim: int, n_pred: int, mode_emb: str = "none", scale: float = 1.0):
        super().__init__()
        assert mode_emb in {"none", "add"}
        self.n_pred = n_pred
        self.mode_emb = mode_emb

        # [K, D] learnable anchor table
        anchors = torch.empty(n_pred, hidden_dim)
        nn.init.xavier_normal_(anchors)
        anchors *= scale
        self.anchors = nn.Parameter(anchors)

        if mode_emb == "add":
            self.ctx_ln = nn.LayerNorm(hidden_dim)

    def _expand_anchors(self, B: int) -> torch.Tensor:
        # Broadcast anchors to batch: [B, K, D]
        return self.anchors.unsqueeze(0).expand(B, -1, -1).contiguous()

    def forward(self, B: int, ctx: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns:
            q: [B, K, D] anchor queries (optionally conditioned on ctx)
        """
        q = self._expand_anchors(B)
        if self.mode_emb == "add" and ctx is not None:
            # Add a normalized pooled context to each anchor (per sample)
            q = q + self.ctx_ln(ctx).unsqueeze(1)  # [B, 1, D] -> [B, K, D]
        return q


class RatTransformer(nn.Module):
    """
    Minimal Wayformer-style decoder over historical tokens using learnable queries.

    Inputs:
      tokens: [B, Th, 2]         # (x, y) per timestep
      mask:   [B, Th]            # 1/True = valid timestep

    Outputs:
      conf_logits: [B, K]        # unnormalized confidence logits per mode
      mean:        [B, K, Tf, 2] # predicted future means
      logvar:      [B, K, Tf, 2] # predicted future log-variances (clamped for safety)

    Notes:
      - If `independent_modes=True`, we pass a `tgt_mask` to the TransformerDecoder
        that disables query-to-query attention off the diagonal, i.e., each mode
        attends ONLY to itself (plus cross-attn to memory).
      - For training, see `mixture_gaussian_nll` below that enforces a variance
        floor via `sigma = softplus(logvar) + eps`.
    """
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,      # number of TransformerDecoder layers
        n_modes: int = 4,
        hist_len: int = 110,
        future_len: int = 33,
        anchor_mode: str = "none",  # "none" or "add"
        ff_mult: int = 4,
        pe_std: float = 0.02,       # init scale for temporal PE
        independent_modes: bool = False,  # NEW: make K queries independent
        logvar_clamp: tuple[float, float] = (-10.0, 5.0),  # safety clamp range
        dropout: float = 0.0,       # optional dropout
    ):
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        self.hist_len = hist_len
        self.future_len = future_len
        self.independent_modes = independent_modes
        self.logvar_clamp = logvar_clamp

        # -------- Input projection via small MLP (2 -> D) --------
        self.input_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # -------- Learnable temporal positional embeddings (history) --------
        self.temporal_pe = nn.Parameter(torch.zeros(hist_len, d_model))
        nn.init.trunc_normal_(self.temporal_pe, std=pe_std)

        # -------- Learnable K queries (anchors) --------
        self.anchors = MultiModalAnchors(
            hidden_dim=d_model, n_pred=n_modes, mode_emb=anchor_mode, scale=1.0
        )

        # -------- Transformer Decoder (anchors as tgt, history as memory) --------
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # -------- Prediction heads --------
        hidden = 2 * d_model

        # Regression branch (for mean/logvar) uses its own trunk
        self.pred_mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden, future_len * 2)
        self.logvar_head = nn.Linear(hidden, future_len * 2)

        # NEW: Separate confidence branch (do NOT share regression trunk)
        self.conf_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conf_head = nn.Linear(d_model, 1)

        # Light normalization on memory tokens
        self.mem_ln = nn.LayerNorm(d_model)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Mean over valid timesteps.
        x:    [B, Th, D]
        mask: [B, Th] (1/True = valid)
        return: [B, D]
        """
        m = mask.float()
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * m.unsqueeze(-1)).sum(dim=1) / denom

    def _make_tgt_mask(self, B: int, K: int, device: torch.device) -> torch.Tensor | None:
        """
        Build a target self-attention mask for [B, K, D] queries.

        If independent_modes:
            mask[i, j] = True for i!=j (block off-diagonal), and False on diagonal.
            That is, each query cannot attend to other queries (no inter-mode info flow).
        Else:
            return None (default behavior: full self-attention among queries).
        """
        if not self.independent_modes:
            return None
        tgt_mask = torch.ones(K, K, dtype=torch.bool, device=device)
        tgt_mask.fill_diagonal_(False)  # allow self, block others
        return tgt_mask  # nn.Transformer accepts bool mask: True = masked

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            tokens: [B, Th, 2]
            mask:   [B, Th]  (1/True = valid)

        Returns:
            conf_logits: [B, K]
            mean:        [B, K, Tf, 2]
            logvar:      [B, K, Tf, 2]  (clamped for safety; see training loss below)
        """
        B, Th, C = tokens.shape
        assert C == 2, f"Expect last dim=2, got {C}"
        assert Th == self.hist_len, f"Expect hist_len={self.hist_len}, got {Th}"


        # ------------------------------------------------------------
        # 1) Encode history tokens
        # ------------------------------------------------------------
        h = self.input_mlp(tokens)                            # [B, Th, D]
        h = h + self.temporal_pe[:Th, :].unsqueeze(0)         # add learnable temporal PE
        emb = self.mem_ln(h)                                  # [B, Th, D]
        memory_key_padding_mask = (mask == 0)                 # True = padding

        # Pooled context for optional anchor conditioning
        ctx = self._masked_mean(emb, mask)                    # [B, D]

        # ------------------------------------------------------------
        # 2) Build K queries (anchors)
        # ------------------------------------------------------------
        anchors = self.anchors(B, ctx=ctx)                    # [B, K, D]

        # Optional target self-attention mask (make modes independent)
        tgt_mask = self._make_tgt_mask(B, self.n_modes, anchors.device)

        # ------------------------------------------------------------
        # 3) Decode: (optional) anchor self-attn + cross-attn over encoded history
        # ------------------------------------------------------------
        dec_out = self.decoder(
            tgt=anchors,                       # [B, K, D]
            memory=emb,                        # [B, Th, D]
            tgt_mask=tgt_mask,                 # [K, K] bool, True = masked
            memory_key_padding_mask=memory_key_padding_mask  # [B, Th]
        )                                       # -> [B, K, D]

        # ------------------------------------------------------------
        # 4) Heads
        # ------------------------------------------------------------
        # Confidence branch (independent small trunk)
        c = self.conf_mlp(dec_out)                         # [B, K, D]
        conf_logits = self.conf_head(c).squeeze(-1)        # [B, K]

        # Regression branch
        z = self.pred_mlp(dec_out)                         # [B, K, 2D]
        mean = self.mean_head(z).view(B, self.n_modes, self.future_len, 2)
        logvar = self.logvar_head(z).view(B, self.n_modes, self.future_len, 2)

        # Safety clamp to prevent NaNs/explosions; training loss still uses softplus floor.
        lo, hi = self.logvar_clamp
        logvar = torch.clamp(logvar, lo, hi)

        return conf_logits, mean, logvar
