# pip install torch einops  (if you haven’t already)

import torch
from einops import rearrange

# ----------------------------------------------------------
# Manual replacement for:
#   rearrange(attn,
#     '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
#      h=H, d=d, hh=hh, ww=ww, ws1=ws, ws2=ws)
# ----------------------------------------------------------
def attn_rearrange(attn, *, B, H, hh, ww, ws, d):
    """
    attn:  [B*hh*ww, H, ws*ws, d]
    returns  [B, H*d, hh*ws, ww*ws]
    """
    # Split the packed axes: (B·hh·ww) and (ws²)
    attn = attn.view(B, hh, ww, H, ws, ws, d)
    # Re-order to b, h, d, hh, ws1, ww, ws2
    attn = attn.permute(0, 3, 6, 1, 4, 2, 5)
    # Merge (h,d) → H*d,   (hh,ws1) → hh*ws,   (ww,ws2) → ww*ws
    attn = attn.reshape(B, H * d, hh * ws, ww * ws)
    return attn
# ----------------------------------------------------------

if __name__ == "__main__":
    # ---------------- hyper-parameters (pick any) ----------
    B   = 2          # batch
    H   = 4          # heads
    ws  = 2          # window size
    d   = 8          # head dim
    hh  = 3          # Hp // ws
    ww  = 3          # Wp // ws
    # ------------------------------------------------------

    # Random input shaped exactly like the rearrange expects
    attn = torch.randn(B * hh * ww, H, ws * ws, d)

    # -------- reference: einops ---------------------------
    out_einops = rearrange(
        attn,
        '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
        b=B, hh=hh, ww=ww, h=H, d=d, ws1=ws, ws2=ws
    )

    # -------- manual PyTorch version ----------------------
    out_manual = attn_rearrange(
        attn, B=B, H=H, hh=hh, ww=ww, ws=ws, d=d
    )

    # --------------- verification -------------------------
    print("shapes equal :", out_einops.shape == out_manual.shape)
    print("max |Δ|       :", (out_einops - out_manual).abs().max().item())
    assert torch.allclose(out_einops, out_manual)
    assert (out_einops - out_manual).sum() == 0
    print("✅  outputs match exactly")
