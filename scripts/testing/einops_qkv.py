import torch

try:
    from einops import rearrange
    einops_available = True
except ImportError:
    einops_available = False
    print("einops is not installed in this environment; cannot run comparison.")

if einops_available:
    # Parameters
    B, H, ws, D, HH, WW = 2, 4, 2, 8, 3, 3
    Hp, Wp = HH * ws, WW * ws
    C = 3 * H * D  # because C = 3 * H * D

    # Create random tensor
    qkv = torch.randn(B, C, Hp, Wp)

    # --- einops version ---
    out_einops = rearrange(
        qkv,
        'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
        qkv=3,
        h=H,
        d=D,
        hh=HH,
        ww=WW,
        ws1=ws,
        ws2=ws
    )

    # --- manual PyTorch version ---
    qkv_manual = qkv.view(B, 3, H, D, HH, ws, WW, ws)            # split packed dims
    qkv_manual = qkv_manual.permute(1, 0, 4, 6, 2, 5, 7, 3)      # reorder axes
    qkv_manual = qkv_manual.reshape(3, B * HH * WW, H, ws * ws, D)  # merge groups

    # Compare
    max_abs_diff = (out_einops - qkv_manual).abs().max().item()
    same_shape = out_einops.shape == qkv_manual.shape

    print("einops output shape :", out_einops.shape)
    print("manual output shape :", qkv_manual.shape)
    print("Shapes match       :", same_shape)
    print("Max absolute diff  :", max_abs_diff)
