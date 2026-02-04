import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNIDecoder(nn.Module):
    def __init__(self, in_channels, channels=(256, 128, 64)):
        super().__init__()
        blocks = []
        prev = in_channels
        for ch in channels:
            blocks.append(ConvBNReLU(prev, ch))
            prev = ch
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, output_size):
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(x)
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class UNISegmentor(nn.Module):
    def __init__(
        self,
        encoder,
        embed_dim=None,
        num_classes=2,
        reg_tokens=8,
        freeze_encoder=True,
        decoder_channels=(256, 128, 64),
        predict_distance=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.reg_tokens = reg_tokens
        self.num_prefix_tokens = int(getattr(encoder, "num_prefix_tokens", reg_tokens))
        self.patch_size = self._get_patch_size()

        if embed_dim is None:
            embed_dim = int(getattr(encoder, "num_features", 0) or getattr(encoder, "embed_dim", 0))
        if embed_dim <= 0:
            raise ValueError(
                "Could not infer encoder feature dimension. "
                "Please set `embed_dim` explicitly in config."
            )

        self.feature_proj = nn.Conv2d(embed_dim, decoder_channels[0], kernel_size=1)
        self.decoder = UNIDecoder(
            in_channels=decoder_channels[0],
            channels=decoder_channels[1:],
        )
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        self.dist_head = (
            nn.Conv2d(decoder_channels[-1], 1, kernel_size=1) if predict_distance else None
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _get_patch_size(self):
        patch_size = getattr(self.encoder.patch_embed, "patch_size", 14)
        if isinstance(patch_size, tuple):
            return patch_size
        return (patch_size, patch_size)

    def _tokens_to_map(self, tokens, image_hw):
        b, n, c = tokens.shape

        patch_tokens = tokens[:, self.num_prefix_tokens:, :]
        n_patches = patch_tokens.shape[1]

        h_tokens = image_hw[0] // self.patch_size[0]
        w_tokens = image_hw[1] // self.patch_size[1]
        if h_tokens * w_tokens != n_patches:
            side = int(math.sqrt(n_patches))
            if side * side != n_patches:
                raise RuntimeError(
                    f"Cannot infer token grid from {n_patches} tokens. "
                    "Please check input size / patch size / prefix token count."
                )
            h_tokens, w_tokens = side, side

        feat = patch_tokens.reshape(b, h_tokens, w_tokens, c).permute(0, 3, 1, 2).contiguous()
        return feat

    def extract_feature_map(self, x):
        feats = self.encoder.forward_features(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        if feats.ndim == 4:
            return feats
        if feats.ndim != 3:
            raise RuntimeError(f"Unexpected encoder output shape: {tuple(feats.shape)}")

        return self._tokens_to_map(feats, image_hw=x.shape[-2:])

    def forward(self, x):
        h, w = x.shape[-2:]
        feat_map = self.extract_feature_map(x)
        y = self.feature_proj(feat_map)
        y = self.decoder(y, output_size=(h, w))

        seg_logits = self.seg_head(y)
        out = {"seg_logits": seg_logits}
        if self.dist_head is not None:
            out["dist_pred"] = torch.sigmoid(self.dist_head(y))
        return out


def create_uni_encoder(enc_name="uni2-h", checkpoint=None, use_hf_pretrained=True):
    import os
    import timm

    presets = {
        "uni": {
            "hf_hub_id": "hf-hub:MahmoodLab/UNI",
            "model_kwargs": {
                "model_name": "vit_large_patch16_224",
                "img_size": 224,
                "patch_size": 16,
                "init_values": 1e-5,
                "num_classes": 0,
                "dynamic_img_size": True,
            },
        },
        "uni2-h": {
            "hf_hub_id": "hf-hub:MahmoodLab/UNI2-h",
            "model_kwargs": {
                "model_name": "vit_giant_patch14_224",
                "img_size": 224,
                "patch_size": 14,
                "depth": 24,
                "num_heads": 24,
                "init_values": 1e-5,
                "embed_dim": 1536,
                "mlp_ratio": 2.66667 * 2,
                "num_classes": 0,
                "no_embed_class": True,
                "mlp_layer": timm.layers.SwiGLUPacked,
                "act_layer": torch.nn.SiLU,
                "reg_tokens": 8,
                "dynamic_img_size": True,
            },
        },
    }
    if enc_name in presets:
        cfg = presets[enc_name]
        if use_hf_pretrained and checkpoint is None:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            if hf_token:
                try:
                    from huggingface_hub import login
                    login(token=hf_token, add_to_git_credential=False)
                except Exception:
                    pass
            try:
                return timm.create_model(cfg["hf_hub_id"], pretrained=True)
            except Exception as e:
                raise RuntimeError(
                    "Failed to load gated Hugging Face weights for UNI2-h. "
                    "Please (1) request access on https://huggingface.co/MahmoodLab/UNI2-h, "
                    "(2) login via `huggingface-cli login` (or set HF_TOKEN), "
                    "or (3) use an open timm model like `vit_base_patch16_224` in config."
                ) from e

        model = timm.create_model(**cfg["model_kwargs"])
        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=True)
        return model

    # Fallback: any open timm encoder (e.g., vit_base_patch16_224, resnet50, convnext_tiny)
    pretrained = bool(use_hf_pretrained and checkpoint is None)
    create_kwargs = {"pretrained": pretrained, "num_classes": 0}
    try:
        model = timm.create_model(enc_name, **create_kwargs)
    except TypeError:
        model = timm.create_model(enc_name, pretrained=pretrained)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    return model
