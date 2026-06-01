from typing import Dict, Any, Optional, List, Set
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
import timm
import clip  # 使用官方 clip 库

CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
    "vit_large_patch14_dinov2.lvd142m": 1024,  # DINOv2
    "vit_large_patch16_siglip_256": 1024,  # SigLIP
}


def _set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


def _parse_qkv_spec(spec: str) -> Set[str]:
    s = spec.lower().replace(" ", "")
    if s in {"all", "qkv"}:
        return {"q", "k", "v"}
    s = s.replace(",", "")
    allowed = set()
    for ch in s:
        if ch in {"q", "k", "v"}:
            allowed.add(ch)
    if len(allowed) == 0:
        raise ValueError(f"Bad lora_qkv spec={spec}, expected subset of q/k/v or 'qkv'")
    return allowed


class LoRAParametrization(nn.Module):
    def __init__(self, out_features: int, in_features: int, r: int = 8, alpha: float = 16.0,
                 row_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.r = int(r)
        self.scaling = float(alpha / r)
        self.enabled = True

        self.A = nn.Parameter(torch.zeros(self.r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, self.r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)

        if row_mask is not None:
            self.register_buffer("row_mask", row_mask.float())
        else:
            self.row_mask = None

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return W
        dW = (self.B @ self.A) * self.scaling
        if self.row_mask is not None:
            dW = dW * self.row_mask
        return W + dW


class CLIP_LORA_PURE(nn.Module):  # 移除了 BaseModel，改为 nn.Module
    def __init__(
            self,
            name: str = "ViT-L/14",
            num_classes: int = 1,
            lora_r: int = 8,
            lora_alpha: float = 16.0,
            lora_qkv: str = "qkv",
            lora_apply_out_proj: bool = True,
            tune_mode: str = "lora",
    ):
        super().__init__()
        self.is_timm = False
        if name in CHANNELS and "ViT" not in name and "RN" not in name:
            self.is_timm = True
            print(f"[Init] Loading via timm: {name}")
            self.model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224)
            data_config = timm.data.resolve_data_config(self.model.default_cfg, model=self.model)
            data_config['input_size'] = (3, 224, 224)
            self.preprocess = timm.data.create_transform(**data_config, is_training=False)
            feat_dim = self.model.num_features
        else:
            self.model, self.preprocess = clip.load(name, device="cpu")
            feat_dim = CHANNELS[name]

        self.fc = nn.Linear(feat_dim, num_classes)  # 使用动态 feat_dim 取代 CHANNELS[name]

        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.lora_apply_out_proj = bool(lora_apply_out_proj)
        self.lora_qkv_set = _parse_qkv_spec(lora_qkv)

        self._lora_params: List[LoRAParametrization] = []
        self._inject_lora_into_visual_attention()

        self.tune_mode = None
        self.set_tune_mode(tune_mode)

    def _make_qkv_row_mask(self, out_features: int, in_features: int) -> torch.Tensor:
        mask = torch.zeros(out_features, 1)
        if out_features % 3 != 0:
            mask[:] = 1.0
            return mask
        d = out_features // 3
        if "q" in self.lora_qkv_set:
            mask[0:d, 0] = 1.0
        if "k" in self.lora_qkv_set:
            mask[d:2 * d, 0] = 1.0
        if "v" in self.lora_qkv_set:
            mask[2 * d:3 * d, 0] = 1.0
        return mask

    def _inject_lora_into_visual_attention(self):
        blocks = None
        if hasattr(self.model, "visual"):
            visual = getattr(self.model, "visual", None)
            if hasattr(visual, "transformer"):
                blocks = visual.transformer.resblocks
        elif hasattr(self.model, "blocks"):
            blocks = self.model.blocks

        if blocks is None:
            return

        for li, blk in enumerate(blocks):
            layer_id = li + 1
            attn_layer = getattr(blk, "attn", None)
            if attn_layer is None:
                continue

            if isinstance(attn_layer, nn.MultiheadAttention):
                if hasattr(attn_layer, "in_proj_weight"):
                    target_weight = attn_layer.in_proj_weight
                    out_features, in_features = target_weight.shape
                    row_mask = self._make_qkv_row_mask(out_features, in_features)
                    lp = LoRAParametrization(out_features, in_features, self.lora_r, self.lora_alpha, row_mask)
                    parametrize.register_parametrization(attn_layer, "in_proj_weight", lp)
                    self._lora_params.append(lp)

                if self.lora_apply_out_proj and hasattr(attn_layer, "out_proj"):
                    target_out = attn_layer.out_proj
                    lp_out = LoRAParametrization(target_out.out_features, target_out.in_features, self.lora_r,
                                                 self.lora_alpha, None)
                    parametrize.register_parametrization(target_out, "weight", lp_out)
                    self._lora_params.append(lp_out)

            elif hasattr(attn_layer, "qkv") and isinstance(attn_layer.qkv, nn.Linear):
                target_linear = attn_layer.qkv
                out_features = target_linear.out_features
                in_features = target_linear.in_features
                row_mask = self._make_qkv_row_mask(out_features, in_features)
                lp = LoRAParametrization(out_features, in_features, self.lora_r, self.lora_alpha, row_mask)
                parametrize.register_parametrization(target_linear, "weight", lp)
                self._lora_params.append(lp)

                if self.lora_apply_out_proj and hasattr(attn_layer, "proj") and isinstance(attn_layer.proj, nn.Linear):
                    lp2 = LoRAParametrization(attn_layer.proj.out_features, attn_layer.proj.in_features, self.lora_r,
                                              self.lora_alpha, None)
                    parametrize.register_parametrization(attn_layer.proj, "weight", lp2)
                    self._lora_params.append(lp2)

    def _set_lora_forward_enabled(self, enabled: bool):
        for lp in self._lora_params:
            lp.enabled = bool(enabled)

    def set_tune_mode(self, mode: str):
        mode = mode.lower().strip()
        self.tune_mode = mode
        if mode == "lp":
            _set_requires_grad(self.model, False)
            self._set_lora_forward_enabled(False)
            for lp in self._lora_params:
                lp.A.requires_grad_(False)
                lp.B.requires_grad_(False)
        elif mode == "lora":
            _set_requires_grad(self.model, False)
            self._set_lora_forward_enabled(True)
            for lp in self._lora_params:
                lp.A.requires_grad_(True)
                lp.B.requires_grad_(True)
        elif mode == "fft":
            _set_requires_grad(self.model, True)
            self._set_lora_forward_enabled(False)
            for lp in self._lora_params:
                lp.A.requires_grad_(False)
                lp.B.requires_grad_(False)
        _set_requires_grad(self.fc, True)

    def forward(self, image, label: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        x = image
        y = None if label is None else label.float()

        if self.is_timm:
            feat = self.model.forward_features(x)
            if feat.ndim == 3:
                feat = feat[:, 0, :]
        else:
            feat = self.model.encode_image(x)

        logit = self.fc(feat)
        logit = logit.squeeze(dim=1) if logit.ndim == 2 and logit.shape[1] == 1 else logit
        pred_label = torch.sigmoid(logit)

        if y is None:
            return {"pred_label": pred_label, "visual_loss": {}}

        bce_main = F.binary_cross_entropy_with_logits(logit, y)
        return {
            "backward_loss": bce_main,
            "pred_label": pred_label,
            "visual_loss": {
                "total_loss": bce_main.detach(),
                "bce_main": bce_main.detach(),
            }
        }
