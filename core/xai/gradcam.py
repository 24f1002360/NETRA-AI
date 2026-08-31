"""
Grad-CAM / Grad-CAM++ on Kanchan's grading backbone.

Usage once the checkpoint + layer name land (Day 7):

    engine = GradCAM(model, target_layer="layer4", variant="gradcam++")
    cam, class_idx = engine(input_tensor, class_idx=None)

Requires torch (add to requirements.txt -- not there yet, Kanchan's stub
doesn't need it). Falls back gracefully if torch isn't installed so the
rest of the repo never breaks on import.
"""
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GradCAM:
    def __init__(self, model, target_layer, variant="gradcam"):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for GradCAM. pip install torch")
        assert variant in ("gradcam", "gradcam++")
        self.model = model
        self.variant = variant
        self.activations = None
        self.gradients = None

        layer = dict(model.named_modules()).get(target_layer)
        if layer is None:
            raise ValueError(f"layer '{target_layer}' not found in model. "
                              f"Confirm exact name with Kanchan.")
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        acts = self.activations[0]    # C,H,W
        grads = self.gradients[0]     # C,H,W

        if self.variant == "gradcam++":
            weights = self._gradcam_pp_weights(acts, grads)
        else:
            weights = grads.mean(dim=(1, 2))

        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_up = F.interpolate(
            cam[None, None], size=input_tensor.shape[-2:],
            mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

        return cam_up.astype(np.float32), class_idx

    @staticmethod
    def _gradcam_pp_weights(acts, grads):
        grads2 = grads ** 2
        grads3 = grads2 * grads
        denom = 2 * grads2 + acts.sum(dim=(1, 2), keepdim=True) * grads3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grads2 / denom
        return (alpha * torch.relu(grads)).sum(dim=(1, 2))