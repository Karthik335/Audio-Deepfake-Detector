"""
Step 3 - Model construction.
ResNet-18 adapted for single-channel Mel-spectrogram input and a single
output logit (real vs fake).
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class DeepfakeResNet(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = ResNet18_Weights.IMAGENET1K_V1
                net = resnet18(weights=weights)
            except Exception as e:   # no internet -> train from scratch
                print(f"[model] Could not load ImageNet weights ({e.__class__.__name__}); "
                      "training from scratch.")
                weights = None
        if weights is None:
            net = resnet18(weights=None)

        # 1) Accept a 1-channel spectrogram instead of a 3-channel RGB image.
        #    When pretrained, keep the learned filters by averaging the RGB kernels.
        old_conv = net.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if weights is not None:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        net.conv1 = new_conv

        # 2) Replace the 1000-class ImageNet head with one output logit.
        in_features = net.fc.in_features
        net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        self.net = net

    def forward(self, x):                 # x: (batch, 1, n_mels, frames)
        return self.net(x).squeeze(1)     # (batch,) raw logits


def load_model(path, device="cpu"):
    model = DeepfakeResNet(pretrained=False)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device).eval()
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
