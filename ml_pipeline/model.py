import torch
import torch.nn as nn
from torchvision import models
from typing import Literal

class CropDiseaseClassifier(nn.Module):
    """
    Edge-optimized vision classification backbone supporting EfficientNet-B0 and ConvNeXt-Tiny.
    """
    def __init__(
        self, 
        num_classes: int, 
        model_name: Literal["efficientnet_b0", "convnext_tiny"] = "efficientnet_b0", 
        pretrained: bool = True,
        dropout_rate: float = 0.3
    ):
        super(CropDiseaseClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        elif model_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def export_to_torchscript(
    model: nn.Module, 
    save_path: str, 
    device: torch.device, 
    img_size: int = 224
) -> None:
    """Exports model to TorchScript (.pt) format via tracing for C++ / mobile edge runtimes."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(save_path)
    print(f"[Export] Successfully generated TorchScript model: {save_path}")

def export_to_onnx(
    model: nn.Module, 
    save_path: str, 
    device: torch.device, 
    img_size: int = 224
) -> None:
    """Exports model to ONNX format with dynamic batch sizing for TensorRT/ONNXRuntime execution."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
    print(f"[Export] Successfully generated ONNX model: {save_path}")