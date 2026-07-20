import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

# Import custom architecture and loader from ML Pipeline (Phase 1)
from ml_pipeline.model import CropDiseaseClassifier
from ml_pipeline.dataset import get_dataloaders

def get_file_size_mb(file_path: str) -> float:
    """Returns file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)

@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Runs evaluation over the validation set and returns (accuracy_percentage, avg_latency_ms)."""
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    start_time = time.perf_counter()
    
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
    total_time = (time.perf_counter() - start_time) * 1000
    avg_latency = total_time / total
    accuracy = (correct / total) * 100.0
    return accuracy, avg_latency

def main():
    parser = argparse.ArgumentParser(description="PyTorch INT8 Dynamic Quantization for Edge Serving")
    parser.add_argument("--model_path", type=str, required=True, help="Path to original FP32 .pth weights")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation image directory")
    parser.add_argument("--num_classes", type=int, default=8, help="Number of target disease classes")
    parser.add_argument("--output_path", type=str, default="efficientnet_b0_int8.pt", help="Export path")
    parser.add_argument("--max_drop", type=float, default=1.0, help="Maximum allowed accuracy drop percentage")
    args = parser.parse_args()

    # Dynamic quantization in PyTorch is optimized for CPU execution backends
    device = torch.device("cpu")
    print(f"[Optimization] Initializing INT8 quantization pipeline on: {device}")

    # 1. Load Baseline FP32 Model
    print("[Optimization] Loading FP32 baseline architecture...")
    fp32_model = CropDiseaseClassifier(num_classes=args.num_classes, model_name="efficientnet_b0", pretrained=False)
    fp32_model.load_state_dict(torch.load(args.model_path, map_location=device))
    fp32_model.eval()

    # Save standalone FP32 state dict for baseline size comparison
    temp_fp32_path = "temp_fp32_baseline.pt"
    torch.save(fp32_model.state_dict(), temp_fp32_path)
    fp32_size = get_file_size_mb(temp_fp32_path)

    # 2. Load Validation Dataset
    _, val_loader, _ = get_dataloaders(
        train_dir=args.val_dir, val_dir=args.val_dir, batch_size=32, num_workers=2
    )

    # 3. Benchmark Baseline FP32 Model
    print("[Benchmark] Evaluating FP32 baseline accuracy and latency...")
    fp32_acc, fp32_lat = evaluate_accuracy(fp32_model, val_loader, device)
    print(f" -> FP32 Baseline | Accuracy: {fp32_acc:.2f}% | Latency: {fp32_lat:.2f} ms/sample | Size: {fp32_size:.2f} MB")

    # 4. Apply INT8 Dynamic Quantization
    # We target nn.Linear layers (vital for classification heads and Vision Transformers)
    print("\n[Optimization] Applying Dynamic Quantization (FP32 -> INT8) on nn.Linear layers...")
    int8_model = torch.quantization.quantize_dynamic(
        model=fp32_model,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8
    )

    # Export quantized TorchScript model (required for saving dynamically quantized graphs)
    dummy_input = torch.randn(1, 3, 224, 224)
    traced_int8_model = torch.jit.trace(int8_model, dummy_input)
    torch.jit.save(traced_int8_model, args.output_path)
    int8_size = get_file_size_mb(args.output_path)

    # 5. Benchmark Quantized INT8 Model
    print("[Benchmark] Evaluating INT8 quantized model accuracy and latency...")
    int8_acc, int8_lat = evaluate_accuracy(int8_model, val_loader, device)
    
    # Calculate compression metrics
    size_reduction = fp32_size / int8_size
    acc_drop = fp32_acc - int8_acc
    speedup = fp32_lat / int8_lat

    print("\n" + "="*60)
    print("                 QUANTIZATION SUMMARY REPORT                 ")
    print("="*60)
    print(f"Metric             | FP32 Baseline     | INT8 Quantized    | Delta / Gain")
    print(f"-------------------|-------------------|-------------------|-----------------")
    print(f"File Size          | {fp32_size:6.2f} MB       | {int8_size:6.2f} MB       | {size_reduction:.2f}x smaller")
    print(f"Accuracy           | {fp32_acc:6.2f}%          | {int8_acc:6.2f}%          | -{acc_drop:.2f}%")
    print(f"Inference Latency  | {fp32_lat:6.2f} ms/img     | {int8_lat:6.2f} ms/img     | {speedup:.2f}x faster")
    print("="*60)

    # Clean up temporary baseline file
    if os.path.exists(temp_fp32_path):
        os.remove(temp_fp32_path)

    # 6. Automated Quality Gate Validation
    if acc_drop > args.max_drop:
        os.remove(args.output_path)
        raise RuntimeError(
            f"[Quality Gate Failed] Accuracy drop of {acc_drop:.2f}% exceeds the threshold limit of {args.max_drop}%. "
            "Quantized artifact has been purged. Consider Static Quantization with calibration instead."
        )
    else:
        print(f"\n[Success] Quality gate passed! Production edge artifact saved to: {args.output_path}")

if __name__ == "__main__":
    main()