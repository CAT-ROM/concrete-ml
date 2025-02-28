import numpy as np
import cv2
import torch
import torch.nn as nn
from concrete.ml.torch.compile import compile_torch_model

# Load and preprocess grayscale images
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))  # Resize for consistency
    return img.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Define FHE-compatible watermark embedding model
class WatermarkEmbedder(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, image, watermark):
        return image + self.alpha * watermark  # Add watermark transparently

# Define FHE-compatible watermark extraction model
class WatermarkExtractor(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, image, original_image):
        return (image - original_image) / self.alpha  # Extract watermark

# Evaluate robustness using Mean Squared Error (MSE)
def evaluate_robustness(image, transformations):
    results = {}
    for transform_name, transform in transformations.items():
        transformed = transform(image)
        mse = np.mean((image - transformed) ** 2)
        results[transform_name] = mse
    return results

# JPEG compression transformation (lossy)
def jpeg_compress(img):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, encoded_img = cv2.imencode('.jpg', img * 255, encode_param)
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
    return decoded_img.astype(np.float32) / 255.0

# Load images
original_image = load_image("original.jpg")
watermark = load_image("watermark.jpg")

# Convert images to PyTorch tensors
original_tensor = torch.from_numpy(original_image).unsqueeze(0)
watermark_tensor = torch.from_numpy(watermark).unsqueeze(0)

# Compile FHE models
embedder_fhe = compile_torch_model(WatermarkEmbedder(), {
    "image": original_tensor, "watermark": watermark_tensor
})
extractor_fhe = compile_torch_model(WatermarkExtractor(), {
    "image": original_tensor, "original_image": original_tensor
})

# Apply watermark embedding (FHE-based)
watermarked = embedder_fhe(original_tensor, watermark_tensor).detach().numpy().squeeze()
cv2.imwrite("watermarked.jpg", watermarked * 255)

# Evaluate robustness to transformations
transformations = {
    "jpeg_compression": jpeg_compress,
    "blur": lambda img: cv2.GaussianBlur(img, (5, 5), 0)
}
robustness_results = evaluate_robustness(watermarked, transformations)

# Extract watermark (FHE-based)
extracted = extractor_fhe(torch.from_numpy(watermarked).unsqueeze(0), original_tensor).detach().numpy().squeeze()
cv2.imwrite("extracted_watermark.jpg", extracted * 255)

# Display FHE circuit details
print("FHE Circuit Embedder:", embedder_fhe.fhe_circuit)
print("FHE Circuit Extractor:", extractor_fhe.fhe_circuit)
print("Watermark robustness results:", robustness_results)
