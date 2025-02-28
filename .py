import concrete.ml as cml
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from concrete.ml.sklearn import LogisticRegression as FHELogisticRegression
import cv2
import pywt

# Define and initialize an sklearn model
sklearn_model = LogisticRegression()
init_params = sklearn_model.get_params()

if "1.1." in sklearn.__version__:
    init_params.pop("solver", None)  # Ensure compatibility

# Instantiate Concrete ML model
fhe_model = FHELogisticRegression()

# Watermark Embedding using Discrete Wavelet Transform (DWT)
def embed_watermark(image, watermark, alpha=0.1):
    """Embed watermark using DWT for robustness."""
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    LL = LL + alpha * watermark  # Embed watermark
    watermarked_image = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
    return np.clip(watermarked_image, 0, 255).astype(np.uint8)

# Watermark Extraction
def extract_watermark(image, original_image, alpha=0.1):
    """Extract watermark from watermarked image using DWT."""
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs
    original_coeffs = pywt.dwt2(original_image, 'haar')
    original_LL, (_, _, _) = original_coeffs
    extracted_watermark = (LL - original_LL) / alpha
    return np.clip(extracted_watermark, 0, 255).astype(np.uint8)

# JPEG compression transformation
def jpeg_compress(img):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    _, encoded_img = cv2.imencode('.jpg', img, encode_param)
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
    return decoded_img

# Robustness Evaluation
def evaluate_robustness(image, transformations):
    """Check resistance to transformations like blurring and JPEG compression."""
    results = {}
    for transform_name, transform in transformations.items():
        transformed = transform(image)
        results[transform_name] = np.mean(np.abs(image - transformed))
    return results

# Example transformations
transformations = {
    "blur": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
    "jpeg": jpeg_compress
}

# Example usage
original_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
watermark = np.random.randint(0, 10, (256, 256), dtype=np.uint8)

watermarked_image = embed_watermark(original_image, watermark)
robustness_results = evaluate_robustness(watermarked_image, transformations)

print("Watermark robustness results:", robustness_results)
