import fitz  # PyMuPDF
import os
import io
from PIL import Image
import numpy as np
import cv2
import logging
from typing import Tuple
from skimage.feature import local_binary_pattern


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fallback_logo_detection(image: np.ndarray) -> Tuple[bool, float]:
    height, width = image.shape[:2]
    if not (50 <= max(height, width) <= 300):
        return False, 0.0
    
    # Convertir a gris si es RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # 1. Detección de bordes con Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (height * width)
    
    # 2. Conteo de contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    
    # 3. Textura con LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()  # normalizar
    
    # Medida simple de uniformidad: sumatoria de bins más altas
    uniformity = lbp_hist[0] + lbp_hist[1]  # suma de patrones muy comunes (fondo uniforme)
    
    # 4. Histograma de colores para detectar homogeneidad (solo si imagen es RGB)
    color_var = 0
    if len(image.shape) == 3:
        color_var = np.mean(np.var(image, axis=(0,1)))
    
    # 5. Combinar en score
    # Mayor edge_ratio, más contornos, menos uniformidad y mayor varianza color = más probable logo
    score = (
        0.4 * min(edge_ratio * 3, 1.0) + 
        0.3 * min(contour_count / 30, 1.0) + 
        0.2 * (1 - uniformity) + 
        0.1 * min(color_var / 1000, 1.0)
    )
    
    is_logo = score > 0.5  # umbral ajustable
    
    return is_logo, score
    
def extract_images(pdf_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    count = 0
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        logger.info(f"Página {page_num+1}: {len(image_list)} imágenes encontradas")
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_array = np.array(pil_image)

            is_logo, score = fallback_logo_detection(image_array)

            label_hint = "logo_candidate" if is_logo else "no_logo_candidate"
            filename = f"page{page_num+1}_img{img_index}_{label_hint}_{score:.2f}.png"
            save_path = os.path.join(output_dir, filename)
            pil_image.save(save_path)
            count += 1
    logger.info(f"Extraídas {count} imágenes en {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extrae imágenes de un PDF para etiquetado manual")
    parser.add_argument("-i", "--input", required=True, help="PDF de entrada")
    parser.add_argument("-o", "--output", default="candidates", help="Carpeta donde guardar imágenes")
    args = parser.parse_args()

    extract_images(args.input, args.output)
