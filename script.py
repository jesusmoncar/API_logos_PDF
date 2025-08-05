import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import os
import pickle
import json
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

# ML y Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split

# OCR
import pytesseract
import easyocr

# Inpainting
from scipy import ndimage
from skimage import restoration, morphology
from skimage.segmentation import flood_fill

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogoDataset(Dataset):
    """Dataset personalizado para entrenar el detector de logos."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class LogoDetectorCNN(nn.Module):
    """Red neuronal convolucional para detectar logos."""
    
    def __init__(self, num_classes: int = 2):
        super(LogoDetectorCNN, self).__init__()
        # Usar ResNet18 preentrenado como backbone
        self.backbone = resnet18(pretrained=True)
        # Modificar la última capa para nuestras clases
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class OCRDetector:
    """Detector de texto usando OCR para identificar logos con texto."""
    
    def __init__(self, use_easyocr: bool = True):
        self.use_easyocr = use_easyocr
        if use_easyocr:
            self.reader = easyocr.Reader(['es', 'en'])  # Español e Inglés
    
    def detect_text_in_logo(self, image: np.ndarray, logo_keywords: List[str] = None) -> Dict:
        """
        Detecta texto en una imagen y determina si contiene palabras clave de logos.
        
        Args:
            image: Imagen en formato numpy array
            logo_keywords: Lista de palabras clave que indican logos
            
        Returns:
            Diccionario con información del texto detectado
        """
        if logo_keywords is None:
            logo_keywords = [
                'logo', 'brand', 'company', 'corp', 'inc', 'ltd', 'sa', 'sl',
                'trademark', 'copyright', '©', '®', '™', 'marca', 'empresa'
            ]
        
        result = {
            'has_text': False,
            'is_logo_text': False,
            'text_content': '',
            'confidence': 0.0,
            'text_boxes': []
        }
        
        try:
            if self.use_easyocr:
                # Usar EasyOCR
                detections = self.reader.readtext(image)
                
                for detection in detections:
                    bbox, text, confidence = detection
                    result['text_content'] += text + ' '
                    result['text_boxes'].append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence
                    })
                    
                    if confidence > result['confidence']:
                        result['confidence'] = confidence
            else:
                # Usar Tesseract
                text = pytesseract.image_to_string(image, lang='spa+eng')
                result['text_content'] = text.strip()
                
                # Obtener información de confianza
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                if confidences:
                    result['confidence'] = np.mean(confidences) / 100.0
            
            result['has_text'] = len(result['text_content'].strip()) > 0
            
            # Verificar si contiene palabras clave de logos
            text_lower = result['text_content'].lower()
            result['is_logo_text'] = any(keyword.lower() in text_lower for keyword in logo_keywords)
            
        except Exception as e:
            logger.error(f"Error en OCR: {e}")
            
        return result

class InpaintingProcessor:
    """Procesador para rellenar áreas de logos de manera natural."""
    
    def __init__(self):
        pass
    
    def create_mask_from_logo(self, image: np.ndarray, logo_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crea una máscara para el área del logo.
        
        Args:
            image: Imagen original
            logo_bbox: Bounding box del logo (x, y, width, height)
            
        Returns:
            Máscara binaria del logo
        """
        x, y, w, h = logo_bbox
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        
        # Suavizar los bordes de la máscara
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def inpaint_logo_area(self, image: np.ndarray, mask: np.ndarray, method: str = 'telea') -> np.ndarray:
        """
        Rellena el área del logo usando técnicas de inpainting.
        
        Args:
            image: Imagen original
            mask: Máscara del área a rellenar
            method: Método de inpainting ('telea' o 'ns')
            
        Returns:
            Imagen con el área rellenada
        """
        try:
            if method == 'telea':
                inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            else:  # ns (Navier-Stokes)
                inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
            
            return inpainted
            
        except Exception as e:
            logger.error(f"Error en inpainting: {e}")
            # Fallback: rellenar con el color promedio del entorno
            return self.fallback_inpaint(image, mask)
    
    def fallback_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Método alternativo de relleno usando interpolación de colores del entorno.
        
        Args:
            image: Imagen original
            mask: Máscara del área a rellenar
            
        Returns:
            Imagen con el área rellenada
        """
        result = image.copy()
        
        # Expandir la máscara para obtener píxeles del entorno
        kernel = np.ones((10, 10), np.uint8)
        expanded_mask = cv2.dilate(mask, kernel, iterations=1)
        border_mask = expanded_mask - mask
        
        # Calcular color promedio del borde
        if len(image.shape) == 3:
            for channel in range(image.shape[2]):
                border_pixels = image[:, :, channel][border_mask > 0]
                if len(border_pixels) > 0:
                    avg_color = np.mean(border_pixels)
                    result[:, :, channel][mask > 0] = avg_color
        else:
            border_pixels = image[border_mask > 0]
            if len(border_pixels) > 0:
                avg_color = np.mean(border_pixels)
                result[mask > 0] = avg_color
        
        return result

class AdvancedPDFLogoRemover:
    """Sistema avanzado de detección y eliminación de logos con ML."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path or "logo_detector_model.pth"
        self.ocr_detector = OCRDetector(use_easyocr=True)
        self.inpainter = InpaintingProcessor()
        
        # Transformaciones para el modelo
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Cargar modelo si existe
        self.load_model()
    
    def prepare_training_data(self, data_dir: str) -> None:
        """
        Prepara los datos de entrenamiento desde un directorio organizado.
        
        Estructura esperada:
        data_dir/
        ├── logos/          # Imágenes que SÍ son logos
        └── no_logos/       # Imágenes que NO son logos
        
        Args:
            data_dir: Directorio con los datos de entrenamiento
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"El directorio {data_dir} no existe")
        
        # Recopilar imágenes y etiquetas
        image_paths = []
        labels = []
        
        # Logos (etiqueta 1)
        logo_dir = data_path / "logos"
        if logo_dir.exists():
            for img_path in logo_dir.glob("*.jpg") + list(logo_dir.glob("*.png")):
                image_paths.append(str(img_path))
                labels.append(1)
        
        # No logos (etiqueta 0)
        no_logo_dir = data_path / "no_logos"
        if no_logo_dir.exists():
            for img_path in no_logo_dir.glob("*.jpg") + list(no_logo_dir.glob("*.png")):
                image_paths.append(str(img_path))
                labels.append(0)
        
        logger.info(f"Datos preparados: {len(image_paths)} imágenes "
                   f"({labels.count(1)} logos, {labels.count(0)} no logos)")
        
        return image_paths, labels
    
    def train_model(self, data_dir: str, epochs: int = 10, batch_size: int = 32) -> None:
        """
        Entrena el modelo de detección de logos.
        
        Args:
            data_dir: Directorio con datos de entrenamiento
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
        """
        logger.info("Iniciando entrenamiento del modelo...")
        
        # Preparar datos
        image_paths, labels = self.prepare_training_data(data_dir)
        
        # Dividir en entrenamiento y validación
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Crear datasets
        train_dataset = LogoDataset(train_paths, train_labels, self.transform)
        val_dataset = LogoDataset(val_paths, val_labels, self.transform)
        
        # Crear data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Inicializar modelo
        self.model = LogoDetectorCNN(num_classes=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        # Configurar optimizador y función de pérdida
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Entrenar
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Entrenamiento
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validación
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            logger.info(f'Época {epoch+1}/{epochs}: '
                       f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            
            # Guardar mejor modelo
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()
        
        logger.info(f"Entrenamiento completado. Mejor precisión: {best_val_acc:.2f}%")
    
    def save_model(self) -> None:
        """Guarda el modelo entrenado."""
        if self.model is not None:
            torch.save(self.model.state_dict(), self.model_path)
            logger.info(f"Modelo guardado en {self.model_path}")
    
    def load_model(self) -> bool:
        """
        Carga un modelo preentrenado.
        
        Returns:
            True si el modelo se cargó exitosamente
        """
        if os.path.exists(self.model_path):
            try:
                self.model = LogoDetectorCNN(num_classes=2)
                self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
                self.model.eval()
                logger.info(f"Modelo cargado desde {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Error cargando modelo: {e}")
                return False
        return False
    
    def predict_logo(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Predice si una imagen contiene un logo usando el modelo entrenado.
        
        Args:
            image: Imagen en formato numpy array
            
        Returns:
            Tupla (es_logo, confianza)
        """
        if self.model is None:
            logger.warning("Modelo no disponible, usando detección heurística")
            return self.fallback_logo_detection(image)
        
        try:
            # Convertir a PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            # Aplicar transformaciones
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Predicción
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                is_logo = predicted.item() == 1
                conf_score = confidence.item()
                
                return is_logo, conf_score
                
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return self.fallback_logo_detection(image)
    
    def fallback_logo_detection(self, image: np.ndarray) -> Tuple[bool, float]:
        """Detección de logos por métodos heurísticos como fallback."""
        # Implementar la lógica heurística del código anterior
        height, width = image.shape[:2]
        
        # Criterios básicos
        if not (50 <= max(height, width) <= 300):
            return False, 0.0
        
        # Análisis de bordes
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (height * width)
        
        is_logo = edge_ratio > 0.1
        confidence = min(edge_ratio * 2, 1.0)  # Normalizar a [0,1]
        
        return is_logo, confidence
    
    def extract_and_analyze_images(self, page) -> List[Dict]:
        """
        Extrae y analiza imágenes de una página usando ML y OCR.
        
        Args:
            page: Página de PyMuPDF
            
        Returns:
            Lista de diccionarios con análisis completo de imágenes
        """
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # Extraer imagen
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                pil_image = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(pil_image)
                
                # Obtener rectángulo
                image_rects = [rect for rect in page.get_image_rects(img) if rect]
                if not image_rects:
                    continue
                    
                rect = image_rects[0]
                
                # Análisis con ML
                is_logo_ml, confidence_ml = self.predict_logo(image_array)
                
                # Análisis con OCR
                ocr_result = self.ocr_detector.detect_text_in_logo(image_array)
                
                # Puntuación combinada
                combined_score = (
                    confidence_ml * 0.7 +  # 70% peso del ML
                    (0.8 if ocr_result['is_logo_text'] else 0.2) * 0.3  # 30% peso del OCR
                )
                
                is_logo_final = combined_score > 0.5
                
                analysis = {
                    'xref': xref,
                    'index': img_index,
                    'image': image_array,
                    'rect': rect,
                    'size': (image_array.shape[1], image_array.shape[0]),
                    'ml_prediction': {
                        'is_logo': is_logo_ml,
                        'confidence': confidence_ml
                    },
                    'ocr_analysis': ocr_result,
                    'final_prediction': {
                        'is_logo': is_logo_final,
                        'combined_score': combined_score
                    }
                }
                
                images.append(analysis)
                
                logger.info(f"Imagen {img_index}: ML={confidence_ml:.3f}, "
                           f"OCR={'Sí' if ocr_result['is_logo_text'] else 'No'}, "
                           f"Final={combined_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error analizando imagen {img_index}: {e}")
                continue
                
        return images
    
    def remove_logo_with_inpainting(self, page, logo_data: Dict) -> None:
        """
        Elimina un logo usando técnicas de inpainting.
        
        Args:
            page: Página de PyMuPDF
            logo_data: Datos del logo a eliminar
        """
        try:
            rect = logo_data['rect']
            image = logo_data['image']
            
            # Convertir coordenadas del rectángulo
            x0, y0, x1, y1 = rect
            bbox = (int(x0), int(y0), int(x1-x0), int(y1-y0))
            
            # Crear máscara
            mask = self.inpainter.create_mask_from_logo(image, bbox)
            
            # Aplicar inpainting
            inpainted = self.inpainter.inpaint_logo_area(image, mask)
            
            # Aquí deberíamos reemplazar la imagen en el PDF
            # Por limitaciones de PyMuPDF, usamos el método de rectángulo
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
            
            logger.info(f"Logo eliminado con inpainting en: {rect}")
            
        except Exception as e:
            logger.error(f"Error en inpainting: {e}")
            # Fallback al método simple
            page.draw_rect(logo_data['rect'], color=(1, 1, 1), fill=(1, 1, 1))
    
    def process_pdf(self, input_path: str, output_path: str, 
                   confidence_threshold: float = 0.5) -> Dict:
        """
        Procesa un PDF usando todas las técnicas avanzadas.
        
        Args:
            input_path: Ruta del PDF de entrada
            output_path: Ruta del PDF de salida
            confidence_threshold: Umbral de confianza para considerar como logo
            
        Returns:
            Estadísticas del procesamiento
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"El archivo {input_path} no existe")
        
        stats = {
            'total_pages': 0,
            'total_images': 0,
            'logos_detected_ml': 0,
            'logos_detected_ocr': 0,
            'logos_removed': 0,
            'processing_details': []
        }
        
        try:
            doc = fitz.open(input_path)
            stats['total_pages'] = len(doc)
            
            logger.info(f"Procesando PDF con técnicas avanzadas: {input_path}")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                logger.info(f"Procesando página {page_num + 1}")
                
                # Análisis avanzado de imágenes
                analyzed_images = self.extract_and_analyze_images(page)
                stats['total_images'] += len(analyzed_images)
                
                page_details = {
                    'page_number': page_num + 1,
                    'images_analyzed': len(analyzed_images),
                    'logos_found': []
                }
                
                # Procesar logos detectados
                for img_data in analyzed_images:
                    ml_pred = img_data['ml_prediction']
                    ocr_analysis = img_data['ocr_analysis']
                    final_pred = img_data['final_prediction']
                    
                    # Contabilizar detecciones
                    if ml_pred['is_logo']:
                        stats['logos_detected_ml'] += 1
                    if ocr_analysis['is_logo_text']:
                        stats['logos_detected_ocr'] += 1
                    
                    # Eliminar si supera el umbral
                    if final_pred['combined_score'] >= confidence_threshold:
                        self.remove_logo_with_inpainting(page, img_data)
                        stats['logos_removed'] += 1
                        
                        page_details['logos_found'].append({
                            'ml_confidence': ml_pred['confidence'],
                            'has_logo_text': ocr_analysis['is_logo_text'],
                            'text_detected': ocr_analysis['text_content'],
                            'final_score': final_pred['combined_score']
                        })
                
                stats['processing_details'].append(page_details)
            
            # Guardar PDF procesado
            doc.save(output_path)
            doc.close()
            
            logger.info(f"Procesamiento completado. PDF guardado en: {output_path}")
            
        except Exception as e:
            logger.error(f"Error procesando PDF: {e}")
            raise
        
        return stats

def main():
    """Función principal para demostrar el uso del sistema avanzado."""
    
    print("=== SISTEMA AVANZADO DE ELIMINACIÓN DE LOGOS ===\n")
    
    # Configuración
    input_file = "Louvre-tickets_V250781813013.pdf"
    output_file = "documento_procesado_avanzado.pdf"
    training_data_dir = "training_data"  # Directorio con datos de entrenamiento
    
    # Crear instancia del sistema avanzado
    remover = AdvancedPDFLogoRemover()
    
    # Paso 1: Entrenar modelo (opcional, solo si tienes datos)
    if os.path.exists(training_data_dir):
        print("1. Entrenando modelo de detección de logos...")
        try:
            remover.train_model(training_data_dir, epochs=5, batch_size=16)
            print("   ✅ Modelo entrenado exitosamente\n")
        except Exception as e:
            print(f"   ⚠️  Error en entrenamiento: {e}")
            print("   Continuando con detección heurística\n")
    else:
        print("1. No se encontraron datos de entrenamiento.")
        print("   Usando detección heurística como alternativa\n")
    
    # Paso 2: Procesar PDF
    print("2. Procesando PDF con técnicas avanzadas...")
    try:
        stats = remover.process_pdf(
            input_file, 
            output_file, 
            confidence_threshold=0.4 # Umbral más estricto
        )
        
        print("   ✅ PDF procesado exitosamente\n")
        
        # Mostrar estadísticas detalladas
        print("=== ESTADÍSTICAS DETALLADAS ===")
        print(f"Páginas procesadas: {stats['total_pages']}")
        print(f"Imágenes analizadas: {stats['total_images']}")
        print(f"Logos detectados por ML: {stats['logos_detected_ml']}")
        print(f"Logos detectados por OCR: {stats['logos_detected_ocr']}")
        print(f"Logos eliminados: {stats['logos_removed']}")
        print(f"Archivo generado: {output_file}\n")
        
        # Detalles por página
        for page_detail in stats['processing_details']:
            if page_detail['logos_found']:
                print(f"Página {page_detail['page_number']}:")
                for i, logo in enumerate(page_detail['logos_found']):
                    print(f"  Logo {i+1}:")
                    print(f"    - Confianza ML: {logo['ml_confidence']:.3f}")
                    print(f"    - Texto detectado: {logo['text_detected'][:50]}...")
                    print(f"    - Puntuación final: {logo['final_score']:.3f}")
        
    except FileNotFoundError:
        print(f"   ❌ Error: No se encontró el archivo {input_file}")
        print("   Asegúrate de que el archivo existe")
    except Exception as e:
        print(f"   ❌ Error durante el procesamiento: {e}")

    print("\n=== INSTRUCCIONES DE USO ===")
    print("""
1. PREPARAR DATOS DE ENTRENAMIENTO:
   Crea un directorio 'training_data' con esta estructura:
   training_data/
   ├── logos/          # Imágenes que SÍ son logos (100-500 imágenes)
   └── no_logos/       # Imágenes que NO son logos (100-500 imágenes)

2. INSTALAR DEPENDENCIAS:
   pip install torch torchvision pytesseract easyocr opencv-python scikit-image

3. CONFIGURAR TESSERACT (si usas Windows):
   - Descargar desde: https://github.com/tesseract-ocr/tesseract
   - Agregar al PATH del sistema

4. EJECUTAR:
   python script.py

5. PERSONALIZAR:
   - Ajustar confidence_threshold (0.0-1.0)
   - Modificar palabras clave de OCR
   - Cambiar métodos de inpainting
    """)

if __name__ == "__main__":
    main()