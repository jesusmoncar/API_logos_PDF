# PDF Logo Removal Tool

## 📋 Descripción
Herramienta automatizada que detecta y elimina logos de documentos PDF utilizando técnicas de machine learning y visión por computadora.

## ✨ Características
- Detección automática de logos usando IA
- Eliminación inteligente de logos sin afectar el contenido
- Procesamiento por lotes de múltiples PDFs
- Entrenamiento personalizable para diferentes tipos de logos
- Interfaz de línea de comandos fácil de usar

## 🚀 Instalación

### Dependencias requeridas
Instala todas las dependencias necesarias ejecutando los siguientes comandos:

```bash
# Dependencias principales
pip install torch torchvision torchaudio
pip install opencv-python
pip install PyMuPDF
pip install Pillow
pip install numpy
pip install scikit-learn
pip install scipy
pip install scikit-image
pip install easyocr

# Dependencias adicionales
pip install anthropic
```

### Instalación alternativa con requirements.txt
Puedes crear un archivo `requirements.txt` con todas las dependencias:

```bash
pip install -r requirements.txt
```

## 📁 Estructura de Carpetas

```
API_logos_PDF/
├── PDF/                           # 📥 Carpeta de PDFs de entrada
│   ├── 1112056.pdf               # Archivos PDF para procesar
│   ├── 1112838.pdf
│   └── ...
├── pdfModificado/                 # 📤 Carpeta de PDFs procesados (salida)
├── training_data/                 # 🎯 Datos de entrenamiento
│   ├── logos/                     # Imágenes con logos para entrenar el modelo
│   └── no_logos/                  # Imágenes sin logos para entrenar el modelo
├── candidates/                    # 🔍 Candidatos de logos extraídos
├── candidates_test/               # 🧪 Candidatos de prueba
├── candidatos_avanzados/          # 🎯 Análisis avanzado de candidatos
│   └── extraction_report.json    # Reporte detallado de extracción
├── script.py                      # 🎯 Script principal de procesamiento
├── extract_candidates.py         # 🔍 Extracción de candidatos de logos
├── train_model.py                # 🤖 Entrenamiento del modelo de IA
├── logo_classifier.pth           # 🧠 Modelo entrenado guardado
├── run_script.bat                 # ▶️ Script para ejecutar el procesamiento
├── limpiar_carpetas.bat          # 🧹 Script para limpiar carpetas
├── requirements.txt               # 📦 Lista de dependencias
└── README.md                      # 📖 Este archivo
```

### Explicación de cada componente:

#### 📂 Carpetas principales:
- **PDF/**: Coloca aquí los archivos PDF que quieres procesar
- **pdfModificado/**: Los PDFs limpiados aparecerán aquí automáticamente
- **training_data/**: Imágenes para entrenar el clasificador de logos
  - `logos/`: Imágenes que contienen logos
  - `no_logos/`: Imágenes que NO contienen logos

#### 📄 Scripts principales:
- **script.py**: Motor principal que procesa los PDFs
- **extract_candidates.py**: Extrae posibles logos de los PDFs
- **train_model.py**: Entrena el modelo de clasificación
- **claude_terminal.py**: Interfaz con IA para análisis avanzado

#### 🔧 Scripts de utilidad:
- **run_script.bat**: Ejecuta el procesamiento automático
- **limpiar_carpetas.bat**: Limpia las carpetas PDF y pdfModificado
- **requirements.txt**: Lista completa de dependencias Python

## 🎮 Uso

### Procesamiento automático (recomendado)
1. Coloca tus archivos PDF en la carpeta `PDF/`
2. Ejecuta el archivo `run_script.bat`
3. Los PDFs procesados aparecerán en `pdfModificado/`

### Procesamiento manual
```bash
# Procesar todos los PDFs
python script.py --batch

# Procesar un PDF específico
python script.py --input "PDF/mi_archivo.pdf" --output "pdfModificado/mi_archivo_limpio.pdf"

# Extraer candidatos de logos
python extract_candidates.py -i archivo.pdf -o ./candidates

# Entrenar modelo personalizado
python train_model.py
```

### Limpiar carpetas
Ejecuta `limpiar_carpetas.bat` para vaciar las carpetas PDF y pdfModificado.

## ⚙️ Comandos disponibles

### Script principal (script.py)
```bash
# Procesamiento por lotes
python script.py --batch

# Procesamiento individual
python script.py --input ARCHIVO.pdf --output SALIDA.pdf

# Extraer candidatos de logos
python script.py --extract-candidates --input ARCHIVO.pdf --candidates_dir CARPETA_SALIDA
```

### Extracción de candidatos
```bash
python extract_candidates.py -i archivo.pdf -o ./candidates
```

### Entrenamiento del modelo
```bash
python train_model.py
```

## ⚙️ Parámetros de configuración

El script principal acepta varios parámetros:
- `--batch`: Procesa todos los PDFs de la carpeta PDF
- `--input`: Especifica el archivo PDF de entrada
- `--output`: Especifica el archivo PDF de salida
- `--extract-candidates`: Modo de extracción de candidatos
- `--candidates_dir`: Directorio para guardar candidatos

## 🔧 Requisitos del sistema
- **Python**: 3.8 o superior
- **RAM**: Mínimo 4GB (recomendado 8GB)
- **Espacio en disco**: Al menos 2GB libres
- **GPU**: Opcional pero recomendada para mejor rendimiento

## 📊 Flujo de trabajo
1. **Análisis**: El script analiza cada página del PDF
2. **Detección**: Identifica posibles logos usando IA
3. **Clasificación**: Determina qué elementos son realmente logos
4. **Eliminación**: Remueve los logos detectados
5. **Generación**: Crea un nuevo PDF limpio

## 🚨 Solución de problemas

### Error: "No se encontró ningún archivo PDF"
- Verifica que hay archivos PDF en la carpeta `PDF/`
- Asegúrate de que los archivos tienen extensión `.pdf`

### Error de dependencias
```bash
# Reinstalar dependencias
pip install --upgrade torch torchvision opencv-python PyMuPDF Pillow numpy
```

### Rendimiento lento
- Considera usar GPU si está disponible
- Reduce el tamaño de los PDFs antes del procesamiento

## 🤝 Contribuciones
Para mejorar la precisión del modelo, puedes:
1. Agregar más imágenes a `training_data/logos/` y `training_data/no_logos/`
2. Ejecutar `python train_model.py` para reentrenar
3. Probar con diferentes PDFs para validar mejoras

## 📄 Licencia
Este proyecto es para uso educativo y personal.