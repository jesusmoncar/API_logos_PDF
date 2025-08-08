@echo off
echo ========================================
echo Instalador de dependencias para API_logos_PDF
echo ========================================
echo.

:: Verificar si Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo Por favor instala Python desde https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python detectado correctamente
python --version
echo.

:: Verificar si pip esta instalado
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip no esta instalado
    echo Instalando pip...
    python -m ensurepip --upgrade
)

echo pip detectado correctamente
pip --version
echo.

:: Actualizar pip a la ultima version
echo Actualizando pip...
python -m pip install --upgrade pip
echo.

:: Instalar PyTorch con soporte CUDA (opcional) o CPU
echo ========================================
echo Instalando PyTorch...
echo ========================================
echo Detectando si CUDA esta disponible...
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo Instalando PyTorch para CPU...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else (
    echo Instalando PyTorch con soporte CUDA...
    pip install torch torchvision torchaudio
)
echo.

:: Instalar el resto de dependencias desde requirements.txt
echo ========================================
echo Instalando dependencias desde requirements.txt...
echo ========================================

if exist requirements.txt (
    pip install -r requirements.txt
    echo.
    echo Dependencias instaladas desde requirements.txt
) else (
    echo requirements.txt no encontrado, instalando dependencias manualmente...
    pip install opencv-python PyMuPDF Pillow numpy scikit-learn scipy scikit-image easyocr anthropic
)

echo.
echo ========================================
echo Verificando instalacion...
echo ========================================

:: Verificar que las dependencias principales esten instaladas
python -c "
import sys
packages = ['torch', 'torchvision', 'cv2', 'fitz', 'PIL', 'numpy', 'sklearn', 'scipy', 'skimage', 'easyocr', 'anthropic']
missing = []
for package in packages:
    try:
        __import__(package)
        print(f'✓ {package} instalado correctamente')
    except ImportError:
        missing.append(package)
        print(f'✗ {package} NO instalado')

if missing:
    print(f'\nERROR: Faltan paquetes: {missing}')
    sys.exit(1)
else:
    print('\n✓ Todas las dependencias estan instaladas correctamente')
"

if errorlevel 1 (
    echo.
    echo ERROR: Algunas dependencias no se instalaron correctamente
    pause
    exit /b 1
)

echo.
echo ========================================
echo Instalacion completada exitosamente!
echo ========================================
echo Ya puedes ejecutar el programa con: python script.py
echo.
pause