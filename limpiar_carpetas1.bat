@echo off
echo ========================================
echo     LIMPIADOR DE CARPETAS SCRIPT 1
echo ========================================
echo Eliminando contenido de input1/ y outputModificado1/...
echo.

if exist "input1\*.*" (
    del /q "input1\*.*"
    echo - Carpeta input1/ limpiada exitosamente.
) else (
    echo - Carpeta input1/ ya estaba vacia.
)

if exist "outputModificado1\*.*" (
    del /q "outputModificado1\*.*"
    echo - Carpeta outputModificado1/ limpiada exitosamente.
) else (
    echo - Carpeta outputModificado1/ ya estaba vacia.
)

echo.
echo Limpieza completada para Script 1.
pause