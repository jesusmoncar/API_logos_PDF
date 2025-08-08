@echo off
echo ========================================
echo     LIMPIADOR DE CARPETAS SCRIPT 3
echo ========================================
echo Eliminando contenido de input3/ y outputModificado3/...
echo.

if exist "input3\*.*" (
    del /q "input3\*.*"
    echo - Carpeta input3/ limpiada exitosamente.
) else (
    echo - Carpeta input3/ ya estaba vacia.
)

if exist "outputModificado3\*.*" (
    del /q "outputModificado3\*.*"
    echo - Carpeta outputModificado3/ limpiada exitosamente.
) else (
    echo - Carpeta outputModificado3/ ya estaba vacia.
)

echo.
echo Limpieza completada para Script 3.
pause