@echo off
echo ========================================
echo     LIMPIADOR DE CARPETAS SCRIPT 2
echo ========================================
echo Eliminando contenido de input2/ y outputModificado2/...
echo.

if exist "input2\*.*" (
    del /q "input2\*.*"
    echo - Carpeta input2/ limpiada exitosamente.
) else (
    echo - Carpeta input2/ ya estaba vacia.
)

if exist "outputModificado2\*.*" (
    del /q "outputModificado2\*.*"
    echo - Carpeta outputModificado2/ limpiada exitosamente.
) else (
    echo - Carpeta outputModificado2/ ya estaba vacia.
)

echo.
echo Limpieza completada para Script 2.
pause