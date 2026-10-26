@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Abrindo o PROTOTIPO do Adam em http://localhost:8604 ...
echo (o Adam real continua no persona_adam.py da pasta de cima, porta 8603)
python -m streamlit run persona_adam.py --server.port 8604
pause
