@echo off
cd /d C:\Users\aryas\Desktop\Poject\reccomendation-engine
set "PYTHONIOENCODING=utf-8"
echo ==== %date% %time% >> data\raw\books\awards\award_gbooks_log.txt
.venv\Scripts\python.exe notebooks\pipeline\lookup_award_books.py --stage gbooks --max-calls 500 >> data\raw\books\awards\award_gbooks_log.txt 2>&1