@echo off
cd /d C:\Users\aryas\Desktop\Poject\reccomendation-engine
set "PYTHONIOENCODING=utf-8"
echo ==== %date% %time% >> data\raw\books\hardcover_v2\gbooks_log.txt
.venv\Scripts\python.exe -u notebooks\pipeline\lookup_award_books.py --stage gbooks --max-calls 950 >> data\raw\books\hardcover_v2\gbooks_log.txt 2>&1
