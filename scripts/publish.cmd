@echo off
REM Un post su Instagram, a giorni alterni alle 18:30.
REM
REM Tre passaggi, in quest'ordine, e ognuno con il suo limite:
REM
REM   approve --auto 1   sceglie UNO shot con le regole di diversita'
REM   render --limit 1   renderizza solo quello, profilo social
REM   publish --limit 1  ne pubblica uno solo
REM
REM Il limite conta in tutti e tre. Senza, `render` renderizzerebbe ogni shot
REM approvato e `publish` manderebbe in coda tutto il pendente in una volta,
REM svuotando settimane di feed in un colpo.
REM
REM Se un passaggio fallisce i successivi non partono: meglio saltare un post
REM che pubblicarne uno sbagliato.

cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
if not exist ".venv\Scripts\python.exe" (
  echo [%date% %time%] .venv non trovato>>"logs\publish.log"
  exit /b 1
)

set PY=.venv\Scripts\python.exe

echo.>>"logs\publish.log"
echo ===== %date% %time% =====>>"logs\publish.log"

%PY% -m pipeline.cli -v approve --auto 1 >>"logs\publish.log" 2>&1
if errorlevel 1 goto :failed

%PY% -m pipeline.cli -v render --limit 1 --profiles social >>"logs\publish.log" 2>&1
if errorlevel 1 goto :failed

%PY% -m pipeline.cli -v publish --limit 1 >>"logs\publish.log" 2>&1
if errorlevel 1 goto :failed

echo [ok]>>"logs\publish.log"
exit /b 0

:failed
echo [FALLITO exit=%errorlevel%] nessun post inviato>>"logs\publish.log"
exit /b %errorlevel%
