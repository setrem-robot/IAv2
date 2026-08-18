# Prepara uma maquina Windows para rodar o RobotEye.
#
# Uso (no PowerShell, a partir da raiz do repositorio):
#   .\scripts\setup.ps1                      instalacao padrao (pergunta o que falta)
#   .\scripts\setup.ps1 -Voice dora          usa outra voz
#   .\scripts\setup.ps1 -Ollama 192.168.1.50 endereco da maquina com a IA
#   .\scripts\setup.ps1 -Model qwen3:8b      modelo de linguagem
#   .\scripts\setup.ps1 -NoLlm               instala sem IA (modo echo)
#   .\scripts\setup.ps1 -Yes                 nao pergunta nada (para automacao)
#   .\scripts\setup.ps1 -Dev                 instala tambem as ferramentas de teste
#
# O script e idempotente: rodar de novo apenas atualiza o que faltar.
#
# Se o Windows recusar a execucao ("scripts is disabled on this system"), rode:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

[CmdletBinding()]
param(
    [string] $Voice = "",
    [string] $Model = "",
    [string] $Ollama = "",
    [switch] $NoLlm,
    [switch] $Yes,
    [switch] $Dev
)

$ErrorActionPreference = "Stop"

$RepoDir = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RepoDir ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"
$RobotEye = Join-Path $VenvDir "Scripts\roboteye.exe"

function Step($texto) { Write-Host ""; Write-Host "==> $texto" -ForegroundColor Cyan }
function Info($texto) { Write-Host "    $texto" }
function Warn($texto) { Write-Host "    [aviso] $texto" -ForegroundColor Yellow }

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "        ROBOT EYE - SETUP" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Info "repositorio: $RepoDir"

# --- 1. ambiente virtual ----------------------------------------------------
Step "[1/3] Preparando o ambiente Python"

if (-not (Get-Command py -ErrorAction SilentlyContinue) -and
    -not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python nao encontrado. Instale o Python 3.10+ de https://python.org e rode de novo."
}

if (-not (Test-Path $Python)) {
    # `py -3` respeita o launcher do Windows, que costuma apontar para a versao
    # mais nova instalada; `python` e o caminho de quem instalou pela Store.
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv $VenvDir }
    else { python -m venv $VenvDir }
    Info "ambiente virtual criado em $VenvDir"
} else {
    Info "ambiente virtual ja existe"
}

& $Python -m pip install --quiet --upgrade pip

# `online` traz as vozes da nuvem; `tts` traz a voz local que assume quando a
# rede cai. As duas juntas fazem o robo continuar falando de qualquer jeito.
$extras = if ($Dev) { "[tts,online,dev]" } else { "[tts,online]" }
& $Python -m pip install --quiet -e "$RepoDir$extras"
Info "pacote instalado (voz local + voz na nuvem)"

# --- 2. configuracao --------------------------------------------------------
# Quem responde "onde roda a IA, qual modelo, qual voz" e o assistente do
# proprio pacote: ele testa o endereco antes de gravar e lista os modelos que a
# maquina realmente tem.
Step "[2/3] Configuracao"

$setupArgs = @()
if ($Voice)  { $setupArgs += @("--voice", $Voice) }
if ($Model)  { $setupArgs += @("--model", $Model) }
if ($Ollama) { $setupArgs += @("--ollama", $Ollama) }
if ($NoLlm)  { $setupArgs += "--no-llm" }
if ($Yes)    { $setupArgs += "--non-interactive" }

& $RobotEye setup @setupArgs

# --- 3. verificacao ---------------------------------------------------------
Step "[3/3] Verificando a instalacao"
& $RobotEye doctor
if ($LASTEXITCODE -ne 0) { Warn "ha pendencias no diagnostico acima" }

Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "    Setup concluido!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host @"

Para usar:

    .venv\Scripts\Activate.ps1
    roboteye                    # face + chat
    roboteye run --fullscreen   # tela cheia
    roboteye chat               # so o terminal

Para trocar de IA, modelo ou voz depois:

    roboteye setup              # o mesmo assistente, de novo
    roboteye models             # o que a maquina da IA tem instalado

"@
