param([string]$Prompt = 'Once upon a time', [int]$Length = 300)
$ErrorActionPreference = 'Stop'
$hopePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $hopePython)) { throw 'Create the .venv environment using README.md first.' }
& $hopePython (Join-Path $PSScriptRoot 'generate.py') --checkpoint (Join-Path $PSScriptRoot 'runs\hope\model.pt') --prompt $Prompt --length $Length
