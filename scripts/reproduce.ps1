[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [Parameter(Mandatory = $true)][string]$RunRoot,
    [Parameter(Mandatory = $true)][string]$Python311
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
function Get-Sha256 {
    param([string]$LiteralPath)
    $stream = [System.IO.File]::OpenRead($LiteralPath)
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    } finally {
        $algorithm.Dispose()
        $stream.Dispose()
    }
}
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\', '/')
$packageRoot = Join-Path $repoRoot 'package'
$runPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RunRoot)
$runPath = [System.IO.Path]::GetFullPath($runPath).TrimEnd('\', '/')
if ($runPath.Equals($repoRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $runPath.StartsWith($repoRoot + [System.IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'RunRoot must be outside the repository; choose a new directory for this run.'
}
if (Test-Path -LiteralPath $runPath) {
    throw 'RunRoot already exists; choose a new absent directory. Existing runs are never overwritten.'
}
# Refuse aliases through junctions/symlinks so the outside-repository check stays meaningful.
$ancestor = [System.IO.Directory]::GetParent($runPath)
while ($null -ne $ancestor) {
    if (Test-Path -LiteralPath $ancestor.FullName) {
        $item = Get-Item -LiteralPath $ancestor.FullName -Force
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'RunRoot ancestors must not be symbolic links or junctions.'
        }
    }
    $ancestor = $ancestor.Parent
}

$dataPath = (Resolve-Path -LiteralPath $DataRoot).ProviderPath
$inputHashes = [ordered]@{
    'tactical/BAR-ATH_tactical_atd.xml' = '327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a'
    'tactical/BAR-ATH_tactical_atd.txt' = 'f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431'
}
foreach ($relative in $inputHashes.Keys) {
    $inputPath = Join-Path $dataPath $relative
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
        throw "Tracking file missing: $relative. DataRoot must contain tactical/."
    }
    if ((Get-Sha256 -LiteralPath $inputPath) -ne $inputHashes[$relative]) {
        throw "Tracking SHA-256 mismatch: $relative. Use the exact authorized tactical files."
    }
}
$pythonPath = (Resolve-Path -LiteralPath $Python311).ProviderPath
$savedEnvironment = @{}
foreach ($name in @('PYTHONPATH', 'PYTHONHOME', 'PYTHONDONTWRITEBYTECODE', 'TEMP', 'TMP')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$startTime = [DateTime]::UtcNow
try {
    [Environment]::SetEnvironmentVariable('PYTHONPATH', $null, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONHOME', $null, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONDONTWRITEBYTECODE', '1', 'Process')
    $runtimeCode = 'import json,platform,struct,sys; r={"implementation":platform.python_implementation(),"python":platform.python_version(),"system":platform.system(),"machine":platform.machine(),"bits":struct.calcsize("P")*8}; print(json.dumps(r)); sys.exit(0 if r == {"implementation":"CPython","python":"3.11.0","system":"Windows","machine":"AMD64","bits":64} else 1)'
    # Stdin avoids Windows PowerShell 5.1 stripping quotes in native -c arguments.
    $runtimeJson = $runtimeCode | & $pythonPath -I -B -
    if ($LASTEXITCODE -ne 0) {
        throw 'Python311 must be CPython 3.11.0 for Windows AMD64 (64-bit).'
    }
    & $pythonPath -I -B (Join-Path $PSScriptRoot 'verify.py')
    if ($LASTEXITCODE -ne 0) { throw 'Frozen package verification failed before environment creation.' }
    if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'verification/reference.json') -PathType Leaf)) {
        throw 'verification/reference.json is required before reproduction.'
    }
    New-Item -ItemType Directory -Path $runPath -ErrorAction Stop | Out-Null
    $temporaryRoot = Join-Path $runPath 'temp'
    $pipCacheRoot = Join-Path $runPath 'pip-cache'
    New-Item -ItemType Directory -Path $temporaryRoot, $pipCacheRoot | Out-Null
    [Environment]::SetEnvironmentVariable('TEMP', $temporaryRoot, 'Process')
    [Environment]::SetEnvironmentVariable('TMP', $temporaryRoot, 'Process')
    $logRoot = Join-Path $runPath 'logs'
    $outputRoot = Join-Path $runPath 'output'
    New-Item -ItemType Directory -Path $logRoot, $outputRoot | Out-Null

    function Invoke-CheckedPython {
        param([string]$Executable, [string[]]$Arguments, [string]$LogName,
              [string]$StandardInput)
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            if ($PSBoundParameters.ContainsKey('StandardInput')) {
                $StandardInput | & $Executable @Arguments 2>&1 | ForEach-Object { $_.ToString() } |
                    Tee-Object -FilePath (Join-Path $logRoot $LogName) | Out-Host
            } else {
                & $Executable @Arguments 2>&1 | ForEach-Object { $_.ToString() } |
                    Tee-Object -FilePath (Join-Path $logRoot $LogName) | Out-Host
            }
            $nativeExit = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousPreference
        }
        if ($nativeExit -ne 0) { throw "Python step failed ($LogName), exit $nativeExit. See local logs." }
    }

    $environmentPath = Join-Path $runPath 'environment'
    Invoke-CheckedPython -Executable $pythonPath -Arguments @('-I', '-B', '-m', 'venv', $environmentPath) -LogName '01-venv.log'
    $venvPython = Join-Path $environmentPath 'Scripts/python.exe'
    $requirements = Join-Path $packageRoot 'repro/requirements-windows-cp311.lock'
    Invoke-CheckedPython -Executable $venvPython -Arguments @('-I', '-B', '-m', 'pip', '--isolated', '--cache-dir', $pipCacheRoot, 'install', '--require-hashes', '--only-binary=:all:', '-r', $requirements) -LogName '02-install.log'
    Invoke-CheckedPython -Executable $venvPython -Arguments @('-I', '-B', '-m', 'pip', '--isolated', 'check') -LogName '03-pip-check.log'
    Invoke-CheckedPython -Executable $venvPython -Arguments @('-I', '-B', (Join-Path $PSScriptRoot 'verify.py')) -LogName '04-package-check.log'
    $outputCsv = Join-Path $outputRoot 'ochtagon.csv'
    $model = Join-Path $packageRoot 'models/observation_gap_v1.pkl'
    Push-Location -LiteralPath (Join-Path $packageRoot 'source')
    try {
        # -I excludes CWD, so only this frozen source directory is explicitly added.
        $launch = 'import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module("solution.final_observation_ensemble",run_name="__main__")'
        Invoke-CheckedPython -Executable $venvPython -Arguments @('-I', '-B', '-', (Join-Path $packageRoot 'source'), 'infer', '--model', $model, '--expected-sha256', '76c4508ff2091e008de23ddec32ecfc86f989e172916702cadd37511dc68b672', '--data-dir', $dataPath, '--out', $outputCsv) -StandardInput $launch -LogName '05-inference.log'
    } finally {
        Pop-Location
    }
    Invoke-CheckedPython -Executable $venvPython -Arguments @('-I', '-B', (Join-Path $PSScriptRoot 'verify.py'), '--csv', $outputCsv) -LogName '06-submission-check.log'
    $versionsCode = 'import importlib.metadata,json; print(json.dumps({d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},sort_keys=True))'
    $versionsJson = $versionsCode | & $venvPython -I -B -
    if ($LASTEXITCODE -ne 0) { throw 'Could not record installed package versions.' }
    $inferenceReceipt = $outputCsv + '.manifest.json'
    if (-not (Test-Path -LiteralPath $inferenceReceipt -PathType Leaf)) {
        throw 'Frozen inference did not create its output manifest.'
    }
    $receipt = [ordered]@{
        schema = 'ochtagon-reproduction-v1'
        success = $true
        started_utc = $startTime.ToString('o')
        completed_utc = [DateTime]::UtcNow.ToString('o')
        runtime = ($runtimeJson | ConvertFrom-Json)
        installed_versions = ($versionsJson | ConvertFrom-Json)
        input_sha256 = $inputHashes
        package_manifest_sha256 = (Get-Sha256 -LiteralPath (Join-Path $packageRoot 'repro/package-manifest.json'))
        model_sha256 = (Get-Sha256 -LiteralPath $model)
        output = 'output/ochtagon.csv'
        output_sha256 = (Get-Sha256 -LiteralPath $outputCsv)
        inference_manifest_sha256 = (Get-Sha256 -LiteralPath $inferenceReceipt)
        frames = 199319
        official_format_valid = $true
        reference_hash_matched = $true
    }
    $receiptPath = Join-Path $runPath 'reproduction.json'
    $receiptJson = $receipt | ConvertTo-Json -Depth 8
    $stream = [System.IO.File]::Open($receiptPath, [System.IO.FileMode]::CreateNew)
    try {
        $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($receiptJson + [Environment]::NewLine)
        $stream.Write($bytes, 0, $bytes.Length)
    } finally { $stream.Dispose() }
    Write-Output "Reproduction verified. CSV: $outputCsv"
    Write-Output "Receipt: $receiptPath"
} finally {
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}
