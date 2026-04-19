param(
    [string]$PythonExe = "python",
    [string]$DataRoot = "data",
    [string]$FeatureRoot = $null,
    [string]$ProtocolRoot = $null,
    [string]$OutputRoot = "outputs",
    [ValidateSet("cnn", "lcnn", "resnet18")]
    [string]$Model = "cnn",
    [ValidateSet("mfcc", "lfcc", "spectrogram")]
    [string]$Feature = "mfcc",
    [switch]$AllCombinations,
    [int]$BatchSize = 0,
    [int]$NumWorkers = -1,
    [switch]$Eval2021,
    [string]$Eval2021Labels = $null,
    [switch]$SkipTrain,
    [switch]$SkipEval2019,
    [switch]$SkipEval2021
)

$ErrorActionPreference = "Stop"
$script:DetectedGpuMemoryGiB = $null
$script:DetectedGpuFreeMemoryGiB = $null
$batchSizeExplicit = $PSBoundParameters.ContainsKey("BatchSize")
$numWorkersExplicit = $PSBoundParameters.ContainsKey("NumWorkers")

if ([string]::IsNullOrWhiteSpace($env:PYTORCH_CUDA_ALLOC_CONF)) {
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan
    Write-Host "$PythonExe $($Arguments -join ' ')" -ForegroundColor DarkGray

    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title"
    }
}

function Add-OptionalArgument {
    param(
        [System.Collections.Generic.List[string]]$ArgumentList,
        [string]$Name,
        [string]$Value
    )

    if (-not [string]::IsNullOrWhiteSpace($Value)) {
        $ArgumentList.Add($Name)
        $ArgumentList.Add($Value)
    }
}

function Resolve-ValidatedDataRoot {
    param(
        [string]$DataRoot
    )

    $trimmed = $DataRoot.TrimEnd('\', '/')
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
        throw "--data_root must point to the parent data directory, not an empty path."
    }

    $segments = $trimmed -split '[\\/]'
    $leaf = $segments[-1].ToLowerInvariant()
    $tail = @($segments | Select-Object -Last ([Math]::Min(4, $segments.Count)) | ForEach-Object { $_.ToLowerInvariant() })
    $invalidNames = @("features", "output_npy_2019", "output_npy_2021", "output_spectrogram_2019", "output_spectrogram_2021")

    if ($invalidNames -contains $leaf -or (@($tail | Where-Object { $invalidNames -contains $_ }).Count -gt 0)) {
        throw "--data_root must point to the parent data directory, not '$DataRoot'."
    }

    return $trimmed
}

function Get-DefaultBatchSize {
    param(
        [string]$Model,
        [string]$Feature
    )

    $key = "$Model|$Feature"
    $baselineMapping = @{
        "cnn|mfcc" = 64
        "cnn|lfcc" = 64
        "cnn|spectrogram" = 32
        "lcnn|mfcc" = 32
        "lcnn|lfcc" = 32
        "lcnn|spectrogram" = 16
        "resnet18|mfcc" = 32
        "resnet18|lfcc" = 32
        "resnet18|spectrogram" = 8
    }
    $cuda4GiBMapping = @{
        "cnn|mfcc" = 256
        "cnn|lfcc" = 256
        "cnn|spectrogram" = 160
        "lcnn|mfcc" = 192
        "lcnn|lfcc" = 192
        "lcnn|spectrogram" = 128
        "resnet18|mfcc" = 256
        "resnet18|lfcc" = 256
        "resnet18|spectrogram" = 128
    }

    $gpuMemoryGiB = Get-GpuMemoryGiB
    $gpuFreeMemoryGiB = Get-GpuFreeMemoryGiB
    if ($gpuMemoryGiB -ge 3.5 -and ($gpuFreeMemoryGiB -eq 0 -or $gpuFreeMemoryGiB -ge 3.0)) {
        return $cuda4GiBMapping[$key]
    }
    return $baselineMapping[$key]
}

function Get-GpuMemoryGiB {
    if ($null -ne $script:DetectedGpuMemoryGiB) {
        return $script:DetectedGpuMemoryGiB
    }

    try {
        $rawOutput = nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if (-not [string]::IsNullOrWhiteSpace($rawOutput)) {
            $memoryMiB = [double]$rawOutput.Trim()
            $script:DetectedGpuMemoryGiB = $memoryMiB / 1024.0
            return $script:DetectedGpuMemoryGiB
        }
    } catch {
    }

    $script:DetectedGpuMemoryGiB = 0
    return $script:DetectedGpuMemoryGiB
}

function Get-GpuFreeMemoryGiB {
    if ($null -ne $script:DetectedGpuFreeMemoryGiB) {
        return $script:DetectedGpuFreeMemoryGiB
    }

    try {
        $rawOutput = nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if (-not [string]::IsNullOrWhiteSpace($rawOutput)) {
            $memoryMiB = [double]$rawOutput.Trim()
            $script:DetectedGpuFreeMemoryGiB = $memoryMiB / 1024.0
            return $script:DetectedGpuFreeMemoryGiB
        }
    } catch {
    }

    $script:DetectedGpuFreeMemoryGiB = 0
    return $script:DetectedGpuFreeMemoryGiB
}

function Get-EffectiveNumWorkers {
    param(
        [int]$RequestedWorkers
    )

    if ($RequestedWorkers -lt 0) {
        $cpuCount = [Environment]::ProcessorCount
        if ($cpuCount -ge 12) {
            return 4
        }
        if ($cpuCount -ge 8) {
            return 3
        }
        if ($cpuCount -ge 4) {
            return 2
        }
        return 0
    }
    return [Math]::Min($RequestedWorkers, 4)
}

function Get-ProfileAlias {
    param(
        [string]$Model,
        [string]$Feature
    )

    if ($Model -eq "cnn" -and $Feature -eq "mfcc") {
        return "baseline"
    }
    if ($Model -eq "lcnn" -and $Feature -eq "lfcc") {
        return "optimized"
    }
    return $null
}

function Get-CheckpointInfo {
    param(
        [string]$Model,
        [string]$Feature
    )

    $experimentName = "$Model" + "_" + "$Feature"
    $checkpointDir = Join-Path $OutputRoot "checkpoints\$experimentName"
    $profileAlias = Get-ProfileAlias -Model $Model -Feature $Feature
    $legacyDir = $null
    $legacyPth = $null

    if (-not [string]::IsNullOrWhiteSpace($profileAlias)) {
        $legacyName = "$profileAlias" + "_" + "$Model" + "_" + "$Feature"
        $legacyDir = Join-Path $OutputRoot "checkpoints\$legacyName"
        $legacyPth = Join-Path $OutputRoot "checkpoints\$legacyName.pth"
    }

    [pscustomobject]@{
        Experiment = $experimentName
        Model = $Model
        Feature = $Feature
        Directory = $checkpointDir
        Best = Join-Path $checkpointDir "best.ckpt"
        Last = Join-Path $checkpointDir "last.ckpt"
        LegacyDirectory = $legacyDir
        LegacyBest = $(if ($legacyDir) { Join-Path $legacyDir "best.ckpt" } else { $null })
        LegacyLast = $(if ($legacyDir) { Join-Path $legacyDir "last.ckpt" } else { $null })
        LegacyPth = $legacyPth
        Profile = $profileAlias
    }
}

function Test-CheckpointComplete {
    param(
        [string]$CheckpointPath
    )

    if ([string]::IsNullOrWhiteSpace($CheckpointPath) -or -not (Test-Path -LiteralPath $CheckpointPath)) {
        return $false
    }

    $pythonCode = "import sys, torch; ckpt = torch.load(sys.argv[1], map_location='cpu'); sys.stdout.write('true' if ckpt.get('training_complete', False) else 'false')"
    $output = & $PythonExe -c $pythonCode $CheckpointPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect checkpoint metadata: $CheckpointPath"
    }
    return $output.Trim() -eq "true"
}

function Test-CheckpointNormalizeEnabled {
    param(
        [string]$CheckpointPath
    )

    if ([string]::IsNullOrWhiteSpace($CheckpointPath) -or -not (Test-Path -LiteralPath $CheckpointPath)) {
        return $false
    }

    $pythonCode = "import sys, torch; ckpt = torch.load(sys.argv[1], map_location='cpu'); sys.stdout.write('true' if ckpt.get('normalize', False) else 'false')"
    $output = & $PythonExe -c $pythonCode $CheckpointPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect checkpoint normalization metadata: $CheckpointPath"
    }
    return $output.Trim() -eq "true"
}

function Get-CheckpointBatchSize {
    param(
        [string]$CheckpointPath
    )

    if ([string]::IsNullOrWhiteSpace($CheckpointPath) -or -not (Test-Path -LiteralPath $CheckpointPath)) {
        return $null
    }

    $pythonCode = "import sys, torch; ckpt = torch.load(sys.argv[1], map_location='cpu'); value = ckpt.get('batch_size'); sys.stdout.write('' if value is None else str(value))"
    $output = & $PythonExe -c $pythonCode $CheckpointPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect checkpoint batch size: $CheckpointPath"
    }

    $trimmed = $output.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
        return $null
    }
    return [int]$trimmed
}

function Resolve-EvalCheckpoint {
    param(
        [pscustomobject]$CheckpointInfo
    )

    foreach ($candidate in @($CheckpointInfo.Best, $CheckpointInfo.LegacyBest, $CheckpointInfo.LegacyPth)) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    throw "No evaluation checkpoint found for $($CheckpointInfo.Experiment)."
}

function Should-SkipTraining {
    param(
        [pscustomobject]$CheckpointInfo
    )

    foreach ($candidate in @($CheckpointInfo.Best, $CheckpointInfo.LegacyBest, $CheckpointInfo.LegacyPth)) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate) -and (Test-CheckpointComplete -CheckpointPath $candidate)) {
            Write-Host "Skipping training; found completed checkpoint at $candidate" -ForegroundColor Yellow
            return $true
        }
    }
    return $false
}

function Test-CompleteMetrics {
    param(
        [string]$Model,
        [string]$Feature
    )

    # Chi skip evaluate khi metrics.json co du metrics 2019 va 2021.
    $experimentName = "$Model" + "_" + "$Feature"
    $metricsPath = Join-Path $OutputRoot "results\$experimentName\metrics.json"
    if (-not (Test-Path -LiteralPath $metricsPath)) {
        return $false
    }

    try {
        $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
    } catch {
        return $false
    }

    return ($null -ne $metrics.metrics_2019) -and ($null -ne $metrics.metrics_2021)
}

function New-CombinationList {
    if ($AllCombinations) {
        return @(
            @{ Model = "cnn"; Feature = "mfcc" },
            @{ Model = "cnn"; Feature = "lfcc" },
            @{ Model = "cnn"; Feature = "spectrogram" },
            @{ Model = "lcnn"; Feature = "mfcc" },
            @{ Model = "lcnn"; Feature = "lfcc" },
            @{ Model = "lcnn"; Feature = "spectrogram" },
            @{ Model = "resnet18"; Feature = "mfcc" },
            @{ Model = "resnet18"; Feature = "lfcc" },
            @{ Model = "resnet18"; Feature = "spectrogram" }
        )
    }

    return @(@{ Model = $Model; Feature = $Feature })
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$DataRoot = Resolve-ValidatedDataRoot -DataRoot $DataRoot
$runEval2021 = $Eval2021 -and -not $SkipEval2021
$combinations = New-CombinationList

foreach ($combination in $combinations) {
    $currentModel = $combination.Model
    $currentFeature = $combination.Feature
    $checkpointInfo = Get-CheckpointInfo -Model $currentModel -Feature $currentFeature

    if (-not $SkipTrain) {
        if (-not (Should-SkipTraining -CheckpointInfo $checkpointInfo)) {
            $trainArgs = [System.Collections.Generic.List[string]]::new()
            $trainArgs.Add("train.py")
            $trainArgs.Add("--model")
            $trainArgs.Add($currentModel)
            $trainArgs.Add("--feature")
            $trainArgs.Add($currentFeature)
            $trainArgs.Add("--data_root")
            $trainArgs.Add($DataRoot)
            $trainArgs.Add("--output_root")
            $trainArgs.Add($OutputRoot)
            if ($batchSizeExplicit) {
                $trainArgs.Add("--batch_size")
                $trainArgs.Add([string]$BatchSize)
            }
            if ($numWorkersExplicit) {
                $trainArgs.Add("--num_workers")
                $trainArgs.Add([string]$NumWorkers)
            }
            Add-OptionalArgument -ArgumentList $trainArgs -Name "--protocol_root" -Value $ProtocolRoot
            Add-OptionalArgument -ArgumentList $trainArgs -Name "--feature_root" -Value $FeatureRoot

            if (-not [string]::IsNullOrWhiteSpace($checkpointInfo.Profile)) {
                $trainArgs.Add("--profile")
                $trainArgs.Add($checkpointInfo.Profile)
            }

            $resumeAllowed = $false
            if (Test-Path -LiteralPath $checkpointInfo.Last) {
                $resumeAllowed = $true
                if ($currentModel -eq "lcnn" -and $currentFeature -eq "mfcc" -and -not (Test-CheckpointNormalizeEnabled -CheckpointPath $checkpointInfo.Last)) {
                    Write-Host "Skipping resume for $($checkpointInfo.Last) because it was trained without normalize=True." -ForegroundColor Yellow
                    $resumeAllowed = $false
                }
            }

            if ($resumeAllowed) {
                $trainArgs.Add("--resume")
                Write-Host "Resuming from $($checkpointInfo.Last)" -ForegroundColor Yellow
            }

            Invoke-Step -Title "Train $currentModel + $currentFeature" -Arguments $trainArgs.ToArray()
        }
    }

    $shouldCheckEval = (-not $SkipEval2019) -or $runEval2021
    $skipEvaluate = $false
    if ($shouldCheckEval) {
        # Chi kiem tra metrics khi co it nhat mot buoc evaluate duoc goi.
        if (Test-CompleteMetrics -Model $currentModel -Feature $currentFeature) {
            Write-Host "[SKIP EVAL] $($checkpointInfo.Experiment) da co metrics day du" -ForegroundColor Yellow
            $skipEvaluate = $true
        } else {
            Write-Host "[RUN EVAL] $($checkpointInfo.Experiment) chua co metrics day du -> chay evaluate" -ForegroundColor Cyan
        }
    }

    if ((-not $SkipEval2019) -and (-not $skipEvaluate)) {
        $eval2019Args = [System.Collections.Generic.List[string]]::new()
        $eval2019Args.Add("evaluate.py")
        $eval2019Args.Add("--checkpoint")
        $eval2019Args.Add((Resolve-EvalCheckpoint -CheckpointInfo $checkpointInfo))
        $eval2019Args.Add("--eval_2019")
        $eval2019Args.Add("--data_root")
        $eval2019Args.Add($DataRoot)
        $eval2019Args.Add("--output_root")
        $eval2019Args.Add($OutputRoot)
        if ($batchSizeExplicit) {
            $eval2019Args.Add("--batch_size")
            $eval2019Args.Add([string]$BatchSize)
        }
        if ($numWorkersExplicit) {
            $eval2019Args.Add("--num_workers")
            $eval2019Args.Add([string]$NumWorkers)
        }
        Add-OptionalArgument -ArgumentList $eval2019Args -Name "--protocol_root" -Value $ProtocolRoot
        Add-OptionalArgument -ArgumentList $eval2019Args -Name "--feature_root" -Value $FeatureRoot
        Invoke-Step -Title "Evaluate $currentModel + $currentFeature on ASVspoof2019" -Arguments $eval2019Args.ToArray()
    }

    if ($runEval2021 -and -not $skipEvaluate) {
        $eval2021Args = [System.Collections.Generic.List[string]]::new()
        $eval2021Args.Add("evaluate.py")
        $eval2021Args.Add("--checkpoint")
        $eval2021Args.Add((Resolve-EvalCheckpoint -CheckpointInfo $checkpointInfo))
        $eval2021Args.Add("--eval_2021")
        $eval2021Args.Add("--data_root")
        $eval2021Args.Add($DataRoot)
        $eval2021Args.Add("--output_root")
        $eval2021Args.Add($OutputRoot)
        if ($batchSizeExplicit) {
            $eval2021Args.Add("--batch_size")
            $eval2021Args.Add([string]$BatchSize)
        }
        if ($numWorkersExplicit) {
            $eval2021Args.Add("--num_workers")
            $eval2021Args.Add([string]$NumWorkers)
        }
        Add-OptionalArgument -ArgumentList $eval2021Args -Name "--protocol_root" -Value $ProtocolRoot
        Add-OptionalArgument -ArgumentList $eval2021Args -Name "--feature_root" -Value $FeatureRoot
        Add-OptionalArgument -ArgumentList $eval2021Args -Name "--eval_2021_labels" -Value $Eval2021Labels
        Invoke-Step -Title "Evaluate $currentModel + $currentFeature on ASVspoof2021" -Arguments $eval2021Args.ToArray()
    }
}

Write-Host ""
Write-Host "Done. Outputs are in $OutputRoot" -ForegroundColor Green
