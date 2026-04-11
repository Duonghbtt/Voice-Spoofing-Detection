param(
    [string]$PythonExe = "python",
    [string]$DataRoot = "data",
    [string]$ProtocolRoot = $null,
    [string]$OutputRoot = "outputs",
    [switch]$Eval2021,
    [string]$Eval2021Labels = $null,
    [string]$Eval2021Features = $null,
    [switch]$SkipTrain,
    [switch]$SkipEval2019,
    [switch]$SkipEval2021
)

$ErrorActionPreference = "Stop"

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

function Resolve-Eval2021Labels {
    param(
        [string]$DataRoot,
        [string]$ExplicitPath = $null,
        [string]$FeatureRoot = $null
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        if (Test-Path -LiteralPath $ExplicitPath -PathType Leaf) {
            return $ExplicitPath
        }
        throw "Could not locate ASVspoof2021 labels file: $ExplicitPath"
    }

    $candidates = [System.Collections.Generic.List[string]]::new()

    if (-not [string]::IsNullOrWhiteSpace($FeatureRoot)) {
        $candidates.Add((Join-Path $FeatureRoot "labels_eval_2021.csv"))
        $featureParent = Split-Path -Parent $FeatureRoot
        if (-not [string]::IsNullOrWhiteSpace($featureParent)) {
            $candidates.Add((Join-Path $featureParent "labels_eval_2021.csv"))
        }
    }

    $roots = @($DataRoot, (Join-Path $DataRoot "raw"))
    $bundleRoots = @("features", "output_npy", "output_npy_2021")
    $featureDirs = @("mfcc", "lfcc", "spectrogram", "spec", "output_mfcc", "output_lfcc", "output_spec", "output_spectrogram")

    foreach ($root in $roots) {
        $candidates.Add((Join-Path $root "labels_eval_2021.csv"))
        foreach ($bundleRoot in $bundleRoots) {
            $bundlePath = Join-Path $root $bundleRoot
            $candidates.Add((Join-Path $bundlePath "labels_eval_2021.csv"))
            foreach ($featureDir in $featureDirs) {
                $candidates.Add((Join-Path (Join-Path $bundlePath $featureDir) "labels_eval_2021.csv"))
            }
        }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    return $null
}

function Get-CheckpointInfo {
    param(
        [string]$Profile,
        [string]$Model,
        [string]$Feature
    )

    $experimentName = "$Profile" + "_" + "$Model" + "_" + "$Feature"
    $checkpointDir = Join-Path $OutputRoot "checkpoints\$experimentName"
    [pscustomobject]@{
        Experiment = $experimentName
        Directory = $checkpointDir
        Best = Join-Path $checkpointDir "best.ckpt"
        Last = Join-Path $checkpointDir "last.ckpt"
        Legacy = Join-Path $OutputRoot "checkpoints\$experimentName.pth"
    }
}

function Test-CheckpointComplete {
    param(
        [string]$CheckpointPath
    )

    if (-not (Test-Path -LiteralPath $CheckpointPath)) {
        return $false
    }

    $pythonCode = "import sys, torch; ckpt = torch.load(sys.argv[1], map_location='cpu'); sys.stdout.write('true' if ckpt.get('training_complete', False) else 'false')"
    $output = & $PythonExe -c $pythonCode $CheckpointPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect checkpoint metadata: $CheckpointPath"
    }
    return $output.Trim() -eq "true"
}

function Resolve-EvalCheckpoint {
    param(
        [pscustomobject]$CheckpointInfo
    )

    if (Test-Path -LiteralPath $CheckpointInfo.Best) {
        return $CheckpointInfo.Best
    }
    if (Test-Path -LiteralPath $CheckpointInfo.Legacy) {
        return $CheckpointInfo.Legacy
    }

    throw "No evaluation checkpoint found for $($CheckpointInfo.Experiment). Expected $($CheckpointInfo.Best) or $($CheckpointInfo.Legacy)"
}

function Invoke-ProfileTraining {
    param(
        [string]$Title,
        [System.Collections.Generic.List[string]]$ArgumentList,
        [pscustomobject]$CheckpointInfo
    )

    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Cyan

    if (Test-Path -LiteralPath $CheckpointInfo.Legacy) {
        Write-Host "Skipping training; found legacy completed checkpoint at $($CheckpointInfo.Legacy)" -ForegroundColor Yellow
        return
    }

    if ((Test-Path -LiteralPath $CheckpointInfo.Best) -and (Test-CheckpointComplete -CheckpointPath $CheckpointInfo.Best)) {
        Write-Host "Skipping training; completed best checkpoint exists at $($CheckpointInfo.Best)" -ForegroundColor Yellow
        return
    }

    if (Test-Path -LiteralPath $CheckpointInfo.Last) {
        $ArgumentList.Add("--resume")
        Write-Host "Resuming from $($CheckpointInfo.Last)" -ForegroundColor Yellow
    } elseif (Test-Path -LiteralPath $CheckpointInfo.Best) {
        Write-Host "Found incomplete best checkpoint without last.ckpt; starting fresh training." -ForegroundColor Yellow
    }

    Write-Host "$PythonExe $($ArgumentList -join ' ')" -ForegroundColor DarkGray
    & $PythonExe @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title"
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$baselineCheckpointInfo = Get-CheckpointInfo -Profile "baseline" -Model "cnn" -Feature "mfcc"
$optimizedCheckpointInfo = Get-CheckpointInfo -Profile "optimized" -Model "lcnn" -Feature "lfcc"
$resolvedEval2021Labels = Resolve-Eval2021Labels -DataRoot $DataRoot -ExplicitPath $Eval2021Labels -FeatureRoot $Eval2021Features
$runEval2021 = -not $SkipEval2021

if ($runEval2021) {
    if ([string]::IsNullOrWhiteSpace($resolvedEval2021Labels)) {
        if ($Eval2021) {
            throw "Could not auto-detect ASVspoof2021 labels. Provide -Eval2021Labels explicitly."
        }
        Write-Host "Skipping ASVspoof2021 evaluation; could not locate labels_eval_2021.csv." -ForegroundColor Yellow
        $runEval2021 = $false
    } else {
        $runEval2021 = $true
        Write-Host "ASVspoof2021 labels: $resolvedEval2021Labels" -ForegroundColor Yellow
    }
} else {
    $runEval2021 = $false
}

if (-not $SkipTrain) {
    $baselineTrainArgs = [System.Collections.Generic.List[string]]::new()
    $baselineTrainArgs.Add("train.py")
    $baselineTrainArgs.Add("--profile")
    $baselineTrainArgs.Add("baseline")
    $baselineTrainArgs.Add("--data_root")
    $baselineTrainArgs.Add($DataRoot)
    $baselineTrainArgs.Add("--output_root")
    $baselineTrainArgs.Add($OutputRoot)
    Add-OptionalArgument -ArgumentList $baselineTrainArgs -Name "--protocol_root" -Value $ProtocolRoot
    Invoke-ProfileTraining -Title "Train baseline profile (CNN + MFCC)" -ArgumentList $baselineTrainArgs -CheckpointInfo $baselineCheckpointInfo

    $optimizedTrainArgs = [System.Collections.Generic.List[string]]::new()
    $optimizedTrainArgs.Add("train.py")
    $optimizedTrainArgs.Add("--profile")
    $optimizedTrainArgs.Add("optimized")
    $optimizedTrainArgs.Add("--data_root")
    $optimizedTrainArgs.Add($DataRoot)
    $optimizedTrainArgs.Add("--output_root")
    $optimizedTrainArgs.Add($OutputRoot)
    Add-OptionalArgument -ArgumentList $optimizedTrainArgs -Name "--protocol_root" -Value $ProtocolRoot
    Invoke-ProfileTraining -Title "Train optimized profile (LCNN + LFCC)" -ArgumentList $optimizedTrainArgs -CheckpointInfo $optimizedCheckpointInfo
}

if (-not $SkipEval2019) {
    $baselineEval2019Args = [System.Collections.Generic.List[string]]::new()
    $baselineEval2019Args.Add("evaluate.py")
    $baselineEval2019Args.Add("--checkpoint")
    $baselineEval2019Args.Add((Resolve-EvalCheckpoint -CheckpointInfo $baselineCheckpointInfo))
    $baselineEval2019Args.Add("--eval_2019")
    $baselineEval2019Args.Add("--data_root")
    $baselineEval2019Args.Add($DataRoot)
    $baselineEval2019Args.Add("--output_root")
    $baselineEval2019Args.Add($OutputRoot)
    Add-OptionalArgument -ArgumentList $baselineEval2019Args -Name "--protocol_root" -Value $ProtocolRoot
    Invoke-Step -Title "Evaluate baseline on ASVspoof2019" -Arguments $baselineEval2019Args.ToArray()

    $optimizedEval2019Args = [System.Collections.Generic.List[string]]::new()
    $optimizedEval2019Args.Add("evaluate.py")
    $optimizedEval2019Args.Add("--checkpoint")
    $optimizedEval2019Args.Add((Resolve-EvalCheckpoint -CheckpointInfo $optimizedCheckpointInfo))
    $optimizedEval2019Args.Add("--eval_2019")
    $optimizedEval2019Args.Add("--data_root")
    $optimizedEval2019Args.Add($DataRoot)
    $optimizedEval2019Args.Add("--output_root")
    $optimizedEval2019Args.Add($OutputRoot)
    Add-OptionalArgument -ArgumentList $optimizedEval2019Args -Name "--protocol_root" -Value $ProtocolRoot
    Invoke-Step -Title "Evaluate optimized on ASVspoof2019" -Arguments $optimizedEval2019Args.ToArray()
}

if ($runEval2021) {
    $baselineEval2021Args = [System.Collections.Generic.List[string]]::new()
    $baselineEval2021Args.Add("evaluate.py")
    $baselineEval2021Args.Add("--checkpoint")
    $baselineEval2021Args.Add((Resolve-EvalCheckpoint -CheckpointInfo $baselineCheckpointInfo))
    $baselineEval2021Args.Add("--eval_2021")
    $baselineEval2021Args.Add("--eval_2021_labels")
    $baselineEval2021Args.Add($resolvedEval2021Labels)
    $baselineEval2021Args.Add("--data_root")
    $baselineEval2021Args.Add($DataRoot)
    $baselineEval2021Args.Add("--output_root")
    $baselineEval2021Args.Add($OutputRoot)
    Add-OptionalArgument -ArgumentList $baselineEval2021Args -Name "--protocol_root" -Value $ProtocolRoot
    Add-OptionalArgument -ArgumentList $baselineEval2021Args -Name "--eval_2021_features" -Value $Eval2021Features
    Invoke-Step -Title "Evaluate baseline on ASVspoof2021" -Arguments $baselineEval2021Args.ToArray()

    $optimizedEval2021Args = [System.Collections.Generic.List[string]]::new()
    $optimizedEval2021Args.Add("evaluate.py")
    $optimizedEval2021Args.Add("--checkpoint")
    $optimizedEval2021Args.Add((Resolve-EvalCheckpoint -CheckpointInfo $optimizedCheckpointInfo))
    $optimizedEval2021Args.Add("--eval_2021")
    $optimizedEval2021Args.Add("--eval_2021_labels")
    $optimizedEval2021Args.Add($resolvedEval2021Labels)
    $optimizedEval2021Args.Add("--data_root")
    $optimizedEval2021Args.Add($DataRoot)
    $optimizedEval2021Args.Add("--output_root")
    $optimizedEval2021Args.Add($OutputRoot)
    Add-OptionalArgument -ArgumentList $optimizedEval2021Args -Name "--protocol_root" -Value $ProtocolRoot
    Add-OptionalArgument -ArgumentList $optimizedEval2021Args -Name "--eval_2021_features" -Value $Eval2021Features
    Invoke-Step -Title "Evaluate optimized on ASVspoof2021" -Arguments $optimizedEval2021Args.ToArray()
}

Write-Host ""
Write-Host "Done. Outputs are in $OutputRoot" -ForegroundColor Green
