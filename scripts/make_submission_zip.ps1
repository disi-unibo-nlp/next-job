param(
  [string]$Output = "next_occupation_prediction_project_submission.zip"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stageParent = Join-Path $env:TEMP ("next_job_submission_" + [guid]::NewGuid().ToString("N"))
$stage = Join-Path $stageParent "next_occupation_prediction_project"

$excludeDirs = @(
  ".git",
  "__pycache__",
  ".pytest_cache",
  ".mypy_cache",
  ".ruff_cache",
  ".venv",
  "venv"
)

$excludeRelativeFiles = @(
  "results\paper_checks\lateon_only_datafull_full_test\lateon_only_datafull_per_query.jsonl",
  "results\paper_checks\bge_m3_original_formal_per_query_full_test\bge_m3_formal_per_query_ranks.jsonl",
  "results\paper_checks\trained_st_formal_per_query_full_test\trained_st_formal_per_query_ranks.jsonl",
  "results\paper_checks\error_analysis_plus_bge\error_analysis_plus_bge_per_case_long.csv"
)

function Get-RelativePathCompat {
  param(
    [string]$BasePath,
    [string]$TargetPath
  )

  $baseFull = [System.IO.Path]::GetFullPath($BasePath)
  if (-not $baseFull.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
    $baseFull += [System.IO.Path]::DirectorySeparatorChar
  }

  $targetFull = [System.IO.Path]::GetFullPath($TargetPath)
  $baseUri = New-Object System.Uri($baseFull)
  $targetUri = New-Object System.Uri($targetFull)
  $relativeUri = $baseUri.MakeRelativeUri($targetUri).ToString()
  return [System.Uri]::UnescapeDataString($relativeUri).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

function Test-ExcludedPath {
  param([string]$Path)

  $relative = Get-RelativePathCompat -BasePath $root -TargetPath $Path
  $parts = $relative -split '[\\/]+'
  $extension = [System.IO.Path]::GetExtension($Path)

  foreach ($dir in $excludeDirs) {
    if ($parts -contains $dir) {
      return $true
    }
  }

  if ($relative -ieq $Output) {
    return $true
  }

  if ($extension -in @(".zip", ".tar", ".gz", ".7z")) {
    return $true
  }

  foreach ($file in $excludeRelativeFiles) {
    if ($relative -ieq $file) {
      return $true
    }
  }

  return $false
}

New-Item -ItemType Directory -Force -Path $stage | Out-Null

# Copy file-by-file after creating any required parent directories.
Get-ChildItem -LiteralPath $root -Recurse -File -Force | ForEach-Object {
  if (Test-ExcludedPath $_.FullName) {
    return
  }
  $relative = Get-RelativePathCompat -BasePath $root -TargetPath $_.FullName
  $destination = Join-Path $stage $relative
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
  Copy-Item -LiteralPath $_.FullName -Destination $destination -Force
}

$outputPath = Join-Path $root $Output
if (Test-Path $outputPath) {
  Remove-Item -LiteralPath $outputPath -Force
}

Compress-Archive -Path $stage -DestinationPath $outputPath -Force

$sizeMb = [math]::Round((Get-Item $outputPath).Length / 1MB, 2)
Write-Host "WROTE: $outputPath"
Write-Host "SIZE_MB: $sizeMb"

Remove-Item -LiteralPath $stageParent -Recurse -Force
