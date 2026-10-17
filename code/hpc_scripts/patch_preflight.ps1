$ErrorActionPreference = 'Stop'
$P  = 'D:\CodexData\tmp\privtrim-review-20261015\patchpre'
$src= 'D:\CodexData\tmp\privtrim-review-20261015\remote_probe\code'
$py = 'D:\Shijie Zeng\CodeX\cybersec_IoT\.venv-compression-batch1\Scripts\python.exe'

New-Item -ItemType Directory -Force $P | Out-Null
Copy-Item "$src\privcheck.py" $P -Force
Copy-Item "$src\v4_router.py" $P -Force
Copy-Item 'D:\CodexData\tmp\privtrim-review-20261015\remote_probe\data\longctx_cases.jsonl' $P -Force
Copy-Item "$src\longctx\e2e_longctx_r4.py" "$P\e2e_longctx_r4.orig.py" -Force
Copy-Item "$P\e2e_longctx_r4.orig.py" "$P\e2e_longctx_r4.py" -Force
Write-Output "=== staged ==="
Get-ChildItem $P | Select-Object Name, Length | Format-Table -AutoSize

$common = @('--cases', "$P\longctx_cases.jsonl", '--arms', 'all',
            '--budgets', '0.25,0.5,0.75', '--sel', 'lex', '--dry-run')

Write-Output "=== [1/3] BASELINE dry-run (pre-patch) ==="
& $py -u "$P\e2e_longctx_r4.py" @common --out "$P\pre.txt" --json "$P\pre.json" --percase "$P\pre.jsonl" 2>&1 |
  Where-Object { $_ -notmatch 'UserWarning|warnings.warn|FutureWarning' } | Select-Object -Last 2
Write-Output "pre.txt bytes: $((Get-Item "$P\pre.txt").Length)"

Write-Output "=== [2/3] apply patch ==="
& $py -u "$src\longctx\patch_ext_arms.py" "$P\e2e_longctx_r4.py" 2>&1 | Select-Object -Last 3
Write-Output "--- py_compile ---"
& $py -m py_compile "$P\e2e_longctx_r4.py" 2>&1 | Select-Object -Last 5
Write-Output "compile exit: $LASTEXITCODE"

Write-Output "=== [3/3] PATCHED dry-run without --ext-arms (must match baseline) ==="
& $py -u "$P\e2e_longctx_r4.py" @common --out "$P\post.txt" --json "$P\post.json" --percase "$P\post.jsonl" 2>&1 |
  Where-Object { $_ -notmatch 'UserWarning|warnings.warn|FutureWarning' } | Select-Object -Last 2
Write-Output "post.txt bytes: $((Get-Item "$P\post.txt").Length)"

Write-Output "=== REGRESSION DIFF (report / json / percase) ==="
foreach($pair in @(@('pre.txt','post.txt'), @('pre.json','post.json'), @('pre.jsonl','post.jsonl'))){
  $a = Get-FileHash "$P\$($pair[0])" -Algorithm SHA256
  $b = Get-FileHash "$P\$($pair[1])" -Algorithm SHA256
  if($a.Hash -eq $b.Hash){ Write-Output "IDENTICAL  $($pair[0]) == $($pair[1])" }
  else {
    Write-Output "DIFFERS    $($pair[0]) vs $($pair[1])"
    $da = Get-Content "$P\$($pair[0])"; $db = Get-Content "$P\$($pair[1])"
    Compare-Object $da $db | Select-Object -First 12 | Format-Table -AutoSize | Out-String -Width 200 | Write-Output
  }
}
