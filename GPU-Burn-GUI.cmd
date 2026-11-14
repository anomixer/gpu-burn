<# : batch
@echo off
REM ---------------------------------------------------------------------------
REM  GPU Burn GUI v1.0 - batch/PowerShell hybrid launcher.
REM  cmd.exe runs only these few lines; to Windows PowerShell, this whole
REM  header is a block comment. The batch part passes this file's own path to
REM  Windows PowerShell, which reads the rest of the file and opens the window.
REM  The back-end is the original gpu_burn.exe; this file does not modify it.
REM ---------------------------------------------------------------------------
setlocal
set "GPU_BURN_SCRIPT=%~f0"
set "GPU_BURN_PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "GPU_BURN_PS=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%GPU_BURN_PS%" set "GPU_BURN_PS=powershell"
"%GPU_BURN_PS%" -NoProfile -ExecutionPolicy Bypass -STA -Command "iex ([IO.File]::ReadAllText($env:GPU_BURN_SCRIPT, [Text.Encoding]::UTF8))"
if errorlevel 1 pause
exit /b
#>

# =============================================================================
#  GPU Burn GUI v1.0
#
#  A WinForms front-end for the unchanged Windows gpu_burn.exe.
#  Options exposed by gpu_burn.exe:
#    -m X     use X MB of memory            (Memory mode: Fixed MB)
#    -m N%    use N% of available VRAM      (Memory mode: Percent)
#    -d       use doubles                   (checkbox)
#    -tc      try Tensor cores              (checkbox)
#    -l       list all GPUs in the system   (Refresh GPUs button / GPU list tab)
#    -i N     execute only on GPU N         (GPU selection)
#    -c FILE  use FILE as compare kernel   (Advanced tab)
#    -stts T  SIGTERM timeout seconds       (Advanced tab)
#    -h       show gpu_burn.exe help         (Show help button)
#    TIME     burn duration in seconds       (Burn duration)
#
#  The window streams stdout/stderr from gpu_burn.exe, turns its CR redraws
#  into a progress bar, and can interrupt the run with taskkill /T /F.
#  Nothing is burned until you click Run burn and confirm.
#  Logs: %LOCALAPPDATA%\GPU-Burn-GUI
#
#  Test-only environment variables (used by this repository's smoke test):
#    GPU_BURN_GUI_TEST=1        auto-refresh GPUs and auto-start a burn
#    GPU_BURN_TEST_SECONDS=N    duration for that burn (default 2)
#    GPU_BURN_GUI_TEST_KEEP=1   keep the window open after the burn
# =============================================================================

$ErrorActionPreference = 'Stop'
$script:AppVersion = '1.0'
$script:AutoTest = ($env:GPU_BURN_GUI_TEST -eq '1')
$script:AutoTestKeep = ($env:GPU_BURN_GUI_TEST_KEEP -eq '1')
$script:TestSeconds = 2
if ($env:GPU_BURN_TEST_SECONDS -match '^\d+$') { $script:TestSeconds = [int]$env:GPU_BURN_TEST_SECONDS }
if ($script:TestSeconds -lt 1) { $script:TestSeconds = 1 }

$script:GuiDir = $PSScriptRoot
if ($env:GPU_BURN_SCRIPT) { $script:GuiDir = Split-Path -Parent $env:GPU_BURN_SCRIPT }
if (-not $script:GuiDir) { $script:GuiDir = (Get-Location).Path }

# ----- Backend discovery ---------------------------------------------------
# The GUI never replaces gpu_burn.exe. It looks for the original executable
# next to itself (package layout) or in the usual source/build folders.

function Resolve-BurnBackend {
    $dirs = @(
        $script:GuiDir
        Join-Path $script:GuiDir 'win\build\Release'
        Join-Path $script:GuiDir 'win\build\Debug'
        Join-Path $script:GuiDir 'build\Release'
        Join-Path $script:GuiDir 'build\Debug'
        Join-Path $script:GuiDir 'package\gpu-burn-win-v0.3'
        Join-Path $script:GuiDir 'win\package\gpu-burn-win-v0.3'
    )
    foreach ($dir in $dirs) {
        if (-not (Test-Path -LiteralPath $dir -PathType Container)) { continue }
        $exe = Join-Path $dir 'gpu_burn.exe'
        if (Test-Path -LiteralPath $exe -PathType Leaf) {
            $kernel = Join-Path $dir 'compare.fatbin'
            if (-not (Test-Path -LiteralPath $kernel -PathType Leaf)) {
                throw ('gpu_burn.exe found in {0}, but compare.fatbin is missing there. Put both files in the same folder.' -f $dir)
            }
            return [pscustomobject]@{
                Exe    = (Resolve-Path -LiteralPath $exe).Path
                Dir    = (Resolve-Path -LiteralPath $dir).Path
                Kernel = (Resolve-Path -LiteralPath $kernel).Path
            }
        }
    }
    # Last resort: search below the repo root for a built executable.
    $hit = Get-ChildItem -LiteralPath $script:GuiDir -Filter 'gpu_burn.exe' -Recurse -File -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($hit) {
        $dir = Split-Path -Parent $hit.FullName
        $kernel = Join-Path $dir 'compare.fatbin'
        if (-not (Test-Path -LiteralPath $kernel -PathType Leaf)) {
            throw ('gpu_burn.exe found at {0}, but compare.fatbin is missing there. Put both files in the same folder.' -f $hit.FullName)
        }
        return [pscustomobject]@{
            Exe    = (Resolve-Path -LiteralPath $hit.FullName).Path
            Dir    = (Resolve-Path -LiteralPath $dir).Path
            Kernel = (Resolve-Path -LiteralPath $kernel).Path
        }
    }
    throw ('gpu_burn.exe was not found. Put GPU-Burn-GUI.cmd next to gpu_burn.exe and compare.fatbin, or run it from the gpu-burn repository root (it also looks in win\build\Release).')
}

# ----- Log/temp folders ----------------------------------------------------

$script:LogDir = Join-Path $env:LOCALAPPDATA 'GPU-Burn-GUI'
$script:TempDir = Join-Path ([IO.Path]::GetTempPath()) 'GPU-Burn-GUI'
$script:LogFile = $null
try {
    if (-not (Test-Path -LiteralPath $script:LogDir)) { [void](New-Item -ItemType Directory -Path $script:LogDir -Force) }
    if (-not (Test-Path -LiteralPath $script:TempDir)) { [void](New-Item -ItemType Directory -Path $script:TempDir -Force) }
} catch { }
try { $script:LogFile = Join-Path $script:LogDir ('GUI_{0:yyyyMMdd_HHmmss}.log' -f (Get-Date)) } catch { $script:LogFile = $null }

function Write-LogFile([string]$text) {
    if ($script:LogFile) {
        try { [IO.File]::AppendAllText($script:LogFile, $text + "`r`n", [Text.Encoding]::UTF8) } catch { }
    }
}

# ----- Small process helpers ----------------------------------------------

function Quote-WinArg([string]$s) {
    if ($s -match '[\s"]') { return ('"{0}"' -f ($s -replace '"', '\"')) }
    return $s
}

function Get-BurnArgumentsLine([string[]]$argList) {
    ($argList | ForEach-Object { Quote-WinArg ([string]$_) }) -join ' '
}

# Runs gpu_burn.exe with redirected stdout/stderr and returns its text.
# Used for -l (GPU list) and -h (backend help), not for the burn itself.
function Invoke-BurnCapture([string[]]$argList) {
    $tag = [guid]::NewGuid().ToString('N')
    $outFile = Join-Path $script:TempDir ('capture-' + $tag + '.out')
    $errFile = Join-Path $script:TempDir ('capture-' + $tag + '.err')
    $argLine = Get-BurnArgumentsLine $argList
    $result = [pscustomobject]@{ Stdout = ''; Stderr = ''; ExitCode = $null; TimedOut = $false }
    try {
        $p = Start-Process -FilePath $script:Burn.Exe -WorkingDirectory $script:Burn.Dir -ArgumentList $argLine `
            -RedirectStandardOutput $outFile -RedirectStandardError $errFile `
            -WindowStyle Hidden -PassThru -ErrorAction Stop
        if (-not $p.WaitForExit(12000)) {
            $result.TimedOut = $true
            try { & "$env:SystemRoot\System32\taskkill.exe" /T /F /PID $p.Id 2>$null | Out-Null } catch { }
        } else {
            $result.ExitCode = [int]$p.ExitCode
        }
        if (Test-Path -LiteralPath $outFile) { $result.Stdout = ([IO.File]::ReadAllText($outFile, [Text.Encoding]::UTF8)).Trim() }
        if (Test-Path -LiteralPath $errFile) { $result.Stderr = ([IO.File]::ReadAllText($errFile, [Text.Encoding]::UTF8)).Trim() }
        try { $p.Dispose() } catch { }
    } catch {
        $result.Stderr = $_.Exception.Message
    } finally {
        try { Remove-Item -LiteralPath $outFile -Force -ErrorAction SilentlyContinue } catch { }
        try { Remove-Item -LiteralPath $errFile -Force -ErrorAction SilentlyContinue } catch { }
    }
    return $result
}

# ----- GPU discovery -------------------------------------------------------

function Get-GpuList {
    $capture = Invoke-BurnCapture @('-l')
    $gpus = @()
    foreach ($line in @($capture.Stdout -split '\r?\n')) {
        if ($line -match '^ID\s+(?<id>\d+):\s+(?<name>.+?),\s*(?<mb>\d+)MB\s*$') {
            $gpus += [pscustomobject]@{
                Id     = [int]$Matches.id
                Name   = [string]$Matches.name
                Memory = [int]$Matches.mb
            }
        }
    }
    return [pscustomobject]@{ Gpus = @($gpus); Capture = $capture }
}

# ----- Traditional Chinese (zh-TW) ---------------------------------------
# Everything is English unless the Windows UI culture is zh-TW/HK/MO or
# GPU_BURN_GUI_LANG is set. The top-right button switches live.

$script:UiLang = 'en'
try {
    $cc = [System.Globalization.CultureInfo]::CurrentCulture
    $uc = [System.Globalization.CultureInfo]::CurrentUICulture
    if (($cc.Name -match '^zh-(TW|HK|MO)$') -or ($uc.Name -match '^zh-(TW|HK|MO)$')) { $script:UiLang = 'zh-TW' }
} catch { }
if ($env:GPU_BURN_GUI_LANG -in 'zh-TW', 'zh', 'zh-HK', 'zh-MO') { $script:UiLang = 'zh-TW' }
elseif ($env:GPU_BURN_GUI_LANG -eq 'en') { $script:UiLang = 'en' }

$script:Zh = @{
    'GPU Burn GUI'                                      = 'GPU Burn 圖形介面'
    'Ready. Set options and click Run burn.'             = '就緒：設定選項後按下「開始燒機」。'
    'Burning GPU stress test...'                         = 'GPU 燒機測試中…'
    'Finished: GPU test OK.'                            = '完成：GPU 測試結果良好。'
    'Finished: FAULTY GPU detected.'                    = '完成：偵測到 GPU 錯誤。'
    'Finished with errors.'                             = '結束，但有錯誤。'
    'Stopped by user.'                                  = '使用者已中止。'
    'Running...'                                        = '執行中…'
    'All GPUs (backend default)'                        = '所有 GPU（後端預設）'
    'Backend default (90% available VRAM)'              = '後端預設（可用 VRAM 的 90%）'
    'Percent of available VRAM'                         = '可用 VRAM 百分比'
    'Fixed amount (MB)'                                 = '固定容量（MB）'
    'MB'                                                = 'MB'
    '%'                                                 = '%'
    'seconds'                                           = '秒'
    'Burn options'                                      = '燒機選項'
    'Advanced'                                          = '進階選項'
    'GPU list'                                         = 'GPU 清單'
    'Memory to use'                                     = '記憶體用量'
    'Burn duration (seconds)'                           = '燒機時間（秒）'
    'Use doubles (-d)'                                  = '使用雙精確度（-d）'
    'Try Tensor cores (-tc)'                            = '嘗試使用 Tensor Cores（-tc）'
    'GPU selection'                                     = 'GPU 選擇'
    'Compare kernel (-c)'                              = '比較核心檔案（-c）'
    'Browse...'                                        = '瀏覽…'
    'SIGTERM timeout (-stts)'                          = 'SIGTERM 逾時（-stts）'
    'Override default (30 seconds)'                     = '覆寫預設值（30 秒）'
    'Empty compare path means the default compare.fatbin next to gpu_burn.exe.' = '比較檔案留白時，會使用 gpu_burn.exe 旁的預設 compare.fatbin。'
    'Refresh GPUs (-l)'                                 = '重新整理 GPU（-l）'
    'Run burn'                                          = '開始燒機'
    'Stop burn'                                        = '中止燒機'
    'Open log folder'                                   = '開啟記錄資料夾'
    'Show help (-h)'                                    = '顯示說明（-h）'
    'GPU adapter'                                       = 'GPU 型號'
    'Selected GPU'                                      = '選擇的 GPU'
    'Backend executable'                                = '後端程式'
    'Compare kernel'                                    = '比較核心'
    'Last burn'                                        = '最近燒機'
    'GPUs found'                                        = '找到的 GPU'
    'Working directory'                                 = '工作目錄'
    'Memory mode'                                       = '記憶體模式'
    'Burn duration'                                     = '燒機時間'
    'ID'                                               = 'ID'
    'Adapter'                                          = '顯示卡'
    'Memory (MB)'                                      = '記憶體（MB）'
    'No GPU rows were returned by gpu_burn.exe -l.'      = 'gpu_burn.exe -l 沒有回傳 GPU 資料列。'
    'GPU list refreshed.'                               = 'GPU 清單已更新。'
    'Refreshing GPU list...'                            = '正在更新 GPU 清單…'
    'Backend help'                                      = '後端說明'
    'This will run gpu_burn.exe with:'                  = '將以下列選項執行 gpu_burn.exe：'
    'Continue?'                                        = '要繼續嗎？'
    'Cancelled. Nothing was burned.'                   = '已取消，沒有進行燒機。'
    'A burn is still running. Stop it first, then close the window.' = '燒機仍在執行，請先中止再關閉視窗。'
    'Stop requested. Waiting for gpu_burn.exe to exit...' = '已送出中止要求，正在等待 gpu_burn.exe 結束…'
    'No GPU found. Check nvidia-smi and the NVIDIA driver.' = '找不到 GPU，請檢查 nvidia-smi 與 NVIDIA 驅動程式。'
    'Burning finished; see the log below.'              = '燒機已結束，請看下方記錄。'
    'GPU-Burn-GUI.cmd'                                 = 'GPU-Burn-GUI.cmd'
    'The back-end is the original gpu_burn.exe. Stop burn interrupts with taskkill /T /F. Heat the card carefully.' = '後端仍是原版 gpu_burn.exe；中止會用 taskkill /T /F 中斷。請注意顯示卡散熱。'
    'Unexpected error: '                               = '未預期的錯誤：'
    'Line '                                             = '行 '
    'Nothing is burned until you click Run burn and confirm.' = '在按下「開始燒機」並確認前，不會進行任何燒機。'
    'Log file: '                                        = '記錄檔：'
    'Backend: '                                         = '後端：'
    'Command: '                                         = '命令：'
    'Exit code: '                                       = '結束代碼：'
    'Finished.'                                         = '結束。'
    'Last burn: '                                       = '最近燒機：'
    'Working directory: '                               = '工作目錄：'
    'Compare kernel: '                                  = '比較核心：'
    'Do not turn the PC off while it runs. Heat the card carefully.' = '執行期間請勿關機；請注意顯示卡散熱。'
    'Auto-test: starting gpu_burn.exe smoke burn.'       = '自動測試：開始執行 gpu_burn.exe 快速燒機。'
    'The compare kernel file was not found. Fix the path or leave it empty for the default compare.fatbin.' = '找不到比較核心檔案；請修正路徑，或留白使用預設 compare.fatbin。'
    'The packaged gpu_burn.exe may hang after a -d run; use Stop burn if needed.' = '此版 gpu_burn.exe 使用 -d 可能卡住；必要時請按「中止燒機」。'
    'gpu_burn.exe -i fails in this build; GPU selection uses CUDA_VISIBLE_DEVICES instead.' = '此版 gpu_burn.exe 的 -i 目前失效；這裡改用 CUDA_VISIBLE_DEVICES 選 GPU。'
    'Live monitor' = 'nvidia-smi 即時監控'
    'nvidia-smi live monitor' = 'nvidia-smi 即時監控'
    'nvidia-smi query: ' = 'nvidia-smi 查詢：'
    'Waiting for a burn.' = '等待燒機。'
    'nvidia-smi was not found on PATH.' = 'PATH 中找不到 nvidia-smi。'
    'Could not read nvidia-smi output.' = '無法讀取 nvidia-smi 輸出。'
    'Time' = '時間'
    'Temp (C)' = '溫度（C）'
    'Utilization (%)' = '使用率（%）'
    'VRAM (used / total MB)' = 'VRAM（已用 / 總量 MB）'
    'Power (W) / limit' = '功耗（W）／上限'
    'Fan (%)' = '風扇（%）'
    'The monitor refreshes while a burn runs; values come from nvidia-smi --query-gpu.' = '燒機期間會持續更新；數值來自 nvidia-smi --query-gpu。'
}

function T([string]$en) {
    if ($script:UiLang -eq 'zh-TW' -and $script:Zh.ContainsKey($en)) { return $script:Zh[$en] }
    return $en
}

# ----- GUI -----------------------------------------------------------------

try {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [System.Windows.Forms.Application]::EnableVisualStyles()

    # Hide the console that the batch launcher leaves behind.
    try {
        Add-Type -Namespace GpuBurnGui -Name Con -MemberDefinition '[DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow(); [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);'
        [void][GpuBurnGui.Con]::ShowWindow([GpuBurnGui.Con]::GetConsoleWindow(), 0)
    } catch { }

    $script:Burn = Resolve-BurnBackend

    # --- Look: same dark palette and Segoe UI / Consolas fonts as the
    #     Update-SecureBootCert GUI.
    function Rgb($a) { [System.Drawing.Color]::FromArgb($a[0], $a[1], $a[2]) }
    $cBg     = Rgb @(28, 30, 36);  $cBox = Rgb @(20, 22, 27)
    $cText   = Rgb @(235, 236, 240); $cMuted = Rgb @(140, 146, 160)
    $cGreen  = Rgb @(72, 199, 116); $cRed = Rgb @(239, 83, 80)
    $cYellow = Rgb @(250, 204, 21); $cOrange = Rgb @(249, 115, 22); $cBlue = Rgb @(59, 130, 246); $cGrey = Rgb @(75, 80, 94)
    $fText  = New-Object System.Drawing.Font('Segoe UI', 9)
    $fBold  = New-Object System.Drawing.Font('Segoe UI', 10, [System.Drawing.FontStyle]::Bold)
    $fTitle = New-Object System.Drawing.Font('Segoe UI', 13, [System.Drawing.FontStyle]::Bold)
    $fMono  = New-Object System.Drawing.Font('Consolas', 9)

    $AnchorTL  = [System.Windows.Forms.AnchorStyles]'Top, Left'
    $AnchorTLR = [System.Windows.Forms.AnchorStyles]'Top, Left, Right'
    $AnchorTR  = [System.Windows.Forms.AnchorStyles]'Top, Right'
    $AnchorAll = [System.Windows.Forms.AnchorStyles]'Top, Bottom, Left, Right'
    $AnchorBLR = [System.Windows.Forms.AnchorStyles]'Bottom, Left, Right'

    function New-Ctl($type, $x, $y, $w, $h) {
        $c = New-Object ("System.Windows.Forms.$type")
        $c.Location = New-Object System.Drawing.Point($x, $y)
        $c.Size = New-Object System.Drawing.Size($w, $h)
        $c.Anchor = $AnchorTL
        $c
    }

    function New-Btn($text, $x, $y, $w, $h, $color) {
        $b = New-Object System.Windows.Forms.Button
        $b.Location = New-Object System.Drawing.Point($x, $y)
        $b.Size = New-Object System.Drawing.Size($w, $h)
        $b.Anchor = $AnchorTL
        $b.Tag = $text; $b.Text = T $text
        $b.FlatStyle = 'Flat'; $b.BackColor = $color; $b.ForeColor = [System.Drawing.Color]::White
        $b.Font = $fBold; $b.FlatAppearance.BorderSize = 0; $b.Cursor = [System.Windows.Forms.Cursors]::Hand
        $b
    }

    function New-Label($text, $x, $y, $w, $h) {
        $l = New-Ctl 'Label' $x $y $w $h
        $l.Tag = $text; $l.Text = T $text; $l.ForeColor = $cText; $l.AutoEllipsis = $true
        $l
    }

    function New-Check($text, $x, $y, $w, $checked) {
        $c = New-Ctl 'CheckBox' $x $y $w 22
        $c.Tag = $text; $c.Text = T $text; $c.Checked = $checked; $c.ForeColor = $cText
        $c.BackColor = $cBox; $c.Font = $fText
        $c
    }

    function New-Combo($x, $y, $w, $h, $items) {
        $c = New-Object System.Windows.Forms.ComboBox
        $c.Location = New-Object System.Drawing.Point($x, $y)
        $c.Size = New-Object System.Drawing.Size($w, $h)
        $c.Anchor = $AnchorTL
        $c.DropDownStyle = 'DropDownList'; $c.BackColor = $cBox; $c.ForeColor = $cText; $c.Font = $fText
        foreach ($i in $items) { [void]$c.Items.Add($i) }
        $c
    }

    function New-Numeric($x, $y, $w, $h, $min, $max, $value) {
        $n = New-Ctl 'NumericUpDown' $x $y $w $h
        $n.Minimum = $min; $n.Maximum = $max; $n.Value = $value
        $n.DecimalPlaces = 0; $n.BackColor = $cBox; $n.ForeColor = $cText; $n.Font = $fText
        $n.BorderStyle = 'FixedSingle'
        $n
    }

    function New-Text($x, $y, $w, $h, $text) {
        $t = New-Ctl 'TextBox' $x $y $w $h
        $t.Text = $text; $t.BackColor = $cBox; $t.ForeColor = $cText; $t.Font = $fText
        $t.BorderStyle = 'FixedSingle'
        $t
    }

    function New-List($columns) {
        $lv = New-Object System.Windows.Forms.ListView
        $lv.Dock = 'Fill'; $lv.View = 'Details'; $lv.FullRowSelect = $true; $lv.HideSelection = $true
        $lv.BackColor = $cBox; $lv.ForeColor = $cText; $lv.Font = $fText; $lv.HeaderStyle = 'Nonclickable'
        $lv.Tag = $columns
        foreach ($col in $columns) { [void]$lv.Columns.Add($col[0], $col[1]) }
        $lv
    }

    # --- Window ---
    $form = New-Object System.Windows.Forms.Form
    $form.AutoScaleDimensions = New-Object System.Drawing.SizeF(96, 96)
    $form.AutoScaleMode = [System.Windows.Forms.AutoScaleMode]::Dpi
    $form.Text = 'GPU Burn GUI v' + $script:AppVersion
    $screenH = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea.Height
    $form.ClientSize = New-Object System.Drawing.Size(900, ([Math]::Max(620, [Math]::Min(740, $screenH - 60))))
    $form.MinimumSize = New-Object System.Drawing.Size(916, 670)
    $form.StartPosition = 'CenterScreen'; $form.FormBorderStyle = 'Sizable'; $form.MaximizeBox = $true
    $form.BackColor = $cBg; $form.ForeColor = $cText; $form.Font = $fText
    $tip = New-Object System.Windows.Forms.ToolTip

    $lblTitle = New-Label 'GPU Burn GUI' 16 10 640 28
    $lblTitle.Font = $fTitle; $lblTitle.Anchor = $AnchorTL
    $btnLang = New-Btn '繁體中文' 762 8 122 24 $cGrey
    $btnLang.Font = $fText; $btnLang.Tag = ''; $btnLang.Anchor = $AnchorTR
    $btnLang.Text = $(if ($script:UiLang -eq 'zh-TW') { 'English' } else { '繁體中文' })
    $form.Controls.Add($lblTitle); $form.Controls.Add($btnLang); $btnLang.BringToFront()

    $lblSummary = New-Label 'Ready. Set options and click Run burn.' 16 42 868 40
    $lblSummary.Font = $fBold; $lblSummary.Anchor = $AnchorTLR; $lblSummary.AutoEllipsis = $false
    $lblHint = New-Label '' 16 84 868 18
    $lblHint.ForeColor = $cYellow; $lblHint.Anchor = $AnchorTLR
    $form.Controls.Add($lblSummary); $form.Controls.Add($lblHint)

    # --- State grid: 4 rows x 2 columns ---
    $stateCaptionL = @('Selected GPU', 'Backend executable', 'Compare kernel', 'Last burn')
    $stateCaptionR = @('GPUs found', 'Working directory', 'Memory mode', 'Burn duration')
    $capLabels = @(); $valLabels = @()
    for ($i = 0; $i -lt 4; $i++) {
        $y = 106 + $i * 21
        $capL = New-Label $stateCaptionL[$i] 16 $y 176 20
        $capL.ForeColor = $cMuted
        $valL = New-Label '' 194 $y 260 20
        $capR = New-Label $stateCaptionR[$i] 456 $y 176 20
        $capR.ForeColor = $cMuted
        $valR = New-Label '' 634 $y 238 20
        $form.Controls.Add($capL); $form.Controls.Add($valL)
        $form.Controls.Add($capR); $form.Controls.Add($valR)
        $capLabels += $capL; $capLabels += $capR
        $valLabels += $valL; $valLabels += $valR
    }

    # --- Tabs ---
    $tabs = New-Ctl 'TabControl' 16 194 868 242
    $tabs.Anchor = $AnchorTLR
    $tabs.ItemSize = New-Object System.Drawing.Size(110, 24)
    $tabBurn = New-Object System.Windows.Forms.TabPage('Burn options')
    $tabAdv  = New-Object System.Windows.Forms.TabPage('Advanced')
    $tabGpus = New-Object System.Windows.Forms.TabPage('GPU list')
    $tabMonitor = New-Object System.Windows.Forms.TabPage('Live monitor')
    foreach ($tp in $tabBurn, $tabAdv, $tabGpus, $tabMonitor) { $tp.Tag = $tp.Text; $tp.Text = T $tp.Tag; $tp.BackColor = $cBox }
    # Empty tabs are fine; controls below are added directly to each TabPage.
    [void]$tabs.TabPages.Add($tabBurn); [void]$tabs.TabPages.Add($tabAdv); [void]$tabs.TabPages.Add($tabGpus); [void]$tabs.TabPages.Add($tabMonitor)
    $form.Controls.Add($tabs)

    # Burn options tab
    $lblMem = New-Label 'Memory to use' 14 18 190 20
    $comboMem = New-Combo 200 16 360 24 @((T 'Backend default (90% available VRAM)'), (T 'Percent of available VRAM'), (T 'Fixed amount (MB)'))
    $comboMem.SelectedIndex = 0
    $numMem = New-Numeric 570 16 100 24 1 100 90
    $lblMemUnit = New-Label '%' 678 18 80 20
    $numMem.Enabled = $false
    $tabBurn.Controls.Add($lblMem); $tabBurn.Controls.Add($comboMem); $tabBurn.Controls.Add($numMem); $tabBurn.Controls.Add($lblMemUnit)

    $lblDur = New-Label 'Burn duration (seconds)' 14 56 190 20
    $numDur = New-Numeric 200 54 100 24 1 864000 3600
    $lblDurUnit = New-Label 'seconds' 308 56 80 20
    $tabBurn.Controls.Add($lblDur); $tabBurn.Controls.Add($numDur); $tabBurn.Controls.Add($lblDurUnit)

    $chkDoubles = New-Check 'Use doubles (-d)' 14 94 366 $false
    $chkTensors = New-Check 'Try Tensor cores (-tc)' 14 116 366 $true
    $tabBurn.Controls.Add($chkDoubles); $tabBurn.Controls.Add($chkTensors)

    $lblGpu = New-Label 'GPU selection' 14 150 190 20
    $comboGpu = New-Combo 200 148 380 24 @(T 'All GPUs (backend default)')
    $comboGpu.SelectedIndex = 0
    $tabBurn.Controls.Add($lblGpu); $tabBurn.Controls.Add($comboGpu)

    # Advanced tab
    $lblCompare = New-Label 'Compare kernel (-c)' 14 18 180 20
    $txtCompare = New-Text 198 16 440 24 ''
    $btnBrowse = New-Btn 'Browse...' 646 15 130 24 $cGrey
    $tabAdv.Controls.Add($lblCompare); $tabAdv.Controls.Add($txtCompare); $tabAdv.Controls.Add($btnBrowse)
    # (The Browse handler is attached later, after the helper functions exist.)

    $lblStts = New-Label 'SIGTERM timeout (-stts)' 14 58 200 20
    $chkStts = New-Check 'Override default (30 seconds)' 210 58 250 $false
    $numStts = New-Numeric 470 54 90 24 1 3600 30
    $lblSttsUnit = New-Label 'seconds' 570 56 80 20
    $numStts.Enabled = $false
    $tabAdv.Controls.Add($lblStts); $tabAdv.Controls.Add($chkStts); $tabAdv.Controls.Add($numStts); $tabAdv.Controls.Add($lblSttsUnit)

    $lblAdvHint = New-Label 'Empty compare path means the default compare.fatbin next to gpu_burn.exe.' 14 100 780 20
    $lblAdvHint.ForeColor = $cMuted
    $tabAdv.Controls.Add($lblAdvHint)

    # GPU list tab
    $lvGpus = New-List @(
        @(('ID'), 50),
        @(('Adapter'), 520),
        @(('Memory (MB)'), 110)
    )
    $tabGpus.Controls.Add($lvGpus)

    # Live monitor tab: nvidia-smi metrics while gpu_burn.exe is running.
    $txtSmi = New-Ctl 'TextBox' 14 14 (868 - 40) (242 - 32)
    $txtSmi.Multiline = $true; $txtSmi.ReadOnly = $true; $txtSmi.ScrollBars = 'Both'
    $txtSmi.BackColor = $cBox; $txtSmi.ForeColor = Rgb @(210, 210, 210); $txtSmi.Font = $fMono
    $txtSmi.WordWrap = $false; $txtSmi.Anchor = $AnchorAll
    $txtSmi.Text = T 'Waiting for a burn.'
    $tabMonitor.Controls.Add($txtSmi)

    # --- Buttons ---
    $btnRefresh = New-Btn 'Refresh GPUs (-l)' 16 452 160 38 $cGrey
    $btnRun     = New-Btn 'Run burn' 186 452 250 38 $cOrange
    $btnStop    = New-Btn 'Stop burn' 446 452 100 38 $cBlue
    $btnLogs    = New-Btn 'Open log folder' 554 452 150 38 $cGrey
    $btnHelp    = New-Btn 'Show help (-h)' 712 452 120 38 $cGrey
    $btnStop.Enabled = $false
    foreach ($c in $btnRefresh, $btnRun, $btnStop, $btnLogs, $btnHelp) { $form.Controls.Add($c) }
    $tip.SetToolTip($chkDoubles, (T 'The packaged gpu_burn.exe may hang after a -d run; use Stop burn if needed.'))
    $tip.SetToolTip($comboGpu, (T 'gpu_burn.exe -i fails in this build; GPU selection uses CUDA_VISIBLE_DEVICES instead.'))

    # --- Progress and output ---
    $pg = New-Object System.Windows.Forms.ProgressBar
    $pg.Location = New-Object System.Drawing.Point(16, 500)
    $pg.Size = New-Object System.Drawing.Size(868, 8)
    $pg.Anchor = $AnchorTLR
    $pg.Minimum = 0; $pg.Maximum = 100
    $form.Controls.Add($pg)

    $txtLog = New-Ctl 'TextBox' 16 516 868 ($form.ClientSize.Height - 516 - 30)
    $txtLog.Multiline = $true; $txtLog.ReadOnly = $true; $txtLog.ScrollBars = 'Vertical'
    $txtLog.BackColor = $cBox; $txtLog.ForeColor = Rgb @(210, 210, 210); $txtLog.Font = $fMono
    $txtLog.Anchor = $AnchorAll
    $form.Controls.Add($txtLog)

    $lblFoot = New-Label 'The back-end is the original gpu_burn.exe. Stop burn interrupts with taskkill /T /F. Heat the card carefully.' 16 ($form.ClientSize.Height - 26) 868 22
    $lblFoot.ForeColor = $cMuted; $lblFoot.Anchor = $AnchorBLR
    $form.Controls.Add($lblFoot)

    # --- Helpers ---
    function Add-UiLog([string]$t) {
        $txtLog.AppendText($t + "`r`n")
        Write-LogFile $t
    }

    function Set-Summary([string]$text, $color) {
        $lblSummary.Text = $text
        $lblSummary.ForeColor = $color
    }

    function Set-Value($label, [string]$text, $color) {
        $label.Text = $text
        $label.ForeColor = $color
    }

    function Add-Separator {
        Add-UiLog ''
        Add-UiLog ('--- ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ---')
    }

    $script:Gpus = @()
    $script:RunProc = $null
    $script:RunStdout = ''
    $script:RunStderr = ''
    $script:RunOutFile = $null
    $script:RunErrFile = $null
    $script:RunOutOffset = 0
    $script:RunErrOffset = 0
    $script:RunStarted = $null
    $script:RunStopRequested = $false
    $script:LastResult = 'Not run'
    $script:Busy = $false
    $script:UiSwitch = $false

    function Set-MemoryMode([bool]$resetValue = $false) {
        $mode = [int]$comboMem.SelectedIndex
        $numMem.Enabled = ($mode -ne 0) -and -not $script:Busy
        if ($mode -eq 1) {
            $numMem.Minimum = 1; $numMem.Maximum = 100
            if ($resetValue -or $numMem.Value -lt 1 -or $numMem.Value -gt 100) { $numMem.Value = 90 }
            $lblMemUnit.Text = '%'
        } elseif ($mode -eq 2) {
            $numMem.Minimum = 1; $numMem.Maximum = 1000000
            if ($resetValue -or $numMem.Value -lt 1 -or $numMem.Value -gt 1000000) { $numMem.Value = 10240 }
            $lblMemUnit.Text = 'MB'
        } else {
            $lblMemUnit.Text = ''
        }
    }

    function Set-OptionState {
        $running = [bool]$script:RunProc
        $btnRefresh.Enabled = -not $running
        $btnRun.Enabled = -not $running
        $btnStop.Enabled = $running
        $btnLogs.Enabled = $true
        $btnHelp.Enabled = -not $running
        $btnLang.Enabled = -not $running
        $btnBrowse.Enabled = -not $running
        $comboMem.Enabled = -not $running
        $comboGpu.Enabled = -not $running
        $numDur.Enabled = -not $running
        $chkDoubles.Enabled = -not $running
        $chkTensors.Enabled = -not $running
        $txtCompare.Enabled = -not $running
        $chkStts.Enabled = -not $running
        $numStts.Enabled = (-not $running) -and $chkStts.Checked
        Set-MemoryMode
    }

    function Set-Busy([bool]$busy) {
        $script:Busy = $busy
        $form.UseWaitCursor = $busy
        Set-OptionState
        [System.Windows.Forms.Application]::DoEvents()
    }

    function Update-View {
        $vSelected = $(if ($comboGpu.SelectedIndex -ge 1 -and $script:Gpus.Count -ge $comboGpu.SelectedIndex) { $comboGpu.Text } else { T 'All GPUs (backend default)' })
        $vBackend = $script:Burn.Exe
        $vKernel = $(if ($txtCompare.Text.Trim()) { $txtCompare.Text } else { $script:Burn.Kernel })
        $vLast = $script:LastResult
        $vGpus = [string]$script:Gpus.Count
        $vDir = $script:Burn.Dir
        $vMem = $comboMem.Text
        $vDur = ('{0} s' -f [int]$numDur.Value)
        # Layout order is left/right pairs, not two contiguous lists:
        # 0 Selected GPU, 1 GPUs found, 2 Backend executable, 3 Working directory,
        # 4 Compare kernel, 5 Memory mode, 6 Last burn, 7 Burn duration.
        $stateText = @($vSelected, $vGpus, $vBackend, $vDir, $vKernel, $vMem, $vLast, $vDur)
        for ($i = 0; $i -lt 8; $i++) {
            $script:valLabels[$i].Text = [string]$stateText[$i]
            $tip.SetToolTip($script:valLabels[$i], [string]$stateText[$i])
        }
        $script:valLabels[6].ForeColor = $(if ($script:LastResult -match 'OK') { $cGreen } elseif ($script:LastResult -match 'FAULTY|error') { $cRed } else { $cText })
    }

    # Rebuild dynamic combo item strings without losing the selected index.
    function Rebuild-GpuCombo {
        $sel = $comboGpu.SelectedIndex
        if ($sel -lt 0) { $sel = 0 }
        $comboGpu.Items.Clear()
        [void]$comboGpu.Items.Add((T 'All GPUs (backend default)'))
        foreach ($g in $script:Gpus) {
            [void]$comboGpu.Items.Add(('GPU {0}: {1}' -f $g.Id, $g.Name))
        }
        if ($sel -lt $comboGpu.Items.Count) { $comboGpu.SelectedIndex = $sel }
        else { $comboGpu.SelectedIndex = 0 }
    }

    function Refresh-GpuList {
        Set-Summary (T 'Refreshing GPU list...') $cMuted
        $lblHint.Text = ''
        [System.Windows.Forms.Application]::DoEvents()
        $result = Get-GpuList
        $script:Gpus = @($result.Gpus)
        $lvGpus.BeginUpdate(); $lvGpus.Items.Clear()
        foreach ($g in $script:Gpus) {
            $it = New-Object System.Windows.Forms.ListViewItem([string]$g.Id)
            [void]$it.SubItems.Add($g.Name)
            [void]$it.SubItems.Add([string]$g.Memory)
            $it.ForeColor = $cGreen
            [void]$lvGpus.Items.Add($it)
        }
        if ($script:Gpus.Count -eq 0) {
            $it = New-Object System.Windows.Forms.ListViewItem('-')
            [void]$it.SubItems.Add((T 'No GPU rows were returned by gpu_burn.exe -l.'))
            [void]$it.SubItems.Add('')
            $it.ForeColor = $cRed
            [void]$lvGpus.Items.Add($it)
        }
        $lvGpus.EndUpdate()
        Rebuild-GpuCombo
        Add-UiLog ''
        Add-UiLog ('=== ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' - GPU list ===')
        $outLines = @($result.Capture.Stdout -split '\r?\n')
        foreach ($line in $outLines) { if ($line.Trim()) { Add-UiLog ('  ' + $line.Trim()) } }
        foreach ($line in @($result.Capture.Stderr -split '\r?\n')) {
            if ($line.Trim() -and ($line.Trim() -notmatch '^cuInit returned 0 \(no error\)$')) { Add-UiLog ('  [stderr] ' + $line.Trim()) }
        }
        if ($result.Capture.TimedOut) { Add-UiLog ('  WARNING: gpu_burn.exe -l did not exit within 12 seconds.'); Set-Summary (T 'Finished with errors.') $cYellow }
        elseif ($script:Gpus.Count -eq 0) { Add-UiLog ('  ' + (T 'No GPU found. Check nvidia-smi and the NVIDIA driver.')); Set-Summary (T 'Finished with errors.') $cRed }
        else { Set-Summary (T 'GPU list refreshed.' + ' ' + ('{0} GPU(s).' -f $script:Gpus.Count)) $cGreen }
        Update-View
    }

    function Get-ComparePath {
        $raw = $txtCompare.Text.Trim()
        if (-not $raw) { return $null }
        try {
            if (Test-Path -LiteralPath $raw -PathType Leaf) { return (Resolve-Path -LiteralPath $raw).Path }
        } catch { }
        # Relative to the backend directory, like gpu_burn.exe would use.
        try {
            $rel = Join-Path $script:Burn.Dir $raw
            if (Test-Path -LiteralPath $rel -PathType Leaf) { return (Resolve-Path -LiteralPath $rel).Path }
        } catch { }
        return '__MISSING__'
    }

    function Get-SelectedGpu {
        if ($comboGpu.SelectedIndex -gt 0 -and $script:Gpus.Count -ge $comboGpu.SelectedIndex) {
            return $script:Gpus[$comboGpu.SelectedIndex - 1]
        }
        return $null
    }

    function Get-NvidiaSmiPath {
        $cmd = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cmd) { return $cmd.Source }
        $candidates = @(
            (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\nvidia-smi.exe'),
            (Join-Path $env:ProgramFiles 'NVIDIA Corporation\NVSMI\bin\nvidia-smi.exe')
        )
        foreach ($candidate in $candidates) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
        }
        return $null
    }

    function Update-GpuMonitor {
        $smi = Get-NvidiaSmiPath
        if (-not $smi) {
            $txtSmi.Text = (T 'nvidia-smi live monitor') + "`r`n" + (T 'nvidia-smi was not found on PATH.')
            return
        }
        try {
            $selected = Get-SelectedGpu
            $queryFields = 'index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,fan.speed'
            $queryArgs = @(
                ('--query-gpu=' + $queryFields),
                '--format=csv,noheader,nounits'
            )
            if ($selected) { $queryArgs = @('-i', [string]$selected.Id) + $queryArgs }
            $raw = & $smi @queryArgs 2>$null
            $blocks = New-Object System.Collections.Generic.List[string]
            $sampleCount = 0
            foreach ($line in @($raw -split '\r?\n')) {
                if (-not $line.Trim()) { continue }
                $parts = $line -split '\s*,\s*'
                if ($parts.Count -lt 9) { continue }
                $id = $parts[0].Trim(); $name = $parts[1].Trim()
                $temp = $parts[2].Trim(); $util = $parts[3].Trim()
                $used = $parts[4].Trim(); $total = $parts[5].Trim()
                $draw = $parts[6].Trim(); $limit = $parts[7].Trim(); $fan = $parts[8].Trim()
                $sampleCount++
                $blocks.Add('')
                $blocks.Add(('GPU {0}: {1}' -f $id, $name))
                $blocks.Add((T 'Temp (C)') + ': ' + $temp + ' C')
                $blocks.Add((T 'Utilization (%)') + ': ' + $util + '%')
                $blocks.Add((T 'VRAM (used / total MB)') + ': ' + $used + ' / ' + $total + ' MB')
                $blocks.Add((T 'Power (W) / limit') + ': ' + $draw + ' / ' + $limit + ' W')
                $blocks.Add((T 'Fan (%)') + ': ' + $fan + '%')
            }
            if ($sampleCount -eq 0) {
                $txtSmi.Text = (T 'nvidia-smi live monitor') + "`r`n" + (T 'Could not read nvidia-smi output.')
                return
            }
            $header = (T 'nvidia-smi live monitor') + '  ' + (Get-Date -Format 'HH:mm:ss') + "`r`n"
            $header += (T 'The monitor refreshes while a burn runs; values come from nvidia-smi --query-gpu.') + "`r`n"
            $txtSmi.Text = $header + ($blocks -join "`r`n")
        } catch {
            $txtSmi.Text = (T 'nvidia-smi live monitor') + "`r`n" + (T 'Could not read nvidia-smi output.') + "`r`n" + $_.Exception.Message
        }
    }

    function Build-BurnArgs {
        $argsList = New-Object System.Collections.Generic.List[string]
        $mode = [int]$comboMem.SelectedIndex
        if ($mode -eq 1) { $argsList.Add('-m'); $argsList.Add(('{0}%' -f [int]$numMem.Value)) }
        elseif ($mode -eq 2) { $argsList.Add('-m'); $argsList.Add([string][int]$numMem.Value) }
        if ($chkDoubles.Checked) { $argsList.Add('-d') }
        if ($chkTensors.Checked) { $argsList.Add('-tc') }
        # The packaged backend crashes when given -i (it skips cuInit for that path);
        # GPU selection is handled by CUDA_VISIBLE_DEVICES in Start-Burn instead.
        $comparePath = Get-ComparePath
        if ($comparePath -eq '__MISSING__') { return '__MISSING_COMPARE__' }
        if ($comparePath) { $argsList.Add('-c'); $argsList.Add($comparePath) }
        if ($chkStts.Checked) { $argsList.Add('-stts'); $argsList.Add([string][int]$numStts.Value) }
        $argsList.Add([string][int]$numDur.Value)
        $argsList.ToArray()
    }

    function Format-OptionSummary([string[]]$argList) {
        $lines = New-Object System.Collections.Generic.List[string]
        $mode = [int]$comboMem.SelectedIndex
        if ($mode -eq 0) { $lines.Add(('Memory: backend default (90% available VRAM)') -join '') }
        elseif ($mode -eq 1) { $lines.Add((T 'Memory to use') + ': ' + [int]$numMem.Value + '%') }
        else { $lines.Add((T 'Memory to use') + ': ' + [int]$numMem.Value + ' MB') }
        if ($chkDoubles.Checked) { $lines.Add('Doubles: yes (-d)') }
        if ($chkTensors.Checked) { $lines.Add('Tensor cores: try (-tc)') }
        $selected = Get-SelectedGpu
        if ($selected) { $lines.Add('GPU: ' + $comboGpu.Text + ' via CUDA_VISIBLE_DEVICES=' + [string]$selected.Id) }
        else { $lines.Add('GPU: all devices') }
        if ($txtCompare.Text.Trim()) { $lines.Add('Compare kernel: ' + (Get-ComparePath)) }
        if ($chkStts.Checked) { $lines.Add('SIGTERM timeout: ' + [int]$numStts.Value + ' s (-stts)') }
        $lines.Add('Duration: ' + [int]$numDur.Value + ' s')
        $lines.Add('Command: gpu_burn.exe ' + (Get-BurnArgumentsLine $argList))
        $lines.ToArray()
    }

    function Stop-Burn {
        if (-not $script:RunProc) { return }
        try {
            $burnProcessId = $script:RunProc.Id
            $script:RunStopRequested = $true
            Add-UiLog ('=== ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' - stop requested ===')
            & "$env:SystemRoot\System32\taskkill.exe" /T /F /PID $burnProcessId 2>$null | Out-Null
        } catch { Add-UiLog ('WARNING: ' + $_.Exception.Message) }
    }

    function Start-Burn([switch]$SkipConfirm) {
        if ($script:RunProc) { return }
        $argList = Build-BurnArgs
        if (-not $argList) { Set-Summary (T 'No GPU found. Check nvidia-smi and the NVIDIA driver.') $cRed; return }
        if ($argList -eq '__MISSING_COMPARE__') {
            [void][System.Windows.Forms.MessageBox]::Show((T 'The compare kernel file was not found. Fix the path or leave it empty for the default compare.fatbin.'), 'GPU Burn GUI', [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Warning)
            return
        }
        $summary = Format-OptionSummary $argList
        $script:RunStopRequested = $false
        if (-not $SkipConfirm) {
            $q = (T 'This will run gpu_burn.exe with:')
            $q += "`r`n" + ($summary -join "`r`n") + "`r`n`r`n"
            $q += (T 'Do not turn the PC off while it runs. Heat the card carefully.')
        } else { $q = (T 'Auto-test: starting gpu_burn.exe smoke burn.') }
        if (-not $SkipConfirm) {
            $resp = [System.Windows.Forms.MessageBox]::Show($q + "`r`n`r`n" + (T 'Continue?'), 'GPU Burn GUI', [System.Windows.Forms.MessageBoxButtons]::YesNo, [System.Windows.Forms.MessageBoxIcon]::Warning)
            if ($resp -ne [System.Windows.Forms.DialogResult]::Yes) { Add-UiLog (T 'Cancelled. Nothing was burned.'); return }
        }
        try {
            Add-Separator
            $selected = Get-SelectedGpu
            if ($selected) { Add-UiLog ('CUDA_VISIBLE_DEVICES=' + [string]$selected.Id + ' / gpu_burn.exe ' + (Get-BurnArgumentsLine $argList)) }
            else { Add-UiLog ((T 'Command: ') + 'gpu_burn.exe ' + (Get-BurnArgumentsLine $argList)) }
            $tag = [guid]::NewGuid().ToString('N')
            $script:RunOutFile = Join-Path $script:TempDir ('run-' + $tag + '.out')
            $script:RunErrFile = Join-Path $script:TempDir ('run-' + $tag + '.err')
            $script:RunStdout = ''; $script:RunStderr = ''
            $script:RunOutOffset = 0; $script:RunErrOffset = 0
            $script:RunStopRequested = $false
            $script:RunStarted = Get-Date
            $prevCvd = [Environment]::GetEnvironmentVariable('CUDA_VISIBLE_DEVICES', 'Process')
            if ($selected) { [Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', [string]$selected.Id, 'Process') }
            try {
                $script:RunProc = Start-Process -FilePath $script:Burn.Exe -WorkingDirectory $script:Burn.Dir `
                    -ArgumentList (Get-BurnArgumentsLine $argList) `
                    -RedirectStandardOutput $script:RunOutFile `
                    -RedirectStandardError $script:RunErrFile `
                    -WindowStyle Hidden -PassThru -ErrorAction Stop
            } finally {
                [Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES', $prevCvd, 'Process')
            }
            Add-UiLog ((T 'Backend: ') + $script:Burn.Exe)
            Set-Summary (T 'Burning GPU stress test...') $cOrange
            Set-Busy $true
            $pg.Value = 0
            $timer.Start()
            $tabs.SelectedTab = $tabMonitor
            $monTimer.Start()
            Update-GpuMonitor
        } catch {
            Add-UiLog ('ERROR: ' + $_.Exception.Message)
            Set-Summary (T 'Finished with errors.') $cRed
            $script:RunProc = $null
            $monTimer.Stop()
            Set-Busy $false
        }
    }

    function Read-TextDelta([string]$path, [int]$offset, [string]$into) {
        if (-not (Test-Path -LiteralPath $path)) { return $false }
        $text = $null
        $fs = $null
        $reader = $null
        try {
            $fs = [System.IO.File]::Open($path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            $reader = New-Object System.IO.StreamReader -ArgumentList @($fs, [System.Text.Encoding]::UTF8, $true)
            $text = $reader.ReadToEnd()
        } catch { return $false }
        finally {
            if ($reader) { $reader.Dispose() }
            if ($fs) { $fs.Dispose() }
        }
        if ($null -eq $text -or $text.Length -le $offset) { return $false }
        try {
            $chunk = $text.Substring($offset)
            if ($into -eq 'out') {
                $script:RunStdout += $chunk
                Write-StdoutChunk $chunk
                $script:RunOutOffset = $text.Length
            } else {
                $script:RunStderr += $chunk
                Write-StderrChunk $chunk
                $script:RunErrOffset = $text.Length
            }
            return $true
        } catch { return $false }
    }

    function Write-StdoutChunk([string]$chunk) {
        $chunk = $chunk -replace '\r\n', "`n"
        foreach ($part in ($chunk -split '\r')) {
            $part = $part.TrimEnd("`n")
            if ($part -match '^\s*(?<pct>\d+(?:\.\d+)?)%') {
                $pct = [double]$Matches.pct
                $pg.Value = [Math]::Min(100, [Math]::Max(0, [int][Math]::Round($pct)))
                $lblHint.Text = (T 'Running...') + ' ' + $part.Trim()
                $lblHint.ForeColor = $cYellow
                foreach ($line in ($part -split '\r?\n')) {
                    if ($line.Trim()) { Add-UiLog ($line.Trim()) }
                }
                continue
            }
            foreach ($line in ($part -split '\r?\n')) {
                if ($line.Trim()) { Add-UiLog ($line.Trim()) }
            }
        }
    }

    function Write-StderrChunk([string]$chunk) {
        foreach ($line in ($chunk -split '\r?\n')) {
            $line = $line.Trim()
            if ($line.Trim() -and ($line -notmatch '^cuInit returned 0 \(no error\)$')) { Add-UiLog ('[stderr] ' + $line) }
        }
    }

    function Finish-Burn {
        $timer.Stop()
        $monTimer.Stop()
        if (-not $script:RunProc) { Set-Busy $false; return }
        # Drain the remaining text and keep RunStdout/RunStderr complete for verdicts.
        try {
            [void](Read-TextDelta $script:RunOutFile $script:RunOutOffset 'out')
            [void](Read-TextDelta $script:RunErrFile $script:RunErrOffset 'err')
        } catch { }
        $exitCode = $null
        try { $exitCode = [int]$script:RunProc.ExitCode } catch { }
        $stopRequested = $script:RunStopRequested
        $script:RunProc.Dispose()
        $script:RunProc = $null
        $pg.Value = $(if ($script:RunStdout -match 'Tested \d+ GPUs') { 100 } else { $pg.Value })
        $diag = $script:RunStdout + $script:RunStderr
        $failure = $diag -match 'DIED|Couldn''t init a GPU test|No clients are alive|FAULTY'
        $fault = $script:RunStdout -match 'GPU \d+: FAULTY'
        $ok = ($script:RunStdout -match 'Tested \d+ GPUs') -and -not $failure
        if ($stopRequested) {
            Set-Summary ((T 'Stopped by user.') + ' ' + (T 'Burning finished; see the log below.')) $cYellow
            $script:LastResult = 'Stopped'
        } elseif ($fault) {
            Set-Summary ((T 'Finished: FAULTY GPU detected.') + ' ' + (T 'Burning finished; see the log below.')) $cRed
            $script:LastResult = 'FAULTY'
        } elseif ($failure) {
            Set-Summary ((T 'Finished with errors.') + ' ' + (T 'Burning finished; see the log below.')) $cRed
            $script:LastResult = 'Error'
        } elseif ($ok) {
            Set-Summary ((T 'Finished: GPU test OK.') + ' ' + (T 'Burning finished; see the log below.')) $cGreen
            $script:LastResult = 'OK'
        } elseif ($null -ne $exitCode -and $exitCode -eq 0) {
            Set-Summary (T 'Finished.') $cYellow
            $script:LastResult = 'Finished (exit 0)'
        } else {
            Set-Summary ((T 'Finished with errors.') + ' ' + (T 'Burning finished; see the log below.')) $cRed
            $script:LastResult = ('Error (exit {0})' -f $exitCode)
        }
        Add-UiLog ((T 'Exit code: ') + [string]$exitCode)
        Add-UiLog ((T 'Last burn: ') + $script:LastResult)
        $lblHint.Text = (T 'Finished.')
        $lblHint.ForeColor = $cMuted
        Set-Busy $false
        Update-View
        if ($script:AutoTest) {
            Add-UiLog ('--- Auto-test finished. Keeping the window open: ' + [string]$script:AutoTestKeep)
            if (-not $script:AutoTestKeep) { $closeTimer.Start() }
        }
    }

    # Re-render the dynamic summary/hint after a language switch.
    function Refresh-ResultTranslation {
        if ($script:Busy) { return }
        $last = $script:LastResult
        if (-not $last) {
            Set-Summary (T 'Ready. Set options and click Run burn.') $cMuted
            $lblHint.Text = ''
            $lblHint.ForeColor = $cMuted
            return
        }
        if ($last -eq 'Stopped') {
            Set-Summary ((T 'Stopped by user.') + ' ' + (T 'Burning finished; see the log below.')) $cYellow
        } elseif ($last -eq 'FAULTY') {
            Set-Summary ((T 'Finished: FAULTY GPU detected.') + ' ' + (T 'Burning finished; see the log below.')) $cRed
        } elseif ($last -eq 'OK') {
            Set-Summary ((T 'Finished: GPU test OK.') + ' ' + (T 'Burning finished; see the log below.')) $cGreen
        } elseif ($last -eq 'Finished (exit 0)' -or $last -eq 'Finished') {
            Set-Summary (T 'Finished.') $cYellow
        } else {
            Set-Summary ((T 'Finished with errors.') + ' ' + (T 'Burning finished; see the log below.')) $cRed
        }
        $lblHint.Text = (T 'Finished.')
        $lblHint.ForeColor = $cMuted
    }

    # --- Background process polling: no GUI freeze, progress stream ---
    $timer = New-Object System.Windows.Forms.Timer
    $timer.Interval = 250
    $timer.Add_Tick({
        try {
            if (-not $script:RunProc) { $timer.Stop(); return }
            $script:RunProc.Refresh()
            [void](Read-TextDelta $script:RunOutFile $script:RunOutOffset 'out')
            [void](Read-TextDelta $script:RunErrFile $script:RunErrOffset 'err')
            if (-not $script:RunProc.HasExited) {
                $elapsed = ((Get-Date) - $script:RunStarted).TotalSeconds
                $lblHint.Text = (T 'Running...') + ' ' + [string]([int]$elapsed) + ' s'
                return
            }
            Finish-Burn
        } catch { Add-UiLog ('ERROR: ' + $_.Exception.Message); $timer.Stop(); if ($script:RunProc) { Finish-Burn } else { Set-Busy $false } }
    })

    # nvidia-smi monitor refresh while the burn runs.
    $monTimer = New-Object System.Windows.Forms.Timer
    $monTimer.Interval = 1000
    $monTimer.Add_Tick({
        try {
            if ($script:RunProc) { Update-GpuMonitor }
        } catch { $monTimer.Stop(); Add-UiLog ('ERROR nvidia-smi: ' + $_.Exception.Message) }
    })

    # One-shot close timer for the smoke-test path.
    $closeTimer = New-Object System.Windows.Forms.Timer
    $closeTimer.Interval = 1500
    $closeTimer.Add_Tick({ $closeTimer.Stop(); $form.Close() })

    # --- Event handlers ---
    $comboMem.Add_SelectedIndexChanged({
        if (-not $script:Busy -and -not $script:UiSwitch) { Set-MemoryMode $true; Update-View }
    })
    $comboGpu.Add_SelectedIndexChanged({ Update-View })
    $numDur.Add_ValueChanged({ Update-View })
    $numMem.Add_ValueChanged({ Update-View })
    $txtCompare.Add_TextChanged({ Update-View })
    $btnBrowse.Add_Click({
        try {
            $dlg = New-Object System.Windows.Forms.OpenFileDialog
            $dlg.InitialDirectory = $script:Burn.Dir
            $dlg.FileName = ''
            $dlg.Filter = 'Compare kernel (*.fatbin;*.cubin)|*.fatbin;*.cubin|All files (*.*)|*.*'
            if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $txtCompare.Text = $dlg.FileName; Update-View }
        } catch { Add-UiLog ('ERROR: ' + $_.Exception.Message) }
    })
    $btnRefresh.Add_Click({
        try { Set-Busy $true; Refresh-GpuList; Set-Busy $false } catch { Add-UiLog ('ERROR: ' + $_.Exception.Message); Set-Busy $false }
    })
    $btnRun.Add_Click({
        try { Start-Burn } catch { Add-UiLog ('ERROR: ' + $_.Exception.Message); Set-Busy $false }
    })
    $btnStop.Add_Click({ try { Stop-Burn } catch { Add-UiLog ('ERROR: ' + $_.Exception.Message) } })
    $btnLogs.Add_Click({ try { Start-Process -FilePath 'explorer.exe' -ArgumentList ('"' + $script:LogDir + '"') } catch { Add-UiLog ('ERROR: ' + $_.Exception.Message) } })
    $btnHelp.Add_Click({
        try {
            Add-Separator
            $capture = Invoke-BurnCapture @('-h')
            foreach ($line in @($capture.Stdout -split '\r?\n')) { if ($line.Trim()) { Add-UiLog ($line.Trim()) } }
            foreach ($line in @($capture.Stderr -split '\r?\n')) { if ($line.Trim() -and ($line -notmatch '^cuInit returned 0 \(no error\)$')) { Add-UiLog ('[stderr] ' + $line.Trim()) } }
        } catch { Add-UiLog ('ERROR: ' + $_.Exception.Message) }
    })
    $chkStts.Add_CheckedChanged({ $numStts.Enabled = $chkStts.Checked -and -not $script:Busy })
    $btnLang.Add_Click({
        if ($script:Busy) { return }
        if ($script:UiLang -eq 'zh-TW') { $script:UiLang = 'en' } else { $script:UiLang = 'zh-TW' }
        try {
            $staticControls = @(
                $lblTitle, $lblFoot, $lblMem, $lblDur, $lblDurUnit, $lblGpu,
                $lblCompare, $lblStts, $lblSttsUnit, $lblAdvHint, $lblMemUnit,
                $chkDoubles, $chkTensors, $chkStts, $btnRefresh, $btnRun,
                $btnStop, $btnLogs, $btnHelp, $btnBrowse
            )
            foreach ($c in $staticControls) { if ($c.Tag) { $c.Text = T ([string]$c.Tag) } }
            foreach ($c in $capLabels) { if ($c.Tag) { $c.Text = T ([string]$c.Tag) } }
            foreach ($tp in $tabs.TabPages) { if ($tp.Tag) { $tp.Text = T $tp.Tag } }
            $script:UiSwitch = $true
            $selMem = $comboMem.SelectedIndex
            $comboMem.Items.Clear()
            [void]$comboMem.Items.Add((T 'Backend default (90% available VRAM)'))
            [void]$comboMem.Items.Add((T 'Percent of available VRAM'))
            [void]$comboMem.Items.Add((T 'Fixed amount (MB)'))
            if ($selMem -ge 0) { $comboMem.SelectedIndex = $selMem } else { $comboMem.SelectedIndex = 0 }
            Rebuild-GpuCombo
            $script:UiSwitch = $false
            for ($i = 0; $i -lt $lvGpus.Columns.Count; $i++) {
                $lvGpus.Columns[$i].Text = T ([string]$lvGpus.Tag[$i][0])
            }
            $btnLang.Text = $(if ($script:UiLang -eq 'zh-TW') { 'English' } else { '繁體中文' })
            $form.Text = 'GPU Burn GUI v' + $script:AppVersion
            Update-View
            Refresh-ResultTranslation
            Update-GpuMonitor
        } catch { Add-UiLog ('ERROR: ' + $_.Exception.Message) }
    })

    $form.Add_FormClosing({
        param($src, $evtArgs)
        if ($script:RunProc) {
            [void][System.Windows.Forms.MessageBox]::Show((T 'A burn is still running. Stop it first, then close the window.'), 'GPU Burn GUI', [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Information)
            $evtArgs.Cancel = $true
        }
    })

    # Log opening lines
    Add-UiLog ('GPU Burn GUI v' + $script:AppVersion)
    Add-UiLog (T 'Nothing is burned until you click Run burn and confirm.')
    Add-UiLog ((T 'Backend: ') + $script:Burn.Exe)
    Add-UiLog ((T 'Working directory: ') + $script:Burn.Dir)
    Add-UiLog ((T 'Compare kernel: ') + $script:Burn.Kernel)
    if ($script:LogFile) { Add-UiLog ((T 'Log file: ') + $script:LogFile) }

    $form.Add_Shown({
        try {
            Set-Busy $false
            Update-View
            Set-Summary (T 'Refreshing GPU list...') $cMuted
            Refresh-GpuList
            Update-GpuMonitor
            if ($script:AutoTest) {
                $comboMem.SelectedIndex = 1
                Set-MemoryMode
                $numMem.Value = 40
                $numDur.Value = [int]$script:TestSeconds
                $chkDoubles.Checked = $false
                $chkTensors.Checked = $true
                $chkStts.Checked = $true
                $numStts.Value = 30
                $txtCompare.Text = $script:Burn.Kernel
                if ($script:Gpus.Count -gt 0) { $comboGpu.SelectedIndex = 1 } else { $comboGpu.SelectedIndex = 0 }
                Update-View
                Start-Burn -SkipConfirm
            }
        } catch {
            Add-UiLog ('ERROR: ' + $_.Exception.Message)
            Set-Summary (T 'Finished with errors.') $cRed
        }
    })

    [void]$form.ShowDialog()
    exit 0
} catch {
    $msg = (T 'Unexpected error: ') + $_.Exception.Message + "`r`n" + (T 'Line ') + $_.InvocationInfo.ScriptLineNumber
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction SilentlyContinue
        [void][System.Windows.Forms.MessageBox]::Show($msg, 'GPU Burn GUI', [System.Windows.Forms.MessageBoxButtons]::OK, [System.Windows.Forms.MessageBoxIcon]::Error)
    } catch { Write-Host $msg -ForegroundColor Red }
    exit 1
}
