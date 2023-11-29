#Requires -Version 2.0

# Soli Deo gloria

# Run this in a PowerShell window:
# 
# conda activate base; cd $Env:UserProfile\Documents\GitHub\itm-analysis-reporting\ps1; cls; .\create_itm_analysis_reporting_temp_environment_yml_file.ps1

# Set up global variables
$DisplayName = "ITM Analysis Reporting"
$RepositoryName = "itm-analysis-reporting"
$EnvironmentName = "itm_analysis_reporting"

$HomeDirectory = $Env:UserProfile
$EnvironmentsDirectory = "${HomeDirectory}\anaconda3\envs"
$RepositoriesDirectory = "${HomeDirectory}\Documents\GitHub"
$EnvironmentPath = "${EnvironmentsDirectory}\${EnvironmentName}"
$OldPath = Get-Location

conda info --envs
."${RepositoriesDirectory}\${RepositoryName}\ps1\create_temp_environment_yml_file.ps1"

Write-Host ""
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
Write-Host "       Running Compare It! to compare the old and new yml files" -ForegroundColor Green
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
Start-Process "C:\Program Files (x86)\Compare It!\wincmp3.exe" -ArgumentList "${RepositoriesDirectory}\${RepositoryName}\environment.yml ${RepositoriesDirectory}\${RepositoryName}\tmp_environment.yml"

Write-Host ""
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
Write-Host "       Running Notepad++ so you can sort the new yml file" -ForegroundColor Green
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
Start-Process "C:\Program Files\Notepad++\notepad++.exe" -ArgumentList "${RepositoriesDirectory}\${RepositoryName}\tmp_environment.yml"

cd $OldPath