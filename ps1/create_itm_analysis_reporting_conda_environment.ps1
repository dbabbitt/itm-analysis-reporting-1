#Requires -Version 2.0
# Soli Deo gloria

# You have to manually stop the jupyter server before you run this in a PowerShell window
# if you are deleting the environment before recreating it:
# Run this in a PowerShell window:
# 
# cd $Env:UserProfile\Documents\GitHub\itm-analysis-reporting\ps1; cls; .\create_itm_analysis_reporting_conda_environment.ps1

# Set up global variables
$DisplayName = "ITM Analysis Reporting"
$RepositoryName = "itm-analysis-reporting"
$EnvironmentName = "itm_analysis_reporting"

$HomeDirectory = $Env:UserProfile
$EnvironmentsDirectory = "${HomeDirectory}\anaconda3\envs"
$RepositoriesDirectory = "${HomeDirectory}\Documents\GitHub"
$RepositoryPath = "${RepositoriesDirectory}\${RepositoryName}"
$PowerScriptsDirectory = "${RepositoryPath}\ps1"
$EnvironmentPath = "${EnvironmentsDirectory}\${EnvironmentName}"
$OldPath = Get-Location

# Prep conda
conda deactivate
conda config --set auto_update_conda true
conda config --set report_errors false

conda activate base

# Delete environment folder
."${PowerScriptsDirectory}\delete_conda_environment.ps1"

# Update the base conda packages
# ."${PowerScriptsDirectory}\update_base_conda_packages.ps1"

# Update Node.js
# ."${PowerScriptsDirectory}\update_node_js.ps1"

# Create the conda environment
."${PowerScriptsDirectory}\create_or_update_conda_environment.ps1"

."${PowerScriptsDirectory}\function_definitions.ps1"

# Fix ZMQ bug
# Reinstall-ZMQ $EnvironmentPath -Verbose

# Create the environment folder
."${PowerScriptsDirectory}\update_jupyterlab_environment.ps1"

cd $OldPath