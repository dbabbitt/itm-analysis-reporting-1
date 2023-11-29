
#!/bin/bash
# Soli Deo gloria

# You have to manually stop the jupyter server before you run this in a PowerShell window
# if you are deleting the environment before recreating it:
# Run this in a PowerShell window:
# 
# cd /mnt/c/Users/DaveBabbitt/Documents/GitHub/itm-analysis-reporting/sh; clear; ./create_itm_analysis_reporting_conda_environment.sh

# Set up global variables
DisplayName="ITM Analysis Reporting"
RepositoryName="itm-analysis-reporting"
EnvironmentName="itm_analysis_reporting"

HomeDirectory="$HOME"
EnvironmentsDirectory="${HomeDirectory}/anaconda3/envs"
RepositoriesDirectory="/mnt/c/Users/DaveBabbitt/Documents/GitHub"
RepositoryPath="${RepositoriesDirectory}/${RepositoryName}"
BashScriptsDirectory="${RepositoryPath}/sh"
EnvironmentPath="${EnvironmentsDirectory}/${EnvironmentName}"
OldPath=$(pwd)

# Prep conda
conda deactivate
conda config --set auto_update_conda true
conda config --set report_errors false

conda activate base

# Delete environment folder
"${BashScriptsDirectory}/delete_conda_environment.sh"

# Update the base conda packages
# "${BashScriptsDirectory}/update_base_conda_packages.sh"

# Update Node.js
# "${BashScriptsDirectory}/update_node_js.sh"

# Create the conda environment
"${BashScriptsDirectory}/create_or_update_conda_environment.sh"

"${BashScriptsDirectory}/function_definitions.sh"

# Create the environment folder
"${BashScriptsDirectory}/update_jupyterlab_environment.sh"

cd "$OldPath"