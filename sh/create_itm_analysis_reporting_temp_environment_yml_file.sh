
#!/bin/bash
# Soli Deo gloria

# Run this in a PowerShell window:
# 
# conda activate base; cd /mnt/c/Users/DaveBabbitt/Documents/GitHub/itm-analysis-reporting/sh; clear; ./create_itm_analysis_reporting_temp_environment_yml_file.sh

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

conda info --envs

# Create the temporary conda environment.yml file
echo ""
echo "-------------------------------------------------------------------------------"
echo "                Creating the temporary conda environment.yml file"
echo "-------------------------------------------------------------------------------"
cd "${RepositoryPath}"
conda activate "${EnvironmentPath}"
conda info --envs
conda env export -f "${RepositoriesDirectory}/${RepositoryName}/tmp_environment.yml" --no-builds
conda deactivate

cd "$OldPath"