#!/bin/bash

# Create the conda environment
cd "$RepositoryPath"
echo ""
echo "--------------------------------------------------------------------------------"  # Assuming echo provides similar functionality to Write-Host
echo -e "\e[32mUpdating the ${DisplayName} conda environment (${EnvironmentName})\e[0m"
echo "--------------------------------------------------------------------------------" 

# Assume here that if the environment folder is missing, the environment was already deleted
if [ -d "$EnvironmentPath" ]; then
    echo -e "\e[32mUpdating the ${DisplayName} conda environment (${EnvironmentName})\e[0m"
else
    echo -e "\e[32mCreating the ${DisplayName} conda environment (${EnvironmentName})\e[0m"
fi
echo "--------------------------------------------------------------------------------"

# You can control where a conda environment lives by providing a path to a target directory when creating the environment.
FilePath="${RepositoryPath}/sh_environment.yml"
if [ -f "$FilePath" ]; then
    conda env update --prefix "$EnvironmentPath" --file "$FilePath" --prune --quiet
fi

if conda info --envs | grep -q "$EnvironmentName"; then
    conda update --all --yes --name "$EnvironmentName"
else
    conda create --yes --name "$EnvironmentName" python=3.7
fi
conda info --envs
