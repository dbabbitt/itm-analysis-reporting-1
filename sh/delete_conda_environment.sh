#!/bin/bash

source "${PowerScriptsDirectory}/function_definitions.sh"

# Delete conda environment
echo ""
echo "--------------------------------------------------------------------------------"
echo " Deleting the ${DisplayName} conda environment (${EnvironmentName})"
echo "--------------------------------------------------------------------------------"
conda remove --name "$EnvironmentName" --all --yes

# You have to manually delete the folder if you don't manually stop the server
# `jupyter notebook stop 8888` Won't work on Windows as of 2020-11-19
if [ -d "$EnvironmentPath" ]; then
    TokenString=$(Get-Token-String)
    if [ "$TokenString" != "" ]; then
        read -p "Stop the Jupyter server manually, then press ENTER to continue..."
    fi
    echo ""
    echo "---------------------------------------------------------------------------------"
    echo " Recursively removing ${EnvironmentPath}"
    echo "---------------------------------------------------------------------------------"
    rm -r -f "$EnvironmentPath"
fi

# Delete the kernel from the Launcher
KernelPath="${HomeDirectory}/anaconda3/share/jupyter/kernels/${EnvironmentName}"
if [ -d "$KernelPath" ]; then
    TokenString=$(Get-Token-String)
    if [ "$TokenString" != "" ]; then
        read -p "Stop the Jupyter server manually, then press ENTER to continue..."
    fi
    echo ""
    echo "--------------------------------------------------------------------------------------------------"
    echo " Recursively removing ${KernelPath}"
    echo "--------------------------------------------------------------------------------------------------"
    rm -r -f "$KernelPath"
fi
conda info --envs