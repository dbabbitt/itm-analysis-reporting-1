#!/bin/bash

Format_Json() {
    local Json="$1"
    local Minify=false
    local Indentation=4
    local AsArray=false

    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -Minify)
                Minify=true
                shift
                ;;
            -Indentation)
                Indentation="$2"
                shift 2
                ;;
            -AsArray)
                AsArray=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    if [ "$Minify" = true ]; then
        echo "$Json" | jq -c .
    else
        echo "$Json" | jq --indent $Indentation .
    fi
}

Add_Python_Executable_To_Path() {
    local EnvironmentPath="$1"
    local ExecutablePath="${EnvironmentPath}/bin/python"
    local PathArray=(${ExecutablePath//\\/ })
    local PythonFolder=$(IFS='\\'; echo "${PathArray[*]:0:${#PathArray[@]}-1}")

    if [[ ! "$PATH" =~ "$PythonFolder" ]]; then
        echo "The ${EnvironmentPath} executable path is not in PATH" >&2
        export PATH="${EnvironmentPath}:${PATH}"
    fi
}

Get_Python_Version() {
    local EnvironmentPath="$1"
    local CommandString="cd ${EnvironmentPath} && python --version"
    local PythonVersion=$(eval "$CommandString" 2>&1)
    echo "$PythonVersion" | tr -d '\r\n'
}

Add_Kernel_To_Launcher() {
    local EnvironmentPath="$1"
    local DisplayName="${2:-Python}"

    local PythonVersion=$(Get_Python_Version "$EnvironmentPath")

    # Fix LookupError: unknown encoding: cp65001
    local CommandString="cd ${EnvironmentPath} && python -c \"import os; os.environ['PYTHONIOENCODING'] = 'UTF-8'\""
    eval "$CommandString" 2>&1

    local EnvironmentName=$(basename "$EnvironmentPath")
    local CommandString="cd ${EnvironmentPath} && conda activate ${EnvironmentName} && python -m ipykernel install --user --name ${EnvironmentName} --display-name '${DisplayName} (${PythonVersion})'"
    eval "$CommandString" 2>&1
}

Reinstall_ZMQ() {
    local EnvironmentPath="$1"

    if [ -d "$EnvironmentPath" ]; then
        local EnvironmentName=$(basename "$EnvironmentPath")

        # Uninstall ZMQ
        echo ""
        echo "-------------------------------------------------------------------------------"
        echo "                   Uninstalling the inexplicably corrupted pyzmq"
        echo "-------------------------------------------------------------------------------"
        conda activate "$EnvironmentName" && conda uninstall --prefix "$EnvironmentPath" --yes pyzmq 2>&1

        # Reinstall ZMQ
        echo ""
        echo "-------------------------------------------------------------------------------"
        echo "                               Reinstalling pyzmq"
        echo "-------------------------------------------------------------------------------"
        conda activate "$EnvironmentName" && conda install --prefix "$EnvironmentPath" --yes pyzmq jupyterlab 2>&1
    else
        echo "The conda environment (${EnvironmentName}) is missing" >&2
    fi
}

Import_Workspace_File() {
    local EnvironmentPath="$1"
    local RepositoryPath="${2:-}"

    local WorkspacePath=""
    local JsonPath="${RepositoryPath}/workspace.json"

    if [ -f "$JsonPath" ]; then
        local ExecutablePath="${EnvironmentPath}/python.exe"
        local CommandString="${ExecutablePath} -m jupyterlab workspaces import ${JsonPath}"
        WorkspacePath=$(eval "$CommandString" 2>&1 | awk '{print $NF}')
    fi

    echo "$WorkspacePath"
}

Add_Logos_To_Kernel_Folder() {
    local EnvironmentName="$1"
    local RepositoryName="${2:-}"
    local RepositoriesDirectory="${3:-D:/mnt/c/Users/DaveBabbitt/Documents/GitHub}"

    local KernelFolder="${HOME}/.local/share/jupyter/kernels/${EnvironmentName}"
    local LogoFolder="${RepositoriesDirectory}/${RepositoryName}/saves/png"
    local SmallPath="${LogoFolder}/logo-32x32.png"

    if [ -f "$SmallPath" ]; then
        cp "$SmallPath" "$KernelFolder"
    fi

    local LargePath="${LogoFolder}/logo-64x64.png"

    if [ -f "$LargePath" ]; then
        cp "$LargePath" "$KernelFolder"
    fi
}

Get_Token_String() {
    local ListResults=$(python -m jupyter server list)
    local TokenString=$(echo "$ListResults" | grep -oP 'http://localhost:8888/\?token=\K[^ ]+')
    echo "$TokenString"
}

Get_Active_Conda_Environment() {
    local CondaInfo=$(conda info --envs)
    local ActiveEnvironment=$(echo "$CondaInfo" | awk '/\*/ {print $NF}' | rev | cut -d'\' -f1 | rev)
    [ -n "$ActiveEnvironment" ] && echo "$ActiveEnvironment" || echo "None"
}
