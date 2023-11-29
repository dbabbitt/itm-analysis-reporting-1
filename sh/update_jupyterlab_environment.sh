#!/bin/bash

# Add the kernel to the Launcher
echo ""
echo "-------------------------------------------------------------------------------" 
echo "                        Adding the kernel to the Launcher"
echo "-------------------------------------------------------------------------------"
Add_Python_Executable_To_Path "$EnvironmentPath"
Add_Kernel_To_Launcher "$EnvironmentPath" -DisplayName "$DisplayName" -Verbose
KernelPath="${HomeDirectory}/.local/share/jupyter/kernels/${EnvironmentName}/kernel.json"
if [ -f "$KernelPath" ]; then
    Add_Logos_To_Kernel_Folder "$EnvironmentName" -RepositoryName "$RepositoryName"
    cat "$KernelPath" | jq .
fi

# Add a workspace file for bookmarking. You can create a temporary workspace file in the 
# $HOME/.jupyter/lab/workspaces folder by going to this URL:
# http://localhost:8888/lab/?clone=$EnvironmentName
echo ""
echo "-------------------------------------------------------------------------------" 
echo "                        Importing the workspace file"
echo "-------------------------------------------------------------------------------"
WorkspacePath=$(Import-Workspace-File "$EnvironmentPath" -RepositoryPath "$RepositoryPath")
if [ "$WorkspacePath" != "null" ]; then
    if [ -f "$WorkspacePath" ]; then
        cat "$WorkspacePath" | jq .
    fi
else
    WorkspacesFolder="$HOME/.jupyter/lab/workspaces"
    if [ -d "$WorkspacesFolder" ]; then
        rm -rf "$WorkspacesFolder"
        mkdir -p "$WorkspacesFolder"
        start "http://localhost:8888/lab/?clone=${EnvironmentName}"
        FilesList=("$WorkspacesFolder"/*)
        for FileObj in "${FilesList[@]}"; do
            echo "$FileObj"
        done
    fi
fi

# Clean up the mess
# (Deprecated) Updating extensions with the jupyter labextension
# update command is now deprecated and will be 
# removed in a future major version of JupyterLab.
: '
echo ""
echo "-------------------------------------------------------------------------------"
echo "                          Cleaning the staging area"
echo "-------------------------------------------------------------------------------"

jupyter-lab clean
# CommandString="${EnvironmentPath}/Scripts/jupyter-lab clean"
# echo "CommandString = '${CommandString}'"
# eval "$CommandString"

# jupyter labextension list
CommandString="${EnvironmentPath}/Scripts/jupyter-labextension list"
echo "CommandString = '${CommandString}'"
ExtensionsList=$(eval "$CommandString")
if [[ ! "$ExtensionsList" =~ "No installed extensions" ]]; then
    echo ""
    echo "-------------------------------------------------------------------------------"
    echo "                     Updating the Jupyter Lab extensions"
    echo "-------------------------------------------------------------------------------"
    # jupyter labextension update --all
    CommandString="${EnvironmentPath}/Scripts/jupyter-labextension update --all"
    eval "$CommandString"
fi
echo ""
echo "-------------------------------------------------------------------------------"
echo "                       Rebuilding the Jupyter Lab assets"
echo "-------------------------------------------------------------------------------"

#,"${HomeDirectory}/anaconda3/etc/jupyter","C:\ProgramData\jupyter"
ConfigFoldersList=("${HomeDirectory}/.jupyter")
for ConfigFolder in "${ConfigFoldersList[@]}"; do
    OldConfigPath="${ConfigFolder}/old_jupyter_notebook_config.py"
    if [ -f "$OldConfigPath" ]; then
        read -p "You better rescue your old_jupyter_notebook_config.py in the ${ConfigFolder} folder; we are about to overwrite it. Then press ENTER to continue..."
    fi
    NewConfigPath="${ConfigFolder}/jupyter_notebook_config.py"
    cp "$NewConfigPath" "$OldConfigPath"
    ConfigPath="${RepositoriesDirectory}/${RepositoryName}/jupyter_notebook_config.py"
    if [ -f "$ConfigPath" ]; then
        cp "$ConfigPath" "$NewConfigPath"
    fi
done

# jupyter-lab build
CommandString="${EnvironmentPath}/Scripts/jupyter-lab build"
echo "CommandString = '${CommandString}'"
eval "$CommandString"
'

# Copy the favicon asset to the static directory
IconPath="${EnvironmentPath}/saves/ico/notebook_static_favicon.ico"
if [ -f "$IconPath" ]; then
    FaviconsFoldersList=("$HOME/anaconda3/share/jupyter/lab/static/favicons" "$EnvironmentPath/share/jupyter/lab/static/favicons")
    for FaviconsFolder in "${FaviconsFoldersList[@]}"; do
        NewIconPath="${FaviconsFolder}/favicon.ico"
        if [ ! -f "$NewIconPath" ]; then
            mkdir -p "$FaviconsFolder"
            cp "$IconPath" "$NewIconPath"
        fi
    done
fi

cd "$OldPath"