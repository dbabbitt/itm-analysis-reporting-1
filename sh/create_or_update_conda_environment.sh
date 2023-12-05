
#!/bin/bash

# Create the conda environment
cd "$RepositoryPath"

# Assume here that if the environment folder is missing, the environment was already deleted
echo ""
echo "--------------------------------------------------------------------------------"
if [ -d "$EnvironmentPath" ]; then
    echo -e "              \e[32mUpdating the ${DisplayName} conda environment (${EnvironmentName})\e[0m"
else
    echo -e "              \e[32mCreating the ${DisplayName} conda environment (${EnvironmentName})\e[0m"
fi
echo "--------------------------------------------------------------------------------"

# You can control where a conda environment lives by providing a path to a target directory when creating the environment.
YamlFilePath="${RepositoryPath}/sh_environment.yml"
if [ -f "$YamlFilePath" ]; then
    echo -e "\e[32mconda env update --prefix $EnvironmentPath --file $YamlFilePath --prune --quiet\e[0m"
    conda env update --prefix "$EnvironmentPath" --file "$YamlFilePath" --prune --quiet
fi

if conda info --envs | grep -q "$EnvironmentName"; then
    echo -e "\e[32mconda update --all --yes --name $EnvironmentName\e[0m"
    conda update --all --yes --name "$EnvironmentName"
else
    echo -e "\e[32mconda create --yes --name $EnvironmentName\e[0m"
    conda create --yes --name "$EnvironmentName"
fi
conda info --envs