
# Create the conda environment
cd $RepositoryPath
Write-Host ""
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Green

# Assume here that if the environment folder is missing, the environment was already deleted
If (Test-Path -Path $EnvironmentPath -PathType Container) {
	Write-Host " Updating the ${DisplayName} conda environment (${EnvironmentName})" -ForegroundColor Green
} Else {
	Write-Host " Creating the ${DisplayName} conda environment (${EnvironmentName})" -ForegroundColor Green
}
Write-Host "--------------------------------------------------------------------------------" -ForegroundColor Green

# You can control where a conda environment lives by providing a path to a target directory when creating the environment.
$FilePath = "${RepositoryPath}\environment.yml"
If (Test-Path $FilePath) {
	conda env update --prefix $EnvironmentPath --file $FilePath --prune --quiet
}

If (conda env list | findstr $EnvironmentName) {
	conda update --all --yes --name $EnvironmentName
} Else {
	conda create --yes --name $EnvironmentName python=3.7
}
conda info --envs