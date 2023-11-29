
# Add the kernel to the Launcher
Write-Host ""
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
Write-Host "                        Adding the kernel to the Launcher" -ForegroundColor Green
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
Add-Python-Executable-To-Path $EnvironmentPath
Add-Kernel-To-Launcher $EnvironmentPath -DisplayName $DisplayName -Verbose
$KernelPath = "${HomeDirectory}\AppData\Roaming\jupyter\kernels\${EnvironmentName}\kernel.json"
If (Test-Path -Path $KernelPath -PathType Leaf) {
	Add-Logos-To-Kernel-Folder $EnvironmentName -RepositoryName $RepositoryName
	(Get-Content $KernelPath) | ConvertFrom-Json | ConvertTo-Json -depth 7 | Format-Json -Indentation 2
}

# Add a workspace file for bookmarking. You can create a temporary workspace file in the 
# $Env:UserProfile\.jupyter\lab\workspaces folder by going to this URL:
# http://localhost:8888/lab/?clone=$EnvironmentName
Write-Host ""
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
Write-Host "                        Importing the workspace file" -ForegroundColor Green
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
$WorkspacePath = Import-Workspace-File $EnvironmentPath -RepositoryPath $RepositoryPath
If ($WorkspacePath -Ne $null) {
	If (Test-Path -Path $WorkspacePath -PathType Leaf) {
		(Get-Content $WorkspacePath) | ConvertFrom-Json | ConvertTo-Json -depth 7 | Format-Json -Indentation 2
	}
} else {

	# Get the path to the Jupyter workspaces folder
	$WorkspacesFolder = "$Env:UserProfile\.jupyter\lab\workspaces"

	# Check if the folder exists
	if (Test-Path $WorkspacesFolder) {
		
		# Empty the folder
		Remove-Item $WorkspacesFolder -Recurse -Force
		if (-not (Test-Path $WorkspacesFolder)) {
			 New-Item -ItemType Directory -Path $WorkspacesFolder
		}
		
		# Create the workspace
		Start-Process "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" -ArgumentList "http://localhost:8888/lab/?clone=${EnvironmentName}"
		
		# Get a list of the files in the workspaces folder
		$FilesList = Get-ChildItem $WorkspacesFolder
		
		# Loop through the files list
		if ($FilesList -ne $null) {
			foreach ($FileObj in $FilesList) {
				Write-Host $FileObj.FullName
			}
		}
		
	}
	
}

# Clean up the mess
# (Deprecated) Updating extensions with the jupyter labextension
# update command is now deprecated and will be 
# removed in a future major version of JupyterLab.
<#
Write-Host ""
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
Write-Host "                          Cleaning the staging area" -ForegroundColor Green
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green

# jupyter-lab clean
$CommandString = "${EnvironmentPath}\Scripts\jupyter-lab.exe clean"
Write-Verbose "CommandString = '${CommandString}'" -Verbose
Invoke-Expression $CommandString

# jupyter labextension list
$CommandString = "${EnvironmentPath}\Scripts\jupyter-labextension.exe list"
Write-Verbose "CommandString = '${CommandString}'" -Verbose
$ExtensionsList = Invoke-Expression $CommandString
if (!($ExtensionsList -Like "*No installed extensions*")) {
	Write-Host ""
	Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
	Write-Host "                     Updating the Jupyter Lab extensions" -ForegroundColor Green
	Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
	# jupyter labextension update --all
	$CommandString = "${EnvironmentPath}\Scripts\jupyter-labextension.exe update --all"
	Invoke-Expression $CommandString
}
Write-Host ""
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
Write-Host "                       Rebuilding the Jupyter Lab assets" -ForegroundColor Green
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green

#,"${HomeDirectory}\anaconda3\etc\jupyter","C:\ProgramData\jupyter"
$ConfigFoldersList = @("${HomeDirectory}\.jupyter")
ForEach ($ConfigFolder in $ConfigFoldersList) {
	$OldConfigPath = "${ConfigFolder}\old_jupyter_notebook_config.py"
	If (Test-Path -Path $OldConfigPath -PathType Leaf) {
		Read-Host "You better rescue your old_jupyter_notebook_config.py in the ${ConfigFolder} folder; we are about to overwrite it. Then press ENTER to continue..."
	}
	$NewConfigPath = "${ConfigFolder}\jupyter_notebook_config.py"
	Copy-Item $NewConfigPath -Destination $OldConfigPath
	$ConfigPath = "${RepositoriesDirectory}\${RepositoryName}\jupyter_notebook_config.py"
	If (Test-Path -Path $ConfigPath -PathType Leaf) {
		Copy-Item $ConfigPath -Destination $NewConfigPath
	}
}

# jupyter-lab build
$CommandString = "${EnvironmentPath}\Scripts\jupyter-lab.exe build"
Write-Verbose "CommandString = '${CommandString}'" -Verbose
Invoke-Expression $CommandString
#>


# Copy the favicon asset to the static directory
$IconPath = "${EnvironmentPath}\saves\ico\notebook_static_favicon.ico"
If (Test-Path -Path $IconPath -PathType Leaf) {
	$FaviconsFoldersList = @("${HomeDirectory}\anaconda3\share\jupyter\lab\static\favicons","${EnvironmentPath}\share\jupyter\lab\static\favicons")
	ForEach ($FaviconsFolder in $FaviconsFoldersList) {
		$NewIconPath = "${FaviconsFolder}\favicon.ico"
		If (!(Test-Path -Path $NewIconPath -PathType Leaf)) {
			If (!(Test-Path -Path $FaviconsFolder -PathType Container)) {
				New-Item -ItemType Directory -Path $FaviconsFolder
			}
			Copy-Item $IconPath -Destination $NewIconPath
		}
	}
}

cd $OldPath