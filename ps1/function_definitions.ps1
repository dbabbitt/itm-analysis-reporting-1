
function Format-Json {
	<#
	.SYNOPSIS
		Prettifies JSON output.
	.DESCRIPTION
		Reformats a JSON string so the output looks better than what ConvertTo-Json outputs.
	.PARAMETER Json
		Required: [string] The JSON text to prettify.
	.PARAMETER Minify
		Optional: Returns the json string compressed.
	.PARAMETER Indentation
		Optional: The number of spaces (1..1024) to use for indentation. Defaults to 4.
	.PARAMETER AsArray
		Optional: If set, the output will be in the form of a string array, otherwise a single string is output.
	.EXAMPLE
		$json | ConvertTo-Json | Format-Json -Indentation 2
	#>
	[CmdletBinding(DefaultParameterSetName = 'Prettify')]
	Param(
		[Parameter(Mandatory = $true, Position = 0, ValueFromPipeline = $true)]
		[string]$Json,

		[Parameter(ParameterSetName = 'Minify')]
		[switch]$Minify,

		[Parameter(ParameterSetName = 'Prettify')]
		[ValidateRange(1, 1024)]
		[int]$Indentation = 4,

		[Parameter(ParameterSetName = 'Prettify')]
		[switch]$AsArray
	)

	if ($PSCmdlet.ParameterSetName -eq 'Minify') {
		return ($Json | ConvertFrom-Json) | ConvertTo-Json -Depth 100 -Compress
	}

	# If the input JSON text has been created with ConvertTo-Json -Compress
	# then we first need to reconvert it without compression
	if ($Json -notmatch '\r?\n') {
		$Json = ($Json | ConvertFrom-Json) | ConvertTo-Json -Depth 100
	}

	$indent = 0
	$regexUnlessQuoted = '(?=([^"]*"[^"]*")*[^"]*$)'

	$result = $Json -split '\r?\n' |
		ForEach-Object {
			# If the line contains a ] or } character, 
			# we need to decrement the indentation level unless it is inside quotes.
			if ($_ -match "[}\]]$regexUnlessQuoted") {
				$indent = [Math]::Max($indent - $Indentation, 0)
			}

			# Replace all colon-space combinations by ": " unless it is inside quotes.
			$line = (' ' * $indent) + ($_.TrimStart() -replace ":\s+$regexUnlessQuoted", ': ')

			# If the line contains a [ or { character, 
			# we need to increment the indentation level unless it is inside quotes.
			if ($_ -match "[\{\[]$regexUnlessQuoted") {
				$indent += $Indentation
			}

			$line
		}

	if ($AsArray) { return $result }
	return $result -Join [Environment]::NewLine
}

function Add-Python-Executable-To-Path {
	<#
	.SYNOPSIS
		Adds python's executable path associated with the environment to the PATH if necessary.
	.EXAMPLE
		Add-Python-Executable-To-Path $EnvironmentPath
	#>
	[CmdletBinding(DefaultParameterSetName = 'EnvironmentPath')]
	Param(
		[Parameter(Mandatory = $true, Position = 0, ValueFromPipeline = $true)]
		[string]$EnvironmentPath
	)
	$ExecutablePath = "${EnvironmentPath}\python.exe"
	$PathArray = $ExecutablePath -Split "\\"
	$PythonFolder = $PathArray[0..($PathArray.count - 2)] -Join "\"
	if (!($env:Path -Like "*$PythonFolder*")) {
		Write-Host "The ${EnvironmentPath} executable path is not in PATH" -ForegroundColor Red
		$env:Path = "${EnvironmentPath};" + $env:Path
	}
}

function Get-Python-Version {
	<#
	.SYNOPSIS
		Get the version of python associated with the environment.
	.EXAMPLE
		$PythonVersion = Get-Python-Version $EnvironmentPath
	#>
	[CmdletBinding(DefaultParameterSetName = 'EnvironmentPath')]
	Param(
		[Parameter(Mandatory = $true, Position = 0, ValueFromPipeline = $true)]
		[string]$EnvironmentPath
	)
	$CommandString = "cd ${EnvironmentPath} & python --version"
	$PythonVersion = cmd /c $CommandString '2>&1'
	$PythonVersion = $PythonVersion.Trim()
	
	Return $PythonVersion
}

function Add-Kernel-To-Launcher {
	<#
	.SYNOPSIS
		Adds python's executable path associated with the environment to the PATH if necessary.
	.EXAMPLE
		Add-Kernel-To-Launcher $EnvironmentPath -DisplayName $DisplayName -Verbose
	#>
	[CmdletBinding(DefaultParameterSetName = 'EnvironmentPath')]
	Param(
		[Parameter(Mandatory = $true, Position = 0, ValueFromPipeline = $true)]
		[string]$EnvironmentPath,
		[Parameter(ParameterSetName = 'DisplayName')]
		[string]$DisplayName = "Python"
	)
	$PythonVersion = Get-Python-Version $EnvironmentPath
	
	# Fix LookupError: unknown encoding: cp65001
	$CommandString = -Join('cd ', $EnvironmentPath, ' & python -c "import os; os.environ[', "'PYTHONIOENCODING'", "] = 'UTF-8'", '"')
	Write-Verbose "CommandString = '${CommandString}'" -Verbose
	cmd /c $CommandString '2>&1'
	
	$PathArray = $EnvironmentPath -Split "\\"
	$EnvironmentName = $PathArray[$PathArray.count - 1]
	$CommandString = -Join('cd ', $EnvironmentPath, ' & conda activate ', $EnvironmentName, ' & python -m ipykernel install --user --name ', $EnvironmentName, ' --display-name "', $DisplayName, ' (', $PythonVersion, ')"')
	Write-Verbose "CommandString = '${CommandString}'" -Verbose
	cmd /c $CommandString '2>&1'
	# Invoke-Expression $CommandString
}

function Reinstall-ZMQ {
	<#
	.SYNOPSIS
		Uninstalls and reinstalls the inexplicably corrupted pyzmq.
	.EXAMPLE
		Reinstall-ZMQ $EnvironmentPath -Verbose
	#>
	[CmdletBinding(DefaultParameterSetName = 'EnvironmentPath')]
	Param(
		[Parameter(Mandatory = $true, Position = 0, ValueFromPipeline = $true)]
		[string]$EnvironmentPath
	)
	
	# Assume here that if the environment folder is missing, the environment was already deleted
	If (Test-Path -Path $EnvironmentPath -PathType Container) {
		
		# Get the environment name
		$PathArray = $EnvironmentPath -Split "\\"
		$EnvironmentName = $PathArray[$PathArray.count - 1]
		
		# Uninstall ZMQ
		Write-Verbose ""
		Write-Verbose "-------------------------------------------------------------------------------" -Verbose
		Write-Verbose "                   Uninstalling the inexplicably corrupted pyzmq" -Verbose
		Write-Verbose "-------------------------------------------------------------------------------" -Verbose
		# $CommandString = -Join('cd ', $EnvironmentPath, ' & conda activate ', $EnvironmentName, ' & conda remove --prefix ', $EnvironmentPath, ' --force-remove --yes pyzmq')
		$CommandString = -Join('cd ', $EnvironmentPath, ' & conda activate ', $EnvironmentName, ' & conda uninstall --prefix ', $EnvironmentPath, ' --yes pyzmq')
		Write-Verbose "CommandString = '${CommandString}'" -Verbose
		cmd /c $CommandString '2>&1'
		# Invoke-Expression $CommandString
		
		# Reinstall ZMQ
		Write-Verbose ""
		Write-Verbose "-------------------------------------------------------------------------------" -Verbose
		Write-Verbose "                               Reinstalling pyzmq" -Verbose
		Write-Verbose "-------------------------------------------------------------------------------" -Verbose
		# $CommandString = -Join('cd ', $EnvironmentPath, ' & conda activate ', $EnvironmentName, ' & conda install --prefix ', $EnvironmentPath, ' --force-reinstall --yes pyzmq')
		$CommandString = -Join('cd ', $EnvironmentPath, ' & conda activate ', $EnvironmentName, ' & conda install --prefix ', $EnvironmentPath, ' --yes pyzmq jupyterlab')
		Write-Verbose "CommandString = '${CommandString}'" -Verbose
		cmd /c $CommandString '2>&1'
		# Invoke-Expression $CommandString
	} Else {
		Write-Host "The conda environment (${EnvironmentName}) if missing" -ForegroundColor Red
	}
}

function Import-Workspace-File {
	<#
	.SYNOPSIS
		Import the local workspace file into the Jupyter Lab workspaces.
	.DESCRIPTION
		Returns the newly created workspace path.
	.EXAMPLE
		$WorkspacePath = Import-Workspace-File $EnvironmentPath -RepositoryPath $RepositoryPath
	#>
	[CmdletBinding(DefaultParameterSetName = 'EnvironmentPath')]
	Param(
		[Parameter(Mandatory = $true, Position = 0, ValueFromPipeline = $true)]
		[string]$EnvironmentPath,
		
		[Parameter(ParameterSetName = 'RepositoryPath')]
		[string]$RepositoryPath
	)
	$WorkspacePath = $null
	$JsonPath = "${RepositoryPath}\workspace.json"
	If (Test-Path -Path $JsonPath -PathType Leaf) {
		$ExecutablePath = "${EnvironmentPath}\python.exe"
		$CommandString = "${ExecutablePath} -m jupyterlab workspaces import ${JsonPath}"
		Write-Verbose "CommandString = '${CommandString}'" -Verbose
		# $WorkspacePath = (jupyter-lab workspaces import workspace.json) | Out-String
		$WorkspacePath = cmd /c $CommandString '2>&1'
		# Write-Host "WorkspacePath = '${WorkspacePath}'" -Verbose
		If ($WorkspacePath -Ne $null) {
			$WorkspacePath = $WorkspacePath.Trim()
			$WorkspacePath = $WorkspacePath -csplit ' '
			$WorkspacePath = $WorkspacePath[$WorkspacePath.Count - 1]
		}
	}
	
	Return $WorkspacePath
}

function Add-Logos-To-Kernel-Folder {
	<#
	.SYNOPSIS
		Copy the 32x32 and 64x64 PNGs to the kernel folder if available.
	.EXAMPLE
		Add-Logos-To-Kernel-Folder $EnvironmentName -RepositoryName $RepositoryName
	#>
	Param(
		[Parameter(Mandatory = $true, Position = 0, ValueFromPipeline = $true)]
		[string]$EnvironmentName,
		
		[Parameter(ParameterSetName = 'RepositoryName')]
		[string]$RepositoryName,
		
		[Parameter(ParameterSetName = 'RepositoriesDirectory')]
		[string]$RepositoriesDirectory = "D:\Documents\GitHub"
	)
	$KernelFolder = "${Env:UserProfile}\AppData\Roaming\jupyter\kernels\${EnvironmentName}"
	$LogoFolder = "${RepositoriesDirectory}\${RepositoryName}\saves\png"
	$SmallPath = "${LogoFolder}\logo-32x32.png"
	If (Test-Path -Path $SmallPath -PathType Leaf) {
		Copy-Item $SmallPath -Destination $KernelFolder
	}
	$LargePath = "${LogoFolder}\logo-64x64.png"
	If (Test-Path -Path $LargePath -PathType Leaf) {
		Copy-Item $LargePath -Destination $KernelFolder
	}
}

function Get-Token-String {
	<#
	.SYNOPSIS
		Get the token string from the running Jupyter Lab.
	.EXAMPLE
		$TokenString = Get-Token-String
	#>
	$ListResults = (python -m jupyter server list) | Out-String
	$TokenRegex = [regex] '(?m)http://localhost:8888/\?token=([^ ]+) :: '
	$TokenString = $TokenRegex.Match($ListResults).Groups[1].Value
	
	Return $TokenString
}

function Get-Active-Conda-Environment {
	<#
	.SYNOPSIS
	Get the name of the active conda environment.

	.DESCRIPTION
	This function runs the `conda info --envs` command to get a list of all conda environments.
	It then splits the output of the command into a list of lines. For each line,
	the function checks if it contains an asterisk (*). If it does,
	the function splits the line on the asterisk and returns the name of the environment.
	If no line contains an asterisk, the function returns "None".

	.PARAMETER
	None.

	.OUTPUTS
	System.String. The name of the active conda environment.

	.EXAMPLE
		$ActiveEnvironment = Get-Active-Conda-Environment
	#>

	$CondaInfo = & conda info --envs
	$SplitLines = $CondaInfo.Split([Environment]::NewLine)
	foreach ($line in $SplitLines) {
		if ($line.Contains("*")) {
			# Write-Host "line: ${line}"
			$SplitInfo = $line.Split("*")
			$EnvironmentName = $SplitInfo[-1]
			$SplitInfo = $line.Split("\")
			$EnvironmentName = $SplitInfo[-1]
			return $EnvironmentName
		}
	}
	return "None"
}
