
# Update Node.js
Write-Host ""
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
Write-Host "                Checking all NPM packages for potential updates" -ForegroundColor Green
Write-Host "-------------------------------------------------------------------------------" -ForegroundColor Green
$CommandString = "npm install -g npm"
Invoke-Expression $CommandString
$CommandString = "npm update -g"
Invoke-Expression $CommandString