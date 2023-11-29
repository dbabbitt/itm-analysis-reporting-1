# Import the required namespace
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public class PopupWindowUtils
{
    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern int GetWindowText(IntPtr hWnd, System.Text.StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern int GetClassName(IntPtr hWnd, System.Text.StringBuilder lpClassName, int nMaxCount);
}
"@

# Define the callback function for enumerating windows
function EnumWindowsCallback($hWnd, $lParam) {
    # Get window title
    $titleBuilder = New-Object System.Text.StringBuilder(256)
    [PopupWindowUtils]::GetWindowText($hWnd, $titleBuilder, $titleBuilder.Capacity) | Out-Null
    $title = $titleBuilder.ToString()

    # Get window class name
    $classNameBuilder = New-Object System.Text.StringBuilder(256)
    [PopupWindowUtils]::GetClassName($hWnd, $classNameBuilder, $classNameBuilder.Capacity) | Out-Null
    $className = $classNameBuilder.ToString()

    # Check if the window is visible and has a non-empty title
    if ([PopupWindowUtils]::IsWindowVisible($hWnd) -and ![string]::IsNullOrWhiteSpace($title)) {
        $windowInfo = @{
            "Handle" = $hWnd
            "Title" = $title
            "ClassName" = $className
        }
        $windowList = [System.Runtime.InteropServices.Marshal]::GetObjectForIUnknown($lParam)
        $windowList.Add($windowInfo) | Out-Null
    }

    return $true
}

# Create a list to store window information
$popupWindows = New-Object System.Collections.Generic.List[Hashtable]

# Enumerate all desktop windows
$desktopHandle = [IntPtr]::Zero
$enumWindowsCallback = [PopupWindowUtils]::EnumWindowsProc
$enumWindowsCallbackDelegate = [System.Runtime.InteropServices.Marshal]::GetFunctionPointerForDelegate($enumWindowsCallback)
$enumWindowsCallbackDelegateIntPtr = [System.Runtime.InteropServices.Marshal]::AllocHGlobal([IntPtr]::Size)
[System.Runtime.InteropServices.Marshal]::WriteIntPtr($enumWindowsCallbackDelegateIntPtr, $enumWindowsCallbackDelegate)
[System.Runtime.InteropServices.Marshal]::WriteIntPtr($enumWindowsCallbackDelegateIntPtr, [System.Runtime.InteropServices.Marshal]::GetIUnknownForObject($popupWindows))

[PopupWindowUtils]::EnumWindows($enumWindowsCallbackDelegateIntPtr, $desktopHandle) | Out-Null

[System.Runtime.InteropServices.Marshal]::Release($enumWindowsCallbackDelegateIntPtr)

# Output the list of popup windows
$popupWindows
