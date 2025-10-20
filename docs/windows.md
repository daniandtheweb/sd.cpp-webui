# Setup instructions for Windows

## Dependencies:

- Python 3.13 (3.14 is currently unsupported)
- Git (Recommended)

### Python install:

1. Open Terminal (PowerShell)

2. Paste this and press enter to install Python 3.13:

    ```powershell
    winget install --id=Python.Python.3.13  -e
    ```

### Git install (optional):

1. Open Terminal (PowerShell)

2.  Paste this and press enter to install Git:
    ```powershell
    winget install --id=Git.Git  -e
    ```

## Setup:

- Using Terminal (PowerShell):

0. Ensure that execution of scripts is allowed (this is just required to run the `sdcpp_webui_windows.ps1` script; I'm not responsable for any other scripts that you may run in your system. You're encouraged to ALWAYS read the content of a script, even this, before executing it).

   - Run PowerShell as an Administrator and execute this:
       ```powershell
       Set-ExecutionPolicy Unrestricted -Scope CurrentUser
       ```
   - Close PowerShell Administrator.

1.  Get the code using one of the following methods:
- 1A: Clone the repository with Git (recommended):

    - Open Terminal (PowerShell) in a location of your choice and run:
        ```powershell
        git clone https://github.com/daniandtheweb/sd.cpp-webui
        ```

- 1B: Manual Download

    - [Download](https://www.google.com/search?q=https://github.com/daniandtheweb/sd.cpp-webui/archive/refs/heads/main.zip) and extract in a location of your choice (e.g., your Desktop).

2. Move inside of `sd.cpp-webui`'s directory:
    ```powershell
    cd sd.cpp-webui
    ```

3. Execute the launch script, this will automatically create a virtual environment and install all the necessary Python packages:
    ```powershell
    .\sdcpp_webui_windows.ps1 --autostart
    ```

- To see all available arguments and options, run:
    ```powershell
    .\sdcpp_webui_windows.ps1 --help
    ```

## Updating the WebUI:

1. Open Terminal (PowerShell) inside of `sd.cpp-webui`'s directory and execute this:
    ```powershell
    git pull
    ```