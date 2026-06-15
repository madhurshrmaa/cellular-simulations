## How to run this code

First, download the project folder and extract the ZIP files to your desired directory.

Second, open a terminal and navigate to the project directory:

```bash
cd path/to/project
```

Then, activate the Virtual Environment:

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

If you are using **Windows Terminal**, the activation command depends on the shell you are running:

**PowerShell**

```powershell
.\venv\Scripts\Activate.ps1
```

**Command Prompt (cmd)**

```cmd
venv\Scripts\activate.bat
```

**Git Bash**

```bash
source venv/Scripts/activate
```

To verify if venv activated or not, you can check if the virtual environment name (e.g., `(venv)`) appears at the beginning of your terminal prompt.

Install the following Python packages:

```bash
pip install -r requirements.txt
```

Finally, execute the desired script. For example:

```bash
python comparison_ipd.py
```

If your system requires `python3` instead of `python`, use:

```bash
python3 comparison_ipd.py
```

#### Troubleshooting

* Ensure Python 3 is installed and available in your system PATH.
* Verify that the virtual environment (`venv`) has been created before attempting to activate it.
* If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

and then try activating the environment again.
