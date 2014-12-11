powershell copy-item $env:RECIPE_DIR\..\..\* $env:SRC_DIR -recurse
"%PYTHON%" setup.py clean
"%PYTHON%" setup.py install
if errorlevel 1 exit 1
