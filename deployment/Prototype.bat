cd ..
cd main
cmd /k "%UserProfile%\myenv\Scripts\activate.bat & python mco.py & python ../navigator/navigator.py & %UserProfile%\myenv\Scripts\deactivate.bat & EXIT"
cd ..
cd deployment