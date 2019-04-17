cd ..
cd navigator
cmd /k "%UserProfile%\myenv\Scripts\activate.bat & python navigator.py & %UserProfile%\myenv\Scripts\deactivate.bat & EXIT"
