cd /D %UserProfile%
py -3 -m pip install --upgrade virtualenv
virtualenv myenv
cmd /k "myenv\Scripts\activate.bat & pip install --upgrade cython numpy matplotlib scipy sympy pip wheel setuptools docutils pygments pypiwin32 kivy.deps.sdl2 kivy.deps.glew & pip install --upgrade kivy.deps.gstreamer & pip install --upgrade kivy & pip install --upgrade kivy-garden & garden install --upgrade graph & myenv\Scripts\deactivate.bat & EXIT"
cd Desktop\FORCE-BDSS-Prototype\force-bdss-plugin-itwm-example\deployment\setup_files
PAUSE
