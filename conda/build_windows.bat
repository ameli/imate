@echo off
setlocal EnableDelayedExpansion

set wheel_installed=false
echo %wheel_installed%

set root_dir=%CD%\dist
echo %root_dir%

for /R %root_dir% %%i in (*.txt) do (
	if "!wheel_installed!"=="false" (
		echo Try installing %%i
		rem call python -m pip install --force-reinstall %%i > nul 2> nul
		rem if errorlevel equ 0 (
		if 0 equ 0 (
			echo SETTING TO TRUE
			set wheel_installed=true
		)
	)
	echo Wheel insatlled: !wheel_installed!
)

if %wheel_installed% equ false (
	echo No wheel could be installed.
	exit /b 1
)
