# How to run program for COM495 on a Raspberry Pi

## First install

1. Open the terminal on the raspberry pi and type the command `sudo apt-get update`
2. When finished, type `sudo apt-get upgrade`
3. Check that the Camera option in the raspberry pi configuration menu is enabled
4. Then, type `cd Desktop` to move to the Desktop folder
4. Download this repo by typing `git clone https://github.com/zackbeucler/Research495.git`
5. After the download finsihes, type `mv Research495 clean_code && cd clean_code` to move to the TensorFlow folder
6. Then, type `sudo pip3 install virtualenv` to install virtual enviroments for Python3
7. Next, create a virtual enviroment by typing `python3 -m venv env` to create an enviroment called `tflite1-env`
8. Start that enviroment by typing `source env/bin/activate`
9. Download additional requirements but typing `bash get_pi_requirements.sh`
10. Run the detection program by typing `python3 multi_wrks_detection_UI.py`
11. You can quit the program by typing `q` when it's running

## After first install

1. Open terminal and type `cd Desktop/clean_code`
2. Run the virtual enviroment by typing `source env/bin/activate`
3. Run the program by typing `python3 multi_wrks_detection_UI.py`

## Git stuff

1. Add all files `git add .`
2. Commit files `git commit -m "message here"`
3. push files `git push origin main`

## File info
- `multi_wrks_detection_UI.py` This file is for multiple workstations and has a UI to help setup
- `multi_wrks_detection.py` This file is for multiple workstations with no UI
- `workstation.py` contains the workstation class


## Special Thanks
- Prof. Tarimo
- Prof. Lee
- Evan Juras for the real-time detection [code](https://www.youtube.com/watch?v=aimSGOAUI8Y)

Last updated: December 5, 2021
