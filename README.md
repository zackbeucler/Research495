# How to run program for COM495 on a Raspberry Pi

[Tutorial Link](https://www.youtube.com/watch?v=aimSGOAUI8Y)

## First install

1. Open the terminal on the raspberry pi and type the command `sudo apt-get update`
2. When finished, type `sudo apt-get upgrade`
3. Check that the Camera option in the raspberry pi configuration menu is enabled
4. Then, type `cd Desktop` to move to the Desktop folder
4. Download this repo by typing `git clone https://github.com/zackbeucler/Research495.git`
5. After the download finsihes, type `mv Research495 tflite1 && cd tflite1` to move to the TensorFlow folder
6. Then, type `sudo pip3 install virtualenv` to install virtual enviroments for Python3
7. Next, create a virtual enviroment by typing `python3 -m venv tflite1-env` to create an enviroment called `tflite1-env`
8. Start that enviroment by typing `source tflite1-env/bin/activate`
9. Download additional requirements but typing `bash get_pi_requirements.sh`
10. Run the detection program by typing `python3 TFLite_detecton_webcam.py --modeldir=Sample_TFLite_model`
11. You can quit the program by typing `q` when it's running

## After first install

1. Open terminal and type `cd Desktop/tflite1`
2. Run the virtual enviroment by typing `source tflite1-env/bin/activate`
3. Run the program by typing `python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model`

## Git stuff

1. Add all files `git add .`
2. Commit files `git commit -m "message here"`
3. push files `git push origin main`

## File info
- `webcam_updated.py` is the final code from last semester where we only had a single work station
- `webcam_updated2.py` is the code we are working on this semester for multiple workstations
- `workstation.py` contains the workstation class

Last updated: March 22, 2021
