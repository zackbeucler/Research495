# How to run program for COM495 on a Raspberry Pi

[Tutorial Video Link](https://www.youtube.com/watch?v=aimSGOAUI8Y)

[Tutorial Written Link](https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md)

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
10. Then, Add coral package to your apt-get distro `echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - sudo apt-get update`
11. Next, install the libedgetpu library `sudo apt-get install libedgetpu1-std` or the over-clocked version `sudo apt-get install libedgetpu1-max` (CAN ONLY HAVE ONE)
12. Run the detection program by typing `python3 webcam_updated.py --modeldir=Sample_TFLite_model --edgetpu`
13. You can quit the program by typing `q` when it's running

## After first install

1. Open terminal and type `cd Desktop/tflite1`
2. Run the virtual enviroment by typing `source tflite1-env/bin/activate`
3. Run the program by typing `python3 webcam_updated.py --modeldir=Sample_TFLite_model`



Last updated: May 4th, 2021
