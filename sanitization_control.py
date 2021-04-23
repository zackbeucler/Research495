import time
#import os

counter = 0
while counter < 60:
    time.sleep(1) # wait 1 second
    counter += 1
    print(counter)
    #if detection == True:
    #    break