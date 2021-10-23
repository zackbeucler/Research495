from threading import*
import time

"""
# list all active threads
for thread in threading.enumerate():
    print(thread.name)
"""

class Workstation():
    def __init__(self, name):
        #threading.Thread.__init__(self,name='wk'+str(i))
        self.name = name  #workstation number
        self.active = False   #When UV-C strip is on
        self.in_use = False     #The person is using the workstation or not
        self.ready = False      #ready for sanitization
        self.dirty = False      #keyboard status
        self.re_enter = False    #See if the person re-entered or not
        self.in_wait = False
        self.waited = False
        
    def cleanThread(self):
        idk = Thread(target=self.clean)
        idk.start()
        
    def waitThread(self):
        idk = Thread(target=self.wait)
        idk.start()
        
    def clean(self): # this method needs a thread
        self.active = True
        self.dirty = False
        self.ready = False
        self.waited = False
        #Red_LED.on() #TURN ON RED LED
        #Green_LED.off() # TURN OFF GREEN LED
        print("Cleaning ------------------workstation"+str(self.name))
        #LED_Strip.off() #this is opposite
        time.sleep(10) #Turn on UV-C # clean for 60 seconds
        #LED_Strip.on() #Turn off UV-C
        print("Workstation is now clean -------------------workstation"+str(self.name))
        self.active = False
        
        
    def wait(self): # this method needs a thread
        print("waiting **************workstation"+str(self.name))  #wait to see if the person really left
        self.in_wait = True
        self.waited = False
        time.sleep(10)
        print("Checking for new person ***************workstation"+str(self.name))
        self.in_wait = False
        self.waited = True

        
        
    def getAllStatus(self):
        return 'dirty: %s  in_use: %s  ready: %s in_wait %s waited %s active %s' % (self.dirty, self.in_use, self.ready, self.in_wait, self.waited, self.active)
        
    def update(self, stat_name, boolean): 
        #Update flags & trigger run/cleaning
        if stat_name == "dirty":
            self.dirty = boolean
        elif stat_name == "ready":
            self.ready = boolean
        elif stat_name == "in_use":
            self.in_use = boolean
        elif stat_name == "re_enter":
            self.re_enter = boolean
        elif stat_name == "in_wait":
            self.in_wait = boolean
        elif stat_name == "waited":
            self.waited = boolean
        
        
        
        
        
    def getStatus(self,stat_name): #stat_name is which status u want to get Ex)dirty
        if stat_name == "dirty":
            return self.dirty
        elif stat_name == "ready":
            return self.ready
        elif stat_name == "in_use":
            return self.in_use
        elif stat_name == "re_enter":
            return self.re_enter
        elif stat_name == "in_wait":
            return self.in_wait
        elif stat_name == "waited":
            return self.waited
        #print("Please type in one of the status: dirty, ready, in_use, re_enter")

        
        
        
        
        
        
        
        
        
        
        
        
    
#   #will sanitize when the workstation is ready
#     def run(self):
#     
#         if self.ready:
#             self.active = True
#             #Red_LED.on() #TURN ON RED LED
#             #Green_LED.off() # TURN OFF GREEN LED
#             print("Cleaning...")
#             #LED_Strip.off() #this is opposite
#             time.sleep(10) #Turn on UV-C # clean for 60 seconds
#             #LED_Strip.on() #Turn off UV-C
#             print("Workstation is now clean")
#             self.active = False
#             self.dirty = False
#             self.ready = False
#             return
# 
# 
# 
#   #status = {"dirty":False, "ready":False, "in_use":False}"
#     def update(self, stat_name, boolean): 
#         #Update flags & trigger run/cleaning
#         if stat_name == "dirty":
#             self.dirty = boolean
#         elif stat_name == "ready":
#             self.ready = boolean
#         elif stat_name == "in_use":
#             self.in_use = boolean
#         elif stat_name == "re_enter":
#             self.re_enter = boolean
#         print('updating')
#       
#       
#     def getAllStatus(self):
#         return 'dirty: %s  in_use: %s  ready: %s' % (self.dirty, self.in_use, self.ready)
#   
#   
#     def getStatus(self,stat_name): #stat_name is which status u want to get Ex)dirty
#         if stat_name == "dirty":
#             return self.dirty
#         elif stat_name == "ready":
#             return self.ready
#         elif stat_name == "in_use":
#             return self.in_use
#         elif stat_name == "re_enter":
#             return self.re_enter
#         print("Please type in one of the status: dirty, ready, in_use, re_enter")
# 
# 
#   #check if the person re-enters the room
#     def check(self):
#         print("waiting...")  #wait to see if the person really left
#         time.sleep(10)
#         print("Checking for new person")
#     

          

          

  


