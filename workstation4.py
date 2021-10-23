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

    #This function updates the status, wait, or clean 
    #For the parameter 'keyword' can take 'person','no_person,'wait','clean' to do actions in certain conditions
    def action(self, keyword):
        
        if keyword == 'person':
            print("Person at workstation*********************workstation"+str(self.name))
            self.update("in_use", True) # method update(dirty,ready,in_use,re_enter) has all boolean variables
            self.update('ready',False)
            self.update('waited', False)
        
        if keyword == 'no_person' and self.getStatus('in_use'):
            print("Workstation is dirty****************workstation"+str(self.name))
            self.update('dirty', True)
            self.update('in_use',False)
        
        if keyword == 'wait' and self.getStatus('dirty') and not self.getStatus('in_use') and not self.getStatus('active') and not self.getStatus('in_wait') and not self.getStatus('waited'):
            Thread(target=self.wait).start()

        
        if keyword == 'clean' and self.getStatus('ready') and self.getStatus('dirty') and not self.getStatus('in_use') and not self.getStatus('in_wait') and self.getStatus('waited') and not self.getStatus('active'):
            Thread(target=self.clean).start()
        
        
        
        
    


          

  



