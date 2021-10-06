import threading
import time

class Workstation(threading.Thread):    
  def __init__(self, i):
    threading.Thread.__init__(self,name='wk'+str(i))
    self.number = i  #workstation number
    self.active = False   #When UV-C strip is on
    self.in_use = False     #The person is using the workstation or not
    self.ready = False      #ready for sanitization
    self.dirty = False      #keyboard status
    self.re_enter = False    #See if the person re-entered or not
    
  #will sanitize when the workstation is ready
  def run(self):
    for thread in threading.enumerate():
        print(thread.name)
    if self.ready:
      self.active = True
      
      #TURN ON RED LED
      #Red_LED.on()
      # TURN OFF GREEN LED
      #Green_LED.off()
      
      #Turn on UV-C # clean for 60 seconds
      print("Cleaning...")
      #LED_Strip.off() #this is opposite
      time.sleep(10)
      #Turn off UV-C
      #LET_Strip.on()
      
      print("Workstation is now clean")
      self.active = False
      self.ready = False
      self.dirty = False


  #status = {"dirty":False, "ready":False, "in_use":False}"
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
    print('updating')
      
  def getAllStatus(self):
      return 'dirty: %s  in_use: %s  ready: %s' % (self.dirty, self.in_use, self.ready)
  
  
  def getStatus(self,stat_name): #stat_name is which status u want to get Ex)dirty
    if stat_name == "dirty":
      return self.dirty
    
    elif stat_name == "ready":
      return self.ready

    elif stat_name == "in_use":
      return self.in_use

    elif stat_name == "re_enter":
      return self.re_enter  

    return print("Please type in one of the status: dirty, ready, in_use, re_enter")

  #check if the person re-enters the room
  def check(self):
    print("waiting...")  #wait to see if the person really left
    time.sleep(10)
    print("Checking for new person")
    

          

          

  