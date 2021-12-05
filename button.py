
from graphics import *
class Button:
    """A button is a labeled rectangle in a window.  It is enabled
    or disabled with the activate() and deactivate() methods.  The
    clicked(pt) method will return True if and only if the button
    is enabled and pt is inside it."""

    #constructor method
    #called when a Button object is created
    #e.g. button1 = Button(win, centerPoint, width, height, text)
    def __init__(self, win, centerPt, width, height, text):
        """Creates a rectangular button, where:
        win is the GraphWin object where the button will be drawn,
        center is a Point object the button will be centered on
        width is an integer specifying the width of the button
        height is an integer specifying the height of the button
        text is a string that will appear on the button"""
        self.xmin = centerPt.getX() - width/2 #instance var
        self.xmax = centerPt.getX() + width/2 #instance var
        self.ymin = centerPt.getY() - height/2 #instance var
        self.ymax = centerPt.getY() + height/2 #instance var
        p1 = Point(self.xmin, self.ymax)
        p2 = Point(self.xmax, self.ymin)
        self.rect = Rectangle(p1,p2)    #instance var
        self.rect.draw(win)             #instance var
        self.rect.setFill('white')  #instance var
        self.label = Text(centerPt,text.upper()) #instance var
        self.label.draw(win)            #instance var
        self.active = True
        
    def activate(self):
        """Sets this button to 'active'"""
        #(re)color the text black
        self.label.setFill("black")
        #set the outline to look bolder
        self.rect.setWidth(3)
        #set the boolean flag (attribute) that tracks "active"-ness to True
        self.active = True

    def deactivate(self):
        """Sets this button to 'de-active' - unclickable"""
        #(re)color the text gray
        self.label.setFill("gray")
        #set the outline to look thin
        self.rect.setWidth(1)
        #set the boolean flag (attribute) that tracks "active"-ness to True
        self.active = False

    def isClicked(self,pt):
        if self.active:
            return (self.activate and self.xmin <= pt.getX() <= self.xmax and self.ymin <= pt.getY() <= self.ymax)
        else:
            return False

    def undraw(self):
        self.rect.undraw()
        self.label.undraw()
        
    


