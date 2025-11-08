import math
import random
from itertools import combinations
from collections import defaultdict 
import numpy as np
from scipy.ndimage import median_filter

class Human_1:

    def __init__(self, color, min_tolerance, tolerance):
        self.tolerance = tolerance          #percentuale massima di vicini diversi
        self.min_tolerance = min_tolerance
        self.color = color
        self.wanna_move = False
        self.empty = (255,255,255)
        self.contact = 0.1
        self.conflict = 0.3
        self.movements=0
        self.asterisk=False #marcatore di adattamento
        self.completely_isolated=False
        
    def compareNeighbourhood(self, neighbourhood):
        """if self!=self.empty and self.movements > 700: #dopo troppi traslochi si rassegna
            self.wanna_move = False
            self.tolerance = 1
            self.asterisk = True
            
            return"""
        # Reset asterisk se l'agente si muove
        #if self.wanna_move:
         #   self.asterisk = False
        if self.color != self.empty:
            countColor = 0
            countOccupied = 0
            for neighbour in neighbourhood:
                if (neighbour.color !=self.empty):
                    countOccupied+=1
                    if (self.color == neighbour.color):
                        countColor+=1
    
            if countOccupied == 0:
                # Nessun vicino: non ha senso spostarsi
                self.wanna_move = False
                self.completely_isolated=True
                return
            
            self.completely_isolated = False
            similarity = countColor / countOccupied
                  
    
            #non mi muovo, perfetta similarità
            if abs(similarity - 1)<1e-6:
                self.contact = max(0.1, self.contact - 0.05) #modificato da 0.01 a 0.1
                self.conflict = min(0.2, self.conflict + 0.1) #solo se agente fortemente isolato
                self.wanna_move = False
            #mi muovo, sono sotto la tolleranza
            elif similarity < (1 - self.tolerance):
                self.contact = min(0.3, self.contact + 0.03)
                self.conflict = min(0.4, self.conflict + 0.05)
                #self.tolerance += (self.contact - self.conflict) 
                self.wanna_move = True

            else:
                self.contact = min(0.4, self.contact + 0.05) #modificato da 0.01 a 0.1
                self.conflict = max(0.1, self.conflict - 0.05) #modificato da 0.05 a 0.2
                self.wanna_move = False
        
        else:
            pass
        
    def updateTolerance(self):
        
        if not self.wanna_move and not self.completely_isolated:
             # Modifica la tolleranza in base a contact e conflict
            self.tolerance += (self.contact - self.conflict) 
            
            
            # Limita il cambiamento della tolleranza: non può scendere sotto 0.25
            
        #else:
         #   self.conflict = min(0.4, self.conflict + 0.05)
          #  self.contact = min (0.3, self.contact + 0.05)
           #self.tolerance += (self.contact - self.conflict) 
        
        self.tolerance = max(self.min_tolerance, min(0.9, self.tolerance))
        
        
            
    def resetAfterMove(self):
        self.wanna_move = False
        self.asterisk = False  # se vuoi anche azzerare l'adattamento


        
    def getTolerance(self):
        return self.tolerance
    
    def getWanna_move(self):
        return self.wanna_move
    
    def getAsterisk(self):
        return self.asterisk

            
        
  

class Human_2(Human_1):

    def __init__(self, color, age, gender, min_tolerance, tolerance = 0.7):
        super().__init__(color, min_tolerance, tolerance)
        self.age = age
        self.gender = gender
        
        self.alpha, self.a = 1, 0.01 #gender  0.01
        self.beta, self.b = 1, 0.05 #age    0.05
        self.rinormalization = self.alpha*(1-self.a*self.gender)*self.beta*(1-self.b*self.age)
        self.tolerance = self.rinormalization*super().getTolerance() 
        
    def updateTolerance(self):
         
         if not self.wanna_move and not self.completely_isolated:
              # Modifica la tolleranza in base a contact e conflict
             self.tolerance += (self.contact - self.conflict) 
             
             
             # Limita il cambiamento della tolleranza: non può scendere sotto 0.25
             
        # else:
          #   self.conflict = min(0.4, self.conflict + 0.05)
           #  self.contact = min (0.3, self.contact + 0.05)
             #self.tolerance += (self.contact - self.conflict)
             
         self.tolerance = self.rinormalization*max(self.min_tolerance, min(0.9, self.tolerance))
         
    def getGender(self):
        return self.gender
    
    def getAge(self):
        return self.age



#preferences is a list of colors
class Human_3(Human_2):

    def __init__(self, color, tolerance, age, gender, min_tolerance, preferences):
        super().__init__(color, age, gender, min_tolerance, tolerance)
        self.preferences = preferences
    
    def compareNeighbourhood(self, neighbourhood):
        
        w_pref = 0.75
        countColor = 0
        countPreferences = 0
        countOccupied = 0
        for neighbour in neighbourhood:
            if neighbour.color != self.empty:
                countOccupied += 1
                if neighbour.color == self.color:
                    countColor += 1
                elif neighbour.color in self.preferences:
                    countPreferences +=1   


        if countOccupied == 0:
            # Nessun vicino: non ha senso spostarsi
            self.wanna_move = False
            self.completely_isolated = True
            return
        self.completely_isolated = False
        
        rate_preference = w_pref*countPreferences / countOccupied
        similarity = countColor / countOccupied
        threeshold = similarity + rate_preference
         

        if abs(threeshold+rate_preference-1)<1e-6: #threesholf = 1-rate_preferenceù
            self.contact = max(0.1, self.contact - 0.05) #modificato da 0.01 a 0.1
            self.conflict = min(0.2, self.conflict + 0.1) #solo se agente fortemente isolato
            self.wanna_move = False
        elif similarity + rate_preference < (1 - self.tolerance):
            self.contact = min(0.3, self.contact + 0.03)
            self.conflict = min(0.4, self.conflict + 0.05)
            #self.tolerance += (self.contact - self.conflict) 
            self.wanna_move = True
        else:
            self.contact = min(0.9, self.contact + 0.1)
            self.conflict = max(0.1, self.conflict - 0.2)
            self.wanna_move = False




        
