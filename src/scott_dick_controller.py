# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick
from pickle import FALSE

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt

class ScottDickController(KesslerController):
    
    
        
    def __init__(self):
        self.eval_frames = 0 #What is this?

        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        
        
        
        
        """ Aiden """
        asteroid_velocity = ctrl.Antecedent(np.arange(0, 90 + 1, 1), 'asteroid_velocity')
        asteroid_distance = ctrl.Antecedent(np.arange(0,500,1), 'asteroid_distance')
        #heading_theta = ctrl.Antecedent(np.arange(-361, 361, 1), 'heading_theta')
        asteroid_theta = ctrl.Antecedent(np.arange(-181, 181, 1), 'asteroid_theta')
        ship_speed = ctrl.Antecedent(np.arange(-240 - 1, 240 + 1, 1), 'ship_speed')
        ship_thrust = ctrl.Consequent(np.arange(-480 - 1, 480 + 1, 1), 'ship_thrust')
        
        ## Define triangle relations for fuzzy set
        ship_thrust['NL'] = fuzz.trimf(ship_thrust.universe, [-480, -480, -320])
        ship_thrust['NM'] = fuzz.trimf(ship_thrust.universe, [-480, -320, -160])
        ship_thrust['NS'] = fuzz.trimf(ship_thrust.universe, [-320, -160, 160])
        ship_thrust['NZ'] = fuzz.trimf(ship_thrust.universe, [-160, 0, 160])  # Zero fuzzy set
        ship_thrust['PS'] = fuzz.trimf(ship_thrust.universe, [-160, 160, 320])
        ship_thrust['PM'] = fuzz.trimf(ship_thrust.universe, [160, 320, 480])
        ship_thrust['PL'] = fuzz.trimf(ship_thrust.universe, [320, 480, 480])
        
        """ Antecedents """
        """ First Solution 
        # Define asteroid velocity relations
        asteroid_velocity['S'] = fuzz.trimf(asteroid_velocity.universe, [0, 22.5, 45]) 
        asteroid_velocity['M'] = fuzz.trimf(asteroid_velocity.universe, [22.5, 45, 67.5]) 
        asteroid_velocity['L'] = fuzz.trimf(asteroid_velocity.universe, [67.5, 90, 90]) 

        # Define asteroid ship distance x relations
        asteroid_distance['S'] = fuzz.trimf(asteroid_distance.universe, [0, 62.5, 125])
        asteroid_distance['M'] = fuzz.trimf(asteroid_distance.universe, [62.5, 187.5, 312.5])
        asteroid_distance['L'] = fuzz.trimf(asteroid_distance.universe, [187.5, 375, 500])
        
        asteroid_theta['NFA'] = fuzz.trimf(asteroid_theta.universe, [-180, -180, -135]) 
        asteroid_theta['NCA'] = fuzz.trimf(asteroid_theta.universe, [-135, -90, -45]) 
        asteroid_theta['T'] = fuzz.trimf(asteroid_theta.universe, [-90, 0, 90])
        asteroid_theta['PCA'] = fuzz.trimf(asteroid_theta.universe, [45, 90, 135])
        asteroid_theta['PFA'] = fuzz.trimf(asteroid_theta.universe, [135, 180, 180])
        """
        
        """ Second Solution """
        
        # Define asteroid angle compared to ship relations (NL, NM, PL, PM: asteroid towards ship)
        asteroid_velocity['S'] = fuzz.trimf(asteroid_velocity.universe, [0, 22.5, 45]) 
        asteroid_velocity['M'] = fuzz.trimf(asteroid_velocity.universe, [22.5, 45, 67.5]) 
        asteroid_velocity['L'] = fuzz.trimf(asteroid_velocity.universe, [67.5, 90, 90]) 
        
        ship_speed['NL'] = fuzz.trimf(ship_speed.universe, [-240, -240, -160])
        ship_speed['NM'] = fuzz.trimf(ship_speed.universe, [-240, -160, -80])
        ship_speed['NS'] = fuzz.trimf(ship_speed.universe, [-160, -80, 80])
        ship_speed['NZ'] = fuzz.trimf(ship_speed.universe, [-80, 0, 80])
        ship_speed['PS'] = fuzz.trimf(ship_speed.universe, [-80, 80, 160])
        ship_speed['PM'] = fuzz.trimf(ship_speed.universe, [80, 160, 240])
        ship_speed['PL'] = fuzz.trimf(ship_speed.universe, [160, 240, 240])

        # Define asteroid ship distance x relations
        asteroid_distance['S'] = fuzz.trimf(asteroid_distance.universe, [0, 62.5, 125])
        asteroid_distance['M'] = fuzz.trimf(asteroid_distance.universe, [62.5, 187.5, 312.5])
        asteroid_distance['L'] = fuzz.trimf(asteroid_distance.universe, [187.5, 375, 500])
        


        # Define heading angle compared to asteroid relations (for NZ to PM, asteroid heading toward ship)
        """ Not needed for current solution 
        heading_theta['NL'] = fuzz.trimf(heading_theta.universe, [-360, -360, -240]) 
        heading_theta['NM'] = fuzz.trimf(heading_theta.universe, [-360, -240, -120]) 
        heading_theta['NS'] = fuzz.trimf(heading_theta.universe, [-240, -120, -120]) 
        heading_theta['NZ'] = fuzz.trimf(heading_theta.universe, [-120, -0, 120]) 
        heading_theta['PS'] = fuzz.trimf(heading_theta.universe, [-120, 120, 240]) 
        heading_theta['PM'] = fuzz.trimf(heading_theta.universe, [120, 240, 360]) 
        heading_theta['PL'] = fuzz.trimf(heading_theta.universe, [240, 360, 360]) 
        """


        """ First solution: Incorporates distance, asteroid velocity, and asteroid theta. Current issue, ship picks up too much speed and sometimes runs into asteroids
        #Moves away from asteroids heading towards it but will chase asteroids moving away from it
        rule_mov1 = ctrl.Rule(
                        asteroid_distance['L'] & asteroid_velocity['L'] & asteroid_theta['T'],
                        ship_thrust['PM']
                    )
        rule_mov2 = ctrl.Rule(
                        asteroid_distance['L'] & asteroid_velocity['M'] & asteroid_theta['T'],
                        ship_thrust['PS']
                    )
        rule_mov3 = ctrl.Rule(
                        asteroid_distance['L'] & asteroid_velocity['S'] & asteroid_theta['T'],
                        ship_thrust['PM']
                    )
        rule_mov4 = ctrl.Rule(
                        asteroid_distance['M'] & asteroid_velocity['L'] & asteroid_theta['T'],
                        ship_thrust['PS']
                    )
        rule_mov5 = ctrl.Rule(
                        asteroid_distance['M'] & asteroid_velocity['M'] & asteroid_theta['T'],
                        ship_thrust['PS']
                    )
        rule_mov6 = ctrl.Rule(
                        asteroid_distance['M'] & asteroid_velocity['S'] & asteroid_theta['T'],
                        ship_thrust['PS']
                    )
        
        rule_mov7 = ctrl.Rule(
                        asteroid_distance['S'] & asteroid_velocity['L'] & asteroid_theta['T'],
                        ship_thrust['NM']
                    )
        rule_mov8 = ctrl.Rule(
                        asteroid_distance['S'] & asteroid_velocity['M'] & asteroid_theta['T'],
                        ship_thrust['NS']
                    )
        rule_mov9 = ctrl.Rule(
                        asteroid_distance['S'] & asteroid_velocity['S'] & asteroid_theta['T'],
                        ship_thrust['NS']
                    )    

        rule_mov10 = ctrl.Rule(
                        asteroid_distance['L'] & asteroid_velocity['L'] & (asteroid_theta['NCA'] | asteroid_theta['PCA']),
                        ship_thrust['PM']
                    )
        rule_mov11 = ctrl.Rule(
                        asteroid_distance['L'] & asteroid_velocity['M'] & (asteroid_theta['NCA'] | asteroid_theta['PCA']),
                        ship_thrust['PM']
                    )
        rule_mov12 = ctrl.Rule(
                        asteroid_distance['L'] & asteroid_velocity['S'] & (asteroid_theta['NCA'] | asteroid_theta['PCA']),
                        ship_thrust['PM']
                    )
        rule_mov13 = ctrl.Rule(
                        asteroid_distance['M'] & asteroid_velocity['L'] & (asteroid_theta['NCA'] | asteroid_theta['PCA']),
                        ship_thrust['PM']
                    )
        rule_mov14 = ctrl.Rule(
                        asteroid_distance['M'] & asteroid_velocity['M'] & (asteroid_theta['NCA'] | asteroid_theta['PCA']),
                        ship_thrust['PS']
                    )
        rule_mov15 = ctrl.Rule(
                        asteroid_distance['M'] & asteroid_velocity['S'] & (asteroid_theta['NCA'] | asteroid_theta['PCA']),
                        ship_thrust['PS']
                    )
        rule_mov16 = ctrl.Rule(
                        asteroid_distance['S'] & asteroid_velocity['L'] & (asteroid_theta['NCA'] | asteroid_theta['PCA']),
                        ship_thrust['NM']
                    )
        rule_mov17 = ctrl.Rule(
                        asteroid_distance['S'] & asteroid_velocity['M'] & (asteroid_theta['NCA'] | asteroid_theta['PCA']),
                        ship_thrust['NM']
                    )
        rule_mov18 = ctrl.Rule(
                        asteroid_distance['S'] & asteroid_velocity['S'] & (asteroid_theta['NCA'] | asteroid_theta['PCA']),
                        ship_thrust['NS']
                    )
        
        rule_mov19 = ctrl.Rule(
                        asteroid_distance['L'] & asteroid_velocity['L'] & (asteroid_theta['PFA'] | asteroid_theta['NFA']),
                        ship_thrust['PL']
                    )
        rule_mov20 = ctrl.Rule(
                        asteroid_distance['L'] & asteroid_velocity['M'] & (asteroid_theta['PFA'] | asteroid_theta['NFA']),
                        ship_thrust['PM']
                    )
        rule_mov21 = ctrl.Rule(
                        asteroid_distance['L'] & asteroid_velocity['S'] & (asteroid_theta['PFA'] | asteroid_theta['NFA']),
                        ship_thrust['PS']
                    )
        rule_mov22 = ctrl.Rule(
                        asteroid_distance['M'] & asteroid_velocity['L'] & (asteroid_theta['PFA'] | asteroid_theta['NFA']),
                        ship_thrust['PM']
                    )
        rule_mov23 = ctrl.Rule(
                        asteroid_distance['M'] & asteroid_velocity['M'] & (asteroid_theta['PFA'] | asteroid_theta['NFA']),
                        ship_thrust['PS']
                    )
        rule_mov24 = ctrl.Rule(
                        asteroid_distance['M'] & asteroid_velocity['S'] & (asteroid_theta['PFA'] | asteroid_theta['NFA']),
                        ship_thrust['NZ']
                    )
        rule_mov25 = ctrl.Rule(
                        asteroid_distance['S'] & asteroid_velocity['L'] & (asteroid_theta['PFA'] | asteroid_theta['NFA']),
                        ship_thrust['NM']
                    )
        rule_mov26 = ctrl.Rule(
                        asteroid_distance['S'] & asteroid_velocity['M'] & (asteroid_theta['PFA'] | asteroid_theta['NFA']),
                        ship_thrust['NS']
                    )
        rule_mov27 = ctrl.Rule(
                        asteroid_distance['S'] & asteroid_velocity['S'] & (asteroid_theta['PFA'] | asteroid_theta['NFA']),
                        ship_thrust['NS']
                    )
        """
        
        """ Second solution: Distance, current ship speed, and asteroid velocity. Moves towards all asteroids but backs away when getting too close """
        rule_sol2_mov1 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['L'] & ship_speed['NL'],
                        ship_thrust['PL']
        )
        rule_sol2_mov2 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['L'] & ship_speed['NM'],
                        ship_thrust['PL']
        )
        rule_sol2_mov3 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['L'] & ship_speed['NS'],
                        ship_thrust['PM']
        )
        rule_sol2_mov4 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['L'] & ship_speed['NZ'],
                        ship_thrust['PM']
        )
        rule_sol2_mov5 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['L'] & ship_speed['PS'],
                        ship_thrust['PS']
        )
        rule_sol2_mov6 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['L'] & ship_speed['PM'],
                        ship_thrust['NZ']
        )
        rule_sol2_mov7 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['L'] & ship_speed['PL'],
                        ship_thrust['NS']
        )
        
        rule_sol2_mov8 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['M'] & ship_speed['NL'],
                        ship_thrust['PL']
        )
        rule_sol2_mov9 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['M'] & ship_speed['NM'],
                        ship_thrust['PM']
        )
        rule_sol2_mov10 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['M'] & ship_speed['NS'],
                        ship_thrust['PS']
        )
        rule_sol2_mov11 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['M'] & ship_speed['NZ'],
                        ship_thrust['PM']
        )
        rule_sol2_mov12 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['M'] & ship_speed['PS'],
                        ship_thrust['PS']
        )
        rule_sol2_mov13 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['M'] & ship_speed['PM'],
                        ship_thrust['NZ']
        )
        rule_sol2_mov14 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['M'] & ship_speed['PL'],
                        ship_thrust['NS']
        )
        
        rule_sol2_mov15 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['S'] & ship_speed['NL'],
                        ship_thrust['PL']
        )
        rule_sol2_mov16 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['S'] & ship_speed['NM'],
                        ship_thrust['PM']
        )
        rule_sol2_mov17 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['S'] & ship_speed['NS'],
                        ship_thrust['PM']
        )
        rule_sol2_mov18 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['S'] & ship_speed['NZ'],
                        ship_thrust['PL']
        )
        rule_sol2_mov19 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['S'] & ship_speed['PS'],
                        ship_thrust['PS']
        )
        rule_sol2_mov20 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['S'] & ship_speed['PM'],
                        ship_thrust['NZ']
        )
        rule_sol2_mov21 = ctrl.Rule(
            asteroid_distance['L'] & asteroid_velocity['S'] & ship_speed['PL'],
                        ship_thrust['NS']
        )
        
        rule_sol2_mov22 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['L'] & ship_speed['NL'],
                        ship_thrust['PM']
        )
        rule_sol2_mov23 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['L'] & ship_speed['NM'],
                        ship_thrust['PS']
        )
        rule_sol2_mov24 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['L'] & ship_speed['NS'],
                        ship_thrust['NZ']
        )
        rule_sol2_mov25 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['L'] & ship_speed['NZ'],
                        ship_thrust['PM']
        )
        rule_sol2_mov26 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['L'] & ship_speed['PS'],
                        ship_thrust['PS']
        )
        rule_sol2_mov27 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['L'] & ship_speed['PM'],
                        ship_thrust['NS']
        )
        rule_sol2_mov28 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['L'] & ship_speed['PL'],
                        ship_thrust['NM']
        )
        
        rule_sol2_mov29 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['M'] & ship_speed['NL'],
                        ship_thrust['PM']
        )
        rule_sol2_mov30 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['M'] & ship_speed['NM'],
                        ship_thrust['PS']
        )
        rule_sol2_mov31 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['M'] & ship_speed['NS'],
                        ship_thrust['NZ']
        )
        rule_sol2_mov32 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['M'] & ship_speed['NZ'],
                        ship_thrust['PM']
        )
        rule_sol2_mov33 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['M'] & ship_speed['PS'],
                        ship_thrust['NS']
        )
        rule_sol2_mov34 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['M'] & ship_speed['PM'],
                        ship_thrust['NS']
        )
        rule_sol2_mov35 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['M'] & ship_speed['PL'],
                        ship_thrust['NM']
        )
        
        rule_sol2_mov36 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['S'] & ship_speed['NL'],
                        ship_thrust['PM']
        )
        rule_sol2_mov37 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['S'] & ship_speed['NM'],
                        ship_thrust['PS']
        )
        rule_sol2_mov38 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['S'] & ship_speed['NS'],
                        ship_thrust['NZ']
        )
        rule_sol2_mov39 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['S'] & ship_speed['NZ'],
                        ship_thrust['PS']
        )
        rule_sol2_mov40 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['S'] & ship_speed['PS'],
                        ship_thrust['NS']
        )
        rule_sol2_mov41 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['S'] & ship_speed['PM'],
                        ship_thrust['NS']
        )
        rule_sol2_mov42 = ctrl.Rule(
            asteroid_distance['M'] & asteroid_velocity['S'] & ship_speed['PL'],
                        ship_thrust['NM']
        )
        
        rule_sol2_mov43 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['L'] & ship_speed['NL'],
                        ship_thrust['PS']
        )
        rule_sol2_mov44 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['L'] & ship_speed['NM'],
                        ship_thrust['NZ']
        )
        rule_sol2_mov45 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['L'] & ship_speed['NS'],
                        ship_thrust['NS']
        )
        rule_sol2_mov46 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['L'] & ship_speed['NZ'],
                        ship_thrust['NM']
        )
        rule_sol2_mov47 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['L'] & ship_speed['PS'],
                        ship_thrust['NL']
        )
        rule_sol2_mov48 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['L'] & ship_speed['PM'],
                        ship_thrust['NL']
        )
        rule_sol2_mov49 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['L'] & ship_speed['PL'],
                        ship_thrust['NL']
        )
        
        rule_sol2_mov50 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['M'] & ship_speed['NL'],
                        ship_thrust['PS']
        )
        rule_sol2_mov51 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['M'] & ship_speed['NM'],
                        ship_thrust['NZ']
        )
        rule_sol2_mov52 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['M'] & ship_speed['NS'],
                        ship_thrust['NS']
        )
        rule_sol2_mov53 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['M'] & ship_speed['NZ'],
                        ship_thrust['NM']
        )
        rule_sol2_mov54 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['M'] & ship_speed['PS'],
                        ship_thrust['NM']
        )
        rule_sol2_mov55 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['M'] & ship_speed['PM'],
                        ship_thrust['NL']
        )
        rule_sol2_mov56 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['M'] & ship_speed['PL'],
                        ship_thrust['NL']
        )
        
        rule_sol2_mov57 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['S'] & ship_speed['NL'],
                        ship_thrust['PM']
        )
        rule_sol2_mov58 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['S'] & ship_speed['NM'],
                        ship_thrust['PS']
        )
        rule_sol2_mov59 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['S'] & ship_speed['NS'],
                        ship_thrust['NZ']
        )
        rule_sol2_mov60 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['S'] & ship_speed['NZ'],
                        ship_thrust['NM']
        )
        rule_sol2_mov61 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['S'] & ship_speed['PS'],
                        ship_thrust['NM']
        )
        rule_sol2_mov62 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['S'] & ship_speed['PM'],
                        ship_thrust['NL']
        )
        rule_sol2_mov63 = ctrl.Rule(
            asteroid_distance['S'] & asteroid_velocity['S'] & ship_speed['PL'],
                        ship_thrust['NL']
        )
        
        
        """ Aiden """
        """ Solution 1:
        self.movement_control = ctrl.ControlSystem()
        self.movement_control.addrule(rule_mov1)
        self.movement_control.addrule(rule_mov2)
        self.movement_control.addrule(rule_mov3)
        self.movement_control.addrule(rule_mov4)
        self.movement_control.addrule(rule_mov5)
        self.movement_control.addrule(rule_mov6)
        self.movement_control.addrule(rule_mov7)
        self.movement_control.addrule(rule_mov8)
        self.movement_control.addrule(rule_mov9)
        self.movement_control.addrule(rule_mov10)
        self.movement_control.addrule(rule_mov11)
        self.movement_control.addrule(rule_mov12)
        self.movement_control.addrule(rule_mov13)
        self.movement_control.addrule(rule_mov14)
        self.movement_control.addrule(rule_mov15)
        self.movement_control.addrule(rule_mov16)
        self.movement_control.addrule(rule_mov17)
        self.movement_control.addrule(rule_mov18)
        self.movement_control.addrule(rule_mov19)
        self.movement_control.addrule(rule_mov20)
        self.movement_control.addrule(rule_mov21)
        self.movement_control.addrule(rule_mov22)
        self.movement_control.addrule(rule_mov23)
        self.movement_control.addrule(rule_mov24)
        self.movement_control.addrule(rule_mov25)
        self.movement_control.addrule(rule_mov26)
        self.movement_control.addrule(rule_mov27)
        """

        """ Solution 2 """
        self.movement_control = ctrl.ControlSystem()
        self.movement_control.addrule(rule_sol2_mov1)
        self.movement_control.addrule(rule_sol2_mov2)
        self.movement_control.addrule(rule_sol2_mov3)
        self.movement_control.addrule(rule_sol2_mov4)
        self.movement_control.addrule(rule_sol2_mov5)
        self.movement_control.addrule(rule_sol2_mov6)
        self.movement_control.addrule(rule_sol2_mov7)
        self.movement_control.addrule(rule_sol2_mov8)
        self.movement_control.addrule(rule_sol2_mov9)
        self.movement_control.addrule(rule_sol2_mov10)
        self.movement_control.addrule(rule_sol2_mov11)
        self.movement_control.addrule(rule_sol2_mov12)
        self.movement_control.addrule(rule_sol2_mov13)
        self.movement_control.addrule(rule_sol2_mov14)
        self.movement_control.addrule(rule_sol2_mov15)
        self.movement_control.addrule(rule_sol2_mov16)
        self.movement_control.addrule(rule_sol2_mov17)
        self.movement_control.addrule(rule_sol2_mov18)
        self.movement_control.addrule(rule_sol2_mov19)
        self.movement_control.addrule(rule_sol2_mov20)
        self.movement_control.addrule(rule_sol2_mov21)
        self.movement_control.addrule(rule_sol2_mov22)
        self.movement_control.addrule(rule_sol2_mov23)
        self.movement_control.addrule(rule_sol2_mov24)
        self.movement_control.addrule(rule_sol2_mov25)
        self.movement_control.addrule(rule_sol2_mov26)
        self.movement_control.addrule(rule_sol2_mov27)
        self.movement_control.addrule(rule_sol2_mov28)
        self.movement_control.addrule(rule_sol2_mov29)
        self.movement_control.addrule(rule_sol2_mov30)
        self.movement_control.addrule(rule_sol2_mov31)
        self.movement_control.addrule(rule_sol2_mov32)
        self.movement_control.addrule(rule_sol2_mov33)
        self.movement_control.addrule(rule_sol2_mov34)
        self.movement_control.addrule(rule_sol2_mov35)
        self.movement_control.addrule(rule_sol2_mov36)
        self.movement_control.addrule(rule_sol2_mov37)
        self.movement_control.addrule(rule_sol2_mov38)
        self.movement_control.addrule(rule_sol2_mov39)
        self.movement_control.addrule(rule_sol2_mov40)
        self.movement_control.addrule(rule_sol2_mov41)
        self.movement_control.addrule(rule_sol2_mov42)
        self.movement_control.addrule(rule_sol2_mov43)
        self.movement_control.addrule(rule_sol2_mov44)
        self.movement_control.addrule(rule_sol2_mov45)
        self.movement_control.addrule(rule_sol2_mov46)
        self.movement_control.addrule(rule_sol2_mov47)
        self.movement_control.addrule(rule_sol2_mov48)
        self.movement_control.addrule(rule_sol2_mov49)
        self.movement_control.addrule(rule_sol2_mov50)
        self.movement_control.addrule(rule_sol2_mov51)
        self.movement_control.addrule(rule_sol2_mov52)
        self.movement_control.addrule(rule_sol2_mov53)
        self.movement_control.addrule(rule_sol2_mov54)
        self.movement_control.addrule(rule_sol2_mov55)
        self.movement_control.addrule(rule_sol2_mov56)
        self.movement_control.addrule(rule_sol2_mov57)
        self.movement_control.addrule(rule_sol2_mov58)
        self.movement_control.addrule(rule_sol2_mov59)
        self.movement_control.addrule(rule_sol2_mov60)
        self.movement_control.addrule(rule_sol2_mov61)
        self.movement_control.addrule(rule_sol2_mov62)
        self.movement_control.addrule(rule_sol2_mov63)




        
        
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)
        
        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)
        
        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 
                
        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))
     
        #DEBUG
        #bullet_time.view()
        #theta_delta.view()
        #ship_turn.view()
        #ship_fire.view()
     
     
        
        # Declare the fuzzy controller, add the rules 
        # This is an instance variable, and thus available for other methods in the same object. See notes.                         
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
             
        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        self.targeting_control.addrule(rule16)
        self.targeting_control.addrule(rule17)
        self.targeting_control.addrule(rule18)
        self.targeting_control.addrule(rule19)
        self.targeting_control.addrule(rule20)
        self.targeting_control.addrule(rule21)
        
    
    
    def movement(self, asteroid_vel, closest_asteroid_dist, ship_ast_difference, ship_speed):
        """
        In charge of passing the inputs needed to produce movement for the kessler ship. Uses a fuzzy system to decide an appropriate thrust for the ship based on its own speed and asteroid properties.
        Inputs: asteroid_vel: Scalar value of the velocity of the asteroid
                closest_asteroid_dist: Distance between the closest asteroid and the kessler ship
                ship_ast_difference: Difference in angles between the asteroid's direction and the ships heading. 0 degrees indicates moving towards each other, 180/-180 indicates moving in the same direction.
                ship_speed: Current speed of the kessler ship
        """
        """ Solution 1 
        thrust = ctrl.ControlSystemSimulation(self.movement_control,flush_after_run=1)
        thrust.input["asteroid_velocity"] = asteroid_vel
        thrust.input["asteroid_distance"] = closest_asteroid_dist
        thrust.input["asteroid_theta"] = ship_ast_difference
        """
        
        """ Solution 2 """
        thrust = ctrl.ControlSystemSimulation(self.movement_control,flush_after_run=1)
        thrust.input["asteroid_velocity"] = asteroid_vel
        thrust.input["asteroid_distance"] = closest_asteroid_dist
        thrust.input["ship_speed"] = ship_speed
        
        thrust.compute()    
        
        thrust = thrust.output['ship_thrust']
        return thrust
        
        
    
        

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and shooting.
        #thrust = 0 <- How do the values scale with asteroid velocity vector?
        #turn_rate = 90 <- How do the values scale with asteroid velocity vector?
        
        # Answers: Asteroid position and velocity are split into their x,y components in a 2-element ?array each.
        # So are the ship position and velocity, and bullet position and velocity. 
        # Units appear to be meters relative to origin (where?), m/sec, m/sec^2 for thrust.
        # Everything happens in a time increment: delta_time, which appears to be 1/30 sec; this is hardcoded in many places.
        # So, position is updated by multiplying velocity by delta_time, and adding that to position.
        # Ship velocity is updated by multiplying thrust by delta time.
        # Ship position for this time increment is updated after the the thrust was applied.
        

        # My demonstration controller does not move the ship, only rotates it to shoot the nearest asteroid.
        # Goal: demonstrate processing of game state, fuzzy controller, intercept computation 
        # Intercept-point calculation derived from the Law of Cosines, see notes for details and citation.

        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]       
        closest_asteroid = None
        
        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_asteroid = dict(aster = a, dist = curr_dist)
                
            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_asteroid["dist"] > curr_dist:
                    # New minimum found
                    closest_asteroid["aster"] = a
                    closest_asteroid["dist"] = curr_dist

        # closest_asteroid is now the nearest asteroid object. 
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.
        
        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!
        
        
        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * (closest_asteroid["dist"]**2))
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        # Velocities are in m/sec, so bullet_t is in seconds. Add one tik, hardcoded to 1/30 sec.
        intrcpt_x = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/30)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/30)

        
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi
        
        # Pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control,flush_after_run=1)
        
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        
        shooting.compute()
        
        # Get the defuzzified outputs
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
               
               
               
               
               
        ## Aiden Teal code, will eventually move to its own controller
        ## closest_asteroid_dist: Scalar distance between ship and asteroid
        ## ship_ast_difference: Angle that the asteroid is heading in in relation to the ship (0 degrees means complete opposite (going towards each other), 180 and -180
        ## mean asteroid and ship are going in same direction)
        ## asteroid_vel: Scalar value of the asteroid's velocity. Don't care about direction as we use ship_ast_difference to indicate if asteroid is moving towards ship
        ## ship_speed: Current speed of the ship: [-240, 240]
        
        ## Get angle difference between the asteroids direction and the asteroid_ship angle
        ship_ast_difference = asteroid_ship_theta - asteroid_direction
        ship_ast_difference = (ship_ast_difference + math.pi) % (2 * math.pi) - math.pi 
        ship_ast_difference = math.degrees(ship_ast_difference) # Convert to degrees
        
        print(f"Angle difference between the asteroids direction and the angle between the asteroid and ship: {ship_ast_difference}")
        
        #Get relative angle to asteroid ( Not needed for current solution )
        #ship_heading_theta = (math.pi/180)*ship_state["heading"]
        #heading_ast_difference = asteroid_ship_theta - ship_heading_theta
        #heading_ast_difference = (heading_ast_difference + math.pi) % (2 * math.pi) - math.pi # Get angle to determine if ship is facing the asteroid.
        #heading_ast_difference = math.degrees(heading_ast_difference) # convert to degrees

        closest_asteroid_dist = closest_asteroid["dist"]
        shooting_degrees = ship_state["heading"]
        asteroid_direction_radian = math.cos(asteroid_direction)
        ship_speed = ship_state["speed"]
        print(f"Asteroid direction degrees: {math.degrees(asteroid_direction_radian)}")
        print(f"Shooting direction degrees: {shooting_degrees}")
        print(f"Difference between asteroid direction and shooting direction: {math.degrees(asteroid_direction_radian) - shooting_degrees}")
        
        print(f"Closest asteroid: {closest_asteroid_dist}, ship-asteroid theta: {ship_ast_difference}, asteroid velocity: {asteroid_vel}, ship speed: {ship_speed}")
        
        thrust = self.movement(asteroid_vel, closest_asteroid_dist, ship_ast_difference, ship_speed)
        print(f"Thrust: {thrust}")
        
        ## Aiden Teal code end
               
               
               
               
               
               
               
               
        # And return your three outputs to the game simulation. Controller algorithm complete.
        #thrust = 0.0

        drop_mine = False
        
        self.eval_frames +=1
        
        #DEBUG
        print(thrust, bullet_t, shooting_theta, turn_rate, fire)
        
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "ScottDick Controller"