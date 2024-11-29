# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick
from typing import Dict, Tuple, Any
from immutabledict import immutabledict

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import config

Gene = dict[str, tuple[float, float, float]]
Chromosome = list[Gene] # MUST be a list due to implementation of EasyGA
ConvertedChromosome = dict[str, Gene]

class TeamCAMController(KesslerController): 
    def __init__(self):
        self.__current_frame = 0

        bullet_time: ctrl.Antecedent
        theta_delta: ctrl.Antecedent
        ship_turn: ctrl.Consequent
        ship_fire: ctrl.Consequent

        bullet_time, theta_delta, ship_turn, ship_fire= self.__setup_fuzzy_sets([])
        self.__rules: list[ctrl.Rule] = self.__get_rules(bullet_time, theta_delta, ship_turn, ship_fire)

        targeting_control = ctrl.ControlSystem(self.__rules)
        self.__control_system_simulation = ctrl.ControlSystemSimulation(
            targeting_control,
            cache=config.USE_SIMULATION_CACHE,
            flush_after_run=config.FLUSH_SIMULATION_CACHE_AFTER_RUN
        )

    @staticmethod
    def __setup_fuzzy_sets(chromosome: Chromosome) -> tuple[ctrl.Antecedent, ctrl.Antecedent, ctrl.Consequent, ctrl.Consequent]:
        """sets up the fuzzy sets with the genes defined in the Chromosome

        Args:
            chromosome (Chromosome): contains the genes with which to setup the fuzzy sets

        Returns:
            tuple[ctrl.Antecedent, ctrl.Antecedent, ctrl.Consequent, ctrl.Consequent]: bullet_time, theta_delta, ship_turn, ship_fire
        """
        bullet_time: ctrl.Antecedent = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta: ctrl.Antecedent = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python
        ship_turn: ctrl.Consequent = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire: ctrl.Consequent = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')

        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)

        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        # theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)

        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        # ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])

        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1])

        return (bullet_time, theta_delta, ship_turn, ship_fire)

    @staticmethod
    def __get_rules(
            bullet_time: ctrl.Antecedent,
            theta_delta: ctrl.Antecedent,
            ship_turn: ctrl.Consequent,
            ship_fire: ctrl.Consequent
        ) -> list[ctrl.Rule]:

        rules: list[ctrl.Rule] = [
            ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            #ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            #ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            #ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))
        ]

        return rules

    def actions(self, ship_state: Dict[str, Any], game_state: immutabledict[Any, Any]) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller.
        """
        # These were the constant actions in the basic demo, just spinning and self.__control_system_simulation.
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
        ship_pos_x: float = ship_state["position"][0]     # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y: float = ship_state["position"][1]       
        closest_asteroid: None | dict = None

        for a in game_state["asteroids"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist: float = math.sqrt((ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
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
        assert (closest_asteroid is not None)
        # Calculate intercept time given ship & asteroid position, asteroid velocity vector, bullet speed (not direction).
        # Based on Law of Cosines calculation, see notes.

        # Side D of the triangle is given by closest_asteroid.dist. Need to get the asteroid-ship direction
        #    and the angle of the asteroid's current movement.
        # REMEMBER TRIG FUNCTIONS ARE ALL IN RADAINS!!!


        asteroid_ship_x: float = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y: float = ship_pos_y - closest_asteroid["aster"]["position"][1]

        asteroid_ship_theta: float = math.atan2(asteroid_ship_y,asteroid_ship_x)

        asteroid_direction: float = math.atan2(closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2: float = asteroid_ship_theta - asteroid_direction
        cos_my_theta2: float = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel: float = math.sqrt(closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py

        # Determinant of the quadratic formula b^2-4ac
        targ_det: float = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid["dist"])

        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1: float = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2: float = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))

        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        bullet_t: float
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
        intrcpt_x: float = closest_asteroid["aster"]["position"][0] + closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/30)
        intrcpt_y: float = closest_asteroid["aster"]["position"][1] + closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/30)


        my_theta1: float = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))

        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta: float = my_theta1 - ((math.pi/180)*ship_state["heading"])

        # Wrap all angles to (-pi, pi)
        shooting_theta: float = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # Pass the inputs to the rulebase and fire it
        self.__control_system_simulation.input['bullet_time'] = bullet_t
        self.__control_system_simulation.input['theta_delta'] = shooting_theta

        self.__control_system_simulation.compute()

        # Get the defuzzified outputs
        turn_rate: float = self.__control_system_simulation.output['ship_turn']

        if self.__control_system_simulation.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False

        ## Aiden Teal code, will eventually move to its own controller
        thrust = 50

        # And return your three outputs to the game simulation. Controller algorithm complete.
        #thrust = 0.0

        drop_mine = False

        self.__current_frame +=1

        #DEBUG
        print(thrust, bullet_t, shooting_theta, turn_rate, fire)

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "ScottDick Controller"
