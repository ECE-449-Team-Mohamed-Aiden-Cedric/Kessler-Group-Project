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
import config as config
from logger import Logger

from gene import Gene
from chromosome import Chromosome
from converted_chromosome import ConvertedChromosome

class TeamCAMController(KesslerController): 
    def __init__(self, chromosome: Chromosome):
        self.__current_frame = 0
        
        self.__logger: Logger = Logger(config.LOG_FILE_PATH)

        bullet_time: ctrl.Antecedent
        theta_delta: ctrl.Antecedent
        ship_speed: ctrl.Antecedent
        ship_stopping_distance: ctrl.Antecedent
        mine_distance: ctrl.Antecedent
        ship_turn: ctrl.Consequent
        ship_fire: ctrl.Consequent
        drop_mine: ctrl.Consequent
        ship_thrust: ctrl.Consequent

        self.__bullet_time_range: tuple[float, float] = (0, 1)
        self.__theta_delta_range: tuple[float, float] = (-1*math.pi/30, math.pi/30) # Radians due to Python
        self.__ship_speed_range: tuple[float, float] = (-240, 240) # m/s
        self.__ship_stopping_distance_range: tuple[float, float] = (0, 60) # m
        self.__mine_distance_range: tuple[float, float] = (0, 1000) # m
        self.__ship_turn_range: tuple[float, float] = (-180, 180) # Degrees due to Kessler
        self.__ship_fire_range: tuple[float, float] = (-1, 1)
        self.__ship_drop_mine_range: tuple[float, float] = (-1, 1)
        self.__ship_thrust_range: tuple[float, float] = (-480.0, 480.0) # m/s^2

        converted_chromosome: ConvertedChromosome = self.__convert_chromosome(chromosome)
        self.__logger.log(f"converted_chromosome: {converted_chromosome}")

        bullet_time, theta_delta, ship_speed, ship_stopping_distance, mine_distance, ship_turn, ship_fire, drop_mine, ship_thrust = self.__setup_fuzzy_sets(converted_chromosome)
        self.__rules: list[ctrl.Rule] = self.__get_rules(bullet_time, theta_delta, ship_speed, ship_stopping_distance, mine_distance, ship_turn, ship_fire, drop_mine, ship_thrust)

        targeting_control = ctrl.ControlSystem(self.__rules)
        self.__control_system_simulation = ctrl.ControlSystemSimulation(
            targeting_control,
            cache=config.USE_SIMULATION_CACHE,
            flush_after_run=config.FLUSH_SIMULATION_CACHE_AFTER_RUN
        )

    def __convert_chromosome(self, chromosome: Chromosome) -> ConvertedChromosome:
        """converts a list of floats into something usable by setup_fuzzy_sets

        Args:
            chromosome (Chromosome): a list of floats

        Returns:
            ConvertedChromosome: something in a format usable in trimf functions
        """
        chromosome_list: list[float] = chromosome.tolist()
        # bullet time
        values: list[float] = chromosome_list[0:3]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        bullet_time_gene: Gene = { # type: ignore
            "S": tuple(values[0:3]),
            "M": tuple(values[1:4]),
            "L": tuple(values[2:5])
        }
        bullet_time_gene = self.__scale_gene(
            bullet_time_gene,
            self.__bullet_time_range[0],
            self.__bullet_time_range[1]
        )

        values: list[float] = chromosome_list[3:10]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        theta_delta_gene: Gene = { # type: ignore
            "NL": tuple(values[0:3]),
            "NM": tuple(values[1:4]),
            "NS": tuple(values[2:5]),
            "Z": tuple(values[3:6]),
            "PS": tuple(values[4:7]),
            "PM": tuple(values[5:8]),
            "PL": tuple(values[6:9])
        }
        theta_delta_gene = self.__scale_gene(
            theta_delta_gene,
            self.__theta_delta_range[0],
            self.__theta_delta_range[1]
        )

        values: list[float] = chromosome_list[10:17]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        ship_turn_gene: Gene = { # type: ignore
            "NL": tuple(values[0:3]),
            "NM": tuple(values[1:4]),
            "NS": tuple(values[2:5]),
            "Z": tuple(values[3:6]),
            "PS": tuple(values[4:7]),
            "PM": tuple(values[5:8]),
            "PL": tuple(values[6:9])
        }
        ship_turn_gene = self.__scale_gene(
            ship_turn_gene,
            self.__ship_turn_range[0],
            self.__ship_turn_range[1]
        )

        values: list[float] = chromosome_list[17:19]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        ship_fire_gene: Gene = { # type: ignore
            "N": tuple(values[0:3]),
            "Y": tuple(values[1:4])
        }
        ship_fire_gene = self.__scale_gene(
            ship_fire_gene,
            self.__ship_fire_range[0],
            self.__ship_fire_range[1]
        )

        values: list[float] = chromosome_list[19:21]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        drop_mine_gene: Gene = { # type: ignore
            "N": tuple(values[0:3]),
            "Y": tuple(values[1:4])
        }
        drop_mine_gene = self.__scale_gene(
            drop_mine_gene,
            self.__ship_drop_mine_range[0],
            self.__ship_drop_mine_range[1]
        )

        values: list[float] = chromosome_list[21:28]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        ship_thrust_gene: Gene = { # type: ignore
            "NL": tuple(values[0:3]),
            "NM": tuple(values[1:4]),
            "NS": tuple(values[2:5]),
            "Z": tuple(values[3:6]),
            "PS": tuple(values[4:7]),
            "PM": tuple(values[5:8]),
            "PL": tuple(values[6:9])
        }
        ship_thrust_gene = self.__scale_gene(
            ship_thrust_gene,
            self.__ship_turn_range[0],
            self.__ship_turn_range[1]
        )

        values: list[float] = chromosome_list[28:35]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        ship_speed_gene: Gene = { # type: ignore
            "NL": tuple(values[0:3]),
            "NM": tuple(values[1:4]),
            "NS": tuple(values[2:5]),
            "Z": tuple(values[3:6]),
            "PS": tuple(values[4:7]),
            "PM": tuple(values[5:8]),
            "PL": tuple(values[6:9])
        }
        ship_speed_gene = self.__scale_gene(
            ship_speed_gene,
            self.__ship_speed_range[0],
            self.__ship_speed_range[1]
        )

        values: list[float] = chromosome_list[35:39]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        ship_stopping_distance_gene: Gene = { # type: ignore
            "Z": tuple(values[0:3]),
            "PS": tuple(values[1:4]),
            "PM": tuple(values[2:5]),
            "PL": tuple(values[3:6])
        }
        ship_stopping_distance_gene = self.__scale_gene(
            ship_stopping_distance_gene,
            self.__ship_stopping_distance_range[0],
            self.__ship_stopping_distance_range[1]
        )

        values: list[float] = chromosome_list[39:43]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        mine_distance_gene: Gene = { # type: ignore
            "Z": tuple(values[0:3]),
            "PS": tuple(values[1:4]),
            "PM": tuple(values[2:5]),
            "PL": tuple(values[3:6])
        }
        mine_distance_gene = self.__scale_gene(
            mine_distance_gene,
            self.__mine_distance_range[0],
            self.__mine_distance_range[1]
        )

        converted_chromosome: ConvertedChromosome = {
            "bullet_time": bullet_time_gene,
            "theta_delta": theta_delta_gene,
            "ship_speed": ship_speed_gene,
            "ship_stopping_distance": ship_stopping_distance_gene,
            "mine_distance": mine_distance_gene,
            "ship_turn": ship_turn_gene,
            "ship_fire": ship_fire_gene,
            "drop_mine": drop_mine_gene,
            "ship_thrust": ship_thrust_gene
        }

        return converted_chromosome

    @staticmethod
    def __scale_gene(gene: Gene, minimum: float, maximum: float) -> Gene:
        scaled_gene: Gene = dict()
        for key in gene.keys():
            scaled_gene[key] = tuple([(gene[key][i] * (maximum - minimum)) + minimum for i in range(3)]) # type: ignore

        return scaled_gene

    def __setup_fuzzy_sets(self, chromosome: ConvertedChromosome) -> tuple[ctrl.Antecedent, ctrl.Antecedent, ctrl.Antecedent, ctrl.Antecedent, ctrl.Antecedent, ctrl.Consequent, ctrl.Consequent, ctrl.Consequent, ctrl.Consequent]:
        """sets up the fuzzy sets with the genes defined in the Chromosome

        Args:
            chromosome (Chromosome): contains the genes with which to setup the fuzzy sets

        Returns:
            tuple[ctrl.Antecedent, ctrl.Antecedent, ctrl.Consequent, ctrl.Consequent]: bullet_time, theta_delta, ship_turn, ship_fire
        """
        bullet_time: ctrl.Antecedent = ctrl.Antecedent(np.arange(self.__bullet_time_range[0], self.__bullet_time_range[1], 0.002), 'bullet_time')
        theta_delta: ctrl.Antecedent = ctrl.Antecedent(np.arange(self.__theta_delta_range[0], self.__theta_delta_range[1], 0.1), 'theta_delta')
        ship_speed: ctrl.Antecedent = ctrl.Antecedent(np.arange(self.__ship_speed_range[0], self.__ship_speed_range[1], 0.1), 'ship_speed')
        ship_stopping_distance: ctrl.Antecedent = ctrl.Antecedent(np.arange(self.__ship_stopping_distance_range[0], self.__ship_stopping_distance_range[1], 0.1), 'ship_stopping_distance')
        mine_distance: ctrl.Antecedent = ctrl.Antecedent(np.arange(self.__mine_distance_range[0], self.__mine_distance_range[1], 0.1), 'mine_distance')
        ship_turn: ctrl.Consequent = ctrl.Consequent(np.arange(self.__ship_turn_range[0], self.__ship_turn_range[1], 1), 'ship_turn')
        ship_fire: ctrl.Consequent = ctrl.Consequent(np.arange(self.__ship_fire_range[0], self.__ship_fire_range[1], 0.1), 'ship_fire')
        drop_mine: ctrl.Consequent = ctrl.Consequent(np.arange(self.__ship_drop_mine_range[0], self.__ship_drop_mine_range[1], 0.1), 'drop_mine')
        ship_thrust: ctrl.Consequent = ctrl.Consequent(np.arange(self.__ship_thrust_range[0], self.__ship_thrust_range[1], 1), 'ship_thrust')

        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time_gene: Gene = chromosome["bullet_time"]
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, bullet_time_gene["S"])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, bullet_time_gene["M"])
        bullet_time['L'] = fuzz.trimf(bullet_time.universe, bullet_time_gene["L"])

        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta_gene: Gene = chromosome["theta_delta"]
        theta_delta['NL'] = fuzz.trimf(theta_delta.universe, theta_delta_gene["NL"])
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, theta_delta_gene["NM"])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, theta_delta_gene["NS"])
        theta_delta['Z']  = fuzz.trimf(theta_delta.universe, theta_delta_gene["Z"])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, theta_delta_gene["PS"])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, theta_delta_gene["PM"])
        theta_delta['PL'] = fuzz.trimf(theta_delta.universe, theta_delta_gene["PL"])

        ship_speed_gene: Gene = chromosome["ship_speed"]
        ship_speed['NL'] = fuzz.trimf(ship_speed.universe, ship_speed_gene["NL"])
        ship_speed['NM'] = fuzz.trimf(ship_speed.universe, ship_speed_gene["NM"])
        ship_speed['NS'] = fuzz.trimf(ship_speed.universe, ship_speed_gene["NS"])
        ship_speed['Z']  = fuzz.trimf(ship_speed.universe, ship_speed_gene["Z"])
        ship_speed['PS'] = fuzz.trimf(ship_speed.universe, ship_speed_gene["PS"])
        ship_speed['PM'] = fuzz.trimf(ship_speed.universe, ship_speed_gene["PM"])
        ship_speed['PL'] = fuzz.trimf(ship_speed.universe, ship_speed_gene["PL"])

        ship_stopping_distance_gene: Gene = chromosome["ship_stopping_distance"]
        ship_stopping_distance['Z']  = fuzz.trimf(ship_stopping_distance.universe, ship_stopping_distance_gene["Z"])
        ship_stopping_distance['PS'] = fuzz.trimf(ship_stopping_distance.universe, ship_stopping_distance_gene["PS"])
        ship_stopping_distance['PM'] = fuzz.trimf(ship_stopping_distance.universe, ship_stopping_distance_gene["PM"])
        ship_stopping_distance['PL'] = fuzz.trimf(ship_stopping_distance.universe, ship_stopping_distance_gene["PL"])

        mine_distance_gene: Gene = chromosome["mine_distance"]
        mine_distance['Z']  = fuzz.trimf(mine_distance.universe, mine_distance_gene["Z"])
        mine_distance['PS'] = fuzz.trimf(mine_distance.universe, mine_distance_gene["PS"])
        mine_distance['PM'] = fuzz.trimf(mine_distance.universe, mine_distance_gene["PM"])
        mine_distance['PL'] = fuzz.trimf(mine_distance.universe, mine_distance_gene["PL"])

        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn_gene: Gene = chromosome["ship_turn"]
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, ship_turn_gene["NL"])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, ship_turn_gene["NM"])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, ship_turn_gene["NS"])
        ship_turn['Z']  = fuzz.trimf(ship_turn.universe, ship_turn_gene["Z"])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, ship_turn_gene["PS"])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, ship_turn_gene["PM"])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, ship_turn_gene["PL"])

        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire_gene: Gene = chromosome["ship_fire"]
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, ship_fire_gene["N"])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, ship_fire_gene["Y"])

        drop_mine_gene: Gene = chromosome["drop_mine"]
        drop_mine['N'] = fuzz.trimf(drop_mine.universe, drop_mine_gene["N"])
        drop_mine['Y'] = fuzz.trimf(drop_mine.universe, drop_mine_gene["Y"])

        ship_thrust_gene: Gene = chromosome["ship_thrust"]
        ship_thrust['NL'] = fuzz.trimf(ship_thrust.universe, ship_thrust_gene["NL"])
        ship_thrust['NM'] = fuzz.trimf(ship_thrust.universe, ship_thrust_gene["NM"])
        ship_thrust['NS'] = fuzz.trimf(ship_thrust.universe, ship_thrust_gene["NS"])
        ship_thrust['Z']  = fuzz.trimf(ship_thrust.universe, ship_thrust_gene["Z"])
        ship_thrust['PS'] = fuzz.trimf(ship_thrust.universe, ship_thrust_gene["PS"])
        ship_thrust['PM'] = fuzz.trimf(ship_thrust.universe, ship_thrust_gene["PM"])
        ship_thrust['PL'] = fuzz.trimf(ship_thrust.universe, ship_thrust_gene["PL"])

        return (bullet_time, theta_delta, ship_speed, ship_stopping_distance, mine_distance, ship_turn, ship_fire, drop_mine, ship_thrust)

    @staticmethod
    def __get_rules(
            bullet_time: ctrl.Antecedent,
            theta_delta: ctrl.Antecedent,
            ship_speed: ctrl.Antecedent,
            ship_stopping_distance: ctrl.Antecedent,
            mine_distance: ctrl.Antecedent,
            ship_turn: ctrl.Consequent,
            ship_fire: ctrl.Consequent,
            drop_mine: ctrl.Consequent,
            ship_thrust: ctrl.Consequent
        ) -> list[ctrl.Rule]:

        rules: list[ctrl.Rule] = [
            ctrl.Rule(theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'])),

            ctrl.Rule(ship_stopping_distance['PL'] & bullet_time['S'], ship_thrust['NL']),
            ctrl.Rule(ship_stopping_distance['PL'] & bullet_time['M'], ship_thrust['NM']),
            ctrl.Rule(ship_stopping_distance['PL'] & bullet_time['L'], ship_thrust['NS']),
            ctrl.Rule(ship_stopping_distance['PM'] & bullet_time['S'], ship_thrust['NM']),
            ctrl.Rule(ship_stopping_distance['PM'] & bullet_time['M'], ship_thrust['NS']),
            ctrl.Rule(ship_stopping_distance['PM'] & bullet_time['L'], ship_thrust['Z']),
            ctrl.Rule(ship_stopping_distance['PS'] & bullet_time['S'], ship_thrust['NS']),
            ctrl.Rule(ship_stopping_distance['PS'] & bullet_time['M'], ship_thrust['Z']),
            ctrl.Rule(ship_stopping_distance['PS'] & bullet_time['L'], ship_thrust['PS']),
            ctrl.Rule(ship_stopping_distance['Z'] & bullet_time['S'], ship_thrust['Z']),
            ctrl.Rule(ship_stopping_distance['Z'] & bullet_time['M'], ship_thrust['PS']),
            ctrl.Rule(ship_stopping_distance['Z'] & bullet_time['L'], ship_thrust['PM']),

            ctrl.Rule(ship_speed['NL'], drop_mine['Y']),
            ctrl.Rule(ship_speed['NM'], drop_mine['Y']),
            ctrl.Rule(ship_speed['NS'], drop_mine['N']),
            ctrl.Rule(ship_speed['Z'], drop_mine['N']),
            ctrl.Rule(ship_speed['PS'], drop_mine['N']),
            ctrl.Rule(ship_speed['PM'], drop_mine['Y']),
            ctrl.Rule(ship_speed['PL'], drop_mine['Y']),

            ctrl.Rule(mine_distance['Z'], ship_thrust['PL']),
            ctrl.Rule(mine_distance['PS'], ship_thrust['PM'])
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

        ship_speed: float = math.sqrt(ship_state["velocity"][0]**2 + ship_state["velocity"][1]**2)
        ship_heading: float = ship_state['heading']
        if (ship_heading >= 0 and ship_heading < 180):
            # ship is heading upwards in some amount
            if (ship_state['velocity'][1] < 0):
                ship_speed *= -1
        else:
            if (ship_state['velocity'][1] >= 0):
                ship_speed *= -1
        stopping_time: float = ship_speed / -self.__ship_thrust_range[0]
        stopping_distance: float = (ship_speed * stopping_time) + (self.__ship_thrust_range[0] * (stopping_time**2) / 2)

        closest_mine: None | dict = None
        for mine in game_state["mines"]:
            #Loop through all asteroids, find minimum Eudlidean distance
            curr_dist: float = math.sqrt((ship_pos_x - mine["position"][0])**2 + (ship_pos_y - mine["position"][1])**2)
            if closest_mine is None :
                # Does not yet exist, so initialize first asteroid as the minimum. Ugh, how to do?
                closest_mine = dict(mine = mine, dist = curr_dist)

            else:    
                # closest_asteroid exists, and is thus initialized. 
                if closest_mine["dist"] > curr_dist:
                    # New minimum found
                    closest_mine["mine"] = mine
                    closest_mine["dist"] = curr_dist

        if closest_mine is None:
            # there were no mines on the field
            closest_mine = dict(mine = None, dist = self.__mine_distance_range[1])
        closest_mine_distance: float = closest_mine["dist"]

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
        self.__control_system_simulation.input['ship_speed'] = ship_speed
        self.__control_system_simulation.input['ship_stopping_distance'] = stopping_distance
        self.__control_system_simulation.input['mine_distance'] = closest_mine_distance

        self.__control_system_simulation.compute()

        # Get the defuzzified outputs
        turn_rate: float = self.__control_system_simulation.output['ship_turn']

        fire: bool
        if self.__control_system_simulation.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False

        drop_mine: bool
        if self.__control_system_simulation.output['drop_mine'] >= 0:
            drop_mine = True
        else:
            drop_mine = False

        ## Aiden Teal code, will eventually move to its own controller
        thrust: float = self.__control_system_simulation.output['ship_thrust']

        self.__current_frame +=1

        #DEBUG
        self.__logger.log(
            "Simulation Results\n\tThrust: {:.1f}\n\tBullet Time: {:.3f}\n\tShooting Theta: {:.3f}\n\tTurn Rate: {:.2f}\n\tFire: {}\n\tDrop Mine: {}".format(
                thrust, bullet_t, shooting_theta, turn_rate, fire, drop_mine
            )
        )

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "ScottDick Controller"
