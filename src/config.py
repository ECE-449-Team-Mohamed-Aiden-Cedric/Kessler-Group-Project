from kesslergame.scenario import Scenario

LOG_FILE_PATH: str = "./KesslerController.log"

RUN_WITH_GRAPHICS: bool = False

USE_SIMULATION_CACHE: bool = True # default is True
FLUSH_SIMULATION_CACHE_AFTER_RUN: int = 1000 # default is 1000

SCENARIOS: list[Scenario] = [ # TODO define asteroid states too
    Scenario(
        name = "Easy Scenario",
        num_asteroids = 10,
        ship_states = [
            {
                "position": (1700, 1000),
                "angle": 90,
                "lives": 3,
                "team": 1,
                "mines_remaining": 3,
            }
        ],
        map_size = (3400, 2000),
        time_limit = 240,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = 42
    ),
    Scenario(
        name = "Medium Scenario",
        num_asteroids = 50,
        ship_states = [
            {
                "position": (1700, 1000),
                "angle": 90,
                "lives": 3,
                "team": 1,
                "mines_remaining": 3,
            }
        ],
        map_size = (3400, 2000),
        time_limit = 240,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = 69
    ),
    Scenario(
        name = "Hard Scenario",
        num_asteroids = 100,
        ship_states = [
            {
                "position": (1700, 1000),
                "angle": 90,
                "lives": 3,
                "team": 1,
                "mines_remaining": 3,
            }
        ],
        map_size = (3400, 2000),
        time_limit = 240,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = 420
    ),
    Scenario(
        name = "Very Hard Scenario",
        num_asteroids = 250,
        ship_states = [
            {
                "position": (1700, 1000),
                "angle": 90,
                "lives": 3,
                "team": 1,
                "mines_remaining": 3,
            }
        ],
        map_size = (3400, 2000),
        time_limit = 240,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = 666
    ),
    Scenario(
        name = "Extremely Hard Scenario",
        num_asteroids = 500,
        ship_states = [
            {
                "position": (1700, 1000),
                "angle": 90,
                "lives": 3,
                "team": 1,
                "mines_remaining": 3,
            }
        ],
        map_size = (3400, 2000),
        time_limit = 240,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = 666
    )
]

# fitness of unoptimized Dr. Dick model on seed 42069: 40.91447368421052

GA_POPULATION_SIZE: int = 20
GA_GENERATION_GOAL: int = 10000
GA_FITNESS_GOAL: float = 10000
GA_NUMBER_OF_PARENTS: int = 5
GA_NUMBER_OF_GENES_TO_MUTATE: int = 1

GA_NUMBER_OF_THREADS: int = 10 # number of threads/processes to use for GA

GA_CHROMOSOME_LENGTH: int = 47

GA_MODEL_FILE: str = "team_cam_controller"

GA_RESTART_FROM_SCRATCH: bool = False # if False, continues from saved model file
GA_STOP_FLAG_FILE: str = "safely_stop_genetic_learner.txt"
