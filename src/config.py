LOG_FILE_PATH: str = "./KesslerController.log"

RUN_WITH_GRAPHICS: bool = False

USE_SIMULATION_CACHE: bool = True # default is True
FLUSH_SIMULATION_CACHE_AFTER_RUN: int = 1000 # default is 1000

SEED: int = 42069

# fitness of unoptimized Dr. Dick model on seed 42069: 40.91447368421052

GA_POPULATION_SIZE: int = 20
GA_GENERATION_GOAL: int = 10
GA_FITNESS_GOAL: float = 10000
GA_NUMBER_OF_PARENTS: int = 5
GA_NUMBER_OF_GENES_TO_MUTATE: int = 1

GA_NUMBER_OF_THREADS: int = 10 # number of threads/processes to use for GA

GA_CHROMOSOME_LENGTH: int = 28

GA_MODEL_FILE: str = "team_cam_controller"
