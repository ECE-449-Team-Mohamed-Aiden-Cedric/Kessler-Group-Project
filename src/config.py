LOG_FILE_PATH: str = "./KesslerController.log"

RUN_WITH_GRAPHICS: bool = True

USE_SIMULATION_CACHE: bool = True # default is True
FLUSH_SIMULATION_CACHE_AFTER_RUN: int = 1000 # default is 1000

SEED: int = 42069

# fitness of unoptimized Dr. Dick model on seed 42069: 40.91447368421052

GA_POPULATION_SIZE: int = 42
GA_GENERATION_GOAL: int = 100
GA_FITNESS_GOAL: float = 1
GA_NUMBER_OF_PARENTS: int = 21
GA_NUMBER_OF_GENES_TO_MUTATE: int = 1

GA_CHROMOSOME_LENGTH: int = 1

GA_MODEL_FILE: str = "team_cam_controller.pygad"
