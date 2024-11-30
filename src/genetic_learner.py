import random
from kesslergame.score import Score
from kesslergame.kessler_game import KesslerGame
from kesslergame.scenario import Scenario
from kesslergame.controller import KesslerController
from kesslergame.team import Team
from team_cam_controller import TeamCAMController
from kesslergame.graphics import GraphicsType
from kesslergame.kessler_game import TrainerEnvironment
import config

from typing import Any

import pygad

Gene = dict[str, tuple[float, float, float]]
Chromosome = list[float] # due to implementation of pygad
ConvertedChromosome = dict[str, Gene]

def execute_fuzzy_inference(
    kessler_game: KesslerGame,
    scenario: Scenario,
    controller: KesslerController
    ) -> Team:
    """executes the fuzzy system and returns the results we care about

    Returns:
        tuple[int, float]: number of asteroids hit, accuracy
    """
    score: Score
    score, _ = kessler_game.run(scenario=scenario, controllers=[controller])

    team_score: Team = score.teams[0] # get this team's score (assuming we're the only team)

    return team_score

def fitness_score_function(score: Team) -> float:
    """function to compute a fitness score to be maximized

    Args:
        score (Team): the Team object containing the score parameters from the game

    Returns:
        float: fitness score to be maximized
    """
    fitness_score: float = (
        score.asteroids_hit * score.accuracy
        + score.deaths * -30
    )

    return fitness_score

def fitness(ga_instance: pygad.GA, chromosome: Chromosome, solution_idx: int) -> float:
    """runs the controller with the given chromosome
    and returns a fitness score to be maximized

    Args:
        ga_instance (pygad.GA): pygad.GA instance
        chromosome (Chromosome): chromosome to use for the controller fuzzy system
        solution_idx (int): idk lol

    Returns:
        float: fitness score to be maximized
    """
    controller: TeamCAMController = TeamCAMController(chromosome)
    scenario: Scenario = Scenario(
        name = "Fitness Scenario",
        num_asteroids = 10,
        ship_states = [
            {
                "position": (400, 400),
                "angle": 90,
                "lives": 3,
                "team": 1,
                "mines_remaining": 3
            }
        ],
        map_size = (1000, 800),
        time_limit = 60,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = config.SEED
    )

    game_settings: dict[str, Any] = {
        "perf_tracker": True,
        "graphics_type": GraphicsType.Tkinter,
        "realtime_multiplier": 1,
        "graphics_obj": None,
        "frequency": 30
    }
    
    game: KesslerGame = TrainerEnvironment(settings = game_settings)

    score: Team = execute_fuzzy_inference(game, scenario, controller)

    final_fitness_score: float = fitness_score_function(score)

    print(f"iteration fitness: {final_fitness_score}")

    return final_fitness_score

def on_generation(ga_instance: pygad.GA):
    print("Generation: {:d}".format(ga_instance.generations_completed))
    print("Fitness of best solution: {:.2f}".format(ga_instance.best_solution(ga_instance.last_generation_fitness)[1]))

def run_genetic_algorithm():
    ga_instance: pygad.GA = pygad.GA(
        num_generations=config.GA_GENERATION_GOAL,
        num_parents_mating=config.GA_NUMBER_OF_PARENTS,
        fitness_func=fitness,
        sol_per_pop=config.GA_POPULATION_SIZE,
        num_genes=config.GA_CHROMOSOME_LENGTH,
        on_generation=on_generation,
        mutation_num_genes=config.GA_NUMBER_OF_GENES_TO_MUTATE,
        gene_type=float,
        gene_space={"low": 0, "high": 1},
        parallel_processing=["process", config.GA_NUMBER_OF_THREADS]
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
    ga_instance.save(config.GA_MODEL_FILE)
    ga_instance.plot_fitness()


if __name__ == "__main__":
    run_genetic_algorithm()
