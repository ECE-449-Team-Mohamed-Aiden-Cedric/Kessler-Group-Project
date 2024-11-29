import random
from kesslergame.score import Score
from kesslergame.kessler_game import KesslerGame
from kesslergame.scenario import Scenario
from kesslergame.controller import KesslerController
from kesslergame.team import Team
from team_cam_controller import TeamCAMController
from kesslergame.graphics import GraphicsType
from kesslergame.kessler_game import TrainerEnvironment

from typing import Any

Gene = dict[str, tuple[float, float, float]]
Chromosome = list[Gene] # MUST be a list due to implementation of EasyGA
ConvertedChromosome = dict[str, Gene]

def gene_generation() -> Gene:
    values: list[float] = [random.random() for _ in range(7)]
    values.extend([-0.01, 1.01])
    values = sorted(values)
    gene: Gene = { # type: ignore
        "NL": tuple(values[0:3]),
        "NM": tuple(values[1:4]),
        "NS": tuple(values[2:5]),
        "Z": tuple(values[3:6]),
        "PS": tuple(values[4:7]),
        "PM": tuple(values[5:8]),
        "PL": tuple(values[6:9])
    }

    return gene

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
    """function to compute a fitness score to be minimized

    Args:
        score (Team): the Team object containing the score parameters from the game

    Returns:
        float: fitness score to be minimized
    """
    fitness_score: float = (
        score.asteroids_hit * score.accuracy /
        + score.deaths * -30
    )

    return fitness_score

def fitness(chromosome: Chromosome) -> float:
    """runs the controller with the given chromosome
    and returns a fitness score to be minimized

    Args:
        chromosome (Chromosome): chromosome to use for the controller fuzzy system

    Returns:
        float: fitness score to be minimized
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
        stop_if_no_ammo = False
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

    return fitness_score_function(score)
