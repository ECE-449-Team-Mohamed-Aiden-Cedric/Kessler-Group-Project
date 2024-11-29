import random
from kesslergame.score import Score
from kesslergame.kessler_game import KesslerGame
from kesslergame.scenario import Scenario
from kesslergame.controller import KesslerController

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
    ) -> Score:
    """executes the fuzzy system and returns the results we care about

    Returns:
        tuple[int, float]: number of asteroids hit, accuracy
    """
    score: Score
    score, _ = kessler_game.run(scenario=scenario, controllers=[controller])

    return score
