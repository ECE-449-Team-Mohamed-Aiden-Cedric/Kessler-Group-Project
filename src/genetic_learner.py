import random

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
