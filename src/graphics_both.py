
from kesslergame import KesslerGraphics
from kesslergame.graphics.graphics_tk import GraphicsTK
from kesslergame.graphics.graphics_ue import GraphicsUE

from typing import List

from kesslergame.ship import Ship
from kesslergame.asteroid import Asteroid
from kesslergame.bullet import Bullet
from kesslergame.mines import Mine
from kesslergame.score import Score
from kesslergame.scenario import Scenario

class GraphicsBoth(KesslerGraphics):
    def __init__(self) -> None:
        self.ue = GraphicsUE()
        self.tk = GraphicsTK({})

    def start(self, scenario: Scenario) -> None:
        self.ue.start(scenario)
        self.tk.start(scenario)

    def update(self, score: Score, ships: List[Ship], asteroids: List[Asteroid], bullets: List[Bullet], mines: List[Mine]) -> None:
        self.ue.update(score, ships, asteroids, bullets)
        self.tk.update(score, ships, asteroids, bullets)

    def close(self) -> None:
        self.ue.close()
        self.tk.close()
