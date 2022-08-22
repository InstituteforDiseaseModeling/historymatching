#! /usr/bin/env python3

from pathlib import Path

WORK_DIR = Path(__file__).parent.absolute()

from history_matching.examples.sir import SIR

z = SIR()
z.sim()
fig, _ = z.plot()
fig.savefig(WORK_DIR / "Stochastic_SIR.png")
