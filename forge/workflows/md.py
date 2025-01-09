from typing import List, Dict, Optional
import numpy as np
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pathlib import Path
import json

class MDSimulator:
    def __init__(self, calculator, temp: float, timestep: float = 1.0,
                 friction: float = 0.002, trajectory_file: Optional[str] = None):
        self.calculator = calculator
        self.temp = temp
        self.timestep = timestep
        self.friction = friction
        self.trajectory_file = trajectory_file
        
    def run_md(self, atoms: Atoms, steps: int, sample_interval: int = 10):
        """Run MD simulation with neural network potential"""
        atoms.calc = self.calculator
        
        # Initialize velocities
        MaxwellBoltzmannDistribution(atoms, temperature_K=self.temp)
        
        # Setup dynamics
        dyn = Langevin(atoms, self.timestep, temperature_K=self.temp,
                      friction=self.friction)
                      
        # Setup trajectory writing
        if self.trajectory_file:
            def write_frame():
                atoms.write(self.trajectory_file, append=True)
            dyn.attach(write_frame, interval=sample_interval)
            
        # Run dynamics
        dyn.run(steps)
        
        return atoms