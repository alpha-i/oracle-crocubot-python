import os
from alcova_init import SimulationEnvironment

if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(current_directory, 'alcova.yml')) as alcova_config_file:

        simulation = SimulationEnvironment(alcova_config_file, current_directory)
        simulation.run()
