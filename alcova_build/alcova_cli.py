import os
from alcova_init import Initializer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    initializer = Initializer(BASE_DIR)
    with open(os.path.join(BASE_DIR, 'alcova.yml')) as alcova_config_file:
        initializer.initialize(alcova_config_file)
        initializer.controller.run()
