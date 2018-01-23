import os
import shutil

from alcova_init import controller, RUNTIME_DIR_PATH

if __name__ == '__main__':
    controller.run()
    
    shutil.rmtree(RUNTIME_DIR_PATH)
    os.mkdir(RUNTIME_DIR_PATH)





