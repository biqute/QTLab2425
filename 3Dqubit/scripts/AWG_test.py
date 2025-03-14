import numpy as np

import sys; sys.path.append("../classes")
from AWG import AWG

# SDG6052X
myAWG = AWG("192.168.40.15")
myAWG.output = True