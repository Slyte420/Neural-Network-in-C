import csv
import math
from random import seed
from random import random

with open('test.csv', mode='w') as file:
    file_writer = csv.writer(
        file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(21):
        x = random()
        y = random()
        f1 = x*y
        f2 = math.sin(math.radians(x)) + math.sin(math.radians(y))
        f3 = math.exp(x-1) * math.exp(y-1)
        f4 = 1.0/(1.0/(x+0.01) + 1.0 / (y+0.001))
        f5 = x*x-x*y+y*y
        if(f5 < 0.0):
            f5 = 0.0
        if(f5 > 1.0):
            f5 = 1.0
        file_writer.writerow([x, y, f1, f2, f3, f4, f5])


file.close()
