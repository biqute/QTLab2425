import numpy as np
import matplotlib.pyplot as plt

table = [["A"], ["B"], ["C"]]
plt.table(cellText=table, cellLoc="center")

# plt.table(cellText=table, rowLabels=("A", "B"), colLabels=("1", "2", "3"))

plt.show()