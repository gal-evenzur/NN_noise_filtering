import matplotlib.pyplot as plt
import numpy as np

x = np.array(range(1,10))

plt.plot(x, x, label='draw a line')
plt.xlabel("X Cords")
plt.ylabel("Y Cords")

plt.plot(x, x**2, label='draw a parabola')
plt.xlabel("X Cords")
plt.ylabel("Y Cords")

# plt.show()


with open("samples.txt", "r") as sample:
    c = sample.read()

lines = c.split("\n")

x2 = [int(l.split(" ")[0]) for l in lines]
y2 = [int(l.split(" ")[1]) for l in lines]

plt.plot(x2, y2, label='draw a data from file')
plt.xlabel("X Cords")
plt.ylabel("Y Cords")
plt.legend()

with open("fdata.csv", 'r') as f:
    c = (f.read()).split("\n")

names = c[0].split(',')


print(c)



