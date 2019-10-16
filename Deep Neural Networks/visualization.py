import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from main import EPOCHS
# import seaborn as sns

style.use("fivethirtyeight")

fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(1, 1, 1)


# graph_data = np.load("data.npy", allow_pickle=True)


def animate(i, frame=0):
    graph_data = np.load("cost_log.npy", allow_pickle=True)

    epoch, cost = graph_data
    frame += 1
    print(frame)
    ax1.clear()
    plt.ylim(0, 1)
    plt.xlim(0, EPOCHS)
    ax1.plot(epoch, cost)
    # ax1.plot(xs, sin)
    # sns.scatterplot(xs, sin)
    # sns.scatterplot(xs, cos)


ani = animation.FuncAnimation(fig, animate, interval=300)

plt.show()

