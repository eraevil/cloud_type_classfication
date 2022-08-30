import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv('loss.csv')
    print(df.info)

    import matplotlib.pyplot as plt
    # plt.scatter(range(len(df)),df)
    plt.plot(df)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("训练次数（每10万次）")
    plt.ylabel("MSE Loss")
    plt.show()