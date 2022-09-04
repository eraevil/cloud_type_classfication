import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    number = "090405"
    df = pd.read_csv('loss.csv',header=2)
    print(df.info)

    import matplotlib.pyplot as plt
    # plt.scatter(range(len(df)),df)
    plt.plot(df)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("No." + number + " Model")
    plt.xlabel("训练次数（每100万次）")
    plt.ylabel("L1 Loss")
    plt.show()
    # plt.savefig("./lossCurve/losscurve_"+number+".png")
    print("ok.")