import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
# 画云类型散点图
def draw_scatter(x, y,c, marker_size=50, savefig_name=""):
    # 创建画图窗口
    fig = plt.figure(1, figsize=(9, 3))
    # 将画图窗口分成2x2, 选择第一块区域作子图
    # subplot1 = fig.add_subplot(2, 2, 1)
    # 画散点图
    plt.scatter(x, y, s=marker_size, c=c, marker='.')
    # 画参考线
    # subplot1.plot((0, 300), (0, 300), linestyle="--", linewidth=0.8, color="b")
    # 调整坐标轴范围
    # plt.xlim((0, 37082))
    # plt.ylim((0, 125))
    # 设置坐标轴刻度
    # xticks = np.arange(0, 126, 50)
    # yticks = np.arange(0, 37083, 50)
    # plt.xticks(xticks)
    # plt.yticks(yticks)
    # 设置标题
    # subplot1.set_title('Scatter Plot')
    # 设置坐标轴名称
    plt.xlabel('Position')
    plt.ylabel('Level')
    # 添加网格线
    plt.grid(linestyle='--', color='grey')
    # 全局修改字体
    plt.rc('font', family='Times New Roman')
    # 显示色带
    # cbar_ticks = ['None','Ci','As','Ac','St','Sc','Cu','Ns','Dc']
    cbar_ticks = c
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 12}
    cbar = plt.colorbar(orientation='horizontal', extend="both", pad=0.2)  # 显示色带
    cbar.set_label("Cloud Type", fontdict=font)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=6)  # 主刻度
    cbar.ax.tick_params(which="minor", direction="in", length=0)  # 副刻度
    # save figure
    # fig.tight_layout()
    # if "" != savefig_name.strip():
    #     plt.savefig(savefig_name, dpi=600)
    plt.show()

x = np.array([1,2])
y = np.array([4,5,6])
c = np.array([[100,120,140],[160,180,200]])

xv,yv = np.meshgrid(x,y,indexing = 'ij')
cv = c.flatten()

# print(xv)
# print(yv)
# print(np.array(xv.flat))
# print(np.array(yv.flat))
# print(cv)

xx = np.array(xv.flat)
yy = np.array(yv.flat)
draw_scatter(xx,yy,c=cv)
plt.show()