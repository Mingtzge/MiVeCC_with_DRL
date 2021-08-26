# -*- coding: utf-8 -*-
from pyheatmap.heatmap import HeatMap
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
# import numpy as np
import seaborn as sns
# import matplotlib.pyplot as plt
#
import matplotlib.pyplot as plt

# 绘制多路口：
signal = False
sns.set()
width = 60
fen = 10
uniform_data = np.zeros((width, width))
veh_v_total = np.zeros((width, width))
test = np.zeros((width, width))
mask_ = np.ones_like(uniform_data)
# mask_[int(width / 4), :] = 0
# mask_[int(width / 4 - 1), :] = 0
# mask_[int(width / 2), :] = 0
# mask_[int(width / 2 - 1), :] = 0
# mask_[int(width * 3 / 4), :] = 0
# mask_[int(width * 3 / 4 - 1), :] = 0
# mask_[:, int(width / 4)] = 0
# mask_[:, int(width / 4 - 1)] = 0
# mask_[:, int(width / 2)] = 0
# mask_[:, int(width / 2 - 1)] = 0
# mask_[:, int(width * 3 / 4)] = 0
# mask_[:, int(width * 3 / 4 - 1)] = 0
if signal:
    ma_json_path = ".\\signal_unsignal_sample_data\\signal_multi_400.txt"
else:
    ma_json_path = ".\\signal_unsignal_sample_data\\unsignal_multi_record_heat_map_data.txt"
collisions_veh_numbers = open(ma_json_path, 'r').readlines()
h_data = []
x = []
y = []
add_n = 1
for item in collisions_veh_numbers:
    data = item.strip().split(" ")
    lane = int(data[0])
    x.append(int((int(float(data[1])) / fen)))
    y.append(int((int(float(data[2])) / fen)))
    # signal
    if signal:
        if lane == 0:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width / 4 - 1)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width / 4 - 1)] += float(data[3])
        if lane == 1:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width / 4)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width / 4)] += float(data[3])
        if lane == 2:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width / 2 - 1)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width / 2 - 1)] += float(data[3])
        if lane == 3:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width / 2)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width / 2)] += float(data[3])
        if lane == 4:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width * 3 / 4 - 1)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width * 3 / 4 - 1)] += float(data[3])
        if lane == 5:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width * 3 / 4)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width * 3 / 4)] += float(data[3])
        if lane == 6:
            uniform_data[int(width / 4 - 1), int(abs(int(float(data[1])) + 150) / fen)] += add_n
            veh_v_total[int(width / 4 - 1), int(abs(int(float(data[1])) + 150) / fen)] += float(data[3])
        if lane == 7:
            uniform_data[int(width / 4), int(abs(int(float(data[1])) + 150) / fen)] += add_n
            veh_v_total[int(width / 4), int(abs(int(float(data[1])) + 150) / fen)] += float(data[3])
        if lane == 8:
            uniform_data[int(width / 2 - 1), int(abs(int(float(data[1])) + 150) / fen)] += add_n
            veh_v_total[int(width / 2 - 1), int(abs(int(float(data[1])) + 150) / fen)] += float(data[3])
        if lane == 9:
            uniform_data[int(width / 2), int(abs(int(float(data[1])) + 150) / fen)] += add_n
            veh_v_total[int(width / 2), int(abs(int(float(data[1])) + 150) / fen)] += float(data[3])
        if lane == 10:
            uniform_data[int(width * 3 / 4 - 1), int(abs(int(float(data[1])) + 150) / fen)] += add_n
            veh_v_total[int(width * 3 / 4 - 1), int(abs(int(float(data[1])) + 150) / fen)] += float(data[3])
        if lane == 11:
            uniform_data[int(width * 3 / 4), int(abs(int(float(data[1])) + 150) / fen)] += add_n
            veh_v_total[int(width * 3 / 4), int(abs(int(float(data[1])) + 150) / fen)] += float(data[3])
    else:
        # unsignal
        if x[-1] == 30 or y[-1] == 30:
            continue
        if lane == 0:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width / 4 - 1)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width / 4 - 1)] += float(data[3])
        if lane == 1:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width / 4)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width / 4)] += float(data[3])
        if lane == 2:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width / 2 - 1)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width / 2 - 1)] += float(data[3])
        if lane == 3:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width / 2)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width / 2)] += float(data[3])
        if lane == 4:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width * 3 / 4 - 1)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width * 3 / 4 - 1)] += float(data[3])
        if lane == 5:
            uniform_data[int(abs(int(float(data[2])) - 300) / fen), int(width * 3 / 4)] += add_n
            veh_v_total[int(abs(int(float(data[2])) - 300) / fen), int(width * 3 / 4)] += float(data[3])
        if lane == 6:
            uniform_data[int(width / 4 - 1), int(abs(int(float(data[1])) + 300) / fen)] += add_n
            veh_v_total[int(width / 4 - 1), int(abs(int(float(data[1])) + 300) / fen)] += float(data[3])
        if lane == 7:
            uniform_data[int(width / 4), int(abs(int(float(data[1])) + 300) / fen)] += add_n
            veh_v_total[int(width / 4), int(abs(int(float(data[1])) + 300) / fen)] += float(data[3])
        if lane == 8:
            uniform_data[int(width / 2 - 1), int(abs(int(float(data[1])) + 300) / fen)] += add_n
            veh_v_total[int(width / 2 - 1), int(abs(int(float(data[1])) + 300) / fen)] += float(data[3])
        if lane == 9:
            uniform_data[int(width / 2), int(abs(int(float(data[1])) + 300) / fen)] += add_n
            veh_v_total[int(width / 2), int(abs(int(float(data[1])) + 300) / fen)] += float(data[3])
        if lane == 10:
            uniform_data[int(width * 3 / 4 - 1), int(abs(int(float(data[1])) + 300) / fen)] += add_n
            veh_v_total[int(width * 3 / 4 - 1), int(abs(int(float(data[1])) + 300) / fen)] += float(data[3])
        if lane == 11:
            uniform_data[int(width * 3 / 4), int(abs(int(float(data[1])) + 300) / fen)] += add_n
            veh_v_total[int(width * 3 / 4), int(abs(int(float(data[1])) + 300) / fen)] += float(data[3])
# x_new = [i - min(x) for i in x]
# y_new = [i - min(y) for i in y]
# for i in range(len(x_new)):
#     test[y_new[i], x_new[i]] += 0.2
for i in range(width):
    if sum(uniform_data[i, :]) > 10:
        mask_[i, :] = 0
    if sum(uniform_data[:, i]) > 10:
        mask_[:, i] = 0
print(min(x), max(x), min(y), max(y))
for i in range(width):
    for j in range(width):
        if uniform_data[i, j] != 0:
            veh_v_total[i, j] = veh_v_total[i, j] / uniform_data[i, j] * add_n
# heat.clickmap(save_as="1.png") #点击图
# plt.axis('off')
# plt.subplot(1, 2, 1)
# plt.plot(x, y, '.')
# plt.axis('off')
# plt.subplot(1, 2, 2)
# ax = sns.heatmap(uniform_data, fmt="d", cmap="rainbow", vmin=0, vmax=0.6, mask=mask_)
# plt.axis('off')
# # heat.heatmap(save_as="signal_single.png") #热图
# plt.savefig("unsignal_multi.png")
# plt.show()
# plt.close()
plt.figure(0)
vx = sns.heatmap(veh_v_total, fmt="d", vmin=0, vmax=13, cmap="rainbow", mask=mask_)
plt.axis('off')
# plt.xlabel("test")
plt.title("Heat Map for Vehicle Velocity")
file_name = "signal_multi_velocity"
if not signal:
    file_name = "unsignal_multi_velocity"
pdf = PdfPages(file_name + '.pdf')
pdf.savefig()
plt.savefig(file_name + ".png")
plt.show()
plt.close()
pdf.close()
plt.figure(1)
# plt.subplot(1, 2, 2)
ax = sns.heatmap(uniform_data, fmt="d", cmap="rainbow", vmin=0, vmax=3, mask=mask_)
# heat.heatmap(save_as="signal_single.png") #热图
plt.axis('off')
plt.title("Heat Map for Vehicle Density")
file_name = "signal_multi_density"
if not signal:
    file_name = "unsignal_multi_density"
pdf = PdfPages(file_name + '.pdf')
pdf.savefig()
plt.savefig(file_name + ".png")
plt.show()
plt.close()
pdf.close()
# 绘制单路口
# sns.set()
# width = 30
# fen = 10
# uniform_data = np.zeros((width, width))
# veh_v_total = np.zeros((width, width))
# mask_ = np.ones_like(uniform_data)
# mask_[int(width / 2), :] = 0
# mask_[int(width / 2 - 1), :] = 0
# mask_[:, int(width / 2)] = 0
# mask_[:, int(width / 2 - 1)] = 0
# ma_json_path = "D:\\temp\\Signal\\single_400.txt"
# collisions_veh_numbers = open(ma_json_path, 'r').readlines()
# h_data = []
# x = []
# y = []
# add_n = 0.2
# for item in collisions_veh_numbers:
#     data = item.strip().split(" ")
#     lane = int(data[0])
#     x.append(int((int(float(data[1]))) / fen))
#     y.append(int((int(float(data[2]))) / fen))
#     if x[-1] == 15 or y[-1] == 15:
#         print(1)
#         continue
#     if lane == 0:
#         uniform_data[int(abs(int(float(data[2])) - 150) / fen), int(width / 2 - 1)] += add_n
#         veh_v_total[int(abs(int(float(data[2])) - 150) / fen), int(width / 2 - 1)] += float(data[3])
#     if lane == 1:
#         uniform_data[int(abs(int(float(data[2])) - 150) / fen), int(width / 2)] += add_n
#         veh_v_total[int(abs(int(float(data[2])) - 150) / fen), int(width / 2)] += float(data[3])
#     if lane == 2:
#         uniform_data[int(width / 2 - 1), int(abs(int(float(data[1])) + 150) / fen)] += add_n
#         veh_v_total[int(width / 2 - 1), int(abs(int(float(data[1])) + 150) / fen)] += float(data[3])
#     if lane == 3:
#         uniform_data[int(width / 2), int(abs(int(float(data[1])) + 150) / fen)] += add_n
#         veh_v_total[int(width / 2), int(abs(int(float(data[1])) + 150) / fen)] += float(data[3])
# print(min(x), max(x), min(y), max(y))
# # heat = HeatMap(h_data)
# for i in range(width):
#     for j in range(width):
#         if uniform_data[i, j] != 0:
#             veh_v_total[i, j] = veh_v_total[i, j] / uniform_data[i, j] * add_n
# # heat.clickmap(save_as="1.png") #点击图
# # plt.subplot(1, 2, 1)
# # plt.plot(x, y, '.')
# plt.figure(0)
# vx = sns.heatmap(veh_v_total, fmt="d", vmin=0, vmax=13, cmap="rainbow", mask=mask_)
# plt.axis('off')
# file_name = "signal_single_velocity"
# pdf = PdfPages(file_name + '.pdf')
# pdf.savefig()
# plt.savefig(file_name + ".png")
# plt.show()
# plt.close()
# pdf.close()
# plt.figure(1)
# # plt.subplot(1, 2, 2)
# ax = sns.heatmap(uniform_data, fmt="d", cmap="rainbow", vmin=0, vmax=0.6, mask=mask_)
# # heat.heatmap(save_as="signal_single.png") #热图
# plt.axis('off')
# file_name = "signal_single_density"
# pdf = PdfPages(file_name + '.pdf')
# pdf.savefig()
# plt.savefig(file_name + ".png")
# plt.show()
# plt.close()
# pdf.close()
# plt.show()
# #
# -*- coding: utf-8 -*-
# from pyheatmap.heatmap import HeatMap
# import numpy as np
# N = 10000
# X = np.random.rand(N) * 255  # [0, 255]
# Y = np.random.rand(N) * 255
# data = []
# for i in range(N):
#   tmp = [int(X[i]), int(Y[i]), 1]
#   data.append(tmp)
# heat = HeatMap(data)
# heat.clickmap(save_as="123.png") #点击图
# heat.heatmap(save_as="2.png") #热图

# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# sns.set()
# np.random.seed(0)
# uniform_data = np.zeros((50, 50))
# ax = sns.heatmap(uniform_data, fmt="d", cmap="rainbow")
# plt.show()
