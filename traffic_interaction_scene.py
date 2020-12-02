import numpy as np
import matplotlib.pyplot as plt
import os
import random
import copy as cp


class MultiTrafficInteraction:
    # vm = 0; % minimum
    # velocity
    # v0 = 10; % initial
    # velocity
    # vM = 13; % maximum
    # velocity
    # am = -3; % minimum
    # acceleration
    # aM = 3; % maximum
    def __init__(self, arrive_time, args, m_num=3, dis_ctl=150, deltaT=0.1, vm=5, vM=13, am=-3, aM=3, v0=10,
                 diff_max=220, lane_cw=5, show_col=False, virtual_l=True, reward_thr=-2, m_thr=0.2, pram1=0.2):
        self.args = args
        self.show_time = np.zeros([6010, 4, 4], dtype=np.float)
        self.pram1 = pram1
        self.m_thr = m_thr
        self.num_intervel = 5
        self.lane_num = m_num * 4
        self.valid_len = dis_ctl * m_num
        self.m_num = m_num
        self.virtual_l = virtual_l
        self.show_col = show_col
        self.collision_thr = args.collision_thr
        self.choose_veh = 15
        self.re_thr = reward_thr
        self.virtual_lane = [[] for i in range(pow(m_num, 2))]
        self.virtual_lane_reg = [[] for i in range(pow(m_num, 2))]
        self.virtual_lane_all = [[] for i in range(self.lane_num)]
        self.virtual_lane_all_index = [[] for i in range(self.lane_num)]
        self.choose_veh_info = [[] for i in range(self.lane_num)]
        self.veh_info_record = [[] for i in range(self.lane_num)]
        self.vm = vm
        self.vM = vM
        self.am = am
        self.aM = aM
        self.v0 = v0
        self.lane_cw = lane_cw
        self.c_mode = args.c_mode
        self.merge_p = [
            [0, -self.lane_cw, 0, self.lane_cw],
            [self.lane_cw, 0, -self.lane_cw, 0],
            [0, self.lane_cw, 0, -self.lane_cw],
            [-self.lane_cw, 0, self.lane_cw, 0]
        ]
        self.lane_section_map = [
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [2, 1, 0],
            [5, 4, 3],
            [8, 7, 6],
            [8, 5, 2],
            [7, 4, 1],
            [6, 3, 0],
            [6, 7, 8],
            [3, 4, 5],
            [0, 1, 2]
        ]
        self.adj_lane = [8, 7, 6, 11, 10, 9, 2, 1, 0, 5, 4, 3]
        self.section_lane_map = [
            [0, 3, 8, 11],
            [1, 3, 7, 11],
            [2, 3, 6, 11],
            [0, 4, 8, 10],
            [1, 4, 7, 10],
            [2, 4, 6, 10],
            [0, 5, 8, 9],
            [1, 5, 7, 9],
            [2, 5, 6, 9]
        ]
        self.relate_lane = [
            [3, 11, 4, 10, 5, 9],
            [6, 2, 7, 1, 8, 0],
            [9, 5, 10, 4, 11, 3],
            [0, 8, 1, 7, 2, 6]
        ]
        self.merge_p_all = [
            [0, 0, 0, 310, 160, 10, 0, 0, 0, - 310, - 160, - 10],
            [0, 0, 0, 160, 10, - 140, 0, 0, 0, - 160, - 10, 140],
            [0, 0, 0, 10, - 140, - 290, 0, 0, 0, - 10, 140, 290],
            [- 310, - 160, - 10, 0, 0, 0, 310, 160, 10, 0, 0, 0],
            [- 160, - 10, 140, 0, 0, 0, 160, 10, - 140, 0, 0, 0],
            [- 10, 140, 290, 0, 0, 0, 10, - 140, - 290, 0, 0, 0],
            [0, 0, 0, - 310, - 160, - 10, 0, 0, 0, 310, 160, 10],
            [0, 0, 0, - 160, - 10, 140, 0, 0, 0, 160, 10, - 140],
            [0, 0, 0, - 10, 140, 290, 0, 0, 0, 10, - 140, - 290],
            [310, 160, 10, 0, 0, 0, - 310, - 160, - 10, 0, 0, 0],
            [160, 10, - 140, 0, 0, 0, - 160, - 10, 140, 0, 0, 0],
            [10, - 140, - 290, 0, 0, 0, - 10, 140, 290, 0, 0, 0]
        ]
        self.merge_p_limit = [
            [0, 0, 0, -10, -10, -10, 0, 0, 0, 300, 300, 300],
            [0, 0, 0, 140, 140, 140, 0, 0, 0, 150, 150, 150],
            [0, 0, 0, 290, 290, 290, 0, 0, 0, 0, 0, 0],
            [300, 300, 300, 0, 0, 0, -10, -10, -10, 0, 0, 0],
            [150, 150, 150, 0, 0, 0, 140, 140, 140, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 290, 290, 290, 0, 0, 0],
            [0, 0, 0, 300, 300, 300, 0, 0, 0, -10, -10, -10],
            [0, 0, 0, 150, 150, 150, 0, 0, 0, 140, 140, 140],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 290, 290, 290],
            [-10, -10, -10, 0, 0, 0, 300, 300, 300, 0, 0, 0],
            [140, 140, 140, 0, 0, 0, 150, 150, 150, 0, 0, 0],
            [290, 290, 290, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        self.arrive_time = arrive_time
        self.current_time = 0
        self.passed_veh = 0
        self.passed_veh_step_total = 0
        self.closer_cars = []
        self.deltaT = deltaT
        self.dis_control = dis_ctl
        self.veh_num = np.zeros(self.lane_num, dtype=np.int)  # 每个车道车的数量
        self.veh_rec = np.zeros(self.lane_num, dtype=np.int)  # 每个车道车的总数量
        self.veh_info = [[] for i in range(self.lane_num)]
        self.diff_max = diff_max
        self.collision = False
        self.id_seq = 0
        self.delete_veh = []
        self.reward_total_s = np.zeros(pow(self.m_num, 2))
        self.ogm_total_s = np.zeros([pow(m_num, 2), self.args.s_agent_num, self.args.s_agent_num, 3])
        self.ogm_total_m = np.zeros([self.args.m_agent_num, self.args.m_agent_num, 3])
        init = True
        self.thr = pow(self.vM - self.vm, 2) / 4 / self.aM + 2.2
        # print("collision threshold: %s" % self.thr)
        while init:
            for i in range(self.lane_num):
                if self.veh_num[i] > 0:
                    init = False
            if init:
                self.scene_update()

    def scene_update(self, count=0):
        self.current_time += self.deltaT
        collisions = 0
        vehicles_v = []
        self.delete_veh.clear()
        self.virtual_lane_reg = [[] for i in range(pow(self.m_num, 2))]
        self.ogm_total_s = np.zeros([pow(self.m_num, 2), self.args.s_agent_num, self.args.s_agent_num, 3])
        self.ogm_total_m = np.zeros([self.args.m_agent_num, self.args.m_agent_num, 3])
        self.reward_total_s = np.zeros(pow(self.m_num, 2))
        for it_, vir_lane in enumerate(self.virtual_lane):
            ogm = np.zeros([self.args.s_agent_num, self.args.s_agent_num, 3], dtype=np.float)  # 构建单一交叉口的状态矩阵
            virtual_lane_4 = [[] for i in range(4)]
            reward = []
            for l_q, lane in enumerate(self.section_lane_map[it_]):
                t_distance = 2
                d_distance = 100
                for _itr in vir_lane:
                    if _itr[1] != self.adj_lane[lane] and self.veh_info[_itr[1]][_itr[2]]["control"] and \
                            self.veh_info[_itr[1]][_itr[2]]["vir_p"] > -self.lane_cw and \
                            self.veh_info[_itr[1]][_itr[2]]["section"] == it_:
                        virtual_lane_4[l_q].append(
                            [_itr[0] + 2 * self.merge_p[self.section_lane_map[it_].index(_itr[1])][l_q], _itr[1],
                             _itr[2], self.veh_info[_itr[1]][_itr[2]]["v"]])
                virtual_lane_4[l_q] = sorted(virtual_lane_4[l_q], key=lambda item: item[0])  # 对虚拟车道的车辆重新通过距离进行排序
                for v_p, car in enumerate(virtual_lane_4[l_q]):
                    closer = [-1, -1]
                    closer_p = 150
                    dis_front = 100
                    if v_p == 0 and car[1] == lane:
                        self.veh_info[car[1]][car[2]]["header"] = True
                    if v_p > 0:
                        dis_front = virtual_lane_4[l_q][v_p][0] - virtual_lane_4[l_q][v_p - 1][0]
                    acc_reward = np.log(pow(min(dis_front / 10, 1.5), 12) + 0.0000001) * \
                                 self.veh_info[car[1]][car[2]]["a"]
                    if dis_front < 10 and self.veh_info[car[1]][car[2]]["a"] < 0:
                        acc_reward = min(acc_reward, 120)
                    back = 1
                    if car[1] == lane:
                        if len(virtual_lane_4[l_q]) >= 2:
                            if v_p == 0:
                                d_distance = abs(virtual_lane_4[l_q][v_p][0] - virtual_lane_4[l_q][v_p + 1][0])
                                closer = [virtual_lane_4[l_q][v_p + 1][1], virtual_lane_4[l_q][v_p + 1][2]]
                                closer_p = virtual_lane_4[l_q][v_p + 1][0]
                                back = -1
                            elif v_p == len(virtual_lane_4[l_q]) - 1:
                                d_distance = abs(virtual_lane_4[l_q][v_p - 1][0] - virtual_lane_4[l_q][v_p][0])
                                closer = [virtual_lane_4[l_q][v_p - 1][1], virtual_lane_4[l_q][v_p - 1][2]]
                                closer_p = virtual_lane_4[l_q][v_p - 1][0]
                            else:
                                d_front = abs(virtual_lane_4[l_q][v_p - 1][0] - virtual_lane_4[l_q][v_p][0])
                                d_back = abs(virtual_lane_4[l_q][v_p][0] - virtual_lane_4[l_q][v_p + 1][0])
                                if d_front < d_back:
                                    d_distance = d_front
                                    closer = [virtual_lane_4[l_q][v_p - 1][1], virtual_lane_4[l_q][v_p - 1][2]]
                                    closer_p = virtual_lane_4[l_q][v_p - 1][0]
                                else:
                                    back = -1
                                    d_distance = d_back
                                    closer = [virtual_lane_4[l_q][v_p + 1][1], virtual_lane_4[l_q][v_p + 1][2]]
                                    closer_p = virtual_lane_4[l_q][v_p + 1][0]
                            if d_distance != 0:
                                t_distance = (self.veh_info[car[1]][car[2]]["vir_p"] + self.merge_p[l_q][
                                    self.section_lane_map[it_].index(closer[0])] - (
                                                      self.veh_info[closer[0]][closer[1]]["vir_p"] +
                                                      self.merge_p[self.section_lane_map[it_].index(closer[0])][
                                                          l_q])) / (self.veh_info[car[1]][car[2]]["v"] -
                                                                    self.veh_info[closer[0]][closer[1]][
                                                                        "v"] + 0.1)
                            self.veh_info[car[1]][car[2]]["closer_p"] = closer_p
                        else:
                            self.veh_info[car[1]][car[2]]["closer_p"] = 150
                        if 0 < t_distance < 1:
                            t_reward = -pow(1.5 / np.tanh(-t_distance), 2)
                        else:
                            t_reward = 2
                        re_b = min(20, max(-20, self.pram1 * acc_reward + np.log(
                            pow(d_distance / 10, 10) + 0.0001) + t_reward))
                        re = min(20, max(-20, min(acc_reward, t_reward)))
                        reward.append([re_b, car[1], car[2], self.veh_info[car[1]][car[2]]["v"]])
                        self.veh_info[car[1]][car[2]]["reward"] = re_b
                        if len(virtual_lane_4[l_q]) >= 2 and virtual_lane_4[l_q][v_p][1] != \
                                virtual_lane_4[l_q][v_p - back][1]:
                            # 不是同一车道，计算欧式距离，来判断是否发生了碰撞
                            d_distance = np.sqrt(
                                np.power(self.veh_info[car[1]][car[2]]["vir_p"] + self.merge_p[l_q][
                                    self.section_lane_map[it_].index(closer[0])], 2) + np.power(
                                    self.veh_info[closer[0]][closer[1]]["vir_p"] +
                                    self.merge_p[self.section_lane_map[it_].index(closer[0])][l_q], 2))
                        if d_distance < self.collision_thr:
                            self.veh_info[car[1]][car[2]]["collision"] += 1
                            collisions += 1
            if it_ == 2:
                v = [i[3] for i in virtual_lane_4[1]]
                vehicles_v = [np.mean(v), len(v)]
            if self.args.type == "s_train":
                reward_seq = sorted(reward, key=lambda item1: item1[0])  # 对每辆车的reward进行排序
            else:
                reward_seq = random.sample(sorted(reward, key=lambda item1: -item1[0]),
                                           min(len(reward), self.args.s_agent_num))
            for i in range(min(len(reward_seq), self.args.s_agent_num)):
                p = self.veh_info[reward_seq[i][1]][reward_seq[i][2]]["vir_p"]
                new_squence = sorted(virtual_lane_4[self.section_lane_map[it_].index(reward_seq[i][1])],
                                     key=lambda item1: abs(p - item1[0]))  # 对虚拟车道的车辆重新通过距离进行排序
                new_squence = np.array([[it[0], self.veh_info[it[1]][it[2]]["v"],
                                         self.veh_info[it[1]][it[2]]["a"]] for it in new_squence])  # 采用虚拟距离
                self.veh_info[reward_seq[i][1]][reward_seq[i][2]]["seq_p"] = [it_, i]
                ogm[i][0:min(len(new_squence), self.args.s_agent_num)][:] = new_squence[
                                                                            0:min(len(new_squence),
                                                                                  self.args.s_agent_num)][:]
            re = [r[0] for r in reward_seq]
            r_v = [r[-1] - 12 for r in reward_seq]
            if len(re) > 0:
                self.reward_total_s[it_] = np.mean(re[0:min(len(reward_seq), self.args.s_agent_num)]) + np.mean(
                    r_v[0:min(len(reward_seq), self.args.s_agent_num)])
            self.ogm_total_s[it_] = ogm
            self.virtual_lane_reg[it_] = cp.deepcopy(virtual_lane_4)
        small_m_interval = []
        for i in range(self.lane_num):
            self.virtual_lane_all[i].clear()
            self.virtual_lane_all[i] += [[item["p"], i, j] for j, item in enumerate(self.veh_info[i]) if item["p"] > 0]
            relate_lane = self.relate_lane[i // 3]
            for r_l in relate_lane:
                self.virtual_lane_all[i] += [[item["p"] + self.merge_p_all[i][r_l], r_l, j] for j, item in
                                             enumerate(self.veh_info[r_l]) if
                                             (item["p"] + self.merge_p_all[i][r_l]) > 0 and item["p"] >
                                             self.merge_p_limit[i][r_l]]
            self.virtual_lane_all[i] = sorted(self.virtual_lane_all[i], key=lambda item: item[0])
            if len(self.virtual_lane_all[i]) > 1:
                mean_interval = (self.virtual_lane_all[i][-1][0] - self.virtual_lane_all[i][0][0]) / (
                        len(self.virtual_lane_all[i]) - 1)
                len_ = len(self.virtual_lane_all[i])
                small_m_interval += [[abs((self.virtual_lane_all[i][min(j + self.num_intervel, len_ - 1)][0] -
                                           self.virtual_lane_all[i][max(j - self.num_intervel, 0)][0]) / (
                                                  min(j + self.num_intervel, len_ - 1) - max(j - self.num_intervel,
                                                                                             0)) - mean_interval),
                                      item[1], item[2]] for j, item in enumerate(self.virtual_lane_all[i]) if
                                     item[1] == i]
            self.virtual_lane_all_index[i] = [[item[1], item[2]] for j, item in enumerate(self.virtual_lane_all[i])]
        interval_total = []
        if len(small_m_interval) > 0:
            small_m_interval = sorted(small_m_interval, key=lambda item: -item[0])[
                               0:min(self.args.m_agent_num, len(small_m_interval))]
            for i, item in enumerate(small_m_interval):
                interval_total.append(item[0])
                index = self.virtual_lane_all_index[item[1]].index([item[1], item[2]])
                p = self.virtual_lane_all[item[1]][index][0]
                new_squence = sorted(self.virtual_lane_all[item[1]],
                                     key=lambda item1: abs(p - item1[0]))  # 对虚拟车道的车辆重新通过距离进行排序
                new_squence = np.array([[it[0], self.veh_info[it[1]][it[2]]["v"],
                                         self.veh_info[it[1]][it[2]]["a"]] for it in new_squence])  # 采用虚拟距离
                self.veh_info[item[1]][item[2]]["m_seq"] = i
                self.ogm_total_m[i][0:min(len(new_squence), self.args.m_agent_num)][:] = new_squence[
                                                                                         0:min(len(new_squence),
                                                                                               self.args.m_agent_num)][
                                                                                         :]
        v_total = []
        for i in range(self.lane_num):
            for j, item in enumerate(self.veh_info[i]):
                if self.veh_info[i][j]["control"]:
                    if self.veh_info[i][j]["finish"]:
                        self.veh_info[i][j]["control"] = False
                    else:
                        v_total.append(self.veh_info[i][j]["v"])
                if self.veh_info[i][j]["p"] < -self.dis_control or self.veh_info[i][j]["collision"] > 0:
                    # 驶出交通路口或者发生碰撞, 删除该车辆
                    self.delete_veh.append([i, j])
                elif self.veh_info[i][j]["p"] < - self.lane_cw and self.veh_info[i][j]["control"]:
                    self.veh_info[i][j]["finish"] = True
                    self.veh_info[i][j]["control"] = False
                    self.passed_veh += 1
                    self.passed_veh_step_total += self.veh_info[i][j]["step"]
            # 添加新车
            self.add_new_veh(i)
        for i in range(self.lane_num):
            for j, item in enumerate(self.veh_info[i]):
                section = 8
                index = 2 - min(2, max(int((self.veh_info[i][j]["p"] + 2 * self.lane_cw) / 150), 0))
                if 3 > index >= 0:
                    vir_item = [self.veh_info[i][j]["p"] - 150 * (2 - index), i, j]
                    self.veh_info[i][j]["vir_p"] = vir_item[0]
                    if self.lane_section_map[i][index] == section:
                        stop = 0
                    if self.lane_section_map[i][index] == section and self.veh_info[i][j]["section"] != section:
                        self.show_time[count][self.section_lane_map[section].index(i)][:] = [
                            self.veh_info[i][j]["vir_p"],
                            self.veh_info[i][j]["v"],
                            self.veh_info[i][j]["a"],
                            self.veh_info[i][j]["reward"]]
                        self.veh_info[i][j]["closer_p"] = 150
                self.veh_info[i][j]["section"] = self.lane_section_map[i][index]
        self.virtual_lane.clear()
        return np.mean(v_total) - 12, collisions, vehicles_v
        # return np.mean(interval_total), collisions, vehicles_v

    def add_new_veh(self, i):
        if self.current_time >= self.arrive_time[self.veh_rec[i]][i]:
            p = self.valid_len
            self.veh_info[i].append(
                {
                    "seq_p": [-1, -1],
                    "m_seq": - 1,
                    "p": p,
                    "m_intervel": -1,
                    "vir_p": 150,
                    "v": self.v0,
                    "a": self.aM,
                    "lane": i,
                    "seq_in_lane": self.veh_rec[i],
                    "control": True,
                    "step": 0,
                    "reward": 10,
                    "closer_p": 150,
                    "collision": 0,
                    "finish": False,
                    "en_braking": False,
                    "estm_collision": 0,
                    "section": -1,
                    "header":False,
                    "estm_arrive_time": abs(p / self.v0),
                    "id_info": [self.id_seq, self.veh_num[i]]
                })
            # "id_info":[在所有车中的出现次序,在当前车道中的出现次序]
            self.veh_num[i] += 1
            self.veh_rec[i] += 1
            self.veh_info_record[i].append([])
            self.id_seq += 1

    def delete_vehicle(self):
        # 删除旧车
        self.delete_veh = sorted(self.delete_veh, key=lambda item: -item[1])
        for d_i in self.delete_veh:
            if len(self.veh_info[d_i[0]]) > d_i[1]:
                self.veh_info[d_i[0]].pop(d_i[1])
                if self.veh_num[d_i[0]] > 0:
                    self.veh_num[d_i[0]] -= 1
            else:
                print("except!!!")

    def judge_fb(self, i, j):
        #  函数功能：判断最邻近车辆在后面还是前面
        back = True
        closer_p = self.veh_info[i][j]["closer_p"]
        if closer_p < self.veh_info[i][j]["vir_p"]:
            back = False
        return back

    def step(self, action, m_action, train_flag):
        v = []
        a = []
        num_total = 0
        co_num = 0
        self.virtual_lane = [[] for i in range(pow(self.m_num, 2))]
        for i in range(self.lane_num):
            for j, item in enumerate(self.veh_info[i]):
                choose = False
                num_total += 1
                rcd_a = self.veh_info[i][j]["reward"] / float(3)
                back = self.judge_fb(i, j)
                if back:
                    rcd_a = abs(rcd_a)
                seq_p = self.veh_info[i][j]["seq_p"]
                m_seq = self.veh_info[i][j]["m_seq"]
                eval_a = rcd_a
                if seq_p[0] != -1 and self.args.type == "s_train" and train_flag == 1:
                    # if rcd_a > 0:
                    eval_a = action[seq_p[0]][seq_p[1]]
                elif seq_p[0] != -1 and self.veh_info[i][j]["reward"] > self.re_thr:
                    if rcd_a*action[seq_p[0]][seq_p[1]] > 0:
                        eval_a = max(rcd_a, action[seq_p[0]][seq_p[1]])
                # else:
                #     eval_a = min(rcd_a, action[seq_p[0]][seq_p[1]])
                if self.args.priori_knowledge:
                    # eval_a = rcd_a
                    if self.veh_info[i][j]["vir_p"] > 50:
                        if j > 0 and self.veh_info[i][j - 1]["p"] > 0 and self.veh_info[i][j]["p"] - \
                                self.veh_info[i][j - 1]["p"] < self.thr:
                            # if self.veh_info[i][j + 1]["p"] - self.veh_info[i][j]["p"] < 6:
                            eval_a = self.am
                            # if self.veh_info[i][j - 1]["en_braking"]:
                            #     eval_a = self.am
                            if self.veh_num[i] > j + 1 and self.veh_info[i][j + 1]["p"] - self.veh_info[i][j][
                                "p"] < self.thr:
                                eval_a = 0
                        elif self.veh_num[i] > j + 1 and self.veh_info[i][j + 1]["p"] - self.veh_info[i][j][
                            "p"] < self.thr:
                            eval_a = self.aM
                if m_seq != -1:
                    co_num += 1
                    eval_a = self.m_thr * m_action[m_seq] + (1 - self.m_thr) * eval_a
                self.veh_info[i][j]["seq_p"] = [-1, -1]
                self.veh_info[i][j]["m_seq"] = -1
                self.veh_info[i][j]["a"] = min(self.aM, max(self.am, eval_a))
                section = self.veh_info[i][j]["section"]
                if self.veh_info[i][j]["header"]:
                    self.veh_info[i][j]["a"] = self.aM
                    self.veh_info[i][j]["header"] = False
                if self.veh_info[i][j]["a"] == self.am:
                    self.veh_info[i][j]["en_braking"] = True
                else:
                    self.veh_info[i][j]["en_braking"] = False
                v_new = min(self.vM,
                            max(self.veh_info[i][j]["v"] + self.veh_info[i][j]["a"] * self.deltaT,
                                self.vm))
                if v_new != self.veh_info[i][j]["v"]:
                    if self.veh_info[i][j]["v"] + self.veh_info[i][j]["a"] * self.deltaT > self.vM or \
                            self.veh_info[i][j]["v"] + self.veh_info[i][j]["a"] * self.deltaT < self.vm:
                        self.veh_info[i][j]["p"] = self.veh_info[i][j]["p"] - (
                                self.veh_info[i][j]["v"] + v_new) * self.deltaT * 0.5
                    else:
                        self.veh_info[i][j]["p"] = self.veh_info[i][j]["p"] - self.veh_info[i][j][
                            "v"] * self.deltaT - 0.5 * \
                                                   self.veh_info[i][j]["a"] * pow(self.deltaT, 2)
                    self.veh_info[i][j]["v"] = v_new
                else:
                    self.veh_info[i][j]["p"] = self.veh_info[i][j]["p"] - self.veh_info[i][j]["v"] * self.deltaT
                    self.veh_info[i][j]["a"] = 0
                # self.veh_info[i][j]["v"] = min(self.vM,
                #                                max(self.veh_info[i][j]["v"] + self.veh_info[i][j]["a"] * self.deltaT,
                #                                    self.vm))
                # self.veh_info[i][j]["p"] = self.veh_info[i][j]["p"] - self.veh_info[i][j]["v"] * self.deltaT - 0.5 * \
                #                            self.veh_info[i][j]["a"] * pow(self.deltaT, 2)
                if self.veh_info[i][j]["p"] > 0:
                    v.append(self.veh_info[i][j]["v"])
                    a.append(self.veh_info[i][j]["a"])
                self.veh_info[i][j]["step"] += 1
                index = 2 - min(2, max(int((self.veh_info[i][j]["p"] + 2 * self.lane_cw) / 150), 0))
                if 3 > index >= 0:
                    vir_item = [self.veh_info[i][j]["p"] - 150 * (2 - index), i, j]
                    self.virtual_lane[self.lane_section_map[i][index]].append(vir_item)
                    self.veh_info[i][j]["vir_p"] = vir_item[0]
                if not self.veh_info[i][j]["control"] or self.veh_info[i][j]["vir_p"] < - self.lane_cw:
                    self.veh_info[i][j]["v"] = self.vM  # 出交叉口之后所有车的速度都变为最大速度离开交叉口
                    # todo
                    # 将速度突改变为渐变
        return np.mean(v), np.mean(a), co_num / num_total


class MultiVisible:
    def __init__(self, lane_w=5, m_num=3, control_dis=300, l_mode="actual", c_mode="closer"):
        self.px = [[] for i in range(4)]
        self.py = [[] for i in range(4)]
        self.lane_w = lane_w
        self.m_num = m_num
        self.color_m = np.zeros((4, 433)) - 1
        self.l_mode = l_mode
        self.c_mode = c_mode
        self.control_dis = control_dis
        self.marker = ["1", "3", "2", "4"]
        self.lpx = []
        self.lpy = []

    def get_p(self, lane, veh):
        if int(lane / 3) == 0:
            p = [-150 * (1 - lane) - self.lane_w, veh["p"] - 150]
        elif int(lane / 3) == 1:
            p = [veh["p"] - 150, 150 * (1 - lane % 3) + self.lane_w]
        elif int(lane / 3) == 2:
            p = [150 * (1 - lane % 3) + self.lane_w, -veh["p"] + 150]
        else:
            p = [-veh["p"] + 150, -150 * (1 - lane % 3) - self.lane_w]
        return p

    def show(self, env, i):
        plt.figure(1, figsize=(12.8, 12.8), dpi=100)
        c_c = ["g", "b", "y", "brown", "gray", "deeppink"]
        point = []
        # self.lpx.clear()
        # self.lpy.clear()
        for k in range(4):
            self.px[k].clear()
            self.py[k].clear()
        if self.l_mode == "actual":
            for lane in range(self.m_num * 4):
                for veh_id, veh in enumerate(env.veh_info[lane]):
                    p = self.get_p(lane, veh)
                    # self.px[int(lane / self.m_num)].append(p[0])
                    # self.py[int(lane / self.m_num)].append(p[1])
                    c_level = max(float(env.veh_info[lane][veh_id]["v"] - 10) / float(env.vM - 10), 0)
                    # c_level = max(float(env.veh_info[veh[1]][veh[2]]["v"] - 11) / float(env.vM - 11), 0)

                    a = 0
                    if env.veh_info[lane][veh_id]["m_seq"] != -1:
                        a = 0
                    plt.plot(p[0], p[1], c=[c_level, a, a], ls='', marker=self.marker[int(lane / self.m_num)],
                             markersize=5)  # 画出当前 ax 列表和 ay 列表中的值的图形
                    # if veh["reward"] < -5 and veh["closer"][0] >= 0 and veh["control"]:
                    #     p_closer = self.get_p(veh["closer"][0],
                    #                           env.veh_info[veh["closer"][0]][veh["closer"][1]])
                    #     self.lpx.append([p[0], p_closer[0]])
                    #     self.lpy.append([p[1], p_closer[1]])
                    # plt.text(p[0], p[1], "%0.f" % veh["v"])
            # for l in range(len(self.lpx)):
            #     plt.plot(self.lpx[l], self.lpy[l])
            # plt.plot(self.px[0], self.py[0], c='r', ls='', marker='1', markersize=5)  # 画出当前 ax 列表和 ay 列表中的值的图形
            # plt.plot(self.px[1], self.py[1], c='r', ls='', marker='3', markersize=5)  # 画出当前 ax 列表和 ay 列表中的值的图形
            # plt.plot(self.px[2], self.py[2], c='r', ls='', marker='2', markersize=5)  # 画出当前 ax 列表和 ay 列表中的值的图形
            # plt.plot(self.px[3], self.py[3], c='r', ls='', marker='4', markersize=5)  # 画出当前 ax 列表和 ay 列表中的值的图形
        elif self.l_mode == "virtual":
            lane = 1
            for veh_id, veh in enumerate(env.virtual_lane_all[lane]):
                if veh[0] < 600:
                    p = p = [-150 * (1 - lane) - self.lane_w, veh[0] - 300]
                    # self.px[int(lane / self.m_num)].append(p[0])
                    # self.py[int(lane / self.m_num)].append(p[1])
                    c_level = max(float(env.veh_info[veh[1]][veh[2]]["v"] - 11) / float(env.vM - 11), 0)
                    a = 0
                    if env.veh_info[veh[1]][veh[2]]["m_seq"] != -1:
                        a = 1
                    plt.plot(p[0], p[1], c=[c_level, 0, 0], ls='', marker=self.marker[int(veh[1] / self.m_num)],
                             markersize=5)  # 画出当前 ax 列表和 ay 列表中的值的图形
                    # if veh[1] in [3, 4, 5]:
                    #     plt.text(p[0] + 10, p[1], "%0.f" % veh[1])
        plt.plot([-self.control_dis, self.control_dis], [self.control_dis / 2, self.control_dis / 2], c='y', ls='--',
                 markersize=1)
        plt.plot([-self.control_dis, self.control_dis], [0, 0], c='y', ls='--', markersize=1)
        plt.plot([-self.control_dis, self.control_dis], [-self.control_dis / 2, -self.control_dis / 2], c='y', ls='--',
                 markersize=1)
        plt.plot([self.control_dis / 2, self.control_dis / 2], [self.control_dis, -self.control_dis], c='y', ls='--',
                 markersize=1)
        plt.plot([0, 0], [self.control_dis, -self.control_dis], c='y', ls='--', markersize=1)
        plt.plot([-self.control_dis / 2, -self.control_dis / 2], [self.control_dis, -self.control_dis], c='y', ls='--',
                 markersize=1)

        plt.plot([-self.control_dis, self.control_dis],
                 [self.control_dis / 2 + 2 * self.lane_w, self.control_dis / 2 + 2 * self.lane_w], c='b', ls='-')
        plt.plot([-self.control_dis, self.control_dis],
                 [self.control_dis / 2 - 2 * self.lane_w, self.control_dis / 2 - 2 * self.lane_w], c='b', ls='-')
        plt.plot([-self.control_dis, self.control_dis],
                 [-self.control_dis / 2 + 2 * self.lane_w, -self.control_dis / 2 + 2 * self.lane_w], c='b', ls='-')
        plt.plot([-self.control_dis, self.control_dis],
                 [-self.control_dis / 2 - 2 * self.lane_w, -self.control_dis / 2 - 2 * self.lane_w], c='b', ls='-')
        plt.plot([-self.control_dis, self.control_dis], [2 * self.lane_w, 2 * self.lane_w], c='b', ls='-')
        plt.plot([-self.control_dis, self.control_dis], [-2 * self.lane_w, - 2 * self.lane_w], c='b', ls='-')

        plt.plot([self.control_dis / 2 + 2 * self.lane_w, self.control_dis / 2 + 2 * self.lane_w],
                 [self.control_dis, -self.control_dis], c='b', ls='--')
        plt.plot([self.control_dis / 2 - 2 * self.lane_w, self.control_dis / 2 - 2 * self.lane_w],
                 [self.control_dis, -self.control_dis], c='b', ls='--')
        plt.plot([2 * self.lane_w, + 2 * self.lane_w],
                 [self.control_dis, -self.control_dis], c='b', ls='--')
        plt.plot([- 2 * self.lane_w, - 2 * self.lane_w],
                 [self.control_dis, -self.control_dis], c='b', ls='--')
        plt.plot([-self.control_dis / 2 + 2 * self.lane_w, -self.control_dis / 2 + 2 * self.lane_w],
                 [self.control_dis, -self.control_dis], c='b', ls='--')
        plt.plot([-self.control_dis / 2 - 2 * self.lane_w, -self.control_dis / 2 - 2 * self.lane_w],
                 [self.control_dis, -self.control_dis], c='b', ls='--')

        plt.xlim((-self.control_dis + 5, self.control_dis + 5))
        plt.ylim((-self.control_dis + 5, self.control_dis + 5))
        if not os.path.exists("results_img"):
            os.makedirs("results_img")
        plt.savefig("results_img/%s.png" % i)
        plt.close()
        # plt.show()
