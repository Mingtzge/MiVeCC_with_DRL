# 模型训练的主代码
import numpy as np
import tensorflow as tf
import os
import scipy.io as scio
import argparse
import cv2
import copy
from PIL import Image
from shutil import copyfile
import matplotlib.pyplot as plt
from traffic_interaction_scene import MultiTrafficInteraction
from traffic_interaction_scene import MultiVisible
import time
from model import MADDPG, M_MADDPG
from replay_buffer import ReplayBuffer
import sys


def create_init_update(oneline_name, target_name, tau=0.99):
    """
    :param oneline_name: the online model name
    :param target_name: the target model name
    :param tau: The proportion of each transfer from the online model to the target model
    :return:
    """
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in
                     zip(online_var, target_var)]  # 按照比例用online更新target

    return target_init, target_update


def get_agents_action(sta, sess, agent, noise_range=0.0):
    """
    :param sta: the state of the agent
    :param sess: the session of tf
    :param agent: the model of the agent
    :param noise_range: the noise range added to the agent model output
    :return: the action of the agent in its current state
    """
    agent1_action = agent.action(state=sta, sess=sess) + np.random.randn(1) * noise_range
    return agent1_action


def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update,
                agent_critic_target_update, sess, summary_writer, args):
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = agent_memory.sample(
        args.batch_size)
    total_next_obs_batch = np.array(total_next_obs_batch)
    total_obs_batch = np.array(total_obs_batch)
    total_act_batch = np.array(total_act_batch)
    act_next = agent_ddpg_target.action(total_next_obs_batch, sess)
    rew_batch = np.array(rew_batch)
    target = rew_batch.reshape(-1, 1) + args.gamma * agent_ddpg_target.Q(
        state=total_next_obs_batch, action=act_next, sess=sess)
    agent_ddpg.train_actor(state=total_obs_batch, action=total_act_batch, sess=sess, summary_writer=summary_writer,
                           lr=args.actor_lr)
    agent_ddpg.train_critic(state=total_obs_batch, action=total_act_batch, target=target, sess=sess,
                            summary_writer=summary_writer, lr=args.critic_lr)

    sess.run([agent_actor_target_update, agent_critic_target_update])  # 从online模型更新到target模型


def parse_args():
    parser = argparse.ArgumentParser("MADDPG experiments for multiagent traffic interaction environments")
    # Environment
    parser.add_argument("--num_episodes", type=int, default=1000, help="number of episodes")  # episode次数
    parser.add_argument("--s_agent_num", type=int, default=15, help="other agent numbers")
    parser.add_argument("--m_agent_num", type=int, default=60, help="other agent numbers")
    parser.add_argument("--actor_lr", type=float, default=1e-3, help="learning rate for Adam optimizer")  # 学习率
    parser.add_argument("--critic_lr", type=float, default=1e-2, help="learning rate for Adam optimizer")  # 学习率
    parser.add_argument("--gamma", type=float, default=0.80, help="discount factor")  # 折扣率
    parser.add_argument("--m_thr", type=float, default=0.2, help="impact factor of coordination model")  # 协作模型影响因子
    parser.add_argument("--trans_r", type=float, default=0.99, help="transfer rate for online model to target model")
    parser.add_argument("--batch_size", type=int, default=48,
                        help="number of episodes to optimize at the same time")  # 经验采样数目
    parser.add_argument("--s_num_units", type=int, default=96, help="number of units in the mlp")
    parser.add_argument("--m_num_units", type=int, default=256, help="number of units in the mlp")
    parser.add_argument("--collision_thr", type=float, default=2, help="the threshold for collision")
    parser.add_argument("--actual_lane", action="store_true", default=False, help="")
    parser.add_argument("--c_mode", type=str, default="closer",
                        help="the way of choosing closer cars, front ,front-end or closer")
    parser.add_argument("--batch_path", type=str, default="",
                        help="the path of batch mat files")
    parser.add_argument("--txt", type=str, default="pt_record",
                        help="the file name of recoding passed time")
    # Checkpointing
    parser.add_argument("--s_exp_name", type=str, default="s_test", help="name of the experiment")  # 单路口模型实验名
    parser.add_argument("--m_exp_name", type=str, default="m_test", help="name of the experiment")  # 多路口模型实验名
    parser.add_argument("--type", type=str, default="m_train", help="type of experiment train or test")
    parser.add_argument("--mat_path", type=str, default="./arvTimeNewVeh_new_900_l.mat", help="the path of mat file")
    parser.add_argument("--save_dir", type=str, default="model_data",
                        help="directory in which training state and model should be saved")  # 模型存储
    parser.add_argument("--save_rate", type=int, default=1,
                        help="save model once every time this many episodes are completed")  # 存储模型的回合间隔
    parser.add_argument("--load_dir", type=str, default="",
                        help="directory in which training state and model are loaded")  # 模型加载目录
    parser.add_argument("--video_name", type=str, default="",
                        help="if it not empty, program will generate a result video (.mp4 format defaultly)with the result imgs")
    parser.add_argument("--visible", action="store_true", default=False, help="visible or not")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)  # 恢复之前的模型，在 load-dir 或 save-dir
    parser.add_argument("--evaluation", action="store_true", default=False)  # 评测
    parser.add_argument("--display", action="store_true", default=False)  # 将训练完成后的测试过程显示出来
    parser.add_argument("--benchmark", action="store_true", default=False)  # 用保存的模型跑测试
    parser.add_argument("--multi_density", action="store_true", default=False)  # 是否采用多密度轮番训练
    parser.add_argument("--priori_knowledge", action="store_true", default=False)  # 用保存的模型跑测试
    parser.add_argument("--benchmark_iters", type=int, default=500, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")  # 训练曲线的目录
    parser.add_argument("--log", type=str, default="output.txt",
                        help="the name of output log file")  # 训练曲线的目录
    # model hyper-parameters
    # parser.add_argument('--image_size', type=int, default=600)
    # # training hyper-parameters
    # parser.add_argument('--cri_img_ch', type=int, default=1)
    # parser.add_argument('--act_img_ch', type=int, default=3)
    # parser.add_argument('--output_ch', type=int, default=1)
    # parser.add_argument('--num_epochs', type=int, default=100)
    # parser.add_argument('--num_epochs_decay', type=int, default=70)
    # parser.add_argument('--num_workers', type=int, default=2)
    # parser.add_argument('--lr', type=float, default=0.0002)
    # parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    # parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    # parser.add_argument('--augmentation_prob', type=float, default=0.4)
    #
    # parser.add_argument('--log_step', type=int, default=2)
    # parser.add_argument('--val_step', type=int, default=2)
    #
    # # misc
    # parser.add_argument('--mode', type=str, default='train')
    # parser.add_argument('--cri_model_type', type=str, default='AlexNet', help="AlexNet/ResNet50/ResNet101/ResNet152")
    # parser.add_argument('--act_model_type', type=str, default='UNet', help="UNet")
    # parser.add_argument('--model_path', type=str, default='./models')
    # parser.add_argument('--result_path', type=str, default='./result/')
    # parser.add_argument('--cuda_idx', type=int, default=1)
    return parser.parse_args()


def benchmark(pt_m_best, s_model, m_model, sess1, sess_m):
    total_c = 0
    c_count = 0
    n = 0
    start = 0
    pt_m = pt_m_best
    data = scio.loadmat(args.mat_path)  # 加载.mat数据
    arrive_time = data["arvTimeNewVeh"]
    env = MultiTrafficInteraction(arrive_time, args)
    while args.benchmark_iters <= 6000:
        while start < args.benchmark_iters:
            with sess1.as_default():
                with sess1.graph.as_default():
                    o_actions = s_model.action(env.ogm_total_s, sess1)
            with sess_m.as_default():
                with sess_m.graph.as_default():
                    m_actions = m_model.action([env.ogm_total_m], sess_m)
            actions = m_actions[0]
            v_m, acc_m, c_rate = env.step(o_actions, actions, start % 2)
            reward, collisions, v_v = env.scene_update()
            c_count += collisions
            if start % 500 == 0:
                pt_m = float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT
                print("benchmark: i: %s collisions_rate: %s pT_m: %s" % (
                    start, float(c_count) / (env.id_seq + total_c), pt_m))
            env.delete_vehicle()
            start += 1
        total_c += env.id_seq
        pt_m = float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT
        print("vehicle number: %s; collisions occurred number: %s; collisions rate: %s; pT_m: %s" % (
            total_c, c_count, float(c_count) / total_c, pt_m))
        # print(actions)
        if pt_m > pt_m_best or args.benchmark_iters >= 6000:
            break
        pt_m_best = pt_m
        args.benchmark_iters += 500
        n += 1
    return pt_m, n


def single_intersection_train():
    agent1_ddpg = MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                         num_units=args.s_num_units, nb_actions=args.s_agent_num)
    agent1_ddpg_target = MADDPG('agent1_target', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                num_units=args.s_num_units, nb_actions=args.s_agent_num)
    saver = tf.train.Saver()  # 为存储模型预备
    agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor',
                                                                              tau=args.trans_r)
    agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic',
                                                                                tau=args.trans_r)
    count_n = 0
    col = tf.Variable(0, dtype=tf.int8)
    collisions_op = tf.summary.scalar('collisions', col)
    etsm_col = tf.Variable(0, dtype=tf.int8)
    etsm_collisions_op = tf.summary.scalar('estimate_collisions', etsm_col)
    v_mean = tf.Variable(0, dtype=tf.float32)
    v_mean_op = tf.summary.scalar('v_mean', v_mean)
    collision_rate = tf.Variable(0, dtype=tf.float32)
    collision_rate_op = tf.summary.scalar('collision_rate', collision_rate)
    acc_mean = tf.Variable(0, dtype=tf.float32)
    acc_mean_op = tf.summary.scalar('acc_mean', acc_mean)
    reward_mean = tf.Variable(0, dtype=tf.float32)
    reward_mean_op = tf.summary.scalar('reward_mean', reward_mean)
    collisions_mean = tf.Variable(0, dtype=tf.float32)
    collisions_mean_op = tf.summary.scalar('collisions_mean', collisions_mean)
    estm_collisions_mean = tf.Variable(0, dtype=tf.float32)
    estm_collisions_mean_op = tf.summary.scalar('estm_collisions_mean', estm_collisions_mean)
    collisions_veh_numbers = tf.Variable(0, dtype=tf.int32)
    collisions_veh_numbers_op = tf.summary.scalar('collision_veh_numbers', collisions_veh_numbers)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init])
    if args.restore:
        model_path = os.path.join(args.save_dir, args.s_exp_name, "test_best.cptk")
        if not os.path.exists(model_path + ".meta"):
            model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.s_exp_name))
        saver.restore(sess, model_path)
        print("load cptk file from " + model_path)

    summary_writer = tf.summary.FileWriter(os.path.join(args.save_dir, args.s_exp_name, "log"),
                                           graph=tf.get_default_graph())

    # 设置经验池最大空间
    agent1_memory = ReplayBuffer(100000)
    collisions_memory = ReplayBuffer(10000)
    merge_summary = tf.summary.merge_all()
    reward_list = []
    collisions_list = []
    estm_collisions_list = []
    statistic_count = 0
    mean_window_length = 10
    state_now = []
    collisions_count = 0
    c_rate = 1.0
    rate_latest = 1.0
    test_rate_latest = 1.0
    # visible = Visible(lane_w=5)
    mat_file = ["arvTimeNewVeh_300.mat", "arvTimeNewVeh_600.mat", "arvTimeNewVeh_900.mat"]
    time_total = []
    data = scio.loadmat(args.mat_path)  # 加载.mat数据
    arrive_time = data["arvTimeNewVeh"]
    env = MultiTrafficInteraction(arrive_time, args)
    for epoch in range(args.num_episodes):
        collisions_count_last = collisions_count
        if args.multi_density:
            mat_path = mat_file[epoch % 3]
            data = scio.loadmat(mat_path)  # 加载.mat数据
            arrive_time = data["arvTimeNewVeh"]
            env = MultiTrafficInteraction(arrive_time, args)
        veh_num = env.id_seq
        for i in range(3000):
            state_now = copy.deepcopy(env.ogm_total_s)
            s_actions = get_agents_action(env.ogm_total_s, sess, agent1_ddpg, noise_range=0.1)
            actions = [0]
            v_m, acc_m, con_rate = env.step(s_actions, actions, i % 2)
            reward, collisions, veh_v = env.scene_update()
            reward_list.append(np.mean(env.reward_total_s))
            collisions_count += collisions
            for seq, re in enumerate(env.reward_total_s):
                if re is not None:
                    agent1_memory.add(np.array(state_now[seq]), np.array(s_actions[seq]), re,
                                      np.array(env.ogm_total_s[seq]),
                                      False)
            if count_n > 5000:
                statistic_count += 1
                time_t = time.time()
                train_agent(agent1_ddpg, agent1_ddpg_target, agent1_memory,
                            agent1_actor_target_update, agent1_critic_target_update, sess, summary_writer, args)
                time_total.append(time.time() - time_t)
                if sum(env.veh_num) > 0:
                    summary_writer.add_summary(sess.run(collisions_op, {col: collisions}), statistic_count)
                    summary_writer.add_summary(sess.run(v_mean_op, {v_mean: v_m}), statistic_count)
                    summary_writer.add_summary(sess.run(acc_mean_op, {acc_mean: acc_m}), statistic_count)
                summary_writer.add_summary(sess.run(reward_mean_op, {reward_mean: np.mean(reward_list)}),
                                           statistic_count)
                summary_writer.add_summary(
                    sess.run(collisions_veh_numbers_op, {collisions_veh_numbers: collisions_count}), statistic_count)
                if i % 100 == 0:
                    print(
                        "reward mean: %s; epoch: %s; i: %s; count: %s; collisions_count: %s; action mean: %s "
                        "; latest_c_rate: %s; test best c_rate: %s;a-lr: %0.6f; c-lr: %0.6f; time_mean: %s" % (
                            np.mean(reward_list), epoch, i, count_n, collisions_count, np.mean(s_actions), rate_latest,
                            test_rate_latest,
                            args.actor_lr, args.critic_lr, np.mean(time_total)))
            env.delete_vehicle()
            count_n += 1
        print('update model to ' + os.path.join(args.save_dir, args.s_exp_name, str(epoch) + '.cptk'))
        saver.save(sess, os.path.join(args.save_dir, args.s_exp_name, str(epoch) + '.cptk'))
        # if epoch % 10 == 0:
        if epoch % args.save_rate == 0:
            if rate_latest > (collisions_count - collisions_count_last) / float(env.id_seq - veh_num):
                rate_latest = (collisions_count - collisions_count_last) / float(env.id_seq - veh_num)
                copyfile(
                    os.path.join(args.save_dir, args.s_exp_name, str(epoch) + '.cptk.data-00000-of-00001'),
                    os.path.join(args.save_dir, args.s_exp_name, 'best.cptk.data-00000-of-00001'))
                copyfile(
                    os.path.join(args.save_dir, args.s_exp_name, str(epoch) + '.cptk.index'),
                    os.path.join(args.save_dir, args.s_exp_name, 'best.cptk.index'))
                copyfile(
                    os.path.join(args.save_dir, args.s_exp_name, str(epoch) + '.cptk.meta'),
                    os.path.join(args.save_dir, args.s_exp_name, 'best.cptk.meta'))
            summary_writer.add_summary(sess.run(collision_rate_op, {
                collision_rate: (collisions_count - collisions_count_last) / float(env.id_seq - veh_num)}),
                                       epoch)
        if epoch % 2 == 0 and args.benchmark:
            c_rate, n = benchmark(env, agent1_ddpg, sess)
            # c_count = 0
            # veh_num = env.id_seq
            # if test_rate_latest == 0:
            #     benchmark_num += 500
            #     test_rate_latest = 1
            # for b_i in range(1, benchmark_num):
            #     o_actions = get_agents_action(env.ogm, sess, agent1_ddpg, noise_range=0.1)
            #     actions = o_actions[0]
            #     v_m, acc_m = env.step(actions)
            #     state_next, reward, collisions, estm_collisions, collisions_per_veh = env.scene_update()
            #     for k in range(len(collisions_per_veh)):
            #         if collisions_per_veh[k][0] > 0:
            #             c_count += 1
            #     if b_i % 500 == 0:
            #         print("benchmark: i: %s collisions_rate: %s" % (b_i, float(c_count) / (env.id_seq - veh_num)))
            #     env.delete_vehicle()
            # c_rate = float(c_count) / (env.id_seq - veh_num)
            # print("vehicle number: %s; collisions occurred number: %s; collisions rate: %s" % (
            #     env.id_seq - veh_num, c_count, c_rate))
            if c_rate <= test_rate_latest or n > 0:
                test_rate_latest = c_rate
                copyfile(
                    os.path.join(args.save_dir, args.s_exp_name, str(epoch) + '.cptk.data-00000-of-00001'),
                    os.path.join(args.save_dir, args.s_exp_name, 'test_best.cptk.data-00000-of-00001'))
                copyfile(
                    os.path.join(args.save_dir, args.s_exp_name, str(epoch) + '.cptk.index'),
                    os.path.join(args.save_dir, args.s_exp_name, 'test_best.cptk.index'))
                copyfile(
                    os.path.join(args.save_dir, args.s_exp_name, str(epoch) + '.cptk.meta'),
                    os.path.join(args.save_dir, args.s_exp_name, 'test_best.cptk.meta'))
            elif np.random.rand() < 0.1 and args.benchmark_iters < 5900:
                model_path = os.path.join(args.save_dir, args.s_exp_name, "test_best.cptk")
                if not os.path.exists(model_path + ".meta"):
                    model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.s_exp_name))
                saver.restore(sess, model_path)
                print("load cptk file from " + model_path)
        if epoch % 30 == 29:
            args.actor_lr = args.actor_lr * 0.9
            args.critic_lr = args.critic_lr * 0.9
        # if len(time_total) > 800000:
        #     break
    sess.close()


def evaluation(n, model1, model_m, sess1, sess_m):
    record_str = str(n)
    for i in range(7):
        mat_file = "arvTimeNewVeh_new_" + str((i + 1) * 300) + "_multi3_3.mat"
        data = scio.loadmat(mat_file)  # 加载.mat数据
        arrive_time = data["arvTimeNewVeh"]
        env = MultiTrafficInteraction(arrive_time, args)
        for b_i in range(1000):
            with sess1.as_default():
                with sess1.graph.as_default():
                    o_actions = model1.action(env.ogm_total_s, sess1)
            with sess_m.as_default():
                with sess_m.graph.as_default():
                    m_actions = model_m.action([env.ogm_total_m], sess_m)
            actions = m_actions[0]
            # print(o_actions)
            v_m, acc_m, c_rate = env.step(o_actions, actions, b_i % 2)
            reward, collisions, v_v = env.scene_update(b_i)
            env.delete_vehicle()
        print("Evaluation: mat_path:%s; passed_veh num:%s; pT-m: %0.4f s" % (args.mat_path,
                                                                             env.passed_veh,
                                                                             float(env.passed_veh_step_total) / (
                                                                                     env.passed_veh + 0.0001) * env.deltaT))
        record_str += " " + str(float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT)
    if args.evaluation:
        efw.write(record_str + "\n")


def multi_intersections_train():
    single_section = tf.Graph()
    multi_sections = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess1 = tf.Session(config=config, graph=single_section)
    sess_m = tf.Session(config=config, graph=multi_sections)
    with sess1.as_default():
        with sess1.graph.as_default():
            agent1_ddpg_test = MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                      num_units=args.s_num_units, nb_actions=args.s_agent_num)
            saver_s = tf.train.Saver()

            sess1.run(tf.global_variables_initializer())
            # saver.restore(sess, './three_ma_weight/40.cptk')
            model_path = os.path.join(args.save_dir, args.s_exp_name, "test_best.cptk")
            if not os.path.exists(model_path + ".meta"):
                model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.s_exp_name))
            saver_s.restore(sess1, model_path)
            print("load single_section cptk file from " + model_path)
    with sess_m.as_default():
        with sess_m.graph.as_default():
            agent_m_ddpg = M_MADDPG('agent_m', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                    num_units=args.m_num_units, nb_actions=args.m_agent_num)
            agent_m_ddpg_target = M_MADDPG('agent_m_target', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                           num_units=args.m_num_units, nb_actions=args.m_agent_num)
            saver_m = tf.train.Saver()  # 为存储模型预备
            agent_m_actor_target_init, agent_m_actor_target_update = create_init_update('agent_m_actor',
                                                                                        'agent_m_target_actor',
                                                                                        tau=args.trans_r)
            agent_m_critic_target_init, agent_m_critic_target_update = create_init_update('agent_m_critic',
                                                                                          'agent_m_target_critic',
                                                                                          tau=args.trans_r)
            col = tf.Variable(0, dtype=tf.int8)
            collisions_op = tf.summary.scalar('collisions', col)
            v_mean = tf.Variable(0, dtype=tf.float32)
            v_mean_op = tf.summary.scalar('v_mean', v_mean)
            pass_time_mean = tf.Variable(0, dtype=tf.float32)
            pass_time_mean_op = tf.summary.scalar('pass_time_mean', pass_time_mean)
            acc_mean = tf.Variable(0, dtype=tf.float32)
            acc_mean_op = tf.summary.scalar('acc_mean', acc_mean)
            reward_mean = tf.Variable(0, dtype=tf.float32)
            reward_mean_op = tf.summary.scalar('reward_mean', reward_mean)
            collisions_veh_numbers = tf.Variable(0, dtype=tf.int32)
            collisions_veh_numbers_op = tf.summary.scalar('collision_veh_numbers', collisions_veh_numbers)
            summary_writer = tf.summary.FileWriter(os.path.join(args.save_dir, args.m_exp_name, "log"),
                                                   graph=tf.get_default_graph())
            sess_m.run(tf.global_variables_initializer())
            sess_m.run([agent_m_actor_target_init, agent_m_critic_target_init])
            if args.restore:
                model_path = os.path.join(args.save_dir, args.m_exp_name, "test_best.cptk")
                if not os.path.exists(model_path + ".meta"):
                    model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.m_exp_name))
                saver_m.restore(sess_m, model_path)
                print("load multi_section cptk file from " + model_path)
    # 设置经验池最大空间
    agent1_memory = ReplayBuffer(50000)
    reward_list = []
    collisions_list = []
    statistic_count = 0
    mean_window_length = 10
    rate_latest = 1.0
    test_pt_smallest = 1000
    data = scio.loadmat(args.mat_path)  # 加载.mat数据
    arrive_time = data["arvTimeNewVeh"]
    env = MultiTrafficInteraction(arrive_time, args)
    collisions_count = 0
    count_n = 0
    time_total = []
    mat_file = ["arvTimeNewVeh_300.mat", "arvTimeNewVeh_600.mat", "arvTimeNewVeh_900.mat"]
    for epoch in range(args.num_episodes):
        if args.evaluation:
            evaluation(epoch, agent1_ddpg_test, agent_m_ddpg, sess1, sess_m)
        if args.multi_density:
            mat_path = mat_file[epoch % 3]
            data = scio.loadmat(mat_path)  # 加载.mat数据
            arrive_time = data["arvTimeNewVeh"]
            env = MultiTrafficInteraction(arrive_time, args)
        for i in range(1000):
            state_now = copy.deepcopy(env.ogm_total_m)
            with sess1.as_default():
                with sess1.graph.as_default():
                    o_actions = agent1_ddpg_test.action(env.ogm_total_s, sess1)
            with sess_m.as_default():
                with sess_m.graph.as_default():
                    # if count_n > 100:
                    #     online_var = [i for i in tf.trainable_variables() if "agent_m_actor" in i.name]
                    #     print(sess_m.run(online_var[0]))
                    m_actions = agent_m_ddpg.action([env.ogm_total_m], sess_m)
            actions = m_actions[0]
            v_m, acc_m, c_rate = env.step(o_actions, actions, i % 2)
            reward, collisions, v_v = env.scene_update()
            reward_list.append(reward)
            reward_list = reward_list[-mean_window_length:]
            collisions_count += collisions
            agent1_memory.add(np.array(state_now), np.array(actions), reward, np.array(env.ogm_total_m),
                              False)
            if count_n > 500:
                statistic_count += 1
                time_t = time.time()
                with sess_m.as_default():
                    with sess_m.graph.as_default():
                        train_agent(agent_m_ddpg, agent_m_ddpg_target, agent1_memory,
                                    agent_m_actor_target_update, agent_m_critic_target_update, sess_m, summary_writer,
                                    args)
                        time_total.append(time.time() - time_t)
                        if sum(env.veh_num) > 0:
                            summary_writer.add_summary(sess_m.run(collisions_op, {col: collisions}), statistic_count)
                            summary_writer.add_summary(sess_m.run(v_mean_op, {v_mean: v_m}), statistic_count)
                            summary_writer.add_summary(sess_m.run(acc_mean_op, {acc_mean: acc_m}), statistic_count)
                            summary_writer.add_summary(sess_m.run(reward_mean_op, {reward_mean: np.mean(reward_list)}),
                                                       statistic_count)
                            summary_writer.add_summary(
                                sess_m.run(collisions_veh_numbers_op, {collisions_veh_numbers: collisions_count}),
                                statistic_count)
                        if i % 100 == 0:
                            print(
                                "reward mean: %s;epoch: %s;i: %s;count: %s;collisions_count: %s co_control rate: %s;"
                                "test best c_rate: %s;a-lr: %0.6f; c-lr: %0.6f; time_mean: %s" % (
                                    np.mean(reward_list), epoch, i, count_n, collisions_count, c_rate,
                                    test_pt_smallest,
                                    args.actor_lr, args.critic_lr, np.mean(time_total)))
            env.delete_vehicle()
            count_n += 1
        if count_n > 500:
            with sess_m.as_default():
                with sess_m.graph.as_default():
                    print('update model to ' + os.path.join(args.save_dir, args.m_exp_name, str(epoch) + '.cptk'))
                    saver_m.save(sess_m, os.path.join(args.save_dir, args.m_exp_name, str(epoch) + '.cptk'))
                    # if epoch % 10 == 0:
                    # online_var = [i for i in tf.trainable_variables() if "agent_m_critic" in i.name]
                    # print(sess_m.run(online_var[0]))
                    if epoch % 2 == 0 and args.benchmark:
                        pt_smallest, n = benchmark(test_pt_smallest, agent1_ddpg_test, agent_m_ddpg, sess1, sess_m)
                        if pt_smallest <= test_pt_smallest or n > 0:
                            test_pt_smallest = pt_smallest
                            copyfile(
                                os.path.join(args.save_dir, args.m_exp_name, str(epoch) + '.cptk.data-00000-of-00001'),
                                os.path.join(args.save_dir, args.m_exp_name, 'test_best.cptk.data-00000-of-00001'))
                            copyfile(
                                os.path.join(args.save_dir, args.m_exp_name, str(epoch) + '.cptk.index'),
                                os.path.join(args.save_dir, args.m_exp_name, 'test_best.cptk.index'))
                            copyfile(
                                os.path.join(args.save_dir, args.m_exp_name, str(epoch) + '.cptk.meta'),
                                os.path.join(args.save_dir, args.m_exp_name, 'test_best.cptk.meta'))
                        else:
                            model_path = os.path.join(args.save_dir, args.m_exp_name, "test_best.cptk")
                            if not os.path.exists(model_path + ".meta"):
                                model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.m_exp_name))
                            saver_m.restore(sess_m, model_path)
                            print("load cptk file from " + model_path)
                        summary_writer.add_summary(sess_m.run(pass_time_mean_op, {pass_time_mean: test_pt_smallest}),
                                                   epoch)
        if epoch % 30 == 29:
            args.actor_lr = args.actor_lr * 0.9
            args.critic_lr = args.critic_lr * 0.9
    sess1.close()
    sess_m.close()


def multi_intersection_batch_test():
    single_section = tf.Graph()
    multi_sections = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess1 = tf.Session(config=config, graph=single_section)
    sess_m = tf.Session(config=config, graph=multi_sections)
    with sess1.as_default():
        with sess1.graph.as_default():
            agent1_ddpg_test = MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                      num_units=args.s_num_units, nb_actions=args.s_agent_num)
            saver_s = tf.train.Saver()

            sess1.run(tf.global_variables_initializer())
            # saver.restore(sess, './three_ma_weight/40.cptk')
            model_path = os.path.join(args.save_dir, args.s_exp_name, "test_best.cptk")
            if not os.path.exists(model_path + ".meta"):
                model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.s_exp_name))
            saver_s.restore(sess1, model_path)
            print("load single_section cptk file from " + model_path)
    with sess_m.as_default():
        with sess_m.graph.as_default():
            agent_m_ddpg = M_MADDPG('agent_m', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                    num_units=args.m_num_units, nb_actions=args.m_agent_num)
            saver_m = tf.train.Saver()  # 为存储模型预备
            sess_m.run(tf.global_variables_initializer())
            model_path = os.path.join(args.save_dir, args.m_exp_name, "test_best.cptk")
            if not os.path.exists(model_path + ".meta"):
                model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.m_exp_name))
            saver_m.restore(sess_m, model_path)
            print("load multi_section cptk file from " + model_path)
    mat_files = os.listdir(args.batch_path)
    txt_w = open(args.txt + ".txt", "w")
    for mat_file in mat_files:
        mat_path = os.path.join(args.batch_path, mat_file)
        data = scio.loadmat(mat_path)  # 加载.mat数据
        arrive_time = data["arvTimeNewVeh"]
        visible = MultiVisible(l_mode="actual")
        env = MultiTrafficInteraction(arrive_time, args)
        size = (1280, 1280)
        fps = 20
        time_total = []
        video_writer = cv2.VideoWriter()
        if args.video_name != "":
            video_writer = cv2.VideoWriter(os.path.join("results_img", args.video_name + ".avi"),
                                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
        collisions_count = 0
        vehicles_v = []
        for i in range(36000):
            with sess1.as_default():
                with sess1.graph.as_default():
                    o_actions = agent1_ddpg_test.action(env.ogm_total_s, sess1)
            with sess_m.as_default():
                with sess_m.graph.as_default():
                    m_actions = agent_m_ddpg.action([env.ogm_total_m], sess_m)
            actions = m_actions[0]
            v_m, acc_m, c_rate = env.step(o_actions, actions, i % 2)
            reward, collisions, v_v = env.scene_update()
            if i % 10 == 0 and v_v[0] > 0:
                vehicles_v.append([len(vehicles_v)] + v_v)
            collisions_count += collisions
            if i == 1000:
                stop = 1
            if i % 1000 == 0:
                print("i: %s collisions_rate: %s; control rate: %s" % (i, float(collisions_count) / env.id_seq, c_rate))
            if env.passed_veh >= 100 and False:
                print("mat_path:%s; passed_veh num:%s; pT-m: %0.4f s" % (args.mat_path,
                                                                         env.passed_veh,
                                                                         float(env.passed_veh_step_total) / (
                                                                                 env.passed_veh + 0.0001) * env.deltaT))
                break
            if (args.visible or args.video_name != "") and i > 4800:
                visible.show(env, i)
                img = cv2.imread("results_img/%s.png" % i)
                cv2.putText(img, "step: " + str(i), (180, 100 + 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                cv2.putText(img, "veh: " + str(env.id_seq), (180, 120 + 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                            1)
                cv2.putText(img, "c-veh: %s" % collisions_count, (180, 140 + 80), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 0, 255),
                            1)
                cv2.putText(img, "c-r: %0.4f" % (float(collisions_count) / env.id_seq), (180, 160 + 80),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (0, 0, 255), 1)
                cv2.putText(img, "p_veh: " + str(env.passed_veh), (180, 180 + 80), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 0, 0),
                            1)
                cv2.putText(img,
                            "pT-m: %0.3f s" % (
                                    float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT),
                            (180, 200 + 80), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (0, 0, 0), 1)
                if args.visible:
                    cv2.imshow("result", img)
                    cv2.waitKey(1)
                if args.video_name != "":
                    video_writer.write(img)
            env.delete_vehicle()
            # if i < 2000:
            #    scio.savemat("test_mat.mat", {"veh_info": env.veh_info_record})
        video_writer.release()
        cv2.destroyAllWindows()
        choose_veh_visible = False
        if choose_veh_visible:
            choose_veh_info = [np.array(item) for item in env.choose_veh_info]
            plt.figure(0)
            color = ['r', 'g', 'b', 'y']
            y_units = ['distance [m]', 'velocity [m/s]', 'accelerate speed [m/s^2]']
            titles = ["The distance of the vehicle varies with the time",
                      "The velocity of the vehicle varies with the time",
                      "The accelerate spped of the vehicle varies with the time"]
            for m in range(len(y_units)):
                for n in range(4):
                    plt.plot(choose_veh_info[n][:, 0], choose_veh_info[n][:, m + 1], color[n])
                plt.legend(["lane-0", "lane-1", "lane-2", "lane-3"])
                plt.xlabel("time [s]")
                plt.ylabel(y_units[m])
                plt.title(titles[m], fontsize='small')
                plt.savefig("exp_result_imgs/%s.png" % (y_units[m].split(" ")[0]), dpi=600)
                plt.close()
        print("vehicle number: %s; collisions occurred number: %s; collisions rate: %s, time_mean: %s" % (
            env.id_seq, collisions_count, float(collisions_count) / env.id_seq, np.mean(time_total)))
        print("mat_path:%s; passed_veh num:%s; pT-m: %0.4f s" % (mat_path,
                                                                 env.passed_veh,
                                                                 float(env.passed_veh_step_total) / (
                                                                         env.passed_veh + 0.0001) * env.deltaT))
        txt_w.write("mat_path: %s collisions occurred number: %s collisions rate: %s pT-m: %0.4f s\n" % (
            mat_path, collisions_count, float(collisions_count) / env.id_seq, float(env.passed_veh_step_total) / (
                    env.passed_veh + 0.0001) * env.deltaT))
    txt_w.close()
    sess1.close()
    sess_m.close()


def get_p(lane, veh, lane_w=5):
    if int(lane / 3) == 0:
        p = [-150 * (1 - lane) - lane_w, veh["p"] - 150]
    elif int(lane / 3) == 1:
        p = [veh["p"] - 150, 150 * (1 - lane % 3) + lane_w]
    elif int(lane / 3) == 2:
        p = [150 * (1 - lane % 3) + lane_w, -veh["p"] + 150]
    else:
        p = [-veh["p"] + 150, -150 * (1 - lane % 3) - lane_w]
    return p


def record_data(env, single=False, density="2100", layer="EEC", n=""):
    seq = [2, 3, 0, 1]
    lane_l = 4
    dst_dir = os.path.join("./heatmap_data_dir", density, layer)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    file_name = os.path.join(dst_dir, "single_record_heat_map_data" + n + ".txt")
    if not single:
        seq = [0, 2, 4, 6, 8, 10, 5, 3, 1, 11, 9, 7]
        lane_l = 12
        file_name = os.path.join(dst_dir, "single_record_heat_map_data" + n + ".txt")
    d_w = open(file_name, "w")
    for lane in range(lane_l):
        for veh_id, veh in enumerate(env.veh_info[lane]):
            p = get_p(lane, veh)
            d_w.write("%s %s %s %s %s\n" % (
                seq[lane], p[0], p[1], float(env.veh_info[lane][veh_id]["v"]), float(env.veh_info[lane][veh_id]["a"])))
    d_w.close()


def multi_intersections_test():
    single_section = tf.Graph()
    multi_sections = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess1 = tf.Session(config=config, graph=single_section)
    sess_m = tf.Session(config=config, graph=multi_sections)
    with sess1.as_default():
        with sess1.graph.as_default():
            agent1_ddpg_test = MADDPG('agent1', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                      num_units=args.s_num_units, nb_actions=args.s_agent_num)
            saver_s = tf.train.Saver()

            sess1.run(tf.global_variables_initializer())
            # saver.restore(sess, './three_ma_weight/40.cptk')
            model_path = os.path.join(args.save_dir, args.s_exp_name, "test_best.cptk")
            if not os.path.exists(model_path + ".meta"):
                model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.s_exp_name))
            saver_s.restore(sess1, model_path)
            print("load single_section cptk file from " + model_path)
    with sess_m.as_default():
        with sess_m.graph.as_default():
            agent_m_ddpg = M_MADDPG('agent_m', actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                                    num_units=args.m_num_units, nb_actions=args.m_agent_num)
            saver_m = tf.train.Saver()  # 为存储模型预备
            sess_m.run(tf.global_variables_initializer())
            model_path = os.path.join(args.save_dir, args.m_exp_name, "test_best.cptk")
            if not os.path.exists(model_path + ".meta"):
                model_path = tf.train.latest_checkpoint(os.path.join(args.save_dir, args.m_exp_name))
            saver_m.restore(sess_m, model_path)
            print("load multi_section cptk file from " + model_path)
    # density = [2100, 1800, 1500, 1200, 900, 600, 300]
    # density = [2100]
    collisions_count = 0
    # for den in density:
    if True:
        # args.mat_path = "arvTimeNewVeh_new_%s_multi3_3.mat" % den
        data = scio.loadmat(args.mat_path)  # 加载.mat数据
        arrive_time = data["arvTimeNewVeh"]
        visible = MultiVisible(l_mode="actual")
        env = MultiTrafficInteraction(arrive_time, args)
        size = (1280, 1280)
        fps = 20
        time_total = []
        video_writer = cv2.VideoWriter()
        if args.video_name != "":
            video_writer = cv2.VideoWriter(os.path.join("results_img", args.video_name + ".avi"),
                                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
        vehicles_v = []
        # if collisions_count > 0:
        #     break
        for i in range(6000):
            with sess1.as_default():
                with sess1.graph.as_default():
                    o_actions = agent1_ddpg_test.action(env.ogm_total_s, sess1)
            with sess_m.as_default():
                with sess_m.graph.as_default():
                    m_actions = agent_m_ddpg.action([env.ogm_total_m], sess_m)
            actions = m_actions[0]
            # print(o_actions)
            v_m, acc_m, c_rate = env.step(o_actions, actions, i % 2)
            reward, collisions, v_v = env.scene_update(i)
            if i % 10 == 0 and v_v[0] > 0:
                vehicles_v.append([len(vehicles_v)] + v_v)
            collisions_count += collisions
            # if collisions_count > 0:
            #     break
            if i == 5390:
                stop = 1
            if i % 100 == 0:
                print("i: %s collisions_rate: %s; control rate: %s" % (
                    i, float(collisions_count) / env.id_seq, c_rate))
            if env.passed_veh >= 100 and False:
                print("mat_path:%s; passed_veh num:%s; pT-m: %0.4f s" % (args.mat_path,
                                                                         env.passed_veh,
                                                                         float(env.passed_veh_step_total) / (
                                                                                 env.passed_veh + 0.0001) * env.deltaT))
                break
            if (args.visible or args.video_name != "") and i > 500:
                visible.show(env, i)
                img = cv2.imread("results_img/%s.png" % i)
                cv2.putText(img, "step: " + str(i), (180, 100 + 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                            1)
                cv2.putText(img, "veh: " + str(env.id_seq), (180, 120 + 80), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 0, 0),
                            1)
                cv2.putText(img, "c-veh: %s" % collisions_count, (180, 140 + 80), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                            (0, 0, 255),
                            1)
                cv2.putText(img, "c-r: %0.4f" % (float(collisions_count) / env.id_seq), (180, 160 + 80),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (0, 0, 255), 1)
                cv2.putText(img, "p_veh: " + str(env.passed_veh), (180, 180 + 80), cv2.FONT_HERSHEY_COMPLEX,
                            0.5,
                            (0, 0, 0),
                            1)
                cv2.putText(img,
                            "pT-m: %0.3f s" % (
                                    float(env.passed_veh_step_total) / (env.passed_veh + 0.0001) * env.deltaT),
                            (180, 200 + 80), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (0, 0, 0), 1)
                if args.visible:
                    cv2.imshow("result", img)
                    cv2.waitKey(1)
                if args.video_name != "":
                    video_writer.write(img)
            env.delete_vehicle()
            # if i == 3000:
            # record_data(env, density=str(den), layer="EE", n=str(i))
                # break
            # if i < 2000:
            #    scio.savemat("test_mat.mat", {"veh_info": env.veh_info_record})
        # np.save("show_time.npy", env.show_time)
        video_writer.release()
        cv2.destroyAllWindows()
        choose_veh_visible = False
        if choose_veh_visible:
            choose_veh_info = [np.array(item) for item in env.choose_veh_info]
            plt.figure(0)
            color = ['r', 'g', 'b', 'y']
            y_units = ['distance [m]', 'velocity [m/s]', 'accelerate speed [m/s^2]']
            titles = ["The distance of the vehicle varies with the time",
                      "The velocity of the vehicle varies with the time",
                      "The accelerate spped of the vehicle varies with the time"]
            for m in range(len(y_units)):
                for n in range(4):
                    plt.plot(choose_veh_info[n][:, 0], choose_veh_info[n][:, m + 1], color[n])
                plt.legend(["lane-0", "lane-1", "lane-2", "lane-3"])
                plt.xlabel("time [s]")
                plt.ylabel(y_units[m])
                plt.title(titles[m], fontsize='small')
                plt.savefig("exp_result_imgs/%s.png" % (y_units[m].split(" ")[0]), dpi=600)
                plt.close()
        print("vehicle number: %s; collisions occurred number: %s; collisions rate: %s, time_mean: %s" % (
            env.id_seq, collisions_count, float(collisions_count) / env.id_seq, np.mean(time_total)))
        print("mat_path:%s; passed_veh num:%s; pT-m: %0.4f s" % (args.mat_path,
                                                                 env.passed_veh,
                                                                 float(env.passed_veh_step_total) / (
                                                                         env.passed_veh + 0.0001) * env.deltaT))
        # plt.figure(1)
        # plt.plot(np.array(vehicles_v)[:, 0], np.array(vehicles_v)[:, 1], "r")
        # plt.plot(np.array(vehicles_v)[:, 0], np.array(vehicles_v)[:, 2], "g")
        # plt.legend(["v_mean", "num_vehicles"])
        # plt.xlabel("time [s]")
        # plt.ylabel("value")
        # plt.show()
        # plt.close(i)
    sess1.close()
    sess_m.close()


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    args = parse_args()
    efw = None
    if args.evaluation:
        efw = open(args.m_exp_name + "_evaluation.txt", "w")
    sys.stdout = Logger(args.log)
    localtime = time.asctime(time.localtime(time.time()))
    print("\n")
    print("time:", localtime)
    if not os.path.exists("results_img"):
        os.makedirs("results_img")
    if not os.path.exists("exp_result_imgs"):
        os.makedirs("exp_result_imgs")
    if not os.path.exists(os.path.join(args.save_dir, args.s_exp_name)):
        os.makedirs(os.path.join(args.save_dir, args.s_exp_name))
    with open(os.path.join(args.save_dir, args.s_exp_name, "args.txt"), "w") as fw:
        fw.write(str(args))
    if not os.path.exists(os.path.join(args.save_dir, args.m_exp_name)):
        os.makedirs(os.path.join(args.save_dir, args.m_exp_name))
    with open(os.path.join(args.save_dir, args.m_exp_name, "args.txt"), "w") as fw:
        fw.write(str(args))
    if args.type == "m_train":
        multi_intersections_train()
    if args.type == "s_train":
        single_intersection_train()
    elif args.batch_path != "":
        multi_intersection_batch_test()
    else:
        multi_intersections_test()
    if args.evaluation:
        efw.close()
