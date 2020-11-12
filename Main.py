from Game import Env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 参数
BATCH_SIZE = 16
LR = 0.005                   # 学习率
EPSILON = 0.9            # 最优选择动作百分比(有0.9的几率是最大选择，还有0.1是随机选择，增加网络能学到的Q值)
GAMMA = 0.8                 # 奖励递减参数（衰减作用，如果没有奖励值r=0，则衰减Q值）
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率100次循环更新一次
MEMORY_CAPACITY = 40      # 记忆库大小
N_ACTIONS = 4  # 棋子的动作0，1，2，3
N_STATES = 1


def trans_torch(list1):
    list1 = np.array(list1)
    l1 = np.where(list1 == 1, 1, 0)
    l2 = np.where(list1 == 2, 1, 0)
    l3 = np.where(list1 == 3, 1, 0)
    b = np.array([l1, l2, l3])
    return b


# 神经网络
class Net(nn.Module):
    def __init__(self, num_input):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(3, 25, 5, 1, 0)
        self.f0 = nn.Linear(num_input, 25)
        self.f1 = nn.Linear(25, 16)
        self.f1.weight.data.normal_(0, 0.1)
        self.f2 = nn.Linear(16, 4)
        self.f2.weight.data.normal_(0, 0.1)
        self.net2 = True


    def forward(self, x):
        if self.net2:
            x = x.view(x.size(0), -1)
            # x = self.c1(x)
            x = self.f0(x)
            x = F.relu(x)
            # x = x.view(x.size(0), -1)
            x = self.f1(x)
            x = F.relu(x)
            action = self.f2(x)
        else:
            x = self.c1(x)
            x = F.relu(x)
            x = x.view(x.size(0), -1)
            x = self.f1(x)
            x = F.relu(x)
            action = self.f2(x)
        return action


class DQN(object):
    def __init__(self,num_input):
        self.eval_net, self.target_net = Net(num_input), Net(num_input)  # DQN需要使用两个神经网络
        # eval为Q估计神经网络 target为Q现实神经网络
        self.learn_step_counter = 0  # 用于 target 更新计时，100次更新一次
        self.memory_counter = 0  # 记忆库记数
        self.memory = list(np.zeros((MEMORY_CAPACITY, 4)))  # 初始化记忆库用numpy生成一个(容量,4)大小的全0矩阵，
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式

    def choose_action(self, x, epsilon):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # ??
        # 这里只输入一个 sample, x为场景
        if np.random.uniform() < epsilon:   # 选最优动作
            actions_value = self.eval_net.forward(x)  # 将场景输入Q估计神经网络
            # torch.max(input,dim)返回dim最大值并且在第二个位置返回位置比如(tensor([0.6507]), tensor([2]))
            action = torch.max(actions_value, 1)[1].data.numpy()  # 返回动作最大值
        else:   # 选随机动作
            action = np.array([np.random.randint(0, N_ACTIONS)])  # 比如np.random.randint(0,2)是选择1或0
        return action

    def store_transition(self, s, a, r, s_):
        # 如果记忆库满了, 就覆盖老数据，2000次覆盖一次
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index] = [s, a, r, s_]
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新,每100次
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            # 将所有的eval_net里面的参数复制到target_net里面
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # 抽取记忆库中的批数据
        # 从2000以内选择32个数据标签
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_s = []
        b_a = []
        b_r = []
        b_s_ = []
        for i in sample_index:
            b_s.append(self.memory[i][0])
            b_a.append(np.array(self.memory[i][1], dtype=np.int32))
            b_r.append(np.array([self.memory[i][2]], dtype=np.int32))
            b_s_.append(self.memory[i][3])
        b_s = torch.FloatTensor(b_s)  # 取出s
        b_a = torch.LongTensor(b_a)  # 取出a
        b_r = torch.FloatTensor(b_r)  # 取出r
        b_s_ = torch.FloatTensor(b_s_)  # 取出s_
        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        # t = self.eval_net(b_s) 个人感觉，gather使得代码更加优雅。实际是一个从t中进行索引的东西。
        # gather 是按照index拿出目标索引的函数，第一个输入为dim.
        # gather 对应了argmax a. DQN是off-policy的。 b_r + q_next - 1_eval
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1) 找到action的Q估计(关于gather使用下面有介绍)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach Q现实
        # p = q_next.max(1)[0]
        # b_r: nsamples*1 q_next.max(1)[0]:nsamples
        # q_eval: nsamples*1
        q_target = b_r + GAMMA * torch.unsqueeze(q_next.max(1)[0], 1)  # shape (batch, 1) DQL核心公式
        # 这步走的不好，将导致下一次判断做大调整。
        loss = self.loss_func(q_eval, q_target)  # 计算误差
        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()  # 反向传递
        self.optimizer.step()

dict_a = {0:'up',1:'down',2:'left',3:'right'}
a_lib= []
env = Env()
dqn = DQN(env.total_grid)  # 定义 DQN 系统
# 400步
study = 1
epsilon = EPSILON
for i_episode in range(2000):
    s = env.start_env()
    s = trans_torch(s)
    his_step = np.zeros_like(s[0, :])
    if epsilon % 100 == 0:
        epsilon += 0.01
    while True:
        env.display()   # 显示实验动画
        a = dqn.choose_action(s, epsilon)  # 选择动作
        a_lib.append(dict_a[int(a)])
        # 选动作, 得到环境反馈
        done, r, s_ = env.step(a, his_step)  # done 表示是否结束。其中掉入陷阱'1'或者走出迷宫'2'都算结束
        s_ = trans_torch(s_)
        this_step = np.where(np.array(s_[0,:]) == 1, 1, 0)
        his_step = np.where((his_step + this_step) >= 1, 1, 0)
        # 存记忆 好的坏的都要存储，不然网络不能学习完整（比如不知道终点就在附近了）
        dqn.store_transition(s, a, r, s_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            if study == 1:
                print('经验池学习')
                study = 0
            dqn.learn()  # 记忆库满了就进行学习
        if done == 1 or done == 2:    # 如果回合结束, 进入下回合
            if done == 1:
                print('epoch', i_episode, r, '失败')
                a_lib = []
            if done == 2:
                print('epoch', i_episode, r, '成功')
                print(a_lib)
                a_lib = []
            break
        s = s_


