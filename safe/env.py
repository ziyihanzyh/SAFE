import copy

import numpy as np
import random
import math
# import gym
import math




class RSU(object):
    def __init__(self, res, B, i2i, w, time_slot):
        self.res = res
        self.B = B
        self.i2i = i2i  # RSU之间的传输速率，定值

        self.workload = w

        self.res_t = [[1 for i in range(slot_s)] for i in range(time_slot)]
        self.B_t = [[1 for i in range(slot_s)] for i in range(time_slot)]

class Vehicle(object):
    def __init__(self, v_rate, start_time, zone, res, model):
        self.v = v_rate  # 行驶速度
        self.res = res  # 资源容量
        self.ratio_b = 0  # 带宽分配
        self.res_alloc = 0 # 资源分配

        self.noise = 174  # 单位dBm
        self.trans = random.randint(5, 10)

        self.zone = zone  # 当前时隙所在区域
        self.model = model
        self.cal_helper = 0
        self.cal_deliver = 0
        self.to_helper_data = 0

        self.start_time = start_time  # 当前任务请求时间
        self.ch = 0  # 当前时隙helper分配的资源量
        self.cd = 0  # 当前时隙deliver分配的资源量

        # 延时
        self.local_cal_time = 0
        self.helper_cal_time = 0
        self.deliver_cal_time = 0
        self.to_helper_time = 0
        self.to_deliver_time = 0
        self.total_time = 0

        self.wait_time = 0
        self.f_time = 0  # 任务交付时间

        # 决策变量
        self.helper = 0
        self.deliver = 0
        self.p1 = 0
        self.p2 = 0

        self.penalty = 0



class Environment(object):
    def __init__(self, args, layer_num):

        self.RSUs_num = args.env['RSUs_num']  # S
        self.zones_num = args.env['zones_num']  # M
        self.RSU_zone = []
        for i in range(self.RSUs_num):
            for j in range(int(self.zones_num / self.RSUs_num)):
                self.RSU_zone.append(i)
        self.slots_num = args.env['slots_num']  # T
        self.model_num = args.env['model_num']
        self.max_vehicle_num = args.env['max_vehicle_num']
        self.location_link = args.env['Lsm']
        self.generate_num = args.env['generate_num']

        self.model_layers = args.env['model_layers']
        self.model_cal = args.env['model_cal_local']
        self.model_cal_rsu = args.env['model_cal_rsu']
        self.model_data = args.env['model_data']
        self.trans_rate = args.env['trans_rate']

        self.RSU_resources = args.env['RSU_resources']
        for i in range(self.RSUs_num):
            #self.RSU_resources[i] = random.uniform(5, 6)  # 系数
            self.RSU_resources[i] = 12

        self.RSU_B = args.env['RSU_B']
        self.RSU_i2i = args.env['RSU_i2i']
        self.penalty = args.env['penalty']
        self.generate = args.env['generate']

        self.action_space = self.RSUs_num

        self.RSUs = []
        for i in range(self.RSUs_num):
            res = self.RSU_resources[i]
            B = self.RSU_B[i]
            i2i_rate = self.RSU_i2i[i]
            workload = 0
            self.RSUs.append(RSU(res, B, i2i_rate, workload, self.slots_num))

        self.t = 0
        # self.sim_timeslot = 2

        self.vehicle_num = 0
        self.Vehicles = []
        self.job, self.job_num = self.generate_job()

        self.RSU_dim = 1
        self.vehicle_dim = 6
        self.state = np.zeros(self.RSU_dim * self.RSUs_num + self.vehicle_dim * self.max_vehicle_num)

        self.action_dim = (self.RSUs_num + layer_num)*2
        self.action = np.zeros(self.action_dim * self.max_vehicle_num)

        # self.action_bound = [0, 1]

        #env = gym.make("CartPole-v1")

        #self.action_space = env.action_space
        #self.action_space.n = len(self.action)
        # self.action_space.shape[0] = 2
        # self.observation_space = env.observation_space
        # self.observation_space.shape[0] = len(self.state)

    def reset(self):
        self.t = 0


        self.job, self.job_num = self.generate_job()
        state = np.zeros(self.RSU_dim * self.RSUs_num + self.vehicle_dim * self.max_vehicle_num)
        for i in range(self.RSUs_num):
            self.RSUs[i].res_t = [[1 for i in range(slot_s)] for i in range(self.slots_num)]
            self.RSUs[i].B_t = [[1 for i in range(slot_s)] for i in range(self.slots_num)]
            self.RSUs[i].workload = 0
            #state[self.RSU_dim * i] = self.RSUs[i].res
            #state[self.RSU_dim * i + 1] = self.RSUs[i].B
            state[self.RSU_dim * i] = self.RSUs[i].workload
        

        self.Vehicles = self.job[self.t]
        self.vehicle_num = len(self.Vehicles)
        '''
        for i in range(self.vehicle_num):
            self.state[self.RSU_dim * self.RSUs_num + self.vehicle_dim * i] = self.Vehicles[i].v
            self.state[self.RSU_dim * self.RSUs_num + self.vehicle_dim * i + 1] = self.Vehicles[i].res
            self.state[self.RSU_dim * self.RSUs_num + self.vehicle_dim * i + 2] = self.Vehicles[i].zone
            self.state[self.RSU_dim * self.RSUs_num + self.vehicle_dim * i + 3] = self.Vehicles[i].model
            self.state[self.RSU_dim * self.RSUs_num + self.vehicle_dim * i + 3] = self.Vehicles[i].trans
        '''

        return state

    def generate_job(self):
        job = {}
        num = 0
        for t in range(self.slots_num):
            job[t] = []
            if t > self.slots_num - 20:
                continue
            # vehicles = []
            k = random.randint(0, 15)
            # if k % 3 != 0:
            #    job.append(vehicles)
            #    continue
            # else:

            if k % self.generate == 0:
                v_num = random.randint(1, int(self.generate_num))
                #v_num = self.max_vehicle_num
                num += v_num
                for i in range(v_num):
                    rate = random.uniform(0.5, 1)
                    #res = random.uniform(1, 1.5)  # 系数
                    res = 1
                    zone = random.randint(0, self.zones_num - 10)
                    model = 0
                    start_time = t

                    v = Vehicle(rate, start_time, zone, res, model)

                    # v = {'rate': rate, 'start_time': start_time, 'zone': zone, 'res': res, 'model': model, }
                    job[t].append(v)
                # job[t] = vehicles
        return job, num

    def get_reward(self):
        reward = []    
        for i in range(self.vehicle_num):
            if self.Vehicles[i].penalty == 0:
                reward.append(-self.Vehicles[i].total_time)
            else:
                reward.append(-self.Vehicles[i].total_time * self.penalty)
        return reward

    def step(self, action):
        helper = []
        deliver = []
        p1 = []
        p2 =[]

        for i in range(self.vehicle_num):
            helper.append(action[i][0])
            deliver.append(action[i][1])
            p1.append(action[i][2])
            p2.append(action[i][3])
            if p1[i] > p2[i]:
                p2[i] = p1[i]


        count = 0  # 已处理任务数
        #bandwidth_t = [[] for i in range(self.slots_num*2)]
        bandwidth_t = {}
        # local cal time
        for i in range(self.vehicle_num):
            model = self.Vehicles[i].model
            layer_num = self.model_layers[model]

            # action normalization
            # h = np.argmax(helper[i])
            # actions[i*self.action_dim + h] = 1

            self.Vehicles[i].helper = helper[i]
            self.Vehicles[i].deliver = deliver[i]
            self.Vehicles[i].p1 = p1[i]
            self.Vehicles[i].p2 = p2[i]

            # cal data
            cal_local = 0
            for j in range(self.Vehicles[i].p1):
                cal_local += self.model_cal[model][j]
            cal_helper = 0
            for j in range(self.Vehicles[i].p1, self.Vehicles[i].p2):
                cal_helper += self.model_cal_rsu[model][j]
            self.Vehicles[i].cal_helper = cal_helper


            cal_deliver = 0
            for j in range(self.Vehicles[i].p2, layer_num):
                cal_deliver += self.model_cal_rsu[model][j]
            self.Vehicles[i].cal_deliver = cal_deliver
                
            self.Vehicles[i].to_helper_data = self.model_data[self.Vehicles[i].model][self.Vehicles[i].p1]

            self.Vehicles[i].local_cal_time = cal_local / self.Vehicles[i].res
            location1 = math.floor(self.Vehicles[i].zone + self.Vehicles[i].local_cal_time * self.Vehicles[i].v)
            if (self.Vehicles[i].helper, location1) not in self.location_link.keys():
                flag = 0
                for w in range(self.RSUs_num):
                    if (w, location1) in self.location_link.keys():
                        self.Vehicles[i].helper = w
                        action[i][0] = w
                        flag = 1
                        break
                if flag == 0:
                    cal_time = sum(self.model_cal[model]) / self.Vehicles[i].res
                    self.Vehicles[i].total_time = cal_time
                    self.Vehicles[i].penalty = 1
                    count += 1
                    continue

            temp_t = self.Vehicles[i].start_time + self.Vehicles[i].local_cal_time
            if self.Vehicles[i].p1 == layer_num:
                self.Vehicles[i].total_time = self.Vehicles[i].local_cal_time
                count += 1
            else:
                one = math.ceil(temp_t*slot_s) // slot_s
                two = math.ceil(temp_t*slot_s) % slot_s
                if (one, two) not in bandwidth_t.keys():
                    bandwidth_t[(one, two)] = []
                bandwidth_t[(one, two)].append(i)
                self.Vehicles[i].channel_gain = 128.1 + 37.6 * math.log10(self.location_link[(self.Vehicles[i].helper, location1)]/1000)
                



        #resource_t = [[] for i in range(self.slots_num*3)]
        resource_t = {}

        for t in range(self.t, self.slots_num):
            if count == self.vehicle_num:
                break
            # to helper time | bandwidth allocation
            for t_1 in range(slot_s):
                if (t, t_1) in bandwidth_t.keys():
                    vehicles = bandwidth_t[(t, t_1)]
                    Q = [[] for i in range(self.RSUs_num)]
                    V = [[] for i in range(self.RSUs_num)]
                    for i in vehicles:
                        Qi = self.Vehicles[i].to_helper_data / (
                                self.RSUs[self.Vehicles[i].helper].B *
                                math.log(
                                    1 + self.Vehicles[i].trans * self.Vehicles[i].channel_gain / self.Vehicles[i].noise,
                                    2))
                        Q[self.Vehicles[i].helper].append(pow(Qi, 0.5))
                        V[self.Vehicles[i].helper].append(i)

                    for s in range(self.RSUs_num):
                        if len(V[s]) == 0:
                            continue
                        for k in range(len(V[s])):
                            i = V[s][k]
                            B = self.RSUs[s].B_t[t][t_1]
                            if B == 0:
                                if t_1 + 1 == slot_s:
                                    if (t + 1, 0) not in bandwidth_t.keys():
                                        bandwidth_t[(t + 1, 0)] = []
                                    bandwidth_t[(t + 1, 0)].append(i)
                                else:
                                    if (t, t_1+1) not in bandwidth_t.keys():
                                        bandwidth_t[(t, t_1+1)] = []
                                    bandwidth_t[(t, t_1+1)].append(i)
                                #bandwidth_t[t + 1].append(i)
                                self.Vehicles[i].wait_time += 1/slot_s
                                continue
                            self.Vehicles[i].ratio_b = Q[s][k] / sum(Q[s])

                            # update RSUs bandwidth
                            self.RSUs[s].B_t[t][t_1] -= self.Vehicles[i].ratio_b * B

                            rate = self.Vehicles[i].ratio_b * B \
                                   * self.RSUs[self.Vehicles[i].helper].B * pow(10, 6) \
                                   * math.log(
                                1 + self.Vehicles[i].trans * self.Vehicles[i].channel_gain / self.Vehicles[i].noise, 2)
                            rate = rate / slot_s
                            # self.Vehicles[i].to_helper_time = self.model_data[self.Vehicles[i].model][self.Vehicles[i].p1] / rate
                            if self.Vehicles[i].to_helper_data > rate:
                                self.Vehicles[i].to_helper_time += 1 /slot_s
                                self.Vehicles[i].to_helper_data -= rate
                                if t_1 + 1 == slot_s:
                                    if (t + 1, 0) not in bandwidth_t.keys():
                                        bandwidth_t[(t + 1, 0)] = []
                                    bandwidth_t[(t + 1, 0)].append(i)
                                else:
                                    if (t, t_1+1) not in bandwidth_t.keys():
                                        bandwidth_t[(t, t_1+1)] = []
                                    bandwidth_t[(t, t_1+1)].append(i)
                                #bandwidth_t[t + 1].append(i)
                            else:
                                self.Vehicles[i].to_helper_time += 1/slot_s  #self.Vehicles[i].to_helper_data / rate
                                self.Vehicles[i].to_helper_data = 0
                                helper_start_time = self.Vehicles[i].start_time + self.Vehicles[i].local_cal_time + \
                                                    self.Vehicles[i].to_helper_time + self.Vehicles[i].wait_time

                                res = helper_start_time
                                if self.Vehicles[i].p1 == self.Vehicles[i].p2:
                                    self.Vehicles[i].helper_cal_time = 0
                                    self.Vehicles[i].to_deliver_time = self.model_data[self.Vehicles[i].model][
                                                                           self.Vehicles[i].p2] / \
                                                                       self.RSUs[self.Vehicles[i].helper].i2i[
                                                                           self.Vehicles[i].deliver]
                                    deliver_start_time = helper_start_time + self.Vehicles[i].to_deliver_time + \
                                                         self.Vehicles[i].wait_time
                                    res = deliver_start_time
                                res_one = math.ceil(res*slot_s) //slot_s
                                res_two = math.ceil(res*slot_s) % slot_s
                                if (res_one, res_two) not in resource_t.keys():
                                    resource_t[(res_one, res_two)] = []
                                resource_t[(res_one, res_two)].append(i)



                # helper cal time | resource allocation
                if (t, t_1) in resource_t.keys():
                    vehicles = copy.deepcopy(resource_t[(t, t_1)])
                    R = [[] for i in range(self.RSUs_num)]
                    V = [[] for i in range(self.RSUs_num)]
                    for i in vehicles:
                        if self.Vehicles[i].total_time != 0:
                            continue
                        # helper or deliver
                        if self.Vehicles[i].cal_helper != 0:  # helper
                            Ri = self.Vehicles[i].cal_helper / self.RSUs[self.Vehicles[i].helper].res
                            R[self.Vehicles[i].helper].append(pow(Ri, 0.5))
                            V[self.Vehicles[i].helper].append(i)
                        else:  # deliver
                            #if self.Vehicles[i].cal_deliver == 0:
                            #    print(t)
                            Ri = self.Vehicles[i].cal_deliver / self.RSUs[self.Vehicles[i].deliver].res
                            R[self.Vehicles[i].deliver].append(pow(Ri, 0.5))
                            V[self.Vehicles[i].deliver].append(i)

                    for s in range(self.RSUs_num):
                        for k in range(len(V[s])):
                            i = V[s][k]
                            model = self.Vehicles[i].model
                            if sum(R[s]) == 0:
                                continue

                            if self.RSUs[s].res_t[t][t_1] == 0:
                                if t_1 + 1 == slot_s:
                                    if (t + 1, 0) not in resource_t.keys():
                                        resource_t[(t + 1, 0)] = []
                                    resource_t[(t + 1, 0)].append(i)
                                else:
                                    if (t, t_1+1) not in resource_t.keys():
                                        resource_t[(t, t_1+1)] = []
                                    resource_t[(t, t_1+1)].append(i)
                                #bandwidth_t[t + 1].append(i)
                                self.Vehicles[i].wait_time += 1/slot_s
                                continue
                            res_t = self.RSUs[s].res_t[t][t_1]
                            self.Vehicles[i].res_alloc = R[s][k] / sum(R[s]) * res_t  # 绝对比例
                            if self.Vehicles[i].cal_helper != 0:  # helper
                                self.Vehicles[i].ch = cal_func[self.Vehicles[i].model](self.Vehicles[i].res_alloc * self.RSUs[s].res) / slot_s
                                # self.Vehicles[i].helper_cal_time = self.Vehicles[i].cal_helper / self.Vehicles[i].ch
                                if self.Vehicles[i].cal_helper > self.Vehicles[i].ch:
                                    self.Vehicles[i].helper_cal_time += 1/slot_s
                                    self.Vehicles[i].cal_helper -= self.Vehicles[i].ch 
                                    if t_1 + 1 == slot_s:
                                        if (t + 1, 0) not in resource_t.keys():
                                            resource_t[(t + 1, 0)] = []
                                        resource_t[(t + 1, 0)].append(i)
                                    else:
                                        if (t, t_1 + 1) not in resource_t.keys():
                                            resource_t[(t, t_1 + 1)] = []
                                        resource_t[(t, t_1 + 1)].append(i)
                                    #if t + 1 not in resource_t.keys():
                                    #    resource_t[t + 1] = []
                                    #resource_t[t + 1].append(i)
                                else:
                                    self.Vehicles[i].helper_cal_time += 1/slot_s #self.Vehicles[i].cal_helper / self.Vehicles[i].ch
                                    self.Vehicles[i].cal_helper = 0

                                    if self.Vehicles[i].p2 == self.model_layers[self.Vehicles[i].model]:
                                        self.Vehicles[i].total_time = self.Vehicles[i].local_cal_time + self.Vehicles[
                                            i].to_helper_time + self.Vehicles[i].helper_cal_time + self.Vehicles[
                                                                          i].wait_time
                                        location2 = math.floor(self.Vehicles[i].zone + self.Vehicles[i].total_time * self.Vehicles[i].v)
                                        if (self.Vehicles[i].helper, location2) not in self.location_link.keys():
                                            self.Vehicles[i].penalty = 1                                
                                        count += 1
                                    else:
                                        self.Vehicles[i].to_deliver_time = self.model_data[model][self.Vehicles[i].p2] / \
                                                                           self.RSUs[self.Vehicles[i].helper].i2i[
                                                                               self.Vehicles[i].deliver]
                                        deliver_start_time = self.Vehicles[i].start_time + self.Vehicles[
                                            i].local_cal_time + \
                                                             self.Vehicles[i].to_helper_time \
                                                             + self.Vehicles[i].helper_cal_time + self.Vehicles[
                                                                 i].to_deliver_time + self.Vehicles[i].wait_time

                                        res_one = math.ceil(deliver_start_time * slot_s) // slot_s
                                        res_two = math.ceil(deliver_start_time * slot_s) % slot_s
                                        if (res_one, res_two) not in resource_t.keys():
                                            resource_t[(res_one, res_two)] = []
                                        resource_t[(res_one, res_two)].append(i)


                                # update helper RSU resource
                                self.RSUs[s].res_t[t][t_1] -= self.Vehicles[i].res_alloc



                            else:  # deliver
                                self.Vehicles[i].cd = cal_func[self.Vehicles[i].model](self.Vehicles[i].res_alloc * self.RSUs[s].res)  / slot_s

                                # self.Vehicles[i].deliver_cal_time = self.Vehicles[i].cal_deliver / self.Vehicles[i].cd
                                if self.Vehicles[i].cal_deliver > self.Vehicles[i].cd:
                                    self.Vehicles[i].deliver_cal_time += 1/slot_s
                                    self.Vehicles[i].cal_deliver -= self.Vehicles[i].cd
                                    #if t + 1 not in resource_t.keys():
                                    #    resource_t[t + 1] = []
                                    #resource_t[t + 1].append(i)
                                    if t_1 + 1 == slot_s:
                                        if (t + 1, 0) not in resource_t.keys():
                                            resource_t[(t + 1, 0)] = []
                                        resource_t[(t + 1, 0)].append(i)
                                    else:
                                        if (t, t_1 + 1) not in resource_t.keys():
                                            resource_t[(t, t_1 + 1)] = []
                                        resource_t[(t, t_1 + 1)].append(i)

                                else:
                                    self.Vehicles[i].deliver_cal_time += 1/slot_s #self.Vehicles[i].cal_deliver / self.Vehicles[i].cd
                                    self.Vehicles[i].cal_deliver = 0

                                    self.Vehicles[i].total_time = self.Vehicles[i].local_cal_time + self.Vehicles[
                                        i].to_helper_time \
                                                                  + self.Vehicles[i].helper_cal_time + self.Vehicles[
                                                                      i].to_deliver_time \
                                                                  + self.Vehicles[i].deliver_cal_time + self.Vehicles[
                                                                      i].wait_time
                                    location2 = math.floor(self.Vehicles[i].zone + self.Vehicles[i].total_time * self.Vehicles[i].v)
                                    if (self.Vehicles[i].deliver, location2) not in self.location_link.keys():
                                        #self.Vehicles[i].deliver = self.RSU_zone[location2]
                                        #action[i][1] = self.Vehicles[i].deliver
                                        flag = 0
                                        for w in range(self.RSUs_num):
                                            if (w, location2) in self.location_link.keys():
                                                self.Vehicles[i].deliver = w
                                                action[i][1] = w
                                                flag = 1
                                                break
                                        if flag == 0:
                                            self.Vehicles[i].penalty = 1
                                        #self.Vehicles[i].penalty = 1                                
                                    count += 1

                                # update deliver RSU resource
                                self.RSUs[s].res_t[t][t_1] -= self.Vehicles[i].res_alloc

        reward = self.get_reward()
        
        self.job[self.t] = self.Vehicles
        
        self.t = self.t + 1
        self.Vehicles = self.job[self.t]
        self.vehicle_num = len(self.Vehicles)
        
        # RSU workload
        for i in range(self.RSUs_num):
            use = 0
            for x in self.RSUs[i].res_t[self.t::]:
                use += sum(x)
            self.RSUs[i].workload = (self.slots_num - self.t) * slot_s - use

        # update state
        self.state = np.zeros(self.RSU_dim * self.RSUs_num + self.vehicle_dim * self.max_vehicle_num)
        for i in range(self.RSUs_num):
            #self.state[self.RSU_dim * i] = self.RSUs[i].res
            #self.state[self.RSU_dim * i + 1] = self.RSUs[i].B
            # self.RSUs[i].workload = max(self.RSUs[i].workload - 1, 0)
            self.state[self.RSU_dim * i] = self.RSUs[i].workload
        '''
        for i in range(self.vehicle_num):
            self.state[self.RSU_dim * self.RSUs_num + self.vehicle_dim * i] = self.Vehicles[i].v
            self.state[self.RSU_dim * self.RSUs_num + self.vehicle_dim * i + 1] = self.Vehicles[i].res
            self.state[self.RSU_dim * self.RSUs_num + self.vehicle_dim * i + 2] = self.Vehicles[i].zone
            self.state[self.RSU_dim * self.RSUs_num + self.vehicle_dim * i + 3] = self.Vehicles[i].model
            self.state[self.RSU_dim * self.RSUs_num + self.vehicle_dim * i + 3] = self.Vehicles[i].trans
        '''
        

        if self.t == self.slots_num - 1:
            done = True
        else:
            done = False

        x = False

        return self.state, reward, done, x, action

