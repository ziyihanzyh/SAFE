import torch
from env import Environment, Setting
import copy
import numpy as np

class Actor(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation, vehicle_num,
                 RSUs_num, layer_num):
        super(Actor, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=hidden)
        self.layer_2 = torch.nn.Linear(in_features=hidden, out_features=hidden // 2)
        self.layer_list = []
        for i in range(vehicle_num):
            self.layer_list.append(torch.nn.Linear(in_features=hidden // 2, out_features=RSUs_num))
            self.layer_list.append(torch.nn.Linear(in_features=hidden // 2, out_features=RSUs_num))
            self.layer_list.append(torch.nn.Linear(in_features=hidden // 2, out_features=layer_num))
            self.layer_list.append(torch.nn.Linear(in_features=hidden // 2, out_features=layer_num))
        self.output_layer = torch.nn.Linear(in_features=hidden // 2, out_features=output_dimension)
        self.output_activation = output_activation
        self.num = vehicle_num

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        # layer_3_output = self.output_layer(layer_2_output)

        # x1, x2 = layer_3_output.split([3, 20], dim=1)
        x = []
        for i in range(self.num):
            x.append(self.output_activation(self.layer_list[4 * i](layer_2_output)))
            x.append(self.output_activation(self.layer_list[4 * i + 1](layer_2_output)))
            x.append(self.output_activation(self.layer_list[4 * i + 2](layer_2_output)))
            x.append(self.output_activation(self.layer_list[4 * i + 3](layer_2_output)))

        output = x[0]
        for i in range(1, len(x)):
            output = torch.cat([output, x[i]], dim=1)
        # x1 = self.output_activation(self.layer_out1(layer_2_output))
        # x2 = self.output_activation(self.layer_out2(layer_2_output))

        # output = torch.cat([x1, x2], dim=1)

        # output = self.output_activation(layer_3_output)
        return output


def get_action_deterministically(action_probabilities, RSUs_num, layer_num, vehicle_num):
    #action_probabilities = self.get_action_probabilities(state)
    discrete_action = []
    temp = RSUs_num + layer_num
    for i in range(vehicle_num):
        # rsu_select = action_probabilities[i * temp: i * temp + self.RSUs_num]
        # layer_select = action_probabilities[i * temp + self.RSUs_num : i * temp + self.RSUs_num + self.layer_num]
        rsu_select_1 = action_probabilities[i * temp: i * temp + RSUs_num]
        rsu_select_2 = action_probabilities[i * temp + RSUs_num: i * temp + RSUs_num * 2]
        layer_select_1 = action_probabilities[
                         i * temp + RSUs_num * 2: i * temp + RSUs_num * 2 + layer_num]
        layer_select_2 = action_probabilities[
                         i * temp + RSUs_num * 2 + layer_num: i * temp + RSUs_num * 2 + layer_num * 2]
        discrete_action.append(np.argmax(rsu_select_1))
        discrete_action.append(np.argmax(rsu_select_2))
        discrete_action.append(np.argmax(layer_select_1))
        discrete_action.append(np.argmax(layer_select_2))
    # discrete_action.append(np.argmax(action_probabilities[0:3]))
    # discrete_action.append(np.argmax(action_probabilities[3:23]))
    # discrete_action = np.argmax(action_probabilities)
    return discrete_action



def use():
    arg = Setting(vehicle, generate, STEPS_PER_EPISODE)
    env = Environment(args=arg, layer_num=layer_num)
    # env_1 = Environment(args=arg, layer_num=layer_num)
    # actor_local = torch.load("./result/max_b=10_v_1_g_3_dc_0.99/run0-episode2999-actor.pt")
    #state_r = env.reset
    env.reset()
    job_num = env.job_num
    # env_1.job = copy.deepcopy(env.job)

    done = False
    st = 0
    while not done and st < STEPS_PER_EPISODE:
        st += 1
        action = []
        state = []
        v_num = env.vehicle_num
        for i in range(v_num):
            
            state.append(copy.deepcopy(state_r))
            state[i][env.RSU_dim * env.RSUs_num] = env.Vehicles[i].v
            state[i][env.RSU_dim * env.RSUs_num + 1] = env.Vehicles[i].res
            state[i][env.RSU_dim * env.RSUs_num + 2] = env.Vehicles[i].zone
            state[i][env.RSU_dim * env.RSUs_num + 3] = env.Vehicles[i].model
            state[i][env.RSU_dim * env.RSUs_num + 4] = env.Vehicles[i].trans
            state[i][env.RSU_dim * env.RSUs_num + 5] = v_num
            state_tensor = torch.tensor(state[i], dtype=torch.float32).unsqueeze(0)
            action_probabilities = actor_local.forward(state_tensor).squeeze(0).detach().numpy()
            action.append(get_action_deterministically(action_probabilities, RSUs_num=env.RSUs_num, layer_num=layer_num,
                                                       vehicle_num=1))
            '''
            rsu_select = env.RSU_zone[env.Vehicles[i].zone]
            if env.Vehicles[i].zone == 14 or env.Vehicles[i].zone == 28:
                #if env.RSUs[rsu_select].workload > env.RSUs[rsu_select-1].workload:
                rsu_select = rsu_select - 1
            
            action.append([rsu_select, rsu_select, 0, 0])
            '''
        # action = agent.get_next_action(state, evaluation_episode=evaluation_episode)
        next_state, reward, done, info = env.step(action)

    job = []
    latency = []
    a_h = []
    a_d = []
    a_p1 = []
    a_p2 = []
    zone = []
    complete = 0
    for t in range(env.slots_num):
        if env.job[t] != []:
            for i in range(len(env.job[t])):
                if env.job[t][i].total_time != 0 and env.job[t][i].penalty == 0:
                    job.append(env.job[t][i])
                    latency.append(env.job[t][i].total_time)
                    a_h.append(env.job[t][i].helper)
                    a_d.append(env.job[t][i].deliver)
                    a_p1.append(env.job[t][i].p1)
                    a_p2.append(env.job[t][i].p2)
                    zone.append(env.job[t][i].zone)
                    # if env.job[t][i].helper != env.job[t][i].deliver:
                    # logging.warning(
                    #    "helper:{}, deliver:{}, p1:{}, p2:{}, total_time_{}, location:{}".format(
                    #        env.job[t][i].helper, env.job[t][i].deliver, env.job[t][i].p1,
                    #        env.job[t][i].p2, env.job[t][i].total_time, env.job[t][i].zone))
                    # else:
                    # logging.info(
                    #    "helper:{}, deliver:{}, p1:{}, p2:{}, total_time_{}, location:{}".format(
                    #        env.job[t][i].helper, env.job[t][i].deliver, env.job[t][i].p1,
                    #        env.job[t][i].p2, env.job[t][i].total_time, env.job[t][i].zone))
                # if env.job[t][i].penalty == 1:
                # logging.error("helper:{}, deliver:{}, p1:{}, p2:{}, total_time_{}, location:{}".format(env.job[t][i].helper, env.job[t][i].deliver, env.job[t][i].p1, env.job[t][i].p2,env.job[t][i].total_time, env.job[t][i].zone))

    if len(latency) == 0:
        complete = 1
    else:
        complete = len(latency)
    # logging.info("episode:%d, reward:%.5f, latency:%.5f, success:%.5f" % (episode_number, episode_reward, sum(latency) / complete, len(latency) / job_num))
    print("latency:%.5f, success:%.5f" % (sum(latency) / complete, len(latency) / job_num))


for i in range(10):
    use()