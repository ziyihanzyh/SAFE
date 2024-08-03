#import gym
import numpy as np
import matplotlib.pyplot as plt
from env_action import Environment, Setting
import time
import os
import copy
import logging


TRAINING_EVALUATION_RATIO = 4
RUNS = 20
EPISODES_PER_RUN = 3000
STEPS_PER_EPISODE = 200
SAVE_EPI_NUM = 50
ALPHA_INITIAL_Initial = 1.
REPLAY_BUFFER_BATCH_SIZE_Initial = 100
DISCOUNT_RATE_Initial = 0.99
LEARNING_RATE_Initial = 10 ** -4
SOFT_UPDATE_INTERPOLATION_FACTOR_Initial = 0.01
TRAIN_PER_STEP = 20

print("vehicle = {} & generate = {}".format(vehicle, generate))
import torch


#from utilities.ReplayBuffer import ReplayBuffer

hidden = 512

class Actor(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation, vehicle_num,
            RSUs_num, layer_num):
        super(Actor, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=hidden)
        self.layer_2 = torch.nn.Linear(in_features=hidden, out_features=hidden//2)
        self.layer_list =[]
        for i in range(vehicle_num):
            self.layer_list.append(torch.nn.Linear(in_features=hidden//2, out_features = RSUs_num))
            self.layer_list.append(torch.nn.Linear(in_features=hidden // 2, out_features=RSUs_num))
            self.layer_list.append(torch.nn.Linear(in_features=hidden//2, out_features = layer_num))
            self.layer_list.append(torch.nn.Linear(in_features=hidden // 2, out_features=layer_num))
        self.output_layer = torch.nn.Linear(in_features=hidden//2, out_features=output_dimension)
        self.output_activation = output_activation
        self.num = vehicle_num

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        x = []
        for i in range(self.num):
            x.append(self.output_activation(self.layer_list[4*i](layer_2_output)))
            x.append(self.output_activation(self.layer_list[4*i + 1](layer_2_output)))
            x.append(self.output_activation(self.layer_list[4 * i + 2](layer_2_output)))
            x.append(self.output_activation(self.layer_list[4 * i + 3](layer_2_output)))
        
        output = x[0]
        for i in range(1, len(x)):
            output = torch.cat([output, x[i]], dim=1)
        return output

class Critic(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=torch.nn.Identity()):
        super(Critic, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=hidden)
        self.layer_2 = torch.nn.Linear(in_features=hidden, out_features=hidden//2)
        self.output_layer = torch.nn.Linear(in_features=hidden//2, out_features=output_dimension)
        self.output_activation = output_activation

    def forward(self, inpt):
        layer_1_output = torch.nn.functional.relu(self.layer_1(inpt))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = self.output_layer(layer_2_output)


        output = self.output_activation(layer_3_output)
        return output

class SACAgent:
    ALPHA_INITIAL = ALPHA_INITIAL_Initial
    REPLAY_BUFFER_BATCH_SIZE = REPLAY_BUFFER_BATCH_SIZE_Initial
    DISCOUNT_RATE = DISCOUNT_RATE_Initial
    LEARNING_RATE = LEARNING_RATE_Initial
    SOFT_UPDATE_INTERPOLATION_FACTOR = SOFT_UPDATE_INTERPOLATION_FACTOR_Initial

    def __init__(self, environment):
        self.environment = environment
        self.vehicle_num = environment.max_vehicle_num
        self.RSUs_num = environment.RSUs_num
        self.layer_num = layer_num
        self.state_dim = len(self.environment.state)
        self.action_dim = len(environment.action)
        self.critic_local = Critic(input_dimension=self.state_dim,
                                    output_dimension=self.action_dim)
        self.critic_local2 = Critic(input_dimension=self.state_dim,
                                     output_dimension=self.action_dim)
        self.critic_optimiser = torch.optim.Adam(self.critic_local.parameters(), lr=self.LEARNING_RATE)
        self.critic_optimiser2 = torch.optim.Adam(self.critic_local2.parameters(), lr=self.LEARNING_RATE)

        self.critic_target = Critic(input_dimension=self.state_dim,
                                     output_dimension=self.action_dim)
        self.critic_target2 = Critic(input_dimension=self.state_dim,
                                      output_dimension=self.action_dim)

        self.soft_update_target_networks(tau=1.)

        self.actor_local = Actor(
            input_dimension=self.state_dim,
            output_dimension=self.action_dim,
            output_activation=torch.nn.Softmax(dim=1),
            vehicle_num = self.vehicle_num,
            RSUs_num = self.RSUs_num,
            layer_num = self.layer_num

        )
        self.actor_optimiser = torch.optim.Adam(self.actor_local.parameters(), lr=self.LEARNING_RATE)

        self.replay_buffer = ReplayBuffer(self.environment)

        self.target_entropy = 0.98 * -np.log(1 / self.action_dim)
        self.log_alpha = torch.tensor(np.log(self.ALPHA_INITIAL), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = torch.optim.Adam([self.log_alpha], lr=self.LEARNING_RATE)

    def get_next_action(self, state, evaluation_episode=False):
        if evaluation_episode:
            discrete_action = self.get_action_deterministically(state)
        else:
            discrete_action = self.get_action_nondeterministically(state)
        return discrete_action

    def get_action_nondeterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action =[]
        temp = (self.RSUs_num + self.layer_num) * 2
        for i in range(self.vehicle_num):
            rsu_select_1 = action_probabilities[i * temp: i * temp + self.RSUs_num]
            rsu_select_2 = action_probabilities[i * temp + self.RSUs_num: i * temp + self.RSUs_num * 2]
            layer_select_1 = action_probabilities[i * temp + self.RSUs_num * 2: i * temp + self.RSUs_num * 2 + self.layer_num]
            layer_select_2 = action_probabilities[
                           i * temp + self.RSUs_num * 2 + self.layer_num: i * temp + self.RSUs_num * 2 + self.layer_num * 2]
            discrete_action.append(np.random.choice(range(0, self.RSUs_num), p=rsu_select_1))
            discrete_action.append(np.random.choice(range(0, self.RSUs_num), p=rsu_select_2))
            discrete_action.append(np.random.choice(range(0, self.layer_num), p=layer_select_1))

            discrete_action.append(np.random.choice(range(0, self.layer_num), p=layer_select_2))
        return discrete_action

    def get_action_deterministically(self, state):
        action_probabilities = self.get_action_probabilities(state)
        discrete_action = []
        temp = self.RSUs_num + self.layer_num
        for i in range(self.vehicle_num):
            rsu_select_1 = action_probabilities[i * temp: i * temp + self.RSUs_num]
            rsu_select_2 = action_probabilities[i * temp + self.RSUs_num: i * temp + self.RSUs_num * 2]
            layer_select_1 = action_probabilities[
                             i * temp + self.RSUs_num * 2: i * temp + self.RSUs_num * 2 + self.layer_num]
            layer_select_2 = action_probabilities[
                             i * temp + self.RSUs_num * 2 + self.layer_num: i * temp + self.RSUs_num * 2 + self.layer_num * 2]
            discrete_action.append(np.argmax(rsu_select_1))
            discrete_action.append(np.argmax(rsu_select_2))
            discrete_action.append(np.argmax(layer_select_1))
            discrete_action.append(np.argmax(layer_select_2))

        return discrete_action


    def train_on_transition(self, state, discrete_action, next_state, reward, done, train):
        count = 0
        for i in range(self.vehicle_num):
            discrete_action[4 * i] = discrete_action[4 * i] + count
            count += self.RSUs_num
            discrete_action[4 * i + 1] = discrete_action[4 * i + 1] + count
            count += self.RSUs_num
            discrete_action[4 * i + 2] = discrete_action[4 * i + 2] + count
            count += self.layer_num
            discrete_action[4 * i + 3] = discrete_action[4 * i + 3] + count
            count += self.layer_num
        transition = (state, discrete_action, reward, next_state, done)
        self.train_networks(transition, train)

    def train_networks(self, transition, train):
        # Set all the gradients stored in the optimisers to zero.
        self.critic_optimiser.zero_grad()
        self.critic_optimiser2.zero_grad()
        self.actor_optimiser.zero_grad()
        self.alpha_optimiser.zero_grad()
        # Calculate the loss for this transition.
        self.replay_buffer.add_transition(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        if self.replay_buffer.get_size() >= self.REPLAY_BUFFER_BATCH_SIZE and train == True:
            # get minibatch of 100 transitions from replay buffer
            minibatch = self.replay_buffer.sample_minibatch(self.REPLAY_BUFFER_BATCH_SIZE)
            minibatch_separated = list(map(list, zip(*minibatch)))

            # unravel transitions to get states, actions, rewards and next states
            states_tensor = torch.tensor(np.array(minibatch_separated[0]), dtype=torch.float32)
            actions_tensor = torch.tensor(np.array(minibatch_separated[1]),dtype=torch.float32)
            rewards_tensor = torch.tensor(np.array(minibatch_separated[2])).float()
            next_states_tensor = torch.tensor(np.array(minibatch_separated[3]))
            done_tensor = torch.tensor(np.array(minibatch_separated[4]))
            #actions_tensor_2 = torch.tensor(np.array(minibatch_separated[5]), dtype=torch.float32)

            critic_loss, critic2_loss = \
                self.critic_loss(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor)

            critic_loss.backward()
            critic2_loss.backward()
            self.critic_optimiser.step()
            self.critic_optimiser2.step()

            actor_loss, log_action_probabilities = self.actor_loss(states_tensor)

            actor_loss.backward()
            self.actor_optimiser.step()

            alpha_loss = self.temperature_loss(log_action_probabilities)

            alpha_loss.backward()
            self.alpha_optimiser.step()
            self.alpha = self.log_alpha.exp()

            self.soft_update_target_networks()

    def critic_loss(self, states_tensor, actions_tensor, rewards_tensor, next_states_tensor, done_tensor):
        with torch.no_grad():
            action_probabilities, log_action_probabilities = self.get_action_info(next_states_tensor)
            next_q_values_target = self.critic_target.forward(next_states_tensor)
            next_q_values_target2 = self.critic_target2.forward(next_states_tensor)
            #soft_state_values = (action_probabilities * (
            #        torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities
            #)).sum(dim=1)

            temp = action_probabilities * (
                    torch.min(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probabilities
            )

            list = []
            for i in range(self.vehicle_num):
                list.append(self.RSUs_num)
                list.append(self.RSUs_num)
                list.append(self.layer_num)
                list.append(self.layer_num)
            soft_state_value = torch.split(temp, list, dim=1)

            next_q_values = []
            for i in range(len(soft_state_value)):
                next_q_values.append(rewards_tensor + ~done_tensor * self.DISCOUNT_RATE*soft_state_value[i].sum(dim=1))

            #next_q_values = rewards_tensor + ~done_tensor * self.DISCOUNT_RATE*soft_state_values

        #actions = []
        #num = self.vehicle_num * 2
        temp = torch.split(actions_tensor, 1, dim=1)
        
        
        #soft_q_values = self.critic_local(states_tensor).gather(1, actions_tensor.type(torch.int64).unsqueeze(-1)).squeeze(-1)
        soft_q_value = self.critic_local(states_tensor)
        soft_q_value2 = self.critic_local2(states_tensor)
        soft_q_values = []
        soft_q_values2 = []

        critic_square_error = 0
        critic2_square_error = 0
        for i in range(len(temp)):
            temp1 = soft_q_value.gather(1, temp[i].type(torch.int64).squeeze().unsqueeze(-1)).squeeze(-1)
            soft_q_values.append(temp1)
            temp2 = soft_q_value2.gather(1, temp[i].type(torch.int64).squeeze().unsqueeze(-1)).squeeze(-1)
            soft_q_values2.append(temp2)
            critic_square_error += torch.nn.MSELoss(reduction="none")(temp1, next_q_values[i])
            critic2_square_error += torch.nn.MSELoss(reduction="none")(temp2, next_q_values[i])


        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic_square_error, critic2_square_error)]
        self.replay_buffer.update_weights(weight_update)
        critic_loss = critic_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        return critic_loss, critic2_loss

    def actor_loss(self, states_tensor,):
        action_probabilities, log_action_probabilities = self.get_action_info(states_tensor)
        q_values_local = self.critic_local(states_tensor)
        q_values_local2 = self.critic_local2(states_tensor)
        inside_term = self.alpha * log_action_probabilities - torch.min(q_values_local, q_values_local2)
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        return policy_loss, log_action_probabilities

    def temperature_loss(self, log_action_probabilities):
        alpha_loss = -(self.log_alpha * (log_action_probabilities + self.target_entropy).detach()).mean()
        return alpha_loss

    def get_action_info(self, states_tensor):
        action_probabilities = self.actor_local.forward(states_tensor)
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action_probabilities, log_action_probabilities

    def get_action_probabilities(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probabilities = self.actor_local.forward(state_tensor)
        return action_probabilities.squeeze(0).detach().numpy()

    def soft_update_target_networks(self, tau=SOFT_UPDATE_INTERPOLATION_FACTOR):
        self.soft_update(self.critic_target, self.critic_local, tau)
        self.soft_update(self.critic_target2, self.critic_local2, tau)

    def soft_update(self, target_model, origin_model, tau):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def predict_q_values(self, state):
        q_values = self.critic_local(state)
        q_values2 = self.critic_local2(state)
        return torch.min(q_values, q_values2)


class ReplayBuffer:

    def __init__(self, environment, capacity=10000):
        transition_type_str = self.get_transition_type_str(environment)
        self.buffer = np.zeros(capacity, dtype=transition_type_str)
        self.weights = np.zeros(capacity)
        self.head_idx = 0
        self.count = 0
        self.capacity = capacity
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None

    def get_transition_type_str(self, environment):
        #state_dim = environment.observation_space.shape[0]
        state_dim = len(environment.state)
        state_dim_str = '' if state_dim == () else str(state_dim)
        #state_type_str = environment.observation_space.sample().dtype.name
        state_type_str = "float32"
        #action_dim = "2"
        action_dim = environment.max_vehicle_num * 4
        #action_dim = environment.action_space.shape
        action_dim_str = '' if action_dim == () else str(action_dim)
        #action_type_str = environment.action_space.sample().__class__.__name__
        action_type_str = "int"

        # type str for transition = 'state type, action type, reward type, state type'
        transition_type_str = '{0}{1}, {2}{3}, float32, {0}{1}, bool'.format(state_dim_str, state_type_str,
                                                                             action_dim_str, action_type_str)

        return transition_type_str

    def add_transition(self, transition):
        self.buffer[self.head_idx] = transition
        self.weights[self.head_idx] = self.max_weight

        self.head_idx = (self.head_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample_minibatch(self, size=100):
        set_weights = self.weights[:self.count] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(self.count), size, p=probabilities, replace=False)
        return self.buffer[self.indices]

    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors

    def get_size(self):
        return self.count

if __name__ == "__main__":


    time = time.time()
    print(time)
    path = str(time)
    dir = os.getcwd() + '\\result\\' + path
    os.makedirs(dir)
    logging.basicConfig(filename="./result/{}/logger.log".format(path), level=logging.INFO)
    #loger_1 = logging.basicConfig(filename="./result/{}/print.log".format(path), level=logging.INFO)
    arg = Setting(vehicle, generate, STEPS_PER_EPISODE)
    env = Environment(args=arg, layer_num=layer_num)
    res = []
    B = []
    link = env.location_link
    for i in range(env.RSUs_num):
        res.append(env.RSUs[i].res)
        B.append(env.RSUs[i].B)
    np.savez("./result/{}/env".format(path), res=res, B=B, link=link)
        
    #env.action_space = a

    agent_results = []
    for run in range(RUNS):
        agent = SACAgent(env)
        run_results = []
        index_results = []
        latency_result = []
        success_result = []
        for episode_number in range(EPISODES_PER_RUN):
            print('\r', f'Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN}', end=' ')
            evaluation_episode = episode_number % TRAINING_EVALUATION_RATIO == 0
            episode_reward = 0
            state_r = env.reset()
            num = env.job_num
            job_num = num * env.max_vehicle_num
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
                    action.append(agent.get_next_action(state[i], evaluation_episode=evaluation_episode))
                #action = agent.get_next_action(state, evaluation_episode=evaluation_episode)
                next_state, reward, done, info, action = env.step(action)
                #action = action[0] * 20 + action[1]
                train = False
                if not evaluation_episode:
                    for i in range(v_num):
                        if st % TRAIN_PER_STEP == 0 and i == 0:
                            train = True
                        agent.train_on_transition(state[i], action[i], next_state, reward[i], done, train)
                else:
                    if v_num > 0:
                        episode_reward += sum(reward)/v_num
                state_r = next_state
            if evaluation_episode:
                job = []
                latency = []
                a_h = []
                a_d = []
                a_p1 = []
                a_p2 = []
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
                                if env.job[t][i].helper != env.job[t][i].deliver:
                                    logging.warning(
                                            "helper:{}, deliver:{}, p1:{}, p2:{}, total_time_{}, location:{}".format(
                                                env.job[t][i].helper, env.job[t][i].deliver, env.job[t][i].p1,
                                                env.job[t][i].p2, env.job[t][i].total_time, env.job[t][i].zone))
                                else:
                                    logging.info(
                                        "helper:{}, deliver:{}, p1:{}, p2:{}, total_time_{}, location:{}".format(
                                            env.job[t][i].helper, env.job[t][i].deliver, env.job[t][i].p1,
                                            env.job[t][i].p2, env.job[t][i].total_time, env.job[t][i].zone))
                            if env.job[t][i].penalty == 1:
                                logging.error(
                                    "helper:{}, deliver:{}, p1:{}, p2:{}, total_time_{}, location:{}".format(
                                        env.job[t][i].helper, env.job[t][i].deliver, env.job[t][i].p1, env.job[t][i].p2,
                                        env.job[t][i].total_time, env.job[t][i].zone))
                if len(latency) == 0:
                    complete = 1
                else:
                    complete = len(latency)
                logging.info("episode:%d, reward:%.5f, latency:%.5f, success:%.5f"%(episode_number, episode_reward, sum(latency)/complete, len(latency)/job_num))
                print("episode:%d, reward:%.5f, latency:%.5f, success:%.5f"%(episode_number, episode_reward, sum(latency)/complete, len(latency)/job_num))
                run_results.append(episode_reward)
                index_results.append(episode_number)
                latency_result.append(sum(latency)/complete)
                success_result.append(len(latency)/job_num)
            if episode_number % SAVE_EPI_NUM == SAVE_EPI_NUM - 1:
                torch.save(agent.actor_local, "./result/{}/run{}-episode{}-actor.pkl".format(path, run, episode_number))
                np.savez("./result/{}/run{}-episode{}-reward".format(path, run, episode_number), episode=index_results,
                         reward=run_results, latency=latency_result, success=success_result)
        agent_results.append(run_results)
    #env.close()

    n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
    results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
    mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

    x_vals = list(range(len(results_mean)))
    x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]


    ax = plt.gca()
    #ax.set_ylim([-30, 0])
    ax.set_ylabel('Episode Score')
    ax.set_xlabel('Training Episode')
    ax.plot(x_vals, results_mean, label='Average Result', color='blue')
    ax.plot(x_vals, mean_plus_std, color='blue', alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='blue')
    ax.plot(x_vals, mean_minus_std, color='blue', alpha=0.1)
    plt.legend(loc='best')
    plt.savefig("./result/{}/1.png".format(path))
    plt.show()
