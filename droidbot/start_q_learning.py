# helper file of droidbot
# it parses command arguments and send the options to droidbot
import argparse
import torch
import input_manager
import input_policy
import env_manager
from droidbot import DroidBot
from droidmaster import DroidMaster
import droidbot_env
import numpy as np
import pickle
import time
from input_event import KeyEvent, TouchEvent, LongTouchEvent, ScrollEvent
import json

import droidbot
import droidbot_env

import torch
import torch.nn as nn
import torch.optim as optim

import gym
from stable_baselines.common.vec_env import DummyVecEnv
#from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN

n_steps = 0 #used for saving model with callback

# save RL model in progress
def callback(_locals, _globals, save_every=1000):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps
    # Print stats every 1000 calls
    if (n_steps + 1) % save_every == 0:
        temp_int = int(time.time())
        print("saving while training model")
        _locals['self'].save('in_progress_model_{}.pkl'.format(temp_int))
    n_steps += 1
    return True


def parse_args():
    """
    parse command line input
    generate options including host name, port number
    """
    parser = argparse.ArgumentParser(description="Start DroidBot to test an Android app.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-d", action="store", dest="device_serial", required=False,
                        help="The serial number of target device (use `adb devices` to find)")
    parser.add_argument("-a", action="store", dest="apk_path", required=True,
                        help="The file path to target APK")
    parser.add_argument("-o", action="store", dest="output_dir",
                        help="directory of output")
    parser.add_argument("-policy", action="store", dest="input_policy", default=input_manager.DEFAULT_POLICY,
                        help='Policy to use for test input generation. '
                             'Default: %s.\nSupported policies:\n' % input_manager.DEFAULT_POLICY +
                             '  \"%s\" -- No event will be sent, user should interact manually with device; \n'
                             '  \"%s\" -- Use "adb shell monkey" to send events; \n'
                             '  \"%s\" -- Explore UI using a naive depth-first strategy;\n'
                             '  \"%s\" -- Explore UI using a greedy depth-first strategy;\n'
                             '  \"%s\" -- Explore UI using a naive breadth-first strategy;\n'
                             '  \"%s\" -- Explore UI using a greedy breadth-first strategy;\n'
                             %
                             (
                                 input_policy.POLICY_NONE,
                                 input_policy.POLICY_MONKEY,
                                 input_policy.POLICY_NAIVE_DFS,
                                 input_policy.POLICY_GREEDY_DFS,
                                 input_policy.POLICY_NAIVE_BFS,
                                 input_policy.POLICY_GREEDY_BFS,
                             ))

    # for distributed DroidBot
    parser.add_argument("-distributed", action="store", dest="distributed", choices=["master", "worker"],
                        help="Start DroidBot in distributed mode.")
    parser.add_argument("-master", action="store", dest="master",
                        help="DroidMaster's RPC address")
    parser.add_argument("-qemu_hda", action="store", dest="qemu_hda",
                        help="The QEMU's hda image")
    parser.add_argument("-qemu_no_graphic", action="store_true", dest="qemu_no_graphic",
                        help="Run QEMU with -nograpihc parameter")

    parser.add_argument("-script", action="store", dest="script_path",
                        help="Use a script to customize input for certain states.")
    parser.add_argument("-count", action="store", dest="count", default=input_manager.DEFAULT_EVENT_COUNT, type=int,
                        help="Number of events to generate in total. Default: %d" % input_manager.DEFAULT_EVENT_COUNT)
    parser.add_argument("-interval", action="store", dest="interval", default=input_manager.DEFAULT_EVENT_INTERVAL,
                        type=int,
                        help="Interval in seconds between each two events. Default: %d" % input_manager.DEFAULT_EVENT_INTERVAL)
    parser.add_argument("-timeout", action="store", dest="timeout", default=input_manager.DEFAULT_TIMEOUT, type=int,
                        help="Timeout in seconds, -1 means unlimited. Default: %d" % input_manager.DEFAULT_TIMEOUT)
    parser.add_argument("-cv", action="store_true", dest="cv_mode",
                        help="Use OpenCV (instead of UIAutomator) to identify UI components. CV mode requires opencv-python installed.")
    parser.add_argument("-debug", action="store_true", dest="debug_mode",
                        help="Run in debug mode (dump debug messages).")
    parser.add_argument("-random", action="store_true", dest="random_input",
                        help="Add randomness to input events.")
    parser.add_argument("-keep_app", action="store_true", dest="keep_app",
                        help="Keep the app on the device after testing.")
    parser.add_argument("-keep_env", action="store_true", dest="keep_env",
                        help="Keep the test environment (eg. minicap and accessibility service) after testing.")
    parser.add_argument("-use_method_profiling", action="store", dest="profiling_method",
                        help="Record method trace for each event. can be \"full\" or a sampling rate.")
    parser.add_argument("-grant_perm", action="store_true", dest="grant_perm",
                        help="Grant all permissions while installing. Useful for Android 6.0+.")
    parser.add_argument("-is_emulator", action="store_true", dest="is_emulator",
                        help="Declare the target device to be an emulator, which would be treated specially by DroidBot.")
    parser.add_argument("-accessibility_auto", action="store_true", dest="enable_accessibility_hard",
                        help="Enable the accessibility service automatically even though it might require device restart\n(can be useful for Android API level < 23).")
    parser.add_argument("-humanoid", action="store", dest="humanoid",
                        help="Connect to a Humanoid service (addr:port) for more human-like behaviors.")
    parser.add_argument("-ignore_ad", action="store_true", dest="ignore_ad",
                        help="Ignore Ad views by checking resource_id.")
    parser.add_argument("-replay_output", action="store", dest="replay_output",
                        help="The droidbot output directory being replayed.")
    options = parser.parse_args()
    # print options
    return options


def main():
    """
    the main function
    it starts a droidbot according to the arguments given in cmd line
    """

    opts = parse_args()
    import os
    if not os.path.exists(opts.apk_path):
        print("APK does not exist.")
        return
    if not opts.output_dir and opts.cv_mode:
        print("To run in CV mode, you need to specify an output dir (using -o option).")

    if opts.distributed:
        if opts.distributed == "master":
            start_mode = "master"
        else:
            start_mode = "worker"
    else:
        start_mode = "normal"

    if start_mode == "master":
        droidmaster = DroidMaster(
            app_path=opts.apk_path,
            is_emulator=opts.is_emulator,
            output_dir=opts.output_dir,
            # env_policy=opts.env_policy,
            env_policy=env_manager.POLICY_NONE,
            policy_name=opts.input_policy,
            random_input=opts.random_input,
            script_path=opts.script_path,
            event_interval=opts.interval,
            timeout=opts.timeout,
            event_count=opts.count,
            cv_mode=opts.cv_mode,
            debug_mode=opts.debug_mode,
            keep_app=opts.keep_app,
            keep_env=opts.keep_env,
            profiling_method=opts.profiling_method,
            grant_perm=opts.grant_perm,
            enable_accessibility_hard=opts.enable_accessibility_hard,
            qemu_hda=opts.qemu_hda,
            qemu_no_graphic=opts.qemu_no_graphic,
            humanoid=opts.humanoid,
            ignore_ad=opts.ignore_ad,
            replay_output=opts.replay_output)
        droidmaster.start()
    else:
        droidbot = DroidBot(
            app_path=opts.apk_path,
            device_serial=opts.device_serial,
            is_emulator=opts.is_emulator,
            output_dir=opts.output_dir,
            # env_policy=opts.env_policy,
            env_policy=env_manager.POLICY_NONE,
            policy_name=opts.input_policy,
            random_input=opts.random_input,
            script_path=opts.script_path,
            event_interval=opts.interval,
            timeout=opts.timeout,
            event_count=opts.count,
            cv_mode=opts.cv_mode,
            debug_mode=opts.debug_mode,
            keep_app=opts.keep_app,
            keep_env=opts.keep_env,
            profiling_method=opts.profiling_method,
            grant_perm=opts.grant_perm,
            enable_accessibility_hard=opts.enable_accessibility_hard,
            master=opts.master,
            humanoid=opts.humanoid,
            ignore_ad=opts.ignore_ad,
            replay_output=opts.replay_output)

        droidbot.start()

    env = DummyVecEnv([lambda: droidbot_env.DroidBotEnv(droidbot)])
    start_time = time.time()
    env.reset()
            
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class ExperienceReplay:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.memory = []

    def add_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        states = torch.tensor([experience[0] for experience in batch], dtype=torch.float32)
        actions = torch.tensor([experience[1] for experience in batch], dtype=torch.long)
        rewards = torch.tensor([experience[2] for experience in batch], dtype=torch.float32)
        next_states = torch.tensor([experience[3] for experience in batch], dtype=torch.float32)
        dones = torch.tensor([experience[4] for experience in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones
    

def events_so_state(env):
        events = env.envs[0].possible_events
        state_now = env.envs[0].device.get_current_state()
        event_ids = []
        probs = []

        for i, event in enumerate(events):
            event_str = str(type(event)) + '_' + event.get_event_str(state_now)
            if event_str in event_ids:
                1/0
            if event:
                event_ids.append(event_str)
                probs.append(env.envs[0].events_probs[i])
        state = state_now.state_str
        probs = np.array(probs)
        return state, probs, event_ids

# Função para verificar o estado no dicionário de estados
def check_state(state_id, Q_TABLE, transitions_matrix, number_of_trans, event_to_id, state_function, max_number_of_actions):
        if state_function.get(state_id) is None:
            if Q_TABLE == []:
                Q_TABLE = np.zeros((1, max_number_of_actions))
                transitions_matrix = np.zeros((1, max_number_of_actions, 1))
            else:
                Q_TABLE = np.concatenate([Q_TABLE, np.zeros((1, max_number_of_actions))], axis=0)
                transition_matrix_new = np.zeros((Q_TABLE.shape[0], max_number_of_actions, Q_TABLE.shape[0]))
                transition_matrix_new[:-1, :, :-1] = transitions_matrix
                transitions_matrix = transition_matrix_new
            event_to_id.append({})
            state_function[state_id] = Q_TABLE.shape[0] - 1
            Q_TABLE[-1][-1] = 1.0
            number_of_trans.append(np.zeros(max_number_of_actions))
        return Q_TABLE, transitions_matrix, number_of_trans, event_to_id, state_function

    # Função para tomar decisões com base nos valores Q
def make_decision(state_i, events, Q_TABLE, event_to_id, max_number_of_actions, EPSILON):
    id_to_action = np.zeros((max_number_of_actions), dtype=np.int32) + 1000
    q_values = np.zeros(max_number_of_actions)
    probs_now = np.zeros(max_number_of_actions)
    for i, event in enumerate(events):
        if i == len(events) - 1:
            q_values[-1] = Q_TABLE[state_i][-1]
            id_to_action[-1] = min(len(events), max_number_of_actions) - 1
            continue
        if event_to_id[state_i].get(event) is None:
            if len(event_to_id[state_i]) >= max_number_of_actions - 1:
                continue
            event_to_id[state_i][event] = int(len(list(event_to_id[state_i].keys())))
            Q_TABLE[state_i][event_to_id[state_i][event]] = 1.0
            q_values[event_to_id[state_i][event]] = Q_TABLE[state_i][event_to_id[state_i][event]]

            id_to_action[event_to_id[state_i][event]] = int(i)

        if np.random.rand() < EPSILON:
            action = max_number_of_actions - 1
            make_action = id_to_action[action]
        else:
            max_q = np.max(q_values)
            actions_argmax = np.arange(max_number_of_actions)[q_values >= max_q - 0.0001]
            probs_unnormed = 1 / (np.arange(actions_argmax.shape[0]) + 1.)
            probs_unnormed /= np.sum(probs_unnormed)
            action = np.random.choice(actions_argmax)
            make_action = id_to_action[action]
        return action, make_action

    # Função para atualizar os valores Q
def update_Q_values(Q_TABLE, transitions_matrix, max_number_of_actions, learning_rate, discount_factor):
    for _ in np.arange(10):
        for i in np.arange(max_number_of_actions):
            transitions = transitions_matrix[:, i, :]
            q_target = np.array([[np.max(Q_TABLE[i])] for i in np.arange(Q_TABLE.shape[0])])
            new_q_values = np.matmul(transitions, q_target) * discount_factor
            good_states = np.sum(transitions, axis=1) > 0.5
            if True in good_states:
                Q_TABLE[good_states, i] = new_q_values[good_states, 0]
            else:
                continue
        return Q_TABLE
    
    
def train(max_episodes, max_steps, max_number_of_actions, learning_rate, discount_factor, epsilon, memory_size):
    Q_TABLE = []
    transitions_matrix = []
    number_of_trans = []
    event_to_id = []
    state_function = {}
    batch_size = 32

    env = DummyVecEnv([lambda: droidbot_env.DroidBotEnv(droidbot)])
    env.reset()

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_network = QNetwork(state_size, action_size)
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    experience_replay = ExperienceReplay(memory_size)

    for episode in range(max_episodes):
        state_pre, probs, event_ids = events_so_state(env)
        Q_TABLE, transitions_matrix, number_of_trans, event_to_id, state_function = check_state(
            state_pre, Q_TABLE, transitions_matrix, number_of_trans, event_to_id, state_function, max_number_of_actions
        )
        state = state_function[state_pre]

        for step in range(max_steps):
            action, make_action = make_decision(state, event_ids, Q_TABLE, event_to_id, max_number_of_actions, epsilon)
            env.step([make_action])
            env.render()  # Atualize o estado do ambiente

            new_state_pre, _, event_ids = events_so_state(env)
            Q_TABLE, transitions_matrix, number_of_trans, event_to_id, state_function = check_state(
                new_state_pre, Q_TABLE, transitions_matrix, number_of_trans, event_to_id, state_function, max_number_of_actions
            )
            new_state = state_function[new_state_pre]

            number_of_trans[state][action] += 1
            transitions_matrix[state, action] *= (number_of_trans[state][action] - 1)
            transitions_matrix[state, action, new_state] += 1
            transitions_matrix[state, action] /= number_of_trans[state][action]

            # Adicionar experiência à memória
            # Obter a recompensa correspondente do ambiente
            reward, done = env.get_reward_done(new_state)  # Chamada à função do ambiente
            experience_replay.add_experience(state, action, reward, new_state, done)

            # Amostragem de lote da memória
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = experience_replay.sample_batch(batch_size)

            # Calcular os valores Q esperados - Treino
            q_values = q_network(batch_states)
            q_values_next = q_network(batch_next_states)
            q_values_target = q_values.clone()
            max_q_values_next = torch.max(q_values_next, dim=1)[0]
            q_values_target[range(batch_size), batch_actions] = batch_rewards + discount_factor * (1 - batch_dones) * max_q_values_next

            # Calcular a perda e otimizar a rede neural
            optimizer.zero_grad()
            loss = loss_function(q_values, q_values_target)
            loss.backward()
            optimizer.step()

            state = new_state

        if episode % 10 == 0:
            np.save('q_function', Q_TABLE)
            np.save('transition_function', transitions_matrix)
            with open('states.json', 'w') as f:
                json.dump(state_function, f)

    return Q_TABLE, transitions_matrix, state_function

# Exemplo de uso
max_episodes = 1000
max_steps = 100
max_number_of_actions = 50
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
memory_size = 10000
batch_size = 32

Q_TABLE, transitions_matrix, state_function = train(max_episodes, max_steps, max_number_of_actions,
                                                    learning_rate, discount_factor, epsilon, memory_size)
                
1/0
droidbot.stop()

if __name__ == "__main__":
    print("Starting droidbot gym env")
    main()

