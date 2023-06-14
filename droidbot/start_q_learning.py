# helper file of droidbot
# it parses command arguments and send the options to droidbot
import argparse
import os
import torch
import numpy as np
import pickle
import time

import json

import torch.nn as nn
import torch.optim as optim

import gym
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import CnnPolicy, MlpPolicy
from stable_baselines import A2C, DQN

from droidbot import DroidBot
import droidbot
from droidmaster import DroidMaster
import droidbot_env

from droidbot_env import DroidBotEnv
import env_manager
import input_manager
import input_policy
from input_event import KeyEvent, TouchEvent, LongTouchEvent, ScrollEvent
from qnetwork import QNetwork
from experience_replay import ExperienceReplay


n_steps = 0  # used for saving model with callback

# save RL model in progress
def callback(_locals, _globals, save_every=1000):
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
                if len(Q_TABLE) == 0:
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
    def make_decision(state_i, events, Q_TABLE, event_to_id, max_number_of_actions, epsilon):
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

            if np.random.rand() < epsilon:
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
       
    def train(max_episodes, max_steps, max_number_of_actions, learning_rate, discount_factor, epsilon, memory_size):

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        q_network = QNetwork(state_size, action_size)
        optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()

        experience_replay = ExperienceReplay(memory_size)

        # Definir o diretório de salvamento
        save_dir = 'saved_model'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Função para salvar o modelo treinado
        def save_model(q_network):
            model_path = os.path.join(save_dir, 'model.pth')
            torch.save(q_network.state_dict(), model_path)

        # Função para carregar o modelo treinado
        def load_model(q_network):
            model_path = os.path.join(save_dir, 'model.pth')
            if os.path.exists(model_path):
                q_network.load_state_dict(torch.load(model_path))
                q_network.eval()

        if os.path.exists('q_function.npy') and os.path.exists('transition_function.npy') and os.path.exists('states.json'):
            Q_TABLE = np.load('q_function.npy')
            transitions_matrix = np.load('transition_function.npy')
            with open('states.json', 'r') as f:
                state_function = json.load(f)

            q_network = QNetwork(state_size, action_size)
            load_model(q_network)
        else:
            Q_TABLE = []
            transitions_matrix = []
            number_of_trans = []
            event_to_id = []
            state_function = {}
            batch_size = 32

            q_network = QNetwork(state_size, action_size)

        for episode in range(max_episodes):
            state_pre, probs, event_ids = events_so_state(env)
            Q_TABLE, transitions_matrix, number_of_trans, event_to_id, state_function = check_state(
                state_pre, Q_TABLE, transitions_matrix, number_of_trans, event_to_id, state_function, max_number_of_actions
            )
            state = state_function[state_pre]

            for step in range(max_steps):
                action, make_action = make_decision(state, event_ids, Q_TABLE, event_to_id, max_number_of_actions, epsilon)
                _, reward, done, _ = env.step([make_action])
                env.render()

                new_state_pre, _, event_ids = events_so_state(env)
                Q_TABLE, transitions_matrix, number_of_trans, event_to_id, state_function = check_state(
                    new_state_pre, Q_TABLE, transitions_matrix, number_of_trans, event_to_id, state_function, max_number_of_actions
                )
                new_state = state_function[new_state_pre]

                number_of_trans[state][action] += 1
                transitions_matrix[state, action] *= (number_of_trans[state][action] - 1)
                transitions_matrix[state, action, new_state] += 1
                transitions_matrix[state, action] /= number_of_trans[state][action]

                # Obter o próximo estado, recompensa e status de conclusão
                next_state_pre, _, _ = events_so_state(env)
                next_state = state_function[next_state_pre]

                experience_replay.add_experience(state, action, reward, next_state, done)

                # Amostragem de lote da memória
                batch_data = experience_replay.sample_batch(batch_size)
                if batch_data is not None:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch_data

                    #batch_states = torch.from_numpy(batch_states).to(torch.float32)
                    batch_states = batch_states.clone().detach().requires_grad_(True).to(torch.float32)
                    batch_actions = torch.tensor(batch_actions.numpy(), dtype=torch.int64).unsqueeze(1)
                    batch_rewards = torch.tensor(batch_rewards.numpy(), dtype=torch.float32)
                    batch_next_states = torch.tensor(batch_next_states.numpy(), dtype=torch.float32)
                    batch_dones = torch.tensor(batch_dones.numpy(), dtype=torch.float32)

                    q_values = q_network(batch_states)
                    q_values_next = q_network(batch_next_states)
                    q_values_target = q_values.clone()
                    max_q_values_next = torch.max(q_values_next, dim=1)[0]
                    q_values_target[range(batch_size), batch_actions.squeeze()] = batch_rewards + discount_factor * (1 - batch_dones) * max_q_values_next

                    optimizer.zero_grad()
                    loss = loss_function(q_values, q_values_target)
                    loss.backward()
                    optimizer.step()

                    state = new_state

                    if done:
                        break

            if episode % 2 == 0:
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
   