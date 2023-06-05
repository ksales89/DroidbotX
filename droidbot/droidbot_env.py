#bibliotecas para criar a rede neural
#Importar os módulos necessários:

import threading
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
from utg import UTG
import networkx as nx
import logging
from input_event import KeyEvent
from collections import deque
from PIL import Image
import cv2
cv2.ocl.setUseOpenCL(False)
#Definir as constantes para as políticas de ação
POLICY_GREEDY_DFS = "dfs_greedy"
POLICY_GREEDY_BFS = "bfs_greedy"
POLICY_NONE = "none"
#Definir a classe DroidBotEnv que herda da classe gym.Env
class DroidBotEnv(gym.Env):

    def __init__(self, droidbot):
        super(DroidBotEnv, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        metadata = {
            'render.modes': ['human'],
            'video.frames_per_second': None
        }
        self.seed()
        # Definir o otimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        #Adam para otimizar os pesos da rede neural, com uma taxa de aprendizado de 0,001.
        self.q_network = QNetwork(self.observation_space.shape[0], self.action_space.n)
        # Instancia da classe ReplayMemory
        # para inicializar a memória de replay com uma capacidade específica.
        self.replay_memory = ReplayMemory(capacity=10000)  # ajuste a capacidade conforme necessário
        # Definir o valor de gamma
        self.gamma = 0.99  # Valor de desconto (ajuste conforme necessário)
        # Definir a função de perda (loss function)
        self.loss_fn = nn.MSELoss()
        self.device = droidbot.device # use device to get current state for assigning actions to events
        self.input_manager = droidbot.input_manager # use input manager to send events to droidbot
        self.policy = droidbot.input_manager.policy
        # use policy to input events, get UTG
        self.possible_events = None # when we generate action_space, this is set
        self.policy_type = POLICY_GREEDY_DFS #set BFS/DFS/NONE for possible additional events
        self.add_unexplored = False # add unexplored actions to list of possible actions
        # Action size can change in be regenerated every step or set fixed
        #self.action_space
        self.action_space = spaces.Discrete(10)
        # Using image stack of past four states for observation space. See Humanoid paper for potential improvements
        self.stack_size = 4
        self.frames = deque([], maxlen=self.stack_size)
        self.height = 320
        self.width = 180
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, self.stack_size))
        # app stats in initial start state, subsequent resets need to stop and start the app
        self.reset_count = 0
        self.adb_connector = ADBConnector()
         
    def step(self, action):
        import time
        start_time = time.time()
        info_dict = {}
        event = self.get_event(action)
        #print(time.time() - start_time)
        # Passar a observação atual pela rede neural usando 
        # self.q_network(state_tensor) e obtendo os Q-values para cada ação possível. 
        # Em seguida, selecionamos a ação com o maior Q-value para o pró
        observation_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(observation_tensor)
        # Selecionar a ação com o maior Q-value
        _, max_action = q_values.max(1)
        max_action = max_action.item()
        # ...
        # returns None if index does not have corresponding event
        if event is not None:
            # do some checks prior to executing gym event
            new_event = self.policy.check_gym_event(event)
            self.input_manager.add_event(new_event)
            if event == new_event:
                info_dict['event_same'] = 1
        else:
            info_dict['event_same'] = 2
        #print(time.time() - start_time)
        state = self.device.get_current_state()
        #print(time.time() - start_time)
        state = self.handle_none_current_state(state)
        #print(time.time() - start_time)
        state = self.device.get_current_state()
        #print(time.time() - start_time)
        reward, done = self.get_reward_done(state)
        #print(time.time() - start_time)
        next_state = self.device.get_current_state()
        # get image for state and return it
        if done:
            img_stack = np.array(self.frames)
        else:
            state_img = self.get_image_state()
            self.frames.append(state_img)
            img_stack = np.array(self.frames)
            img_stack = np.moveaxis(img_stack, 0, -1)
       # print(time.time() - start_time)
        self.set_possible_events()
        # Atualizar a rede neural com base na recompensa e na próxima observação
        next_state, reward, done, info_dict = self.device.get_current_state(), self.get_reward_done(state)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        q_values_next = self.q_network(torch.FloatTensor(next_state).unsqueeze(0))
        max_q_next = q_values_next.max(1)[0].item()

        q_values_target = q_values.clone()
        q_values_target[0][action] = reward + self.gamma * max_q_next

        loss = nn.MSELoss()(q_values, q_values_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.replay_memory.add_transition(state, action, reward, next_state, done)
        state = next_state


        #1/0
        return img_stack, reward, done, info_dict

    def get_event(self, action):
        if action >= len(self.possible_events):
            return None
        return self.possible_events[action]

    # if doing action size of variable number, use these functions
    # def step(self, action):
    #     info_dict = {}
    #     event = self.get_event(action)
    #     if event is not None:
    #         # do some checks prior to executing gym event
    #         new_event = self.policy.check_gym_event(event)
    #         self.input_manager.add_event(new_event)
    #         if event == new_event:
    #             info_dict['event_same'] = 1
    #     else:
    #         info_dict['event_same'] = 2
    #
    #     state = self.device.get_current_state()
    #     self.action_space
    #     reward, done = self.get_reward_done(state)
    #     # get image for state and return it
    #     state_img = self.get_image_state()
    #     self.frames.append(state_img)
    #     img_stack = np.array(self.frames)
    #     img_stack = np.moveaxis(img_stack, 0, -1)
    #
    #     return img_stack, reward, done, info_dict
    #
    # def get_event(self, action):
    #     return self.possible_events[action]
    #
    # # action space size shifts, generate after each step and restart
    # @property
    # def action_space(self):
    #     self.set_possible_events()
    #     return spaces.Discrete(len(self.possible_events))

    # update env with possible events
    def set_possible_events(self):
        state = self.device.get_current_state()
        events = state.get_possible_input()
        #print(events)
        self.possible_events = events
        self.events_probs = list(np.ones(len(events)) / len(events))
        # if humanoid, sort events by humanoid model
        if self.device.humanoid is not None:
            self.possible_events, self.events_probs = self.policy.sort_inputs_by_humanoid(self.possible_events)
        if self.add_unexplored:
            # get first unexplored event and insert it at beginning of events. is pushed to index 1 if using POLICY_GREED_BFS
            # if no unexplored event, None is inserted
            unexplored_event = self.get_unexplored_event()
            self.possible_events.insert(0, unexplored_event)
            self.events_probs.insert(0, 0)
        if self.policy_type == POLICY_GREEDY_BFS:
            self.possible_events.insert(0, KeyEvent(name="BACK"))
            self.events_probs.insert(0, 0)
        elif self.policy_type == POLICY_GREEDY_DFS:
            self.possible_events.append(KeyEvent(name="BACK"))
            self.events_probs.append(0)
        # print('FINAL SOURCE')
        # for event in self.possible_events:
        #     print(event.get_event_str(state))
        # print('END BASIC SOURCE')
    def reset(self):
        if self.reset_count > 0:
            self.logger.info("Resetting env: calling stop ")
            event = self.policy.reset_stop()
            self.input_manager.add_event(event)
            event = self.policy.reset_home()
            self.input_manager.add_event(event)
            event = self.policy.reset_start()
            self.input_manager.add_event(event)
            #self.logger.info("Done resetting env ")

        self.reset_count += 1
        state = self.device.get_current_state()
        state_img = self.get_image_state()
        for _ in range(self.stack_size):
            self.frames.append(state_img)
        img_stack = np.array(self.frames)
        img_stack = np.moveaxis(img_stack,0,-1)
        self.set_possible_events()

        # new state, thus generate new action space and events
        self.action_space
        return img_stack, {}, False, {}

    def get_image_state(self):
        img_path = self.device.take_screenshot()
        img = Image.open(img_path)
        img_array = np.array(img)
         # imagem em escala de cinza
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # redimensionar imagem
        frame = cv2.resize(frame,(self.width,self.height), interpolation=cv2.INTER_AREA)
        return frame

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

#AQUI para baixo precisa se adaptar ao código
    def check_for_errors(response):
        # Verificar se há erros na resposta do aplicativo
        if 'error' in response:
            return True
    # Função Para saber se um evento no estado atual deu errado   
    def is_event_failed(self, event): 
        current_state = self.device.get_current_state()

        # Capturar o estado antes de executar o evento
        pre_event_state = current_state.copy()

        # Executar o evento no dispositivo
        self.device.execute_event(event)

        # Capturar o estado depois de executar o evento
        post_event_state = self.device.get_current_state()

        # Verificar se houve alguma mudança indesejada ou inconsistência no estado
        if self.is_unexpected_state_change(pre_event_state, post_event_state):
            return True
        else:
            return False

    def is_unexpected_state_change(self, state_before, state_after):
        # Comparar os estados antes e depois
        # Implemente a lógica de comparação específica do seu aplicativo
        # Verifique se há mudanças indesejadas ou inconsistências

        # Exemplo: Verificar se há uma diferença nas propriedades relevantes do estado
        if state_before.property != state_after.property:
            return True

        return False


    def is_exception_logged():
        # Verificar se há exceções registradas no log do aplicativo usando o ADB
        process = subprocess.Popen(['adb', 'logcat', '-d', '-s', 'TAG'], stdout=subprocess.PIPE)
        output, _ = process.communicate()
        logcat_output = output.decode('utf-8')
        if 'exception' in logcat_output:
            return True
        else:
            return False
    #Neste exemplo, estamos usando o comando adb logcat para obter o conteúdo do log do aplicativo. 
    # Você precisará substituir 'TAG' pelo valor correto usado para filtrar as mensagens de log do 
    # seu aplicativo.
    #Certifique-se de adaptar o código para a sua situação específica, considerando como você está 
    # registrando os logs do seu aplicativo durante a execução.
    #ESSas 3 funções precisão se adptar ao código de acordo com o 
    # comportamento esperado do seu aplicativo e a forma como as exceções são registradas no log

    #Função de Recompensa
    def get_reward_done(self, state):
        reward = 0.0
        done = 0
        
        error_logs = self.adb_connector.get_error_logs()  # Obter os logs de erro usando o ADBConnector

        # Lógica para atribuir recompensas com base nos erros encontrados nos logs
        if error_logs:
            reward += 5  # Recompensa de 5 para cada erro encontrado nos logs

        if state is not None:
            try:
                # State not visited before
                if not self.policy.utg.is_state_reached(state):
                    reward += 0.04
                # Is an ending state?
                if len(state.get_possible_input()) == 0:
                    done = 1
            except Exception as e:
                self.logger.warning("exception in get_reward_done: %s" % e)
                import traceback
                traceback.print_exc()
                reward = 0.0
                done = 1
                self.logger.error("Erro durante a execução do teste: %s" % e)
                reward -= 1.0  # Recompensa negativa para erros de exceção
    
        # Iniciar a cobertura em uma thread separada
        coverage_tracker = DroidBotCoverage()
        coverage_thread = threading.Thread(target=coverage_tracker.start_coverage)
        coverage_thread.start()

        # Executar o teste do DroidBotX

        # Interromper a cobertura e obter as informações de cobertura
        coverage_tracker.stop_coverage()
        coverage_thread.join()
        coverage_data = coverage_tracker.get_coverage_data()

        # Analisar as informações de cobertura e atribuir recompensa com base na cobertura alcançada
        if coverage_data:
            reward += 3  # Recompensa de 3 para cada erro encontrado na cobertura de código

        return reward, done


    # return first unexplored event
    def get_unexplored_event_index(self):
        current_state = self.device.get_current_state()
        for i in range(len(self.possible_events)):
            if not self.policy.utg.is_event_explored(event=self.possible_events[i], state=current_state):
                #self.logger.info("Found an unexplored event, returning to agent")
                return i
        return None

    # return first unexplored event
    def get_unexplored_event(self):
        current_state = self.device.get_current_state()
        for input_event in self.possible_events:
            if not self.policy.utg.is_event_explored(event=input_event, state=current_state):
                #self.logger.info("Found an unexplored event, returning to agent")
                return input_event
        return None

    # return all unexplored events
    def get_unexplored_event_list(self):
        ret_list = []
        current_state = self.device.get_current_state()
        for input_event in self.possible_events:
            if not self.policy.utg.is_event_explored(event=input_event, state=current_state):
                ret_list.append(ret_list)
        return ret_list


    def handle_none_current_state(self, current_state):
        if current_state is None:
            self.logger.warning("Failed to get current state in handle_none! Sleep for 5 seconds then back event (per droidbot source code)")
            import time
            time.sleep(5)
            new_event = KeyEvent(name="BACK")
            self.input_manager.add_event(new_event)
            current_state = self.device.get_current_state()

        while current_state is None:
            self.logger.warning("Failed to get current state again! Resetting Env")
            self.reset()
            time.sleep(2)
            current_state = self.device.get_current_state()

        return current_state

""" a classe ADBConnector encapsula a lógica para acessar o ADB e obter os logs de erro. 
A função get_logs retorna todos os logs do ADB, enquanto a função get_error_logs filtra 
apenas os logs de erro.
Dentro da função get_reward_done, após calcular a recompensa com base em outros critérios, 
você pode chamar o método get_error_logs da instância adb_connector para obter os logs de 
erro do ADB. Em seguida, você pode analisar esses logs e atribuir uma recompensa adicional
com base na quantidade ou gravidade dos erros encontrados.
Certifique-se de atualizar o caminho para o executável do ADB na variável adb_path dentro da classe 
ADBConnector para corresponder à localização do ADB no seu sistema. """
import subprocess

class ADBConnector:
    def __init__(self):
        # Configurar o caminho para o executável do ADB
        self.adb_path = '/caminho/para/o/adb'

    def get_logs(self):
        # Comando do ADB para obter os logs
        cmd = [self.adb_path, 'logcat']

        try:
            # Executar o comando do ADB e capturar a saída
            result = subprocess.run(cmd, capture_output=True, text=True)
            logs = result.stdout

            return logs
        except Exception as e:
            print(f"Erro ao obter logs do ADB: {e}")
            return None

    def get_error_logs(self):
        # Comando do ADB para filtrar os logs de erro
        cmd = [self.adb_path, 'logcat', '*:E']

        try:
            # Executar o comando do ADB e capturar a saída
            result = subprocess.run(cmd, capture_output=True, text=True)
            error_logs = result.stdout

            return error_logs
        except Exception as e:
            print(f"Erro ao obter logs de erro do ADB: {e}")
            return None

""" a classe DroidBotCoverage para gerenciar a cobertura de instrução. A função start_coverage() 
inicia a cobertura, a função stop_coverage() para e salva as informações de cobertura em um 
arquivo .coverage, e a função get_coverage_data() retorna os dados de cobertura coletados. 
A função get_coverage_percentage() calcula a porcentagem de cobertura alcançada com base nas informações de cobertura. """
#devo importar coverage = pip install coverage
#devo importar threading
import coverage
import threading

class DroidBotCoverage:
    def __init__(self):
        self.cov = coverage.Coverage(source=["path/to/my_module.py"])  # Incluir o módulo relevante no escopo da cobertura
        self.coverage_data = None
        self.coverage_lock = threading.Lock()

    def start_coverage(self):
        self.cov.start()

    def stop_coverage(self):
        self.cov.stop()
        self.coverage_lock.acquire()
        self.coverage_data = self.cov.get_data()
        self.coverage_lock.release()

    def get_coverage_data(self):
        self.coverage_lock.acquire()
        data = self.coverage_data
        self.coverage_lock.release()
        return data

    def get_coverage_percentage(self):
        self.coverage_lock.acquire()
        total_lines = self.cov.total_lines()
        covered_lines = self.cov.covered_lines()
        coverage_percentage = (covered_lines / total_lines) * 100 if total_lines > 0 else 0
        self.coverage_lock.release()
        return coverage_percentage
    
    
import random
from collections import deque

#memória de replay, você pode criar uma classe separada que represente a memória 
# e possua os métodos para adicionar, armazenar e amostrar transições de experiência. 
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)   


#Nova classe chamada QNetwork 
# que herda de nn.Module para representar a rede neural
    
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, replay_memory):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.replay_memory = replay_memory

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    # Esse método é responsável por receber um lote de transições da 
    # memória de replay e realizar o treinamento da rede neural com base nesse lote
    def train_batch(self, batch, optimizer, loss_fn, gamma):
        states, actions, rewards, next_states, dones = batch
        # Convertendo as transições para tensores do PyTorch
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        # Calculando os valores Q-alvo
        q_values = self.forward(states)
        next_q_values = self.forward(next_states)
        q_values_targets = q_values.clone().detach()
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        q_values_targets[range(len(actions)), actions] = rewards + gamma * max_next_q_values * (1 - dones)
        # Calculando a perda
        loss = loss_fn(q_values, q_values_targets)
        # Otimizando os pesos da rede neural
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()
    #chamadas para amostrar o lote de transições
    def train(self, batch_size, optimizer, loss_fn, gamma):
        batch = self.replay_memory.sample_batch(batch_size)
        loss = self.train_batch(batch, optimizer, loss_fn, gamma)
        return loss
    
    """ 
    Aqui posso colocar o código DRLAgent
    foi implementado em DRLAgent.py
    """



