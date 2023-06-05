""" A classe DRLAgent, é responsável pelo treinamento do agente de 
aprendizado por reforço. Nessa classe, foi definido os 
hiperparâmetros, criou-se instâncias do ambiente, memória de replay 
e rede neural, e implementou o loop de treinamento. """
import torch
from droidbot.droidbot_env import DroidBotEnv, QNetwork, ReplayMemory


class DRLAgent:
    def __init__(self):
        # Define hyperparameters
        self.num_episodes = 50
        self.max_steps_per_episode = 500
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0

        # Criar instâncias do ambiente, memória de replay e rede neural
        self.env = DroidBotEnv()
        self.replay_memory = ReplayMemory(capacity=10000)
        self.q_network = QNetwork(input_size, output_size, self.replay_memory)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.MSELoss()

    def predict(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # Converte o estado em um tensor e adiciona uma dimensão extra
            # para que o tensor tenha o formato adequado para a entrada da rede neural
            q_values = self.q_network(state)
            # Calcula os valores Q para o estado atual usando a rede neural
            action = q_values.argmax().item()
            # Seleciona a ação com o maior valor Q
        return action

    def train(self):
          # Loop de treinamento
        for episode in range(self.num_episodes):
            state = self.env.reset()  # Get the initial state of the environment
            total_reward = 0
            # Selecionar uma ação com base na política atual (por exemplo, epsilon-greedy)
            for _ in range(self.max_steps_per_episode):
                action = self.predict(state)    # Executar a ação no ambiente
                next_state, reward, done = self.env.step(action)
                # Armazenar a transição na memória de replay
                self.replay_memory.add_transition(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                # Verificar se a memória de replay tem transições suficientes para amostrar um lote
                if len(self.replay_memory) >= self.batch_size:
                     # Amostrar um lote de transições da memória de replay
                    batch = self.replay_memory.sample_batch(self.batch_size)
                     # Treinar a rede neural com o lote de transições
                    loss = self.q_network.train_batch(batch, self.optimizer, self.loss_fn, self.gamma)

                if done:
                    break

            print(f"Episode: {episode}, Total Reward: {total_reward}")


class Game:
    def __init__(self):
        self.agent = DRLAgent()

    def play(self):
        state = self.agent.env.reset()  # Get the initial state of the environment
        total_reward = 0

        for _ in range(self.agent.max_steps_per_episode):
            action = self.agent.predict(state)  # Predict the next action
            next_state, reward, done = self.agent.env.step(action)

            state = next_state
            total_reward += reward

            if done:
                break

        #print(f"Total Reward: {total_reward}")


# Criar uma instância do jogo para jogá-lo
game = Game()
game.play()
