# Project: RLAgent
# Created: 08.11.17 15:13
# Author: Espen Meidell, Markus Andresen

import gym
from keras.layers import Dense
from keras.models import Sequential, load_model
from collections import deque
import numpy as np
import random
from typing import List, Tuple


def build_model(n_inputs: int, n_outputs: int) -> Sequential:
    model = Sequential()
    model.add(Dense(input_dim=n_inputs, units=20, activation="relu", kernel_initializer='random_uniform'))
    model.add(Dense(units=40, activation="relu", kernel_initializer='random_uniform'))
    model.add(Dense(units=n_outputs, activation="softmax", kernel_initializer='random_uniform'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model


def save_model(model: Sequential, name: str):
    model.save("models/%s.h5" % name)


class Agent:
    def __init__(self, env, n_observations: int, n_actions: int):
        self.environment = env
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.n_observations = n_observations
        self.last_state = self.environment.reset().reshape(1, n_observations)
        self.memory = deque(maxlen=2000)
        self.Q_network = build_model(n_inputs=n_observations, n_outputs=n_actions)

    # Replaced by random sample ?
    def get_random_cases_from_memory(self, n: int) -> List[Tuple]:
        results = []
        indices = np.random.randint(0, len(self.memory), n).tolist()
        for i in indices:
            results.append(self.memory[i])
        return results

    def get_next_action(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.environment.action_space.sample()
        return np.argmax(self.Q_network.predict(self.last_state)[0])
        # return np.random.choice(range(0, self.environment.action_space.n), p=output[0])

    def train(self):
        # TODO: FIND BUG AND REMOVE IT
        if len(self.memory) >= self.batch_size:
            samples = random.sample(self.memory, self.batch_size)
            inputs = []
            targets = []
            for sample in samples:
                state, action, reward, next_state, done = sample
                output = self.Q_network.predict(state)
                q_current = reward
                if not done:
                    q_future = max(self.Q_network.predict(next_state.reshape(1, self.n_observations))[0])
                    output[0][action] = q_current + q_future * self.gamma
                else:
                    output[0][action] = q_current
                inputs.append(state[0])
                targets.append(output[0])
            print(targets)
            self.Q_network.fit(np.array(inputs), np.array(targets), batch_size=1, epochs=1, verbose=0)

    def run(self):
        self.last_state = self.environment.reset().reshape(1, self.n_observations)
        done = False
        total_reward = 0
        print("Epsilon", self.epsilon)
        while not done:
            action = self.get_next_action()
            state, reward, done, _ = self.environment.step(action)
            total_reward += reward
            self.memory.append((self.last_state, action, reward, state, done))
            self.train()
            self.last_state = state.reshape(1, self.n_observations)
        return total_reward


def main():
    env = gym.make("CartPole-v0")
    # env.render()
    agent = Agent(env=env, n_observations=env.observation_space.shape[0], n_actions=env.action_space.n)
    # agent.Q_network = load_model("models/15.4.h5")
    trials = 50
    result = 0
    for trial in range(trials):
        print("Trial %d" % trial)
        result += agent.run()
    print("Average result: %.3f" % (result/trials))
    save_model(agent.Q_network, str(result/trials))


if __name__ == "__main__":
    main()
