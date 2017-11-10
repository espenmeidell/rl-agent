# Project: RLAgent
# Created: 08.11.17 15:13
# Author: Espen Meidell, Markus Andresen

import gym
from keras.layers import Dense
from keras.models import Sequential
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


class Agent:
    def __init__(self, env, n_observations: int, n_actions: int):
        self.environment = env
        self.gamma = 0.95
        self.batch_size = 32
        self.n_observations = n_observations
        self.last_state = self.environment.reset().reshape(1, n_observations)
        self.memory = deque(maxlen=3000)
        self.Q_network = build_model(n_inputs=n_observations, n_outputs=n_actions)

    # Replaced by random sample ?
    def get_random_cases_from_memory(self, n: int) -> List[Tuple]:
        results = []
        indices = np.random.randint(0, len(self.memory), n).tolist()
        for i in indices:
            results.append(self.memory[i])
        return results

    def get_next_action(self):
        output = self.Q_network.predict(self.last_state)
        action = np.argmax(output)
        if np.random.random() <= 0.2:
            action = self.environment.action_space.sample()
        return action

    def train(self):
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
            self.Q_network.fit(np.array(inputs), np.array(targets), batch_size=1, epochs=1, verbose=0)

    def run(self):
        self.last_state = self.environment.reset().reshape(1, self.n_observations)
        done = False
        while not done:
            action = self.get_next_action()
            state, reward, done, _ = self.environment.step(action)
            self.memory.append((self.last_state, action, reward, state, done))
            self.train()
            self.last_state = state.reshape(1, self.n_observations)


def main():
    env = gym.make("CartPole-v0")
    env.render()
    agent = Agent(env=env, n_observations=env.observation_space.shape[0], n_actions=env.action_space.n)
    trials = 0
    for trial in range(trials):
        agent.run()
        print(trial)


if __name__ == "__main__":
    main()
