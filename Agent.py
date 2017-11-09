# Project: RLAgent
# Created: 08.11.17 15:13

import gym
from keras.layers import Dense
from keras.models import Sequential
from collections import deque
import numpy as np


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
        self.last_observation = self.environment.reset().reshape(1, 2)  # TODO make non specific to environment
        self.memory = deque(maxlen=1000)
        self.Q_network: Sequential = build_model(n_inputs=n_observations, n_outputs=n_actions)

    def get_next_action(self):
        output = self.Q_network.predict(self.last_observation)
        action = np.argmax(output)
        if np.random.random() > 0.5:
            action = self.environment.action_space.sample()
        return action

    def run(self):
        done = False
        while not done:
            action = self.get_next_action()
            print("Action:", action)
            observation, reward, done, _ = self.environment.step(action)
            self.last_observation = observation.reshape(1, 2)  # TODO make non specific to environment
            self.environment.render()


def main():
    env = gym.make("MountainCar-v0")
    env.render()
    agent = Agent(env=env, n_observations=env.observation_space.shape[0], n_actions=env.action_space.n)
    agent.run()


if __name__ == "__main__":
    main()
