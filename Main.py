# Project: RLAgent
# Created: 08.11.17 15:13

import gym
from keras.layers import Dense
from keras.models import Sequential
import numpy as np


def build_model(n_observations: int, n_actions: int) -> Sequential:
    model = Sequential()
    model.add(Dense(input_dim=n_observations, units=20, activation="relu", kernel_initializer='random_uniform'))
    model.add(Dense(units=40, activation="relu", kernel_initializer='random_uniform'))
    model.add(Dense(units=n_actions, activation="softmax", kernel_initializer='random_uniform'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    return model


def deploy_agent():
    env = gym.make("MountainCar-v0")
    observation = env.reset()
    env.render()
    model = build_model(n_observations=env.observation_space.shape[0], n_actions=env.action_space.n)
    done = False
    while not done:
        action = np.argmax(model.predict(observation.reshape(1, 2)))
        observation, reward, done, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    deploy_agent()
