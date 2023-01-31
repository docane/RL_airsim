import math
import numpy as np
import random
import time
import torch
from collections import defaultdict
from OOP_environment import Env
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

"""===================================================================="""

""" 학습요소 옵션 """
TERMINAL_REWARD = 10.0
LEARNING_RATE = 0.001
DISCOUNTING_RATE = 0.99

""" 엡실론 값 옵션 """
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

""" 학습 모델 옵션 """
EPISODE = 10000 # 진행할 에피소드 횟수
MEMORY_SIZE = 10000
BATCH_SIZE = 128

# 모델 저장 불러오기 옵션
TARGET_UPDATE = 1000
LOAD_MODEL = False
LOAD_NUM = 10000
SAVE_MODEL = False

"""===================================================================="""

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Tensorboard 사용
writer = SummaryWriter()

class DQNAgent():
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(9, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), LEARNING_RATE)
        self.steps_done = 0
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPS_START

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state,
                            action,
                            torch.FloatTensor([reward]),
                            next_state))

    def get_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.epsilon = eps_threshold
        if random.random() < eps_threshold:
            return torch.LongTensor([[random.randrange(4)]])
        else:
            return self.model(state).max(1)[1].view(1, 1)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random .sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        current_q = self.model(states).gather(1, actions)

        max_next_q = self.model(next_states).detach().max(1)[0]
        expected_q = rewards + (DISCOUNTING_RATE * max_next_q)

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

if __name__ == "__main__":
    env = Env("DQN")
    agent = DQNAgent()

    if LOAD_MODEL:
        agent.model.load_state_dict(torch.load(f'./saved_model/DQN/model_state_dict_ep{LOAD_NUM}.pt'))
    else:
        LOAD_NUM = 0

    for episode in range(EPISODE):
        env.reset()
        action_count = 1

        # 생성된 게임의 상태를 불러옴
        state = env.get_state()
        state = torch.FloatTensor([state])
        # 현재 상태에 대한 행동을 선택
        action = agent.get_action(state)

        epi_text = env.print_text(f"Episode : {episode + 1 + LOAD_NUM}", 15, (-230, 200))

        while True:
            # 행동을 위한 후 다음상태 보상 에피소드의 종료 여부를 받아옴
            next_state, reward, done = env.step(action)
            # 다음 상태에서의 다음 행동 선택
            next_state = torch.FloatTensor([next_state])
            next_action = agent.get_action(next_state)


            if done:
                reward = TERMINAL_REWARD

            agent.memorize(state, action, reward, next_state)
            loss = agent.learn()

            state = next_state
            action = next_action
            action_count += 1

            if done:
                print("episode: {:3d} | actions: {:6d} | epsilon: {:.3f} | loss: {:.6f}".format(
                    episode + 1 + LOAD_NUM, action_count, agent.epsilon, loss))
                #스탭 딜레이
                writer.add_scalar("Action_counts/episode", action_count, episode)
                time.sleep(1)
                break

        if SAVE_MODEL and (episode + 1) % TARGET_UPDATE == 0:
            torch.save(agent.model.state_dict(), f'./saved_model/DQN/model_state_dict_ep{LOAD_NUM + episode + 1}.pt')


        time.sleep(0.1)
        epi_text.clear()
        writer.flush()

        EPS = EPS_START * (1. - (1. / EPISODE))
        if EPS > EPS_END: EPS_START = EPS

    writer.close()