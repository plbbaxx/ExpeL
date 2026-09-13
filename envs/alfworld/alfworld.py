import re
from typing import List, Dict, Any, Tuple
from envs.base import BaseEnv

from alfworld.agents.environment import get_environment
from utils import get_env_name_from_gamefile

class AlfworldEnv(BaseEnv):
    def __init__(self,
                gamefile: str,
                config: Dict[str, Any],
                max_steps: int = 50,
                ):
        self.max_steps = max_steps
        self.gamefile = gamefile
        self.config = config
        # Resolve the configured ALFWorld environment via its supported registry.
        # Recent ALFWorld releases do not re-export ``AlfredTWEnv`` at module scope.
        env_cls = get_environment(self.config.env.type)
        self.main_env = env_cls(self.config, train_eval=self.config.split)
        self.main_env.game_files = [self.gamefile]
        self.task = "housekeeper robot. The agent was placed in a household environment and a task to complete."
        self.env_name = get_env_name_from_gamefile(gamefile)

        self.reset()

    def reset(self):
        self.curr_step = 1
        self.answer = ''
        self.terminated = False
        self.reward = False
        self.is_exhausted = False
        self.env = self.main_env.init_env(batch_size=1)
        _, infos = self.env.reset()
        self.admissible_actions = list(infos.get('admissible_commands', [[]])[0])
        self.last_action = None
        # Kept as a per-task audit trail.  It records the exact model action
        # and the canonical action delivered to ALFWorld without changing the
        # task or action protocol.
        self.action_trace = []

    def step(self, action: str) -> Tuple[str, bool, bool, bool, int]:
        raw_action = action
        if action.startswith('put'):
            # ALFWorld has one canonical placement command.  Qwen may emit
            # ``in``, ``on``, or the official literal ``in/on``; normalize all
            # three forms before dispatching and preserve both in action_trace.
            pattern = r'put (\w+\s*\d+) (?:in/on|in|on) (\w+\s*\d+)'
            match = re.fullmatch(pattern, action.strip())
            if match is not None:
                action = 'put ' + match.group(1) + ' in/on ' + match.group(2)
        
        admissible_before_step = action in self.admissible_actions
        observation, reward, _, info = self.alfworld_run(action)
        self.action_trace.append({
            'step': self.curr_step,
            'raw_action': raw_action,
            'env_action': action,
            'admissible_before_step': admissible_before_step,
            'observation': observation,
            'reward': bool(reward),
        })
        self.admissible_actions = list(info.get('admissible_commands', [[]])[0])
        observation = observation.replace(' In it, you see nothing.', '').replace(', you see nothing', '')
        if self.last_action == action:
            self.truncated = True
            self.terminated = True
        
        self.last_action = action

        if reward:
                observation = 'Task is SOLVED.'
                self.terminated = True
        else:
            if self.is_truncated():
                observation = 'Max steps reached.'
            pass

        self.curr_step += 1
        self.terminated = self.is_terminated()
        self.truncated = self.is_truncated()
        self.reward = reward

        return observation, self.reward, self.terminated, self.truncated, self.curr_step

    def success_fn(self) -> bool:
        return self.reward
    
    def alfworld_run(self, action):
        observation, reward, done, info = self.env.step([action])
        observation, reward, done = process_observation(observation[0]), info['won'][0], done[0]

        return observation, reward, done, info

def process_observation(obs):
    if obs.startswith('You arrive at loc '):
        obs = obs[obs.find('. ')+2:]    
    return obs
