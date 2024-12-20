from .gymma import GymmaWrapper
from prompts.util import *
from utils.logging import get_logger
from copy import deepcopy
# from .multiagentenv import MultiAgentEnv
logger = get_logger()

class LLMWrapper(GymmaWrapper):
    def __init__(self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        compute_reward_fn,
        planning_fn,
        reward_mode,
        env_name,
        **kwargs,
    ):
        super().__init__(key, time_limit,
                        pretrained_wrapper,
                        seed,
                        common_reward, 
                        reward_scalarisation,
                        **kwargs)
        self.reward_mode = reward_mode
        self.set_func(planning_fn, compute_reward_fn)
        self.env_name = env_name
        self.dirname = self.env_name.split("_")[0]
        self.pre_action = [[], []]

        if self.dirname == "rware":
            self.initial_memory = {
                'workstation location': [[4,10], [5,10]],
                'empty_shelves_pos': [],
                'return location': [],
                'status of carried shelf': [False, False],
                'is_carrying_shelf': [False, False]
            }
            self.rware_memory = deepcopy(self.initial_memory)  # initial memory
            self.add_memory2obs =  import_function(
                f"prompts.env_code.rware.rware_memory", "rware_memory")

    def compute_reward(self, observations, tasks):
        raise NotImplementedError
    
    def planning_function(self, observations):
        raise NotImplementedError
    
    def set_func(self, planning_function, compute_reward):
        self.planning_function = planning_function
        self.compute_reward = compute_reward
        logger.critical(f"Planning function and compute reward function({self.reward_mode} mode) set")
    
    def step_train(self, actions):
        # print("memory", self.rware_memory)  
        prev_obs = self._obs
        obs, r, done, truncated, info = super().step(actions)
        process_state =  import_function(
                f"prompts.env_code.{self.dirname}.processed_obs_{self.env_name}", "process_state")

        processed_obs = process_state(prev_obs)
        if self.dirname == "rware":
            processed_obs, self.rware_memory = self.add_memory2obs(processed_obs, self.rware_memory, self.pre_action)

        actions_ = convert_actions(actions)
        self.pre_action = actions
        
        if self.dirname  == "lbf":
            llm_task = self.planning_function(processed_obs)
            llm_actions = lbf_task_to_actions(llm_task, processed_obs)
        elif self.dirname  == "mpe":
            llm_task = self.planning_function(processed_obs)
            llm_actions = mpe_task_to_actions(llm_task, processed_obs)
        elif self.dirname  == "rware":
            llm_task = self.planning_function(processed_obs) 
            llm_actions = rware_task_to_actions(llm_task, processed_obs)

        if  self.reward_mode == "pure":
            # pure llm reward
            reward_dict = self.compute_reward(processed_obs, actions_)
            reward = float(sum(reward_dict.values()))
        elif self.reward_mode == "mixed_constant":
            # original reward + llm constant aligned reward
            reward_dict = constant_reward_signal(
                actions_, llm_actions, llm_reward=0.001, penalty=0.001)
            reward = float(sum(reward_dict.values()))+r
        elif self.reward_mode == "mixed_normalized":
            # original reward + llm normalized code gen reward
            reward_dict = self.compute_reward(processed_obs, llm_actions, actions_)
            reward = normalized_reward(reward_dict, theta=0.01)
            reward = float(sum(reward_dict.values())) + r
        else:
            raise NotImplementedError
    
        return obs, reward, done, truncated, info

    def step_eval(self, actions):
        obs, reward, done, truncated, info = super().step(actions)
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        # print("New episode")
        obs, info = super().reset(seed, options)
        self.pre_action = [[], []]
        if self.dirname == "rware":
            self.rware_memory = deepcopy(self.initial_memory)  # reset memory
        return obs, info