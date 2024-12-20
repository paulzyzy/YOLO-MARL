import re
import numpy as np
import importlib
import os
import glob


def convert_reward(reward_dict):
    '''
    param: reward (dict): dict containing rewards for each agent.
    return: list: list containing rewards for each agent.
    '''
    rewards = []
    for agent_id, reward in reward_dict.items():
        if isinstance(reward, np.ndarray):
            reward = reward.tolist()
        else:
            reward = [reward]
        rewards.append(reward)
    return rewards

def dict2array(dict):
    '''
    param: dict (dict): dict containing rewards for each agent. 
    return: array: array containing rewards for each agent. (nx1)
    '''
    arr = []
    for agent_id, val in dict.items():
        arr.append(val)
    return np.array(arr).reshape(-1, 1)

def convert_actions(action_list):
    n = len(action_list)
    agent_ids = [f"agent_{i}" for i in range(n)]
    action_dict = {}
    for agent_id, action in zip(agent_ids, action_list):
        action_dict[agent_id] = action
    return action_dict


def clean_obs_code(obs_code_str):
    cleaned_code = re.sub(
        r'^(import .+|from .+)', '', obs_code_str, flags=re.MULTILINE).strip()
    return cleaned_code


def process_available_actions(available_actions):
    '''
    param:
        available_actions (list of n list): list of available actions for each agent.
    return:
        action_dict (Dict of n list): dict containing available actions indices for each agent.
    '''
    n_agents = len(available_actions)
    action_dict = {}
    for i in range(n_agents):
        indices = np.where(np.array(available_actions[i]) == 1)[0].tolist()
        action_dict[f"agent_{i}"] = indices
    return action_dict


def get_generated_code(gencode_path):
    with open(gencode_path, 'r') as f:
        code_str = f.read()
    namespace = {**globals()}
    exec(code_str, namespace)
    planning_function = namespace['planning_function']
    compute_reward = namespace['compute_reward']
    return planning_function, compute_reward

def setup_wrapper(env, gencode_path):
    planning_function, compute_reward = get_generated_code(gencode_path)
    env.set_func(planning_function, compute_reward)
    return env

def lbf_task_to_actions(task, processed_obs):
    '''
    param: task (dict): task to be converted to actions, keyed by agent_id (e.g. 'agent_0').
    param: processed_obs: tuple (food_info, other_agents_info) containing information about food and other agents.
    return: dict: dictionary of actions for each agent.
    '''
    food_info, agents_info = processed_obs
    action = {agent: [] for agent in task.keys()}
    
    for agent in task.keys():
        agent_task = task[agent]
        agent_pos = agents_info[agent][0]  # Get the agent's position from agents_info
        
        if agent_task == "No op":
            action[agent].append(0)  # No operation
        
        elif agent_task == "Pickup":
            action[agent].append(5)  # Pickup action
        
        elif agent_task == "Target food 0":
            food_key = 'food_0'  # Use the correct food key
            if food_info[food_key] is not None:  # Ensure the food is present
                food_pos = food_info[food_key][0]  # Position of food 0
                relative_pos = get_relative_position(agent_pos, food_pos)
                action[agent].extend(get_movement_actions(relative_pos))
        
        elif agent_task == "Target food 1":
            food_key = 'food_1'  # Use the correct food key
            if food_info[food_key] is not None:  # Ensure the food is present
                food_pos = food_info[food_key][0]  # Position of food 1
                relative_pos = get_relative_position(agent_pos, food_pos)
                action[agent].extend(get_movement_actions(relative_pos))
    
    return action

def get_relative_position(agent_pos, target_pos):
    '''
    Helper function to calculate the relative position of target to the agent.
    Returns a tuple (dx, dy) where:
    dx: Difference in the x-coordinate.
    dy: Difference in the y-coordinate.
    '''
    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    return dx, dy

def get_movement_actions(relative_pos):
    '''
    Helper function to determine the movement actions based on the relative position.
    Returns a list of movement actions [move1, move2].
    1: Move North (X-)
    2: Move South (X+)
    3: Move West (Y-)
    4: Move East (Y+)
    '''
    dx, dy = relative_pos
    actions = []
    
    # Determine movement in the x-direction
    if dx < 0:
        actions.append(1)  # Move North
    elif dx > 0:
        actions.append(2)  # Move South
    
    # Determine movement in the y-direction
    if dy < 0:
        actions.append(3)  # Move West
    elif dy > 0:
        actions.append(4)  # Move East
    
    return actions


def import_function(module_name, func_name):
    try:
        # Attempt to import the module
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        return func
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        return None

def get_gencode_path(env_name):
    prompt_dir = os.path.dirname(os.path.abspath(__file__))
    gencode_dir = os.path.join(prompt_dir, "gen_code", env_name, "code")
    gencode_path = os.path.join(gencode_dir, f'*_generated_code_{len(os.listdir(gencode_dir)) - 1}.py')
    gencode_path = glob.glob(gencode_path)[0]
    return gencode_path

def constant_reward_signal(actions, llm_actions, llm_reward=0.01, penalty=0.01):
    reward_dict = {agent: 0 for agent in actions.keys()}
    # Reward for following LLM suggestions
    for agent, llm_action in llm_actions.items():
        if actions[agent] in llm_action:
            reward_dict[agent] += llm_reward
        else:
            reward_dict[agent] -= penalty

    return reward_dict

def normalized_reward(reward, theta=0.01):
    min_reward = min(reward.values())
    max_reward = max(reward.values())
    if min_reward != max_reward:
        for agent_id in reward:
            reward[agent_id] = (reward[agent_id] - min_reward) / (max_reward - min_reward) * theta
    return reward

def mpe_task_to_actions(task, processed_obs, N=3):
    '''
    param: task (dict): task to be converted to actions, keyed by agent_id (e.g. 'agent_0').
    param: processed_obs (dict): processed observation containing landmark-agent and agent-agent positions.
    param: N (int): number of landmarks or agents.
    return: dict: dictionary of actions for each agent.
    '''
    action = {agent: [] for agent in task.keys()}

    for agent in task.keys():
        agent_task = task[agent]
        agent_obs = processed_obs[agent]
        
        if agent_task == "No op":
            action[agent].append(0)  # No operation

        elif agent_task.startswith("Landmark_0"):
            landmark_idx = 0  # Extract the landmark index
            landmark_pos = agent_obs[landmark_idx]  # Get the relative position of the landmark to the agent
            action[agent].extend(get_mpe_actions(landmark_pos))

        elif agent_task.startswith("Landmark_1"):
            landmark_idx = 1  # Extract the landmark index
            landmark_pos = agent_obs[landmark_idx]  # Get the relative position of the landmark to the agent
            action[agent].extend(get_mpe_actions(landmark_pos))
        
        elif agent_task.startswith("Landmark_2"):
            landmark_idx = 2  # Extract the landmark index
            landmark_pos = agent_obs[landmark_idx]  # Get the relative position of the landmark to the agent
            action[agent].extend(get_mpe_actions(landmark_pos))
    
    return action

def get_mpe_actions(relative_pos):
    '''
    Helper function to determine the movement actions based on the relative position.
    Returns a list of movement actions [move1, move2].
    1: Move left x-
    2: Move right x+
    3: Move up y+
    4: Move down y-
    '''
    dx, dy = relative_pos
    actions = []
    
    # Determine movement in the x-direction
    if dx < 0:
        actions.append(1)  # Move left
    elif dx > 0:
        actions.append(2)  # Move right
    
    # Determine movement in the y-direction
    if dy < 0:
        actions.append(3)  # Move down
    elif dy > 0:
        actions.append(4)  # Move up
    
    return actions

def get_rware_actions(direction, relative_pos, can_move_forward):
    '''
    Helper function to determine the movement actions based on the relative position.
    Returns a list of movement actions [move1, move2].
    1: Up (X-)
    2: Down (X+)
    3: Left (Y-)
    4: Right (Y+)
    '''
    dx, dy = relative_pos

    actions = []
    if dx == 0 and dy == 0:
        return [4]
    if direction == 'up': 
        if can_move_forward and dy < 0:
            actions.append(1)
        if dx < 0:
            actions.append(2)
        if dx > 0:
            actions.append(3)
        if dx == 0:
            if dy > 0:
                actions.extend([2, 3])
            elif dy < 0 and not can_move_forward:
                actions.extend([2, 3])

    elif direction == 'down':
        if can_move_forward and dy > 0:
            actions.append(1)
        if dx < 0:
            actions.append(3)
        if dx > 0:
            actions.append(2)
        if dx == 0:
            if dy < 0:
                actions.extend([2, 3])
            elif dy > 0 and not can_move_forward:
                actions.extend([2, 3])

    elif direction == 'left':
        if can_move_forward and dx < 0:
            actions.append(1)
        if dy < 0:
            actions.append(3)
        if dy > 0:
            actions.append(2)
        if dy == 0:
            if dx > 0:
                actions.extend([2, 3])
            elif dx < 0 and not can_move_forward:
                actions.extend([2, 3])
        
    elif direction == 'right':
        if can_move_forward and dx > 0:
            actions.append(1)
        if dy < 0:
            actions.append(2)
        if dy > 0:
            actions.append(3)
        if dy == 0:
            if dx < 0:
                actions.extend([2, 3])
            elif dx > 0 and not can_move_forward:
                actions.extend([2, 3])

    return actions

def rware_task_to_actions(task, processed_obs):
    action = {agent: [] for agent in task.keys()}
    agent_infos = processed_obs['agent_infos']

    for agent in task.keys():
        agent_task = task[agent]
        agent_index = int(agent.split('_')[1])  # Extract the index from the agent string
        agent_info = agent_infos[agent_index]
        agent_pos = agent_info['location']
        agent_direction = agent_info['direction']
        agent_can_move_forward = agent_info['can_move_forward']
        agent_can_place_shelf = agent_info['can_place_shelf']

        if agent_task == "random explore":
            if agent_info['can_move_forward']:
                action[agent].extend([1, 2, 3])
            else:
                action[agent].extend([2, 3])

        elif agent_task == "empty shelf":
            empty_shelf_pos = processed_obs['empty_shelves_pos']
            if not empty_shelf_pos:
                if agent_info['can_move_forward']:
                    action[agent].extend([1, 2, 3])
                else:
                    action[agent].extend([2, 3])
            else:
                target = min(empty_shelf_pos, key=lambda pos: sum(abs(a - b) for a, b in zip(pos, agent_pos)))
                relative_pos = get_relative_position(agent_pos, target)
                action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
        
        elif agent_task == "workstation":
            workstation_locations = processed_obs['workstation location']
            closest_workstation = min(workstation_locations, 
                    key=lambda pos: sum(abs(a - b) for a, b in zip(pos, agent_pos)))
            relative_pos = get_relative_position(agent_pos, closest_workstation)
            action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
            # workstation_pos_1 = processed_obs['workstation location'][0]
            # workstation_pos_2 = processed_obs['workstation location'][1]
            # relative_pos_1 = get_relative_position(agent_pos, workstation_pos_1)
            # relative_pos_2 = get_relative_position(agent_pos, workstation_pos_2)
            # action[agent].extend(get_rware_actions(agent_direction, relative_pos_1, agent_can_move_forward))
            # action[agent].extend(get_rware_actions(agent_direction, relative_pos_2, agent_can_move_forward))

        elif agent_task == "return":
            return_locations = processed_obs['return location']
            if not processed_obs['return location']:
                if agent_info['can_move_forward']:
                    action[agent].extend([1, 2, 3])
                else:
                    action[agent].extend([2, 3])
            else:
                closest_return = min(return_locations, 
                        key=lambda pos: sum(abs(a - b) for a, b in zip(pos, agent_pos)))
                relative_pos = get_relative_position(agent_pos, closest_return)
                action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
                # for return_pos in processed_obs['return location']: 
                #     relative_pos = get_relative_position(agent_pos, return_pos)
                #     action[agent].extend(get_rware_actions(agent_direction, relative_pos, agent_can_move_forward))
        
    return action

if __name__ == "__main__":
    print(get_gencode_path("mpe_simple_spread_v3_5"))