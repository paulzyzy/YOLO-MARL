import numpy as np
def process_state(observations, p=3, f=3):
    '''
    Param:
        observation:
                        array of array (p, n): dict('agent_0', 'agent_1', ..., 'agent_p')
                        List:
                        Agent : (n, ) list of observation components
        p: int, number of agents
        f: int, number of foods in the environment
    Return:
        obs: tuples (food_info, agents_info):
            food_info: dictionary that contains information about food in the environment
                        key: food_id ('food_0', 'food_1', ...)
                        value: tuples (food_pos, food_level) or None if the food is already been picked up
            agents_info: dictionary that contains information about agents in the environment
                        key: agent_id ('agent_0', 'agent_1', ...)
                        value: tuples (agent_pos, agent_level)
    '''
    food_info = {}
    agents_info = {}
    obs = observations[0]
    offset = 0
    for food_idx in range(f):
        food_obs = obs[offset:offset+3]
        offset += 3
        curr_food_pos = food_obs[:2]
        curr_food_level = food_obs[2]
        food_id = f'food_{food_idx}'
        # If food level is 0, then the food is already been pickup and not present in the environment
        if curr_food_level == 0 and curr_food_pos[0] < 0:
            food_info[food_id] = None
        # The food is present in the environment
        else:
            food_info[food_id] = (curr_food_pos, curr_food_level)
        
    for agent_idx in range(p):
        agent_obs = obs[offset:offset+3]
        offset += 3
        curr_agent_pos = agent_obs[:2]
        curr_agent_level = agent_obs[2]
        agent_id = f'agent_{agent_idx}'
        agents_info[agent_id] = (curr_agent_pos, curr_agent_level)

    return food_info, agents_info