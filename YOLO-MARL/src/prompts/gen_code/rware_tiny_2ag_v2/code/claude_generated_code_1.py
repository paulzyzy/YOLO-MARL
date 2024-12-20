
def planning_function(processed_state):
    """
    Determines optimal tasks for each agent based on the current state.
    
    Args:
        processed_state: A dictionary containing global information for each agent and the positions of empty shelves observed by any agent.

    Returns:
        dict: Optimal tasks for each agent ('random explore','empty shelf','workstation','return')
    """
    tasks = {}
    for i, agent_info in enumerate(processed_state['agent_infos']):
        agent_name = f'agent_{i}'
        
        if not agent_info['is_carrying_shelf']:
            if processed_state['empty_shelves_pos']:
                tasks[agent_name] = 'empty shelf'
            else:
                tasks[agent_name] = 'random explore'
        else:  # Agent is carrying a shelf
            if not agent_info['status of carried shelf']:  # Carrying an empty shelf
                tasks[agent_name] = 'workstation'
            else:  # Carrying a non-empty shelf
                tasks[agent_name] = 'return'
    
    return tasks

def compute_reward(processed_state, llm_actions, actions):
    """
    Calculate rewards based on the tasks assigned and their outcomes.
    
    Args:
        processed_state: returned from function process_state(state, p, f)
        llm_actions (dict): dictionary of list of integers which means the suggest actions from llm for each agent.
        actions (dict): dictionary of a integer action that actually perform by each agent.
        
    Returns:
        reward: Dict containing rewards for each agent.
    """
    rewards = {}
    for i, agent_info in enumerate(processed_state['agent_infos']):
        agent_name = f'agent_{i}'
        reward = 0
        
        # Reward for following LLM suggestions
        if actions[agent_name] in llm_actions[agent_name]:
            reward += 0.1
        
        # Reward for successful pickup or delivery
        if actions[agent_name] == 4:  # Load/Unload action
            if agent_info['location'] in processed_state['empty_shelves_pos'] and not agent_info['is_carrying_shelf']:
                reward += 1  # Picked up an empty shelf
            elif agent_info['location'] in processed_state['workstation location'] and agent_info['is_carrying_shelf'] and not agent_info['status of carried shelf']:
                reward += 2  # Delivered an empty shelf to workstation
            elif agent_info['location'] in processed_state['return location'] and agent_info['is_carrying_shelf'] and agent_info['status of carried shelf']:
                reward += 1.5  # Returned a non-empty shelf
        
        # Penalty for collisions or invalid actions
        if actions[agent_name] == 1 and not agent_info['can_move_forward']:
            reward -= 0.5
        
        rewards[agent_name] = reward
    
    return rewards
