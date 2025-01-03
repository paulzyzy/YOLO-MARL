
def planning_function(processed_state):
    """
    Determines optimal tasks for each agent based on the current state.
    
    Args:
        processed_state: A dictionary containing global information for each agent and the positions of empty shelves observed by any agent.

    Returns:
        dict: Optimal tasks for each agent ('random explore','empty shelf','workstation','return')
    """
    llm_tasks = {}
    agent_infos = processed_state['agent_infos']
    empty_shelves_pos = processed_state['empty_shelves_pos']
    return_locations = processed_state.get('return location', [])
    workstation_locations = processed_state.get('workstation location', [])

    for i, agent_info in enumerate(agent_infos):
        agent_name = f'agent_{i}'
        
        if agent_info['is_carrying_shelf']:
            if agent_info['status of carried shelf']:
                llm_tasks[agent_name] = 'return'
            else:
                llm_tasks[agent_name] = 'workstation'
        else:
            if empty_shelves_pos:
                llm_tasks[agent_name] = 'empty shelf'
            else:
                llm_tasks[agent_name] = 'random explore'

        # Coordinate agents to avoid conflicts
        if i > 0 and llm_tasks[agent_name] == llm_tasks[f'agent_{i-1}']:
            if llm_tasks[agent_name] == 'empty shelf' and len(empty_shelves_pos) > 1:
                continue  # Allow both agents to go for different empty shelves
            elif llm_tasks[agent_name] == 'workstation' and len(workstation_locations) > 1:
                continue  # Allow both agents to go for different workstations
            elif llm_tasks[agent_name] == 'return' and len(return_locations) > 1:
                continue  # Allow both agents to go for different return locations
            else:
                llm_tasks[agent_name] = 'random explore'  # Avoid conflict by exploring

    return llm_tasks

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
    reward = {}
    agent_infos = processed_state['agent_infos']
    empty_shelves_pos = processed_state['empty_shelves_pos']
    return_locations = processed_state.get('return location', [])
    workstation_locations = processed_state.get('workstation location', [])

    for i, (agent_name, agent_action) in enumerate(actions.items()):
        agent_info = agent_infos[i]
        suggested_actions = llm_actions[agent_name]
        reward[agent_name] = 0

        # Reward for following LLM suggestions
        if agent_action in suggested_actions:
            reward[agent_name] += 0.1

        # Reward for successful loading/unloading
        if agent_action == 4:  # Load/Unload action
            if agent_info['location'] in empty_shelves_pos and not agent_info['is_carrying_shelf']:
                reward[agent_name] += 1  # Successfully loaded an empty shelf
            elif agent_info['location'] in workstation_locations and agent_info['is_carrying_shelf']:
                reward[agent_name] += 2  # Successfully delivered to workstation
            elif agent_info['location'] in return_locations and agent_info['is_carrying_shelf']:
                reward[agent_name] += 1.5  # Successfully returned a shelf

        # Penalty for unnecessary actions
        if agent_action != 0 and agent_info['can_move_forward'] == False:
            reward[agent_name] -= 0.1

        # Reward for exploration when needed
        if not agent_info['is_carrying_shelf'] and not empty_shelves_pos:
            if agent_action in [1, 2, 3]:  # Forward, Left, Right
                reward[agent_name] += 0.05

        # Coordinate rewards
        other_agent = f'agent_{1-i}'  # Get the other agent's name
        if other_agent in actions:
            if actions[agent_name] != actions[other_agent]:  # Reward for diverse actions
                reward[agent_name] += 0.1
            if agent_info['is_carrying_shelf'] != agent_infos[1-i]['is_carrying_shelf']:  # Reward for task division
                reward[agent_name] += 0.2

    return reward
