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