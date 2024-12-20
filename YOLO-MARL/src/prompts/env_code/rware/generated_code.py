
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
