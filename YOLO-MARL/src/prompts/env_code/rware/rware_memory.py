import numpy as np
def rware_memory(processed_obs, memory, actions):  #the action is the action which leads to the processed_obs       
    # Access agent information
    agent_infos = processed_obs['agent_infos']
    current_empty_shelf_pos = processed_obs.get('empty_shelves_pos', [])
    for pos in current_empty_shelf_pos:
        if not any(np.array_equal(pos, mem_pos) for mem_pos in memory['empty_shelves_pos']):
            memory['empty_shelves_pos'].append(pos) # Add new empty shelf positions to memory
    # Iterate over each agent
    for i, (agent_info, action) in enumerate(zip(agent_infos, actions)):
        agent_location = agent_info['location']
        is_carrying_shelf = agent_info['is_carrying_shelf']
        can_place_shelf = agent_info['can_place_shelf']
        # Case 1: Agent loaded a shelf in the last step
        if action == 4 and is_carrying_shelf and can_place_shelf and not memory['is_carrying_shelf'][i] and not any(np.array_equal(agent_location, loc) for loc in memory['workstation location']):
            if not any(np.array_equal(agent_location, empty_pos) for empty_pos in memory['empty_shelves_pos']):# No empty shelves, agent loads a non-empty shelf
                memory['status of carried shelf'][i] = True
                if not any(np.array_equal(agent_location, ret_pos) for ret_pos in memory['return location']):
                    memory['return location'].append(agent_location)
            else: #agent loads an empty shelf
                memory['status of carried shelf'][i] = False
                # Check if agent is at an empty shelf position
                for idx, ep_pos in enumerate(memory['empty_shelves_pos']):
                    if np.array_equal(agent_location, ep_pos):
                        del memory['empty_shelves_pos'][idx]
                        # memory['empty_shelves_pos'].remove(agent_location)
                        if not any(np.array_equal(agent_location, ret_pos) for ret_pos in memory['return location']):
                            memory['return location'].append(agent_location)

        # Case 2: Agent places a shelf at return location
        elif action == 4 and not is_carrying_shelf and can_place_shelf and memory['is_carrying_shelf'][i]:  
            memory['status of carried shelf'][i] = False        
            # Check if agent is at a return location
            for idx, rl_pos in enumerate(memory['return location']):  # Create a copy to avoid modification during iteration
                if np.array_equal(agent_location, rl_pos):
                    del memory['return location'][idx]
                    # memory['return location'].remove(agent_location)
        
        # Case 3: Agent at workstation
        if action == 4 and is_carrying_shelf and any(np.array_equal(agent_location, loc) for loc in memory['workstation location']):
            memory['status of carried shelf'][i] = True

        agent_info['status of carried shelf'] = memory['status of carried shelf'][i]# Update agent_info of carried shelf is empty or not
    # Update processed_obs with memory
    processed_obs['return location'] = memory['return location']
    processed_obs['workstation location'] = memory['workstation location']
    processed_obs['empty_shelves_pos'] = memory['empty_shelves_pos']
    memory['is_carrying_shelf'] = [agent_info['is_carrying_shelf'] for agent_info in agent_infos]

    return processed_obs, memory
