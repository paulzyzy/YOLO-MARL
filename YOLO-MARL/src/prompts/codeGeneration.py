from base import BaseCodeGen
import hydra
import os
import numpy as np
import torch
from util import clean_obs_code
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from openai import OpenAI


class CodeGen(BaseCodeGen):
    def __init__(self, cfg):
        super(CodeGen, self).__init__(cfg)
        self.cfg = cfg            
        if self.cfg.model.find('gpt') != -1 or self.cfg.model.find('o1') != -1:
            self.model_type = "gpt"
        elif self.cfg.model.find('claude') != -1:
            self.model_type = "claude"
        self.use_llm_strategy = True

        if self.cfg.env.name.split("_")[0] == "lbf":
            self.environment_task = f"This Level-Based Foraging (LBF) {self.cfg.env.name} multi-agent reinforcement learning enviroment has {self.cfg.env.n_agents} agents and {self.cfg.env.n_food} food. Your goal is to make agents collaborate and pickup all the food present in the environment."
        elif self.cfg.env.name.split("_")[0] == "mpe":
            self.environment_task = f"This Multi-Agent Particle Environment (MPE) {self.cfg.env.name} multi-agent reinforcement learning enviroment has {self.cfg.env.n_agents} agents and {self.cfg.env.n_landmarks} landmarks. Your goal is to make agents collaborate and to cover all the landmarks."
        elif self.cfg.env.name.split("_")[0] == "rware":
            self.environment_task = f"This multi-robot warehouse (RWARE) environment {self.cfg.env.name} multi-agent reinforcement learning enviroment has {self.cfg.env.n_agents} agents, which simulates a warehouse with agents moving and delivering requested goods. Your goal is to make agents pick-up empty shelves and deliver them to the goal location, and then agents need to return them to the return location."
        self.sys_prompt = "You are an AI expert specializing in multi-agent reinforcement learning."

        self.prompt_dir = os.getcwd()
        with open(os.path.join(self.prompt_dir , "env_code",self.cfg.env.name.split("_")[0], f"processed_obs_{self.cfg.env.name}.py"), "r") as f:
            self.processed_global_states_format = clean_obs_code(f.read())
        if self.cfg.env.name.split("_")[0] == "rware":
            with open(os.path.join(self.prompt_dir , "env_code",self.cfg.env.name.split("_")[0], f"rware_memory.py"), "r") as f:
                self.rware_memory_format = clean_obs_code(f.read())
        with open(os.path.join(self.prompt_dir , "env_code",self.cfg.env.name.split("_")[0], f"task2action.py"), "r") as f:
            self.llm_action_format = clean_obs_code(f.read())
        with open(os.path.join(self.prompt_dir, "tips" ,self.cfg.env.name.split("_")[0], "goal.txt"), "r") as f:
            self.goal = f.read()
        with open(os.path.join(self.prompt_dir, "tips" ,self.cfg.env.name.split("_")[0], "format.txt"), "r") as f:
            self.format = f.read()
        with open(os.path.join(self.prompt_dir, "tips" ,self.cfg.env.name.split("_")[0], "rules.txt"), "r") as f:
            self.rule = f.read().format(time_steps=self.cfg.env.limit)      
        with open(os.path.join(self.prompt_dir, "tips", self.cfg.env.name.split("_")[0], "planning_signature.txt"), "r") as f:
            planning_func_signature = f.read()
        with open(os.path.join(self.prompt_dir, "tips", self.cfg.env.name.split("_")[0], "reward_signature.txt"), "r") as f:
            reward_func_signature = f.read()
        with open(os.path.join(self.prompt_dir, "tips", self.cfg.env.name.split("_")[0], "instruction_think.txt"), "r") as f:
            self.instruction_think = f.read()
        with open(os.path.join(self.prompt_dir, "tips", self.cfg.env.name.split("_")[0], "scenario", f"assignment_{self.cfg.env.name}.txt"), "r") as f:
            self.assignment_class = f.read()

        with open(os.path.join(self.prompt_dir, "tips", self.cfg.env.name.split("_")[0], "scenario_think.txt"), "r") as f:
            self.scenerio_think = f.read()

        self.format = self.format.format(planning_func_signature=planning_func_signature, reward_func_signature=reward_func_signature)
        ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(self.prompt_dir)))
        if self.model_type == "gpt":
            with open(os.path.join(ws_dir, "openai_KEY.txt"), "r") as f:
                api_key = f.read()
            self.client = OpenAI(api_key=api_key)
        elif self.model_type == "claude":
            with open(os.path.join(ws_dir, "claude_KEY.txt"), "r") as f:
                api_key = f.read()
            self.client = Anthropic(api_key=api_key)

    def get_completion(self, messages, temperature=1.0):
        if self.model_type == "gpt":
            response = self.client.chat.completions.create(
                model = self.cfg.model,
                messages=messages,
                temperature=temperature,
                # seed = 123456,
                top_p=0.1
            )
            return response.choices[0].message.content
        elif self.model_type == "claude":
            message = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=8192,
            temperature=temperature,
            system=self.sys_prompt,
            messages=messages
            )
            return message.content[0].text
        else:
            raise ValueError("Model type not supported")

    def generate_strategy(self,):

        strat_user_prompt = f"<environment description>{self.environment_task}</environment description>"
        strat_user_prompt += f"You have to follow the rules of game: {self.rule}"
        strat_user_prompt += f"{self.assignment_class}"
        # if self.cfg.env.name.split("_")[0] == "lbf":
        # strat_user_prompt += f"{self.scenerio_think}"
        strategy = ""
        if self.use_llm_strategy:
            print("Using LLM strategy")
            messages = []
            if self.model_type == "gpt":
                messages += [{"role": "system", "content": self.sys_prompt}]
            messages += [{"role": "user", "content": strat_user_prompt + self.instruction_think}]
            strategy_response = self.get_completion(messages)
            if self.cfg.save_raw:
                self.save_code_to_file(strategy_response, f"{self.model_type}_strategy.txt", "strat")
            strategy += "<tips>" + strategy_response + "</tips>"
        else:
            strategy = ""
        return strategy, strat_user_prompt

    def generate_functions(self,):
        strategy_response, strat_user_prompt = self.generate_strategy()

        functions_prompt = f"<objective>{self.goal}</objective>"
        functions_prompt += f"The enviroment code information is provided as followed: <processed_states code>{self.processed_global_states_format}<\processed_tates code>"
        if self.cfg.env.name.split("_")[0] == "rware":
            functions_prompt += f"The memory function is provided as followed: <memory code>{self.rware_memory_format}<\memory code>"
        functions_prompt += f"The mapping from llm_tasks to llm_actions function as follow: <action code>{self.llm_action_format}</action code>"

        functions_prompt += f"The function generation format are given as follows:{self.format}"
        functions_prompt += f"Think step-by-step before you generate two functions based on all the information given above. First, think what kind of informations are provided in processed_states and how to use them in the functions. Second, please analysis the environment descrition and think about what is the proper strategies to use and what combination of tasks for each agent you want to assign in this situation. Please not only pay attention to how to make two functions correct but also try your best to make agents coordinate in two functions based on the instrcution."
        functions_prompt += f"Generate the response code for two functions in the following format: <code>def ...</code>" 
        # print(strat_user_prompt)
        # print(functions_prompt)
        if self.model_type == "gpt":
            if self.use_llm_strategy:
                total_messages = [
                    {
                        "role": "system",
                        "content": self.sys_prompt
                    },
                    {
                        "role": "assistant",
                        "content": strategy_response 
                    
                    },
                    {
                        "role": "user",
                        "content": strat_user_prompt + functions_prompt
                    
                    }
                ]
            else:
                total_messages = [
                    {
                        "role": "system",
                        "content": self.sys_prompt
                    },
                    {
                        "role": "user",
                        "content": strat_user_prompt + functions_prompt
                    
                    }
                ]
        elif self.model_type == "claude":
            if self.use_llm_strategy:
                total_messages = [
                    {
                        "role": "user",
                        "content": strat_user_prompt
                
                    },
                    {
                        "role": "assistant",
                        "content": strategy_response

                
                    },
                    {
                        "role": "user",
                        "content": functions_prompt
                
                    }
                ]
            else:
                total_messages = [
                    {
                        "role": "user",
                        "content": strat_user_prompt + functions_prompt
                
                    }
                ]

        function_response = self.get_completion(total_messages)
        save_file_name = f"{self.model_type}_generated_code"
        if self.cfg.save_raw:
            self.save_code_to_file(function_response, save_file_name + ".txt", "raw")
        clean_code = self.extract_python_functions(function_response)
        self.save_code_to_file(clean_code, save_file_name + ".py", "code")
        return total_messages

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    print(cfg)
    codegen = CodeGen(cfg)
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    codegen.generate_functions()
    
if __name__ == "__main__":
    main()