import os
import re

class BaseCodeGen:
    def __init__(self, cfg):
        pass

    def get_completion(self, ):
        raise NotImplementedError
    
    def generate_functions(self,):
        raise NotImplementedError

    def save_code_to_file(self, code, name, save_dir):
        """
        Saves the generated code to a Python file.

        Parameters:
            code (str): The Python code to save.
            filename (str): The filename to save the code to.
        """
        filename_dir = os.path.join(self.prompt_dir, "gen_code", self.cfg.env.name, save_dir)
        if not os.path.exists(filename_dir):
            os.makedirs(filename_dir)
        file_num = len(os.listdir(filename_dir))
        self.gen_code_dir = filename_dir
        prefix, suffix = name.split(".")[0], name.split(".")[1]
        filename = f"{prefix}_{file_num}.{suffix}"
        filename_path = os.path.join(filename_dir, filename)
        with open(filename_path, 'w') as file:
            file.write(code)
        print(f"Code saved to {filename_path}")

    def extract_python_functions(self, text):
        """
        Extracts complete Python function definitions from a given raw text string.

        Param:
            text (str): The text to extract Python functions from.
        Returns:
            str: The extracted Python functions.
        """
        patterns = [
            r'```python\s*(.*?)```',   # Matches ```python <code> ```
            r'<code>(.*?)</code>',     # Matches inline code in HTML
        ]
        code = ""
        for pattern in patterns:
            code = re.findall(pattern, text, re.DOTALL)
            if code:
                break
        
        if not code:
            raise ValueError("No Python code found in the text")
        return code[0]
    

# if __name__ == "__main__":
#     with open("/home/eddie880509/src/LLM-copilot-RL/LBF/src/prompts/gen_code/lbf_2p_2f_coop/raw/claude_generated_code_39.txt", "r") as f:
#         code = f.read()

#     clean_code = BaseCodeGen.extract_python_functions(code)
#     print(clean_code)
#     exit()