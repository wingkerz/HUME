#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from ai_agent_local_llm.crew import AiAgentLocalLlm,Generate_function
from ai_agent_local_llm.tools.custom_tool import Init_Pysr

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    
    inputs = {
        # '': str(datetime.now().year),
        'case_name': 'stressstrain',
        'dim': 2,
        'data_num': 20
    }

    
    try:
        AiAgentLocalLlm().crew().kickoff(inputs=inputs)
        pysr_result = Init_Pysr(inputs["case_name"], inputs["dim"], inputs["data_num"])
        prompts = """
Below, I will provide you with a list that contains the best equations currently in the population:
{fx_list}.
To generate equations that are different from those in the given lists (with different operator or variable combinations) and ensure they are not completely identical to any existing equation!
The variables in the equations : {variables}. It is forbidden to generate any variables other than the specified ones.
Operators list : [+, -, *, /, ^, ^(0.5), exp, sqrt, sin, cos, log] (The operators for the equations can only be chosen from the operators list.)
Replace constants with the number 1.
Please analyze the following generation requirements step by step:  
    1. Operator Diversity: Each equation must use at least 3 different operators (chosen from Operators list). Generate an equation no longer than 100 characters!
    2. The generated equations need to satisfy the structure of rational functions, and they should be obtained through the addition, subtraction, multiplication, and division of polynomials.
       Keep the polynomial in factored form. 
{suggest_list}
You don't need to output anything extra. Use your imagination as much as possible! Generated equations must be diverse, fully exploring combinations of variables and operators, as well as all possibilities! 
"""
        # print(pysr_result)
        with open("report1.txt", "r", encoding="utf-8") as f:
            suggest_list = f.read().strip()
        new_string = prompts.format(
            fx_list=pysr_result,
            suggest_list=suggest_list,
            variables=['x0','x1']
        )
        print(new_string)
        # inputs = {

        # }
        Generate_function().crew().kickoff(new_string)

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI Agent",
        'current_year': str(datetime.now().year)
    }
    try:
        AiAgentLocalLlm().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        AiAgentLocalLlm().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI Agent",
        "current_year": str(datetime.now().year)
    }
    
    try:
        AiAgentLocalLlm().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
