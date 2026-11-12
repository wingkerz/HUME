import ast
import copy
from datetime import datetime
from itertools import product
import json
import os
import random
import re
import sys
os.environ["OTEL_SDK_DISABLED"] = "true"
import numpy as np
import sympy as sp
from src.ai_agent_local_llm.tools.custom_tool import Input_data,Input_data_dim, Observations, Observations_island, ResultCollector, aaa, add_parentheses_around_denominator, bbb, convert_unit_list_to_base, dict_to_list, dimensional_analysis, expand_formula, expand_formula_list, expand_formula_list_with_permutations, extract_json_string, extract_operators, get_equation_complexity, get_unique_random_samples, input_data_num, list_to_obs, load_data_SR, load_json_f, load_json_f_to_list, load_json_f_zuhe, load_json_oper_and_var_agent, load_json_structure_agent, load_json_structure_term_agent,multi_constant, multi_constant_parallel, mutate, operator_combinations, process_solution_list, replace_C, replace_C_to_one, replace_numbers, replace_numbers111, to_physic, validate_operators
from langchain_openai import ChatOpenAI
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai import Agent, Crew, Process, Task
from typing import Tuple, Union, Dict, Any
import src.ai_agent_local_llm.tools.search_2nd_phase.gp_tools as gp
from IPython.display import Markdown,display
from pydantic import BaseModel
from crewai import TaskOutput
import asyncio
import concurrent.futures
import warnings
from collections import Counter
from fractions import Fraction
from math import lcm
import multiprocessing
import signal
from src.ai_agent_local_llm.tools.screen_pretrain_knowledge_eq import make_pop_from_file, evolve, multi_processing_eval_pop, \
    multi_processing_opt_constant, opt_constant_one_core, opt_constant_one_core_SR, pset_f, pset_g, converter, tournamentSelection, \
    opt_constant_multi_core, multi_processing_opt_constant_SR,get_parser
project_root = '/home/liqifei/ai_agent_local_llm/src'
if project_root not in sys.path:
    sys.path.append(project_root)
parser = get_parser()
args, unknown = parser.parse_known_args() 
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["SERPER_API_KEY"] = "api-key"
qwen_model = ChatOpenAI(model = "ollama/qwen2.5:7b")
prompts = """
Below, I will provide you with a list that contains the best equations currently in the population:
{fx_list}.
To generate equations that are different from those in the given list (with different operator or variable combinations) and ensure they are not completely identical to any existing equation!
The variables in the equations : {variables}. It is forbidden to generate any variables other than the specified ones.
Operators list : {operators} (The operators for the equations can only be chosen from the operators list.)
Please analyze the following generation requirements step by step:  
    1. Operator Diversity: Each equation must use at least 3 different operators (chosen from Operators list). Generate an equation no longer than 100 characters!
    2. The generated equations need to satisfy the structure of rational functions, and they should be obtained through the addition, subtraction, multiplication, and division of polynomials.
       Keep the polynomial in factored form. 
    3. Replace constants with the number 1.
    4. Here are some suggestions you must refer to when generating equations:
       {suggest_list}
Generated equations must be diverse, fully exploring combinations of variables and operators, as well as all possibilities! You don't need to output anything extra. 
"""
prompts2 = """
Below, I will provide you with a list that contains the best equations currently in the population:
{fx_list}.
The variables in the equations : {variables}. It is forbidden to generate any variables other than the specified ones.
Operators list : {operators} (The operators for the equations can only be chosen from the operators list.)
Please analyze the following generation requirements step by step:  
    1. Operator Diversity: Each equation must use at least 3 different operators (chosen from Operators list). 
       Generate an equation no less than 50 characters!
    2. The generated equations need to satisfy the structure of rational functions, and they should be obtained through the addition, subtraction, multiplication, and division of polynomials.
    3. Replace constants with the number 1.
    4. Generating mathematically illogical expressions (such as ^sqrt, ()exp, x0 +- 1) is prohibited! 
    5. The generated equation should satisfy the following patterns:
       {suggest_list}
You don't need to output anything extra. 
"""
prompts6 = """
The variables in the equations : {variables}. (It is forbidden to generate any variables other than the specified ones.) 
Operators list : {operators} (The operators for the equations can only be chosen from the operators list.)
Below, I will provide you with a list that contains the best sub-equations currently in the population:
{reference}
Please analyze the following generation requirements step by step:  
    1. Generate an equation no more than 40 characters! Don't be too short! Replace constants with the number 1.
    2. {desc}
    3. While conforming to physical dimensions, unleash your imagination as much as possible. Fully exploring combinations of variables ({variables}) and operators in operators list, as well as all possibilities!
You don't need to output anything extra. 
"""
STORED_OBSERVATIONS = Observations()
save_result = True
inputs = {
        'case_name': args.case_name, 
        'dim': 4,
        'len': 3,
        'variables': [],
        'operators' : '[+, -, *, /, ^, sin, cos , sqrt, log]',
        'symbols': [],
        'symbol_x' : [],
        'symbol_y' : None,
        'symbol_descs': [],
        'num': 3,
        'data_num': 10,
        'prompts': prompts2,
        'fx_list':[],
        'suggest_list':[],
        'suggest_list1':[],
        'data':"",
        'list_f_term':[],
        'step_i':0,
        'stuck_num':0,
        'stuck_num1':0,
        'default_eqs_groups': [],
        'pysr_operators':[],
        'pyhsics':True,
        'union_y':[],
        'union_x':[],
        'scene':"",
        'final_unit_list':[]
    }
print(f"case_name: {inputs['case_name']}")
if save_result:
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d-%H-%M")
    save_dir = os.path.join("result_figure",inputs["case_name"], str(inputs["dim"]),formatted_now)
    save_path = os.path.join(save_dir, 'result.txt')
    os.makedirs(save_dir, exist_ok=True)
pysr_result = []
operators_list = [] 
suggest_list = []
inputs['case_name'], X0, Y, inputs['dim'], N, inputs["symbols"], inputs["symbol_descs"] = load_data_SR(inputs["case_name"], inputs["dim"], add_str="")
if "symbols" in inputs and inputs["symbols"]:
    inputs["symbol_y"] = inputs["symbols"][0]
    inputs["symbol_x"] = inputs["symbols"][1:]
    inputs['pyhsics'] = True 
else:
    inputs['pyhsics'] = False
    inputs["symbol_y"] = None        
    inputs["symbol_x"] = None
inputs['variables'] = [f'x{i}' for i in range(inputs['dim'])]
inputs['len'] = inputs['dim'] + 1
inputs['data'] = input_data_num(inputs['case_name'],inputs['dim'],inputs['data_num'])
inputs["pysr_operators"] = inputs["operators"]
print(inputs["symbols"])
print(inputs["symbol_descs"])
class Blog(BaseModel):
    f : str
def validate_json_output(result: TaskOutput) -> Tuple[bool, Any]:
    """Validate and parse JSON output."""
    try:
        data = json.loads(extract_json_string(result.raw))
        return (True, data)
    except json.JSONDecodeError as e:
        return (False, "Invalid JSON format")
def validate_json_output_meta(result: TaskOutput) -> Tuple[bool, Any]:
    """
    Validate and parse JSON output.
    Required format: {"1": "Value"}, where Value is in ['A', 'B', 'C', 'D'].
    """
    try:
        json_str = extract_json_string(result.raw)
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return (False, "Output is not a JSON object (dictionary).")
        if "1" not in data:
            return (False, "Missing required key '1'.")
        valid_values = {'A', 'B', 'C', 'D'}
        value = data["1"]
        if not isinstance(value, str) or value.strip().upper() not in valid_values:
            return (False, f"Invalid value '{value}'. Must be one of {valid_values}")
        data["1"] = value.strip().upper()
        return (True, data)
    except json.JSONDecodeError:
        return (False, "Invalid JSON format (Decode Error).")
    except Exception as e:
        return (False, f"Unexpected validation error: {str(e)}")
def validate_json_output_structure_agent(result: TaskOutput) -> Tuple[bool, Any]:
    try:
        data = json.loads(extract_json_string(result.raw))
    except json.JSONDecodeError:
        return (False, "Invalid JSON format")
    if "global_structure" not in data or "sub_expressions" not in data:
        return (False, "Missing required fields")
    global_structure = data["global_structure"]
    sub_expressions = data["sub_expressions"]
    valid_pattern = re.compile(r"^T\d+$")
    for key in sub_expressions.keys():
        if not valid_pattern.match(key):
            return (False, f"Invalid T key: {key}")
    terms_in_structure = re.findall(r"T\d+", global_structure)
    unique_terms_in_structure = set(terms_in_structure)
    unique_terms_in_subexpr = set(sub_expressions.keys())
    if unique_terms_in_structure != unique_terms_in_subexpr:
        missing_in_structure = unique_terms_in_subexpr - unique_terms_in_structure
        missing_in_subexpr = unique_terms_in_structure - unique_terms_in_subexpr
        return (
            False,
            f"Mismatch between global_structure and sub_expressions. "
            f"Missing in structure: {missing_in_structure}, "
            f"Missing in sub_expressions: {missing_in_subexpr}"
        )
    return (True, data)
def validate_json_output_structure_term_agent(result: TaskOutput) -> Tuple[bool, Any]:
    try:
        data = json.loads(extract_json_string(result.raw))
    except json.JSONDecodeError:
        return (False, "Invalid JSON format")
    if not isinstance(data, dict):
        return (False, "JSON must be an object with T keys")
    valid_pattern = re.compile(r"^T\d+$")
    for key, value in data.items():
        if not valid_pattern.match(key):
            return (False, f"Invalid T key: {key}. Must match 'T<number>'")
        if not isinstance(value, str):
            return (False, f"Value of {key} must be a string description")
    return (True, data)
def validate_json_physics_union(result: TaskOutput) -> Tuple[bool, Any]:
    try:
        data = json.loads(extract_json_string(result.raw))
    except json.JSONDecodeError:
        return (False, "Invalid JSON format")
    if "global_structure" not in data or "sub_expressions" not in data:
        return (False, "Missing required fields")
    sub_expressions = data["sub_expressions"]
    try:
        expected_count = inputs['dim'] + 1
        actual_count = len(sub_expressions)
        if actual_count != expected_count:
            return (
                False, 
                f"Unit count mismatch. Expected {expected_count} (dim={inputs['dim']} + 1), "
                f"but got {actual_count}. Keys found: {list(sub_expressions.keys())}"
            )
    except KeyError:
        return (False, "Missing 'dim' in inputs configuration for validation.")
    except NameError:
        return (False, "Global variable 'inputs' not found for validation.")
    return (True, data)
def validate_simple_unit_json(result: TaskOutput) -> Tuple[bool, Any]:
    try:
        json_str = extract_json_string(result.raw)
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return (False, "Invalid JSON format")
    if not data:
        return (False, "Empty JSON result")
    if len(data) != 1:
        return (False, f"Expected exactly 1 key-value pair, got {len(data)}. Data: {data}")
    unit_value = list(data.values())[0]
    if not isinstance(unit_value, str):
        return (False, f"Unit must be a string, got {type(unit_value)}")
    return (True, data)
def validate_output_physics(result: TaskOutput) -> Tuple[bool, Any]:
    ALLOWED_VARS = {f"x{i}" for i in range(100)}
    ALLOWED_FUNCS = {
        'sqrt','exp','log','sin','cos','tan','abs',
        'tanh','sinh','cosh','arcsin','arccos','arctan',
        'sec','csc','cot','sinc','pi','e'
    }
    try:
        raw = result.raw.strip()
        first_line = raw.splitlines()[0].strip()
        if not first_line:
            return (False, "Empty expression")
        x = to_physic(first_line, inputs["symbols"])
        x_str = str(x)
        print(f"Parsed result: {x_str}")
        found_words = set(re.findall(r"[a-zA-Z_]\w*", x_str))
        illegal = {
            w for w in found_words
            if w not in ALLOWED_VARS and w not in ALLOWED_FUNCS
        }
        if illegal:
            return (False, f"Invalid variable or function: {', '.join(sorted(illegal))}")
        return (True, x_str)
    except json.JSONDecodeError:
        return (False, "Invalid JSON format")
    except Exception as e:
        return (False, f"error: {type(e).__name__} {e}")
data_analyst = Agent(
    role = "Data Analyst",
    goal = "Please use the tools to retrieve and carefully analyze the data relationships in the given {case_name} dataset, and identify the possible operators that may exist between the data",
    backstory = (
        "You are a professional data analyst. You can analyze the given data to identify the operators present in the data"
    ),
    tools=[Input_data],
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
data_analyst_task = Task(
    description = (
           "Data dimension dim={dim}. Please analyze the relationships between variables based on the first {data_num} records\n"
           "Propose specific hypotheses based on the data, and test the simplest arithmetic combinations based on the data:\n"
            "- Variable sums: x0+x1, x0+x2, x1+x3, ...\n"
            "- Variable differences: x0-x1, x1-x2, x2-x3, x3-x0, ...\n"
            "- Find the 3 simple combinations most correlated with y\n"
"1. Which variable has the most direct effect on y? 2. Is there an obvious pairing relationship?\n"
    ),
    expected_output = ("You do not need to output anything extra; only output the analysis results obtained\n"),
    tools=[Input_data],
    agent = data_analyst,
)
data_analyst_1 = Agent(
    role = "Physics Expert",
    goal = "You are a physics expert. You can analyze the data in the given {case_name} dataset and provide the possible physical meaning of each x and y",
    backstory = (
        "You can analyze the given data to determine the possible physical meaning of each x and y in the data"
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
data_analyst_task_1 = Task(
    description = (
            "Below are the data x and y:\n"
            "{data}\n"
           "Data dimension dim={dim}. Please analyze the relationships between variables based on the first {data_num} records\n"
           "Based on the relationships between variables, analyze what physical quantity and meaning each variable may represent\n"
    ),
    expected_output = ("Output the obtained analysis results and the physical meaning of each variable\n"),
    agent = data_analyst_1,
)
crew = Crew(
    agents = [data_analyst_1],
    tasks = [data_analyst_task_1],
)
data_analyst_operators = Agent(
    role = "Mathematical Pattern Recognition Expert",
    goal = "Match the most likely mathematical operators based on data characteristics (such as trends and periodicity)",
    backstory = (
        "You are a professional data analyst, and you focus only on the engine that captures numerical patterns in variable data. My goal is to fit y using all x variables and operators. You determine the mathematical relationships by observing the ranges, growth rates, and volatility of all x and y data."
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
data_analyst_operators_task = Task(
    description = (
    "Data dimension dim={dim}. Please analyze the possible operators between variables based on {data_num} records:\n"
    "1. Operator candidate list: ['/', '^', '^2', '^3', 'sqrt', sin, cos, log, exp]\n"
    "2. Judgment rules:\n"
    "   - If data y shows periodicity with respect to one or some x variables: then 'sin' and 'cos' may be selected.\n"
    "   - If data y grows extremely fast (explosively) with respect to one or some x variables: then '^2' or '^3' may be selected (use exp with caution).\n"
    "   - If data y grows more slowly (gradually flattening) with respect to one or some x variables: then 'sqrt' or 'log' may be selected.\n"
    "   - If data y has an inverse relationship with respect to one or some x variables (y decreases as x increases): prioritize '/'.\n"
    "3. Data overview:\n"
        "{data}\n"
    "Based on the above data patterns, please select the two different operators most likely to fit y\n"
    ),
    expected_output=(
        "Strictly output a Python list containing only 2 strings, without any explanation; only select the two different operators most likely to appear from ['^', '^2', '^3', 'sqrt', 'log', 'sin', 'cos', 'exps']\n"
        "Example format: ['operator1','operator2']\n"
    ),
    agent = data_analyst_operators,
)

data_analyst_operators_crew = Crew(
    agents = [data_analyst_operators],
    tasks = [data_analyst_operators_task],
)
Physics_Agent = Agent(
    role = "You are a physics expert",
    goal = "Output the physical scenario I am currently studying, and output the international standard units of the given variables",
    backstory = (
        "You can analyze the physical scenario I am currently studying and the physical units of each variable based on the given variables and their descriptions"
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
Physics_Agent_task = Task(
    description = "Below are the names of {len} variables: {symbols}. \n"
    "Below is the description corresponding to each variable: {symbol_descs}\n"
    "The specific units are as follows: {final_unit_list}"
    "Please strictly follow the steps below step by step: \n"
    "1. Analyze the physical scenario I am currently studying\n"
    "2. Provide the international standard physical unit for each variable; only the symbol of the unit is needed (if a variable has no unit, use '-')\n",
    expected_output = "Below is an example output format. You do not need to output any extra information; just output the following JSON string (this is only to show you the corresponding JSON format; you need to generate the specific content yourself):\n"
    "{ \n"
    "   \"global_structure\": \"physical scenario\",\n"
    "   \"sub_expressions\": { \n"
    "       \"variable\": \"physical unit (do not output special symbols)\",\n"
    "       \"variable\": \"physical unit (do not output special symbols)\",\n"
    "       \"variable\": \"physical unit (do not output special symbols)\",\n"
    "     }\n"
    "}\n",
    guardrail=validate_json_physics_union,
    max_retries=15, 
    agent = Physics_Agent,
    async_execution = True
)
Physics_Single_Var_Task = Task(
    description = (
        "You are a professional physicist. Please provide the international standard unit (SI Unit) based on the variable name and its physical meaning that I provide.\n"
        "Variable name: {target_symbol}\n"
        "Variable description: {target_desc}\n\n"
        "Requirements: \n"
        "1. If there is no unit, use '-'.\n"
        "2. Use '*' for multiplication and '/' for division (for example: 'kg*m/s^2').\n"
        "3. Output JSON directly, without including any analysis process or extra text.\n\n"
    ),
    expected_output = "Only output a JSON string containing the variable name and unit (just provide the symbol of the unit). Expected output example: \n"
        "{\n"
        "  \"{target_symbol}\": \"fill in the unit symbol here\"\n"
        "}\n",
    guardrail=validate_simple_unit_json,
    max_retries=5, 
    agent=Physics_Agent, 
    async_execution=False 
)
Physics_Single_Var_Crew = Crew(
    agents=[Physics_Agent],
    tasks=[Physics_Single_Var_Task],
    verbose=False
)
Physics_Agent_Crew = Crew(
    agents = [Physics_Agent],
    tasks = [Physics_Agent_task],
)
Trans_Physics_Agent = Agent(
    role = "You are a physicist proficient in dimensional analysis and the International System of Units (SI).",
    goal = "Output the conversion of the units of the variables I provide using only the seven international standard units",
    backstory = (
        "You can convert the units of the variables I provide into unit forms expressed using only the international base units, based on the known international base units"
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
Trans_Physics_Agent_task = Task(
    description = "Below are the variable units: {union_per}. \n"
    "Please strictly follow the steps below step by step: \n"
    "1. Only base units: the result may contain only these 7 base units and their combinations: [kg, m, s, A, K, mol, cd].\n"
    "2. Decompose derived units: all derived units must be fully decomposed.\n"
    "  - For example: 'J' (joule) -> 'kg*m^2*s^-2'\n"
    "  - For example: 'J*s' (joule) -> 'kg*m^2*s^-1'\n"
    "3. Handle dimensionless units: for example, 'rad/s' must be converted to 's^-1'; '-' indicates that the variable has no unit, so output '0'.\n"
    "4. Format requirements: use * for multiplication, use ^ for exponentiation, do not use the division sign /; use negative exponents instead (for example, use s^-1).\n"
    "Please output the derived units according to the above steps.\n",
    expected_output = "Below is an example output format. You do not need to output any extra information; just output the following JSON string (this is only to show you the corresponding JSON format; you need to generate the specific content yourself):\n"
    "{ \n"
    "   \"1\": \"derived base units (do not output special symbols)\",\n"
    "}\n",
    guardrail=validate_json_output,
    max_retries=15, 
    agent = Trans_Physics_Agent,
    async_execution = True
)
Trans_Physics_Agent_Crew = Crew(
    agents = [Trans_Physics_Agent],
    tasks = [Trans_Physics_Agent_task],
)
Physics_generate_Agent = Agent(
    role = "You are a physics expert",
    goal = "Based on the variable information and variable units I provide, give the mathematical expression describing {symbol_y}",
    backstory = (
        "Based on all known variable information and the physical background, give the mathematical expression for {symbol_y}"
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
Physics_generate_task = Task(
    description = "Below are {len} variables: {symbols}. \n"
    "x3, x4, and x5 are all constants. Please use the above three variables arbitrarily; each equation must contain the variables x3, x4, and x5\n"
    "{fx_list}\n"
    "Please use the other {dim} variables to give the mathematical expression for {symbol_y}. Note that dimensional consistency of the variables must be satisfied\n"
    "Please strictly follow the steps below step by step: \n"
    "1. The variables must include the following {dim} variables: {variables}; using other variables that have not appeared is prohibited.\n"
    "2. Please select operators from the following list: {operators}. \n"
    "3. Dimensional consistency must be satisfied, that is, the unit of the final generated mathematical expression must be consistent with the unit of {symbol_y}.\n"
    "4. The length must not exceed 50 characters.\n"
    "Fully consider dimensional consistency; output only expressions that satisfy unit dimensional consistency; all given variables must be used; other variables, operators, and other characters must not appear!\n"
    "Do not output special LaTeX characters! Do not generate illegal variables, functions, or other special characters! You may only use variables and operators to form the equation! Output only one mathematical expression that satisfies dimensional consistency; all given variables must be used!",
    expected_output = "Do not generate any other information (any textual description is prohibited)! Output only one mathematical expression: {symbol_y} = ",
    context=[Physics_Agent_task],
    guardrail=validate_output_physics,
    max_retries=5, 
    agent = Physics_generate_Agent,
    async_execution = False
)
Physics_generate_Agent_Crew = Crew(
    agents = [Physics_generate_Agent],
    tasks = [Physics_generate_task],
)
Meta_Controller_Agent = Agent(
    role = "You are the 'Commander-in-Chief' of a multi-agent symbolic regression system",
    goal = "You have four search strategies (A/B/C/D) available for scheduling. Your goal is to find the equation with the highest fit and the simplest structure at the lowest computational cost.",
    backstory = (
        "You are the 'Commander-in-Chief' of a multi-agent symbolic regression system. Based on some currently known information, you can accurately select the next search strategy."
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
Meta_Controller_Agent_task = Task(
    description = (
        "The strategy descriptions are as follows:\n"
        "Strategy A: Single skeleton with multiple equations. Each iteration generates a new combination of outer skeletons such as T1, T2..., and each inner T generates multiple equations. (Suitable for determining the outer skeleton and exploring inner variable combinations.)\n"
        "Strategy B: Multiple skeletons with a single equation. The grouping of variables is determined, and the variables for each T are determined. Each iteration generates a combination of multiple skeletons such as T1, T2..., exploring the combination forms of outer skeletons. (Suitable for determining inner variable combinations and exploring outer skeletons.)\n"
        "Strategy C: Multiple skeletons with multiple equations. Each iteration generates a combination of multiple outer skeletons such as T1, T2..., and each inner T generates multiple equations. (Suitable for searching the initial situation and exploring the search space.)\n"
        "Strategy D: Single term T1 with multiple equations. The outer skeleton is only T1, and each iteration generates multiple inner equations to explore the relationships among variables. (Suitable when equation complexity is too high.)\n"
        "Current state:\n"
        "Stagnation generations: {stuck_num} (indicating no improvement in the most recent {stuck_num} iterations)\n"
        "Current population diversity: {diversity_score} (similarity among equations in the population; lower means higher similarity)\n"
        "Current best equation: {best_equation} , outer skeleton: {best_term}, inner variable combination: {best_components}, equation complexity: {complexity}\n"
        "History of strategy selections: {Strategy} \n"
        "Please analyze the current state and select the next specific strategy"
    ),
    expected_output = (
        "Output only the letter of the strategy; any extra information is prohibited.\n"
        "{\n"
        "  \"1\": \"letter of the strategy\",\n"
        "}\n"
    ),
    guardrail=validate_json_output_meta,
    max_retries=15, 
    agent = Meta_Controller_Agent,
)
Meta_Controller_Agent_Crew = Crew(
    agents = [Meta_Controller_Agent],
    tasks = [Meta_Controller_Agent_task],
)
Meta_Architect_Agent = Agent(
    role = "Evolutionary Algorithm Strategy Architect",
    goal = "Based on the current optimization state, design a search strategy workflow composed of multiple atomic steps.",
    backstory = "You are proficient in symbolic regression algorithms. You have a toolbox containing: summarizing history, decomposing variables, analyzing operators, generating skeletons, generating subexpressions, and other tools. You need to connect these tools in series according to the difficulties of the task (such as falling into local optima or excessive complexity).",
    llm = qwen_model,
    verbose = True
)
Meta_Architect_Task = Task(
    description = (
        "Current optimization state:\n"
        " - Stagnation generations: {stuck_num}\n"
        " - Historical best equation: {best_equation}\n"
        " - Available variables: {variables}\n\n"
        "Please use the following atomic capability modules to design a new search strategy (Workflow):\n"
        "1. F: Regenerate operators based on the data. Output: new operators\n"
        "2. A: Analyze historical equations and provide suggestions. Output: suggestions in text form\n"
        "3. B: Decompose the problem into subterms (T1, T2...). Output: how many subterms there are and a textual description of each subterm\n"
        "4. C: Determine the specific variables and operator subsets for the subterms. Output: the decomposed variables and operators\n"
        "5. D: Generate the specific mathematical expression for the subterm (or the whole). Output: the specific expression for x\n"
        "6. E: Generate the external skeleton composed of T. Output: the specific expression for T\n"
        "Strategy design principles (must be strictly followed):\n"
        "1. Minimization principle: select only the steps most needed at present (strategies may be skipped). If you only want simple mutation, do not call A.\n"
        "2. You may randomly select a combination strategy from the preceding tools, but the last strategy must end with D or E.\n"
    ),
    expected_output = (
        "Output only a JSON string containing the strategy name, reasoning, and step list. Example format (you need to generate the specific number of steps and strategy usage yourself, and they must not be repeated!):\n"
        "{\n"
        "  \"reasoning\": \"Currently trapped in a local optimum; new structures need to be explored through fine-grained decomposition.\",\n"
        "  \"steps\": [\n"
        "    {\"tool\": \"A\"},\n"
        "    {\"tool\": \"E\"}\n"
        "  ]\n"
        "}"
    ),
    agent = Meta_Architect_Agent,
)
Meta_Architect_Crew = Crew(
    agents = [Meta_Architect_Agent],
    tasks = [Meta_Architect_Task]
)
Structure_Agent = Agent(
    role = "You are a mathematician",
    goal = "You can provide a relatively good expression decomposition plan, including the decomposed sub-expressions and the internal structure of each sub-expression",
    backstory = (
        "You are responsible for designing expression-compliant structures in symbolic regression tasks"
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
Structure_Agent_task = Task(
    description = (
        "Below is the input information for your task:\n"
        "Available variable pool: {variables}. Variables not in the variable pool are prohibited from appearing in T!\n"
        "Reference to the current best expressions: {fx_list} (sorted by loss from low to high)\n"
        "The result of the current best decomposition skeleton is: {list_f_term} (your decomposition result must be different from those in the list! Generating the same decomposition result is prohibited!)\n"
        "Suggested structural hints: {suggest_list}\n"
        "Based on the above information, please execute the following core steps step by step:\n"
        "1. Determine the overall structure: design a completely new skeleton structure with no more than 3 main sub-expressions (T1, T2, T3) (different from known skeletons), and decompose the target variable with reference to the known information, using Ti to represent them.\n"
        "2. Dimensional matching decomposition: decompose the overall structure into sub-expressions Ti\n"
        "   - Constraint: global_structure must contain sub-expressions Ti (T1, T2, T3), and must be different from the given known decomposition skeleton results; repetition is not allowed! global_structure must not contain more than 3 T's.\n"
        "3. Sub-expression description: provide a description for each Ti, and the description must clearly state the variables suggested for that sub-expression.\n"
        "   - Constraint: each T must contain at least one variable; pure constant terms are prohibited.\n"
    ),
    expected_output = (
        "Output only a JSON string; any extra information is prohibited. The description must clearly include sub-term suggestions.\n"
        "{\n"
        "  \"global_structure\": \"T1 + T2 / T3\",\n"
        "  \"sub_expressions\": {\n"
        "    \"T1\": \"Square term; it is recommended to use a combination of x1 and x2.\",\n"
        "    \"T2\": \"Product term; it is recommended to use an exponential form of x0.\",\n"
        "    \"T3\": \"Dimensionless term; it is recommended to use the ratio of two variables.\"\n"
        "  }\n"
        "}\n"
    ),
    guardrail=validate_json_output_structure_agent,
    max_retries=15, 
    agent = Structure_Agent,
)
Structure_Agent_Crew = Crew(
    agents = [Structure_Agent],
    tasks = [Structure_Agent_task],
)
Structure_term_Agent = Agent(
    role = "You are a mathematician",
    goal = "You can provide a relatively good expression decomposition plan",
    backstory = (
        "You are responsible for designing expression-compliant structures in symbolic regression tasks"
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
Structure_term_Agent_task = Task(
    description = (
        "Below is the input information for your task:\n"
        "Available variable pool: {variables}. Variables not in the variable pool are prohibited from appearing in T!\n"
        "Reference to the current best expressions: {fx_list} (sorted by loss from low to high)\n"
        "The result of the current best decomposition skeleton is: {list_f_term} (your decomposition result must be different from those in the list! Generating the same decomposition result is prohibited!)\n"
        "Suggested structural hints: {suggest_list}\n"
        "Based on the above information, please execute the following core steps step by step:\n"
        "1. Split variables: group {variables} into reasonable structures and split them into no more than 4 sub-expressions, represented by Ti. (The number of split sub-expressions must not exceed 4.) \n"
        "    - Constraint: each T_i must contain at least one variable; pure constant terms are prohibited.\n"
        "2. Subterm description: provide a textual description for each Ti, and the description must clearly state the variables contained in that sub-expression and a description of the alignment relationship.\n"
    ),
    expected_output = (
        "Output only a JSON string; any extra information is prohibited. The description must clearly include sub-term suggestions. (For format reference only; you generate the specific content yourself.)\n"
        "{\n"
        "    \"T1\": \"Subterm description 1 (including variables from the variable pool and textual information)\",\n"
        "    \"T2\": \"Subterm description 2 (including variables from the variable pool and textual information)\",\n"
        "    \"T3\": \"Subterm description 3 (including variables from the variable pool and textual information)\"\n"
        "}\n"
    ),
    guardrail=validate_json_output_structure_term_agent,
    max_retries=15, 
    agent = Structure_term_Agent,
)
Structure_term_Agent_Crew = Crew(
    agents = [Structure_term_Agent],
    tasks = [Structure_term_Agent_task],
)
Structure_T_Agent = Agent(
    role = "You are an expression generator",
    goal = "You can generate diverse expression structures based on variables and operators",
    backstory = (
        "You are responsible for generating diverse expressions"
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
Structure_T_Agent_task = Task(
    description = (
        "Below is the input information for your task:\n"
        "Available variable pool: {T}. Variables not in the variable pool are STRICTLY PROHIBITED!\n"
        "Available operators: {operators}\n"
        "List of known best structures: {list_f_term} (repeating any structure in it is prohibited!)\n"
        "Suggested structural hints for reference: {suggest_list}\n"
        "Below are the requirements for the expressions you generate; please follow them one by one:\n"
        "1. Each expression must contain at least 2 different operators.\n"
        "2. The use of composite structures such as nested combinations and fractional division terms is highly encouraged, such as: (T1*T2^2 +T1)\n"
        "Based on the above information, fully explore possible combinations of {T} and {operators} as much as possible, use your imagination, and do not be afraid to experiment! Generate 3 expressions (each expression must use at least 2 operators).\n"
    ),
    expected_output = (
        "Output only a JSON string; any extra information is prohibited. Generate 3 expressions (for format reference only; do not copy the content).\n"
        "{\n"
        "    \"1\": \"Expression 1\",\n"
        "    \"2\": \"Expression 2\",\n"
        "    \"3\": \"Expression 3\"\n"
        "}\n"
    ),
    guardrail=validate_json_output,
    max_retries=15, 
    agent = Structure_T_Agent,
)
Structure_T_Agent_task1 = Task(
    description = (
        "Below is the input information for your task:\n"
        "Available variable pool: {T}. Variables not in the variable pool are prohibited from appearing!\n"
        "Available operators: {operators}\n"
        "List of known best structures: {list_f_term} (your decomposition result must be different from those in the list! Generating the same decomposition result is prohibited!)\n"
        "Suggested structural hints: {suggest_list}\n"
        "Based on the above information, explore possible combinations of {T} and {operators} as much as possible, and use your imagination. Generate 3 expressions\n"),
    expected_output = (
        "Output only a JSON string; any extra information is prohibited. Generate 3 expressions (for format reference only; do not copy the content).\n"
        "{\n"
        "    \"1\": \"Expression 1\",\n"
        "    \"2\": \"Expression 2\",\n"
        "    \"3\": \"Expression 3\"\n"
        "}\n"
    ),
    guardrail=validate_json_output,
    max_retries=15, 
    agent = Structure_T_Agent,
)
Structure_T_Agent_Crew = Crew(
    agents = [Structure_T_Agent],
    tasks = [Structure_T_Agent_task],
)
Structure_T_Agent_Crew1 = Crew(
    agents = [Structure_T_Agent],
    tasks = [Structure_T_Agent_task1],
)
oper_and_var_Agent = Agent(
    role = "Symbolic Equation Structure Analysis and Expression Generation Expert",
    goal = "Select the most appropriate set of variables and operators from the provided pools based on the given sub-expression description to construct a mathematical expression that aligns with the expected physical units.",
    backstory = (
        "You are assisting in a symbolic regression task. Your expertise is needed to select the optimal combinations of variables and operators tailored to specific sub-expression semantics and physical constraints."
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
oper_and_var_Agent_task = Task(
    description = "Task Description:\n"
    "Known Input Information:\n"
    "- Sub-expression Description: {desc}\n"
    "- Available Variables Pool: {variables}\n"
    "- Available Operators Pool: {operators}\n\n"
    "Execute the following instructions based on semantic reasoning from the description to construct an expression conforming to the expected physical units:\n"
    "1. Select a set of variables (MUST select at least one).\n"
    "2. Select a set of operators (MUST select at least two distinct operators).\n"
    "3. If the sqrt operator is selected, you must simultaneously include the ^ operator. Wrap each operator in double quotes.",
    expected_output = "Expected Output Constraints:\n"
    "- Output ANY extra information is STRICTLY PROHIBITED!\n"
    "- Variables MUST strictly come from the {variables} pool. The character 'T' is PROHIBITED!\n"
    "- Output ONLY the following JSON format string:\n\n"
    "Expected Output Format:\n"
    "{\n"
    "  \"desc\": \"Original description\",\n"
    "  \"vars\": [\"Selected variable 1\", \"Selected variable 2\"],\n"
    "  \"ops\": [\"op1\", \"op2\"]\n"
    "}\n",
    guardrail=validate_json_output,
    max_retries=20, 
    agent = oper_and_var_Agent,
)
Oper_And_Var_Agent = Crew(
    agents = [oper_and_var_Agent],
    tasks = [oper_and_var_Agent_task],
)
reduce_dim = Agent(
    role = "You are a planner for generating mathematical equations.",
    goal = "Please provide the lists of variables for A and B after the distribution.",
    backstory = ("You can accurately distribute all the given variables {variables} to A and B based on the known information."),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
reduce_dim_task = Task(
    description = (
    "A and B are sub-functions of the equation. Please distribute all the variables {variables} to A and B. The variables can overlap between A and B (both in A and B). The known information is as follows: {suggest_list} Note that you must allocate all the variables in {variables} to either A or B. Variables can be repeated, but none can be omitted! Only the variables themselves are allowed; no additional characters or operators are permitted."),
    expected_output = "You don't need to output anything extra. Please output the lists of variables for A and B after the distribution of {variables} in the JSON format: \n"
    "{ \n"
    "   \"A\": [\"x0\", ...],\n"
    "   \"B\": [...],\n"
    "}\n",
    agent = reduce_dim,
    guardrail=validate_json_output,
    max_retries=3,  
    async_execution=True 
)
crew_reduce_dim = Crew(
    agents = [reduce_dim],
    tasks = [reduce_dim_task],
)
generate_f = Agent(
    role = "A Mathematical function generator. ",
    goal = "Generate equation skeletons without numerical coefficients in JSON format. ",
    backstory = (
        "Strictly generate the number of equations I need, no less!"
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
generate_f_task = Task(
    description = ("{prompts}"),
    expected_output = "Do not output anything else! Please list two equations in the JSON format: \n"
     "{ \n"
    "   \"1\": \"x0 + 1\", \n"
    "   \"2\": \"x1^2 / x0\"\n"
    "}\n"
    "Do not output anything else!",
    agent = generate_f,
    guardrail=validate_json_output,
    max_retries=20  
)
generate_f_task_one = Task(
    description = ("{prompts}"),
    expected_output = "Do not output anything else! Please list one equation in the JSON format: \n"
     "{ \n"
    "   \"1\": \"x0 + 1\"\n"
    "}\n"
    "Do not output anything else!",
    agent = generate_f,
    guardrail=validate_json_output,
    max_retries=20  
)
check_f_task = Task(
    description = ("You are a mathematical analysis and derivation expert. You need to analyze whether the currently generated equations conform to known variable patterns. Please analyze according to the following steps:\n"
               "1. Obtain the first generated equation\n"
               "2. Sequentially determine whether the equation conforms to the known patterns (whether it is missing variables, or missing operators from the patterns).\n"
               "3. If it conforms, return the equation directly; if it does not conform, modify it into a new equation that conforms to the patterns (you may delete or add variables and operators to conform to the patterns)\n"
               "Repeat the above steps to traverse all generated equations\n"
               ),
    expected_output = "Please list 3 new equations in the JSON format: \n"
    "{ \n"
    "    \"1\": \"x0\",\n"
    "   ......\n"
    "}\n",
    agent = generate_f,
    context=[generate_f_task],
    guardrail=validate_json_output,
    max_retries=3  
)
crew_generate_f = Crew(
    agents = [generate_f],
    tasks = [generate_f_task],
)
crew_generate_f_one = Crew(
    agents = [generate_f],
    tasks = [generate_f_task_one],
)
zongjie = Agent(
    role = "You are a mathematics and structural pattern summarization expert.",
    goal = "You need to summarize high-quality generation patterns from the currently available valid equations. You will combine variables and operators to provide a more reliable structural prior for subsequent equation generation.",
    backstory = (
        "You are proficient in symbolic regression and can propose high-value optimization suggestions based on existing equation structures."
    ),
    llm = qwen_model,
    allow_delegation = False,
    verbose = True
)
zongjie_task = Task(
    description = ("You will receive a set of currently successful equations: {fx_list}. \n"
                   "The operators are: {operators},\n"
                   "The variable list is: {variables} \n,"
        "I hope to obtain an equation that accurately fits y. Please, based on this information, propose two textual improvement suggestions that can enhance the quality of future equation generation (do not output special characters)"     
                   ),
    expected_output = "Please strictly output the following JSON format; outputting special LaTeX characters is prohibited! Outputting special characters like '\(...\)' is prohibited! Do not use brackets or special symbols:\n"
    "{ \n"
    "   \"1\": \"Suggestion 1\",\n"
    "   \"2\": \"Suggestion 2\"\n"
    "}\n",
    agent = zongjie,
    guardrail=validate_json_output,
    max_retries=12  
)
crew_zongjie = Crew(
    agents = [zongjie],
    tasks = [zongjie_task],
)
def get_physic_1_1(inputs):
    n_rounds = 5
    all_unions = []
    final_scene = None 
    for i in range(n_rounds):
        print(f"Running iteration {i+1}/{n_rounds}...")
        scene_union_result = Physics_Agent_Crew.kickoff(inputs=inputs)
        current_scene, current_union = load_json_structure_agent(scene_union_result.raw)
        all_unions.append(current_union)
        scene = current_scene
    union = []
    lengths = [len(u) for u in all_unions]
    if len(set(lengths)) > 1:
        print(f"Warning: Union lists have different lengths: {lengths}. Voting will truncate to shortest.")
    for unit_group in zip(*all_unions):
        count = Counter(unit_group)
        winner = count.most_common(1)[0][0]
        union.append(winner)
    print(union)
    union_list = convert_unit_list_to_base(union)
    print(union_list)
    solution = dimensional_analysis(union_list[0], union_list[1:])
    print([[x] for x in solution["list"]])
    return scene,union,process_solution_list(solution)
def get_physic(inputs):
    n_rounds = 3
    symbols = inputs.get("symbols", [])
    descs = inputs.get("symbol_descs", [])
    final_unit_list = []
    print(f"Starting Independent Unit Extraction for {len(symbols)} variables...")
    for idx, (sym, desc) in enumerate(zip(symbols, descs)):
        print(f"\nProcessing Variable {idx+1}/{len(symbols)}: {sym}")
        current_var_votes = []
        for i in range(n_rounds):
            task_inputs = {
                "target_symbol": sym,
                "target_desc": desc
            }
            try:
                result = Physics_Single_Var_Crew.kickoff(inputs=task_inputs)
                data = json.loads(extract_json_string(result.raw))
                unit_val = list(data.values())[0]
                unit_val = unit_val.strip()
                current_var_votes.append(unit_val)
                print(f"  Round {i+1}: {unit_val}")
            except Exception as e:
                print(f"  Round {i+1} Error: {e}")
                continue
        if not current_var_votes:
            print(f"Warning: No valid units found for {sym}. Defaulting to '-'.")
            final_unit_list.append("-")
        else:
            counter = Counter(current_var_votes)
            winner_unit = counter.most_common(1)[0][0]
            print(f"  => Winner for {sym}: {winner_unit} (Votes: {counter})")
            final_unit_list.append(winner_unit)
    print("\nAll Final Units:", final_unit_list)
    inputs['final_unit_list'] = final_unit_list
    scene_union_result = Physics_Agent_Crew.kickoff(inputs=inputs)
    current_scene, current_union = load_json_structure_agent(scene_union_result.raw)
    final_scene = current_scene
    union_list_base = convert_unit_list_to_base(final_unit_list)
    print("Base Units:", union_list_base)
    solution = dimensional_analysis(union_list_base[0], union_list_base[1:])
    print("Solution List:", [[x] for x in solution["list"]])
    return final_scene, final_unit_list, process_solution_list(solution)
def operators_count_none_physic():
    operator_counter = Counter()
    run_times = 10 
    for i in range(run_times):
        inputs['data'] = input_data_num(inputs['case_name'], inputs['dim'], inputs['data_num'])
        try:
            result = data_analyst_operators_crew.kickoff(inputs=inputs)
            raw_output = result.raw.strip()
            if raw_output.startswith("```"):
                raw_output = raw_output.strip("`").replace("python", "").replace("json", "").strip()
            current_ops_list = ast.literal_eval(raw_output)
            if isinstance(current_ops_list, list):
                operator_counter.update(current_ops_list)
            else:
                print(f"Warning: Run {i+1} did not return a list; skipped.")
        except Exception as e:
            print(f"Warning: Failed to parse run {i+1}: {e}; skipped.")
            continue
    sin_count = operator_counter.get('sin', 0)
    cos_count = operator_counter.get('cos', 0)
    if sin_count > 0 or cos_count > 0:
        max_trig_count = max(sin_count, cos_count)
        operator_counter['sin'] = max_trig_count
        operator_counter['cos'] = max_trig_count
    print("Final voting details:", operator_counter) 
    top_2_voted = [op for op, count in operator_counter.most_common(2)]
    print(top_2_voted)
    return top_2_voted
def filter_expr_by_vars(expr_list, var_list):
    return [
        expr for expr in expr_list
        if any(var in expr for var in var_list)
    ]
class AtomicCapabilities:
    """
    Atomic capability library: encapsulates each Agent call as a standard step
    Input ctx: a dictionary containing 'inputs', 'history', and 'temp_data'
    Return ctx: the updated dictionary
    """
    @staticmethod
    def step_summarize(ctx): 
        """A: Invoke the summarization Agent (zongjie)"""
        print("  [Step] Running Summarizer...")
        agent_inputs = ctx['inputs'].copy()
        result = crew_zongjie.kickoff(inputs=agent_inputs)
        raw_json = extract_json_string(result.raw)
        suggestions = list(json.loads(raw_json).values())
        ctx['inputs']['suggest_list'] = suggestions
        ctx['temp_data']['latest_suggestions'] = suggestions
        return ctx
    @staticmethod
    def step_decompose_structure(ctx):
        """B: Invoke the structure decomposition Agent (Structure_term_Agent)"""
        print("  [Step] Decomposing Structure (Terms)...")
        agent_inputs = ctx['inputs'].copy()
        result = Structure_term_Agent_Crew.kickoff(inputs=agent_inputs)
        t_keys, t_descs = load_json_structure_term_agent(result.raw)        
        print(f"  -> Keys: {t_keys}")
        print(f"  -> Descs: {t_descs}")
        if isinstance(t_keys, list) and isinstance(t_descs, list):
            term_map_dict = dict(zip(t_keys, t_descs))
        else:
            print("  [Warning] Return format mismatch, expected two lists.")
            term_map_dict = {}
            t_keys = []
        ctx['temp_data']['term_map'] = term_map_dict 
        ctx['temp_data']['T_list'] = t_keys
        return ctx
    @staticmethod
    def step_analyze_var_ops(ctx):
        """C: Invoke the operator and variable analysis Agent (oper_and_var_Agent)"""
        print("  [Step] Analyzing Vars & Ops...")
        inputs_array = []
        target_keys = [] 
        bool_fenjie = False
        term_map = ctx['temp_data'].get('term_map', {})
        if term_map:
            print("    -> Mode: Per-Term Analysis (T1, T2...)")
            for t_name, desc in term_map.items():
                inputs_array.append({
                    'desc': desc,
                    'variables': ctx['inputs']['variables'],
                    'operators': ctx['inputs']['operators']
                })
                target_keys.append(t_name)
        else:
            print("    -> Mode: Global Analysis (Based on Suggestions)")
            bool_fenjie = True
            suggestions = ctx['inputs'].get('suggest_list1', [])
            if not suggestions:
                suggestions = ctx['inputs'].get('suggest_list', "No specific suggestions provided.")
            desc_str = str(suggestions) if isinstance(suggestions, list) else str(suggestions)
            target_key = "T1" 
            inputs_array.append({
                'desc': desc_str,
                'variables': ctx['inputs']['variables'],
                'operators': ctx['inputs']['operators']
            })
            target_keys.append(target_key)
        if not inputs_array:
            print("  [Warning] No inputs for analysis.")
            return ctx
        async_results = asyncio.run(Oper_And_Var_Agent.kickoff_for_each_async(inputs=inputs_array))
        term_configs = {}
        for idx, result in enumerate(async_results):
            t_name = target_keys[idx]
            desc, var, ops = load_json_oper_and_var_agent(result.raw)
            if bool_fenjie:
                var = ctx['inputs']['variables']
            term_configs[t_name] = {'vars': var, 'ops': ops, 'desc': desc}
        ctx['temp_data']['term_configs'] = term_configs
        if not term_map and "T1" in term_configs:
             ctx['temp_data']['term_map'] = {"T1": term_configs["T1"]["desc"]}
        return ctx
    @staticmethod
    def step_generate_sub_equations(ctx):
        """Invoke the equation generation Agent (generate_f) to generate sub-expressions"""
        term_map = ctx['temp_data'].get('term_map', {})         
        term_configs = ctx['temp_data'].get('term_configs', {}) 
        inputs_data = ctx['inputs']                             
        inputs_array = []
        target_t_names = [] 
        print(f"  Debug info - Term Map Keys: {list(term_map.keys())}")
        print(f"  Debug info - Config Keys: {list(term_configs.keys())}")
        print(len(list(term_map.keys())))
        if term_map and len(list(term_map.keys())) > 1: 
            print("    -> Mode: Component Generation (Based on Terms)")
            for t_name, t_desc in term_map.items():
                if t_name in term_configs:
                    current_vars = term_configs[t_name]['vars']
                    current_ops = term_configs[t_name]['ops']
                    current_desc = term_configs[t_name]['desc'] 
                else:
                    current_vars = inputs_data['variables']
                    current_ops = inputs_data['operators']
                    current_desc = t_desc 
                prompt_content = prompts6.format(
                    desc=current_desc,
                    variables=current_vars,
                    operators=current_ops,
                    reference = filter_expr_by_vars(ctx['inputs']["best_components_ref"],current_vars),
                    len=50 
                )
                inputs_array.append({"prompts": prompt_content})
                target_t_names.append(t_name)
        else:
            print("    -> Mode: Direct Generation (Whole Equation)")
            parallel_num = inputs_data.get('parallel_num', 5)
            for _ in range(parallel_num):
                if term_configs:
                    inputs_array.append({
                        "prompts": prompts2.format(
                            fx_list=inputs_data.get("fx_list", []),
                            suggest_list=inputs_data.get("suggest_list1", []),
                            variables=term_configs['T1']["vars"],
                            operators=term_configs['T1']['ops']
                        )
                    })
                else:
                    inputs_array.append({
                        "prompts": prompts2.format(
                            fx_list=inputs_data.get("fx_list", []),
                            suggest_list=inputs_data.get("suggest_list1", []),
                            variables=inputs_data["variables"],
                            operators=inputs_data["operators"]
                        )
                    })
        if not inputs_array:
            print("  [Warning] No inputs generated for equation generation.")
            return ctx
        async_results = asyncio.run(crew_generate_f.kickoff_for_each_async(inputs=inputs_array))
        if target_t_names:
            sub_equations_map = {}
            for idx, result in enumerate(async_results):
                t_name = target_t_names[idx] 
                fx_list = []
                fx_list = load_json_f_to_list(result.raw, fx_list, inputs_data['symbols'])
                sub_equations_map[t_name] = fx_list
            ctx['temp_data']['sub_equations_map'] = sub_equations_map
            print(f"    -> Generated sub-expressions: {sub_equations_map}")
        else:
            all_direct_equations = []
            for result in enumerate(async_results):
                res_obj = result[1] 
                fx_list = []
                fx_list = load_json_f_to_list(res_obj.raw, fx_list, inputs_data['symbols'])
                all_direct_equations.extend(fx_list)
            sub_equations_map = {"T1": all_direct_equations}
            ctx['temp_data']['sub_equations_map'] = sub_equations_map
            print(f"    -> Generated {len(all_direct_equations)} direct equations mapped to 'T1'.")
        return ctx
    @staticmethod
    def step_generate_skeleton(ctx):
        """E Invoke the skeleton generation Agent (Structure_T_Agent)"""
        if 'T_list' not in ctx['temp_data']:
            count = len(inputs['default_eqs_groups'])
            ctx['temp_data']['T_list'] = [f'T{i+1}' for i in range(count)]
        print("  [Step] Generating Equation Skeletons...")
        agent_inputs = ctx['inputs'].copy()
        agent_inputs['T'] = ctx['temp_data']['T_list']
        result = Structure_T_Agent_Crew.kickoff(inputs=agent_inputs)
        json_str = extract_json_string(result.raw)
        skeletons = list(json.loads(json_str).values()) 
        ctx['temp_data']['skeletons'] = skeletons
        return ctx
    @staticmethod
    def step_optimize_operators(ctx):
        
        print("  [Step] Analyzing Data to optimize Operators...")
        agent_inputs = ctx['inputs'].copy()
        try:
            result = data_analyst_operators_crew.kickoff(inputs=agent_inputs)
            basic_ops = ['+', '-', '*', '/', '^']
            raw_output = extract_json_string(result.raw) 
            try:
                result_list = ast.literal_eval(raw_output)
            except:
                result_list = ast.literal_eval(result.raw)
            if isinstance(result_list, list):
                final_ops = list(dict.fromkeys(basic_ops + result_list))
                print(f"  -> Operator Optimization Result: {final_ops}")
                ctx['inputs']['operators'] = final_ops
            else:
                print(f"  [Warning] Operator analyst returned {type(result_list)} instead of list. Keeping defaults.")
        except Exception as e:
            print(f"  [Warning] Operator Analysis failed: {e}. Keeping default operators.")
        return ctx
CAPABILITY_REGISTRY = {
    "A": AtomicCapabilities.step_summarize,
    "B": AtomicCapabilities.step_decompose_structure,
    "C": AtomicCapabilities.step_analyze_var_ops,
    "D": AtomicCapabilities.step_generate_sub_equations,
    "E": AtomicCapabilities.step_generate_skeleton,
    "F": AtomicCapabilities.step_optimize_operators 
}
def get_best_components_from_archive(GLOBAL_ARCHIVE):
    """Extract the component list of the best individual from the global archive"""
    try:
        if hasattr(GLOBAL_ARCHIVE["pop"], 'return_best') and len(GLOBAL_ARCHIVE["pop"].pop) > 0:
            best_ind = GLOBAL_ARCHIVE["pop"].return_best()[0]
            components = best_ind[2]
            if isinstance(components, dict):
                return list(components.values())
            return components
    except Exception as e:
        print(f"[Info] Cannot retrieve best components: {e}")
    return []
def Strategy1(inputs, local_store, STORED_OBSERVATIONS,step_i,strategy_name):  
    list_x = []
    for i in range(10):
        try:
            result = Physics_generate_Agent_Crew.kickoff(inputs=inputs) 
            raw = result.raw.strip()
            first_line = raw.splitlines()[0].strip()
            x = to_physic(first_line, inputs["symbols"])
            list_x.append(x)
        except Exception as e:
            print(f"[Warning] Generation attempt {i+1} failed; skipped. Error: {e}")
            continue
    list_x.extend(mutate(list_x,inputs["variables"],inputs["operators"],3))
    generated_init_pop = list_to_obs(list_x, ["T1"], list_x)
    change_Bool, local_store = multi_constant_parallel(generated_init_pop, local_store, STORED_OBSERVATIONS, save_path, step_i,strategy_name)
    return change_Bool, local_store
def Strategy6(inputs, local_store, STORED_OBSERVATIONS, step_i,strategy_name): 
    inputs_array = [
        {
            "prompts": prompts2.format(
                fx_list=list_f,
                suggest_list=inputs["suggest_list1"],
                variables=inputs["variables"],
                operators=inputs["operators"]
            )
        }
        for _ in range(5)
    ]
    async_results = asyncio.run(crew_generate_f.kickoff_for_each_async(inputs=inputs_array))
    generated_init_pop = []
    for result1 in async_results:
        a = extract_json_string(result1.raw)
        list_a = list(json.loads(a).values())
        list_muate = mutate(list_a,inputs["variables"],inputs["operators"],2)
        list_a.extend(list_muate)
        generated_init_pop.extend(list_to_obs(list_a,["T1"],list_a))
    print("done!")
    change_Bool, local_store = multi_constant_parallel(generated_init_pop, local_store, STORED_OBSERVATIONS, save_path, step_i,strategy_name)
    return change_Bool, local_store
def Dynamic_Strategy_Executor(inputs, local_store, STORED_OBSERVATIONS, GLOBAL_ARCHIVE, step_i,strategy_name):

    PLAN_KEY = "active_strategy_plan"
    need_new_strategy = (step_i % 15 == 0) or (PLAN_KEY not in inputs)
    if need_new_strategy:
        print(f"\n[Meta-Architect] Designing new strategy (Trigger: Step {step_i})...")
        meta_inputs = {
            "stuck_num": inputs.get("stuck_num", 0),
            "best_equation": inputs.get("fx_list", ["None"]) if inputs.get("fx_list") else "None",
            "variables": inputs["variables"]
        }
        architect_result = Meta_Architect_Crew.kickoff(inputs=meta_inputs)
        try:
            new_plan = json.loads(extract_json_string(architect_result.raw))
            inputs[PLAN_KEY] = new_plan
            print(f"  -> Reasoning: {new_plan.get('reasoning', 'None')}")
        except Exception as e:
            print(f"  [Error] Failed to parse strategy JSON: {e}")
            if PLAN_KEY not in inputs:
                return False, local_store
            print("  -> Fallback: Reusing previous strategy despite error.")
    else:
        print(f"\n[Meta-Architect] Reusing active strategy (Step {step_i})...")
    strategy_plan = inputs.get(PLAN_KEY)
    if not strategy_plan:
        print("  [Error] No strategy plan available.")
        return False, local_store
    best_comps = []
    if hasattr(GLOBAL_ARCHIVE["pop"], 'return_best') and len(GLOBAL_ARCHIVE["pop"].pop) > 0:
        try:
            best_ind = GLOBAL_ARCHIVE["pop"].return_best()[0]
            raw_comps = best_ind[2] 
            if isinstance(raw_comps, list):
                best_comps = raw_comps
            elif isinstance(raw_comps, dict):
                sorted_keys = sorted(raw_comps.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
                best_comps = [raw_comps[k] for k in sorted_keys]
        except:
            pass
    inputs['best_components_ref'] = best_comps
    print(inputs['best_components_ref'])
    ctx = {
        "inputs": inputs.copy(), 
        "temp_data": {},         
        "history": []
    }
    steps = strategy_plan.get("steps", [])
    if need_new_strategy:
        with open(save_path, "a", encoding="utf-8") as fw:
            fw.write(f"Search strategy : {steps}")
            fw.write("\n")
    success = True
    for step in steps:
        tool_name = step["tool"]
        if tool_name in CAPABILITY_REGISTRY:
            try:
                func = CAPABILITY_REGISTRY[tool_name]
                ctx = func(ctx)
            except Exception as e:
                print(f"  [Error] Step {tool_name} failed: {e}")
                success = False
                break
        else:
            print(f"  [Warning] Unknown tool: {tool_name}")
    if not success:
        return False, local_store
    generated_init_pop = []
    skeletons = ctx['temp_data'].get('skeletons', [])
    sub_eqs_map = ctx['temp_data'].get('sub_equations_map', {})
    t_list = ctx['temp_data'].get('T_list', []) 
    all_eqs_groups = inputs['default_eqs_groups']
    if not sub_eqs_map:
        print("  [Repair] sub_eqs_map is empty (Agent D skipped). Repairing...")
        target_ts = t_list if t_list else ["T1"]
        source_components = []
        if hasattr(GLOBAL_ARCHIVE['pop'], 'return_best') and len(GLOBAL_ARCHIVE['pop'].pop) > 0:
            try:
                best_ind = GLOBAL_ARCHIVE['pop'].return_best()[0]
                raw_comps = best_ind[2]
                if isinstance(raw_comps, dict):
                    source_components = list(raw_comps.values())
                elif isinstance(raw_comps, list):
                    source_components = raw_comps
                print("    -> Sourcing components from Global Best.")
            except:
                source_components = [] 
        if not source_components:
            print("    -> Sourcing components from default all_eqs_groups.")
            source_components = [item[0] if isinstance(item, list) else item for item in all_eqs_groups]
        new_map = {}
        for i, t_key in enumerate(target_ts):
            if i < len(source_components):
                comp_val = source_components[i]
                new_map[t_key] = [str(comp_val)] if not isinstance(comp_val, list) else comp_val
            else:
                new_map[t_key] = ["1"]
        sub_eqs_map = new_map
        ctx['temp_data']['sub_equations_map'] = sub_eqs_map
    if not skeletons:
        print("  [Repair] skeletons is empty (Agent E skipped). Repairing...")
        is_multi_term = len(sub_eqs_map) > 1
        if is_multi_term:
            print(f"    -> Detected Multi-Term ({list(sub_eqs_map.keys())}). Fetching/Constructing structure...")
            fetched_skeleton = None
            if hasattr(GLOBAL_ARCHIVE['pop'], 'return_best') and len(GLOBAL_ARCHIVE['pop'].pop) > 0:
                try:
                    best_ind = GLOBAL_ARCHIVE['pop'].return_best()[0]
                    fetched_skeleton = str(sp.sympify(best_ind[1]))
                    print("    ->  Sourced skeleton from Global Best.")
                except:
                    fetched_skeleton = None
            if not fetched_skeleton:
                sorted_keys = sorted(sub_eqs_map.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
                fetched_skeleton = "+".join(sorted_keys)
                print(f"    -> Constructed default skeleton: {fetched_skeleton}")
            try:
                t_vars = list(sub_eqs_map.keys())
                skeletons = mutate([fetched_skeleton], t_vars, inputs["operators"], 3)
                if fetched_skeleton not in skeletons:
                    skeletons.insert(0, fetched_skeleton)
            except Exception as e:
                print(f"    [Warning] Mutate failed ({e}), using raw skeleton.")
                skeletons = [fetched_skeleton]
        else:
            print("    -> Detected Single-Term. Using default skeleton ['T1'].")
            skeletons = ["T1"]
        ctx['temp_data']['skeletons'] = skeletons
    if skeletons and sub_eqs_map:
        required_indices = set()
        for skel in skeletons:
            found = re.findall(r'T(\d+)', skel)
            required_indices.update(int(x) for x in found)
        if required_indices:
            max_index = max(required_indices)
            for i in range(1, max_index + 1):
                key = f"T{i}"
                if key not in sub_eqs_map:
                    sub_eqs_map[key] = ["1"]
    best_refs = ctx['inputs'].get('best_components_ref', [])
    if best_refs:
        for i, ref_comp in enumerate(best_refs):
            t_key = f"T{i+1}"
            ref_str =  replace_C_to_one(replace_numbers(str(ref_comp)))
            if t_key in sub_eqs_map:
                if ref_str not in sub_eqs_map[t_key]:
                    sub_eqs_map[t_key].insert(0, ref_str)
            else:
                sub_eqs_map[t_key] = [ref_str]
                print(f"    -> Added new key {t_key} to accommodate historical best.")
        ctx['temp_data']['sub_equations_map'] = sub_eqs_map
    if all_eqs_groups:
        print(f"  [Injection] Merging default equation groups...")
        for i, group in enumerate(all_eqs_groups):
            t_key = f"T{i+1}"
            group_list = group if isinstance(group, list) else [group]
            group_list_str = [str(sp.sympify(x)).replace("**", "^") for x in group_list]
            if t_key in sub_eqs_map:
                current_list = sub_eqs_map[t_key]
                for expr in group_list_str:
                    if expr not in current_list:
                        current_list.append(expr) 
            else:
                sub_eqs_map[t_key] = group_list_str
                print(f"    -> Created new key {t_key} for default group.")
        ctx['temp_data']['sub_equations_map'] = sub_eqs_map

    if skeletons and sub_eqs_map:
        print("  [Assembly] Combining fixed Skeletons and Sub-expressions...")
        try:
            sorted_keys = sorted(sub_eqs_map.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
            all_eqs_groups = [sub_eqs_map[k] for k in sorted_keys]
            combined, term_list, all_components_list = expand_formula_list(skeletons, all_eqs_groups)
            generated_init_pop.extend(list_to_obs(combined, term_list, all_components_list))
        except Exception as e:
            import traceback
            print(f"  [Assembly Error] Critical failure during expand: {e}")
            traceback.print_exc()
            return False, local_store
    else:
        print("  [Error] Assembly failed: Data could not be repaired.")
        return False, local_store
    if len(generated_init_pop) > 0:
        print(f"  -> Generated {len(generated_init_pop)} new equations.")
        change_Bool, local_store = multi_constant_parallel(generated_init_pop, local_store, STORED_OBSERVATIONS, save_path, step_i,strategy_name)
        return change_Bool, local_store
    else:
        return False, local_store
def worker_execution_wrapper(strategy_name, inputs, step_i, initial_pop_data = STORED_OBSERVATIONS, GLOBAL_ARCHIVE = STORED_OBSERVATIONS, extra_args=None, pyhsics = False):

    try:
        local_store = Observations()
        success = False
        if strategy_name == "D":
        #     success,selected_pop = Strategy3(inputs, local_store, initial_pop_data, step_i,strategy_name)
        # elif strategy_name == "B":
        #     all_eqs_groups = extra_args
        #     success,selected_pop = Strategy5(inputs, local_store, initial_pop_data, step_i, all_eqs_groups,strategy_name)
        # elif strategy_name == "C":
        #     success,selected_pop = Strategy4(inputs, local_store, initial_pop_data, step_i,strategy_name)
        # elif strategy_name == "D":
            if pyhsics == True:
                success,selected_pop = Strategy1(inputs, local_store, initial_pop_data, step_i,strategy_name)
            else:
                success,selected_pop = Strategy6(inputs, local_store, initial_pop_data, step_i,strategy_name)
        elif strategy_name.startswith("DYNAMIC"):
            success, selected_pop = Dynamic_Strategy_Executor(inputs, local_store, initial_pop_data, GLOBAL_ARCHIVE, step_i,strategy_name)
        return strategy_name, success, selected_pop
    except Exception as e:
        import traceback
        return strategy_name, False, f"Error: {str(e)}\n{traceback.format_exc()}"
class TimeoutException(Exception):
    pass
def _timeout_handler(signum, frame):
    raise TimeoutException()
def print_result(GLOBAL_ARCHIVE,STRATEGY_CONTEXTS,target_strategies,results_summary,save_path,step_i):
    global_best_list = GLOBAL_ARCHIVE["pop"].return_best()
    if global_best_list:
        global_best = global_best_list[0] 
        def safe_tree_to_str(tree_obj, timeout=6):
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
            try:
                return str(sp.simplify(gp.TreeToDeadStr(tree_obj, converter)))
            except TimeoutException:
                return str(gp.TreeToDeadStr(tree_obj, converter))
            except Exception:
                return str(gp.TreeToDeadStr(tree_obj, converter))
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        any_strategy_changed = any(results_summary.values())
        iteration_best_ind = None
        iteration_best_fit = float('inf')
        print("-" * 100)
        print(f"Iteration {step_i} Strategy Breakdown:")
        for strat in target_strategies:
            is_success = results_summary.get(strat, False)
            status_str = "success" if is_success else "stuck"
            ctx = STRATEGY_CONTEXTS[strat]
            s_pop_list = ctx["pop"]
            if len(s_pop_list) > 0:
                s_best = min(s_pop_list, key=lambda x: x[3])
                s_f_str = safe_tree_to_str(s_best[0][0])
                print(f"  Strategy {strat} [{status_str:<7}] | Loss: {s_best[3]:.8f} | f = {s_f_str}")
                if s_best[3] < iteration_best_fit:
                    iteration_best_fit = s_best[3]
                    iteration_best_ind = s_best
            else:
                print(f"  Strategy {strat} [{status_str:<7}]: [Empty Population]")
        print("-" * 100)
        curr_best = iteration_best_ind if iteration_best_ind else global_best
        cb_f_str = safe_tree_to_str(curr_best[0][0])
        cb_term = curr_best[1]
        cb_comp = curr_best[2]
        cb_fit  = curr_best[3]
        cb_strategy_name = curr_best[4]
        gb_f_str = safe_tree_to_str(global_best[0][0])
        gb_term = global_best[1]
        gb_comp = global_best[2]
        gb_fit  = global_best[3]
        gb_strategy_name = global_best[4]
        if any_strategy_changed and cb_fit == gb_fit:
            print()
            print("Change!")
            print()
            GLOBAL_ARCHIVE["inputs"]["stuck_num"] == 0 
            GLOBAL_ARCHIVE["inputs"]["stuck_num1"] == 0 
            with open(save_path, "a", encoding="utf-8") as fw:
                fw.write(f"Change! at step {step_i}!\n")
        GLOBAL_ARCHIVE["inputs"]["stuck_num"] += 1 
        GLOBAL_ARCHIVE["inputs"]["stuck_num1"] += 1 
        print(inputs['case_name'])
        log_str = 'Step %s: curr best is %s | %s | %s | %s | %s | %s, \n global best is %s | %s | %s | %s | %s | %s' % (
            step_i,
            cb_f_str, cb_term, cb_comp, cb_fit, cb_strategy_name,curr_best[5],
            gb_f_str, gb_term, gb_comp, gb_fit, gb_strategy_name,global_best[5]
        )
        print(log_str)
        report = GLOBAL_ARCHIVE["pop"].get_population_status()
        print(report)
        with open(save_path, "a", encoding="utf-8") as fw:
            fw.write(log_str)
            fw.write("\n")
            fw.write(str(report["diversity_score"]))
            fw.write("\n")
        if gb_fit < 1e-10:
            return True
        return False
def change_meta(GLOBAL_ARCHIVE, STRATEGY_CONTEXTS):
    global_source = GLOBAL_ARCHIVE["pop"].pop 
    if len(global_source) >= 2:
        sorted_global = sorted(global_source, key=lambda x: x[3])
        global_elites = sorted_global[:2]
        best_elite = sorted_global[0]
        best_strategy_name = best_elite[4] 
        for strat, context in STRATEGY_CONTEXTS.items():
            strat_inputs = context["inputs"]
            current_stuck = strat_inputs.get("stuck_num", 0)
            current_stuck1 = strat_inputs.get("stuck_num1", 0)
            if current_stuck1 > 20:
                print(f"  >>> [Rescue Event] Strategy {strat} has stagnated for {current_stuck1} generations. Injecting...")
                with open(save_path, "a", encoding="utf-8") as fw:
                    fw.write(f"Strategy {strat} has stagnated for {current_stuck} generations. Injecting global Top 2...")
                    fw.write("\n")
                if best_strategy_name in STRATEGY_CONTEXTS:
                    print(f"      ->  from [{best_strategy_name}] to [{strat}]")
                    source_inputs = STRATEGY_CONTEXTS[best_strategy_name]["inputs"]
                    if "operators" in source_inputs:
                        strat_inputs["operators"] = validate_operators(copy.deepcopy(source_inputs["operators"]))
                        print(f"        -> update: {strat_inputs['operators']}")
                        with open(save_path, "a", encoding="utf-8") as fw:
                            fw.write(f"update: {strat_inputs['operators']}")
                            fw.write("\n")
                    source_suggestions = source_inputs.get("suggest_list1") or source_inputs.get("suggestion1")
                    if source_suggestions:
                        target_key = "suggest_list1" if "suggest_list1" in strat_inputs else "suggest_list"
                        strat_inputs[target_key] = copy.deepcopy(source_suggestions)
                        with open(save_path, "a", encoding="utf-8") as fw:
                            fw.write(f"suggestion update : {source_suggestions}")
                            fw.write("\n")
                        print(f"        -> suggestion update (num: {len(source_suggestions)})")
                else:
                    print(f"      [Warning]  {best_strategy_name}")
                migrants = copy.deepcopy(global_elites)
                migrants = context["pop"].get_pop_str(migrants, simplify_flag=False)
                equations = [p[0][0] for p in migrants]
                equations_G = [p[0][1] for p in migrants]
                skeletons = [p[1] for p in migrants]
                components = [p[2] for p in migrants]
                losses = [p[3] for p in migrants]
                strat_names = [p[4] for p in migrants] 
                if random.random() < 0.8:
                    context["pop"].append_without_print(
                        equations,
                        equations_G,
                        skeletons,
                        components,
                        losses,
                        strat_names
                    )
                strat_inputs["stuck_num1"] = 0
                print(f"      -> Injection complete. Resetting stagnation counter.")
list_f_term = []
selected_strategy_list = []
if __name__ == '__main__':
    pyhsics = inputs["pyhsics"]
    if pyhsics == False:
        result = operators_count_none_physic()
        basic_ops = ['+', '*', '/', '^']
        inputs['operators'] = list(dict.fromkeys(basic_ops + result))
        inputs['operators'] = basic_ops
    if pyhsics == True:
        scene, union, all_eqs_groups = get_physic(inputs)
        inputs['scene'] = scene
        inputs['union_y'] = union[0]
        inputs['union_x'] = union[1:]
        inputs['suggest_list1'] = [] 
        inputs['default_eqs_groups'] = all_eqs_groups
        list_x = []
        for _ in range(20):
            result = Physics_generate_Agent_Crew.kickoff(inputs=inputs)
            x = to_physic(result.raw,inputs["symbols"])
            list_x.append(x)
        inputs["operators"] = validate_operators(extract_operators(list_x),base_ops = ['+', '-', '*', '/', '^'])
        inputs["pysr_operators"] = inputs["operators"]
        print(inputs['default_eqs_groups'])
        print(inputs["operators"])
    else:
        inputs['default_eqs_groups'] = [['x0']]
        suggest_list = []
        all_eqs_groups = inputs['default_eqs_groups'] 
        inputs["suggest_list"] = suggest_list
        inputs["suggest_list1"] = suggest_list
    base_inputs = inputs.copy()
    GLOBAL_ARCHIVE = {"pop": STORED_OBSERVATIONS, "inputs": copy.deepcopy(base_inputs)}
    target_strategies = ["D","DYNAMIC1","DYNAMIC2","DYNAMIC3"]
    STRATEGY_CONTEXTS = {}
    for strat in target_strategies:
        STRATEGY_CONTEXTS[strat] = {
            "pop": Observations(), 
            "inputs": copy.deepcopy(base_inputs)
        }
        STRATEGY_CONTEXTS[strat]["inputs"]["stuck_num"] = 0
        STRATEGY_CONTEXTS[strat]["inputs"]["stuck_num1"] = 0
        STRATEGY_CONTEXTS[strat]["inputs"]["strategy_name"] = strat
    for step_i in range(300):
        if step_i == 0:
            list_f = pysr_result
        if step_i > 20:
            change_meta(GLOBAL_ARCHIVE,STRATEGY_CONTEXTS)
        STRATEGY_CONTEXTS_1 = {}
        for strat in target_strategies:
            STRATEGY_CONTEXTS_1[strat] = {
                "pop": [],  
            }
        print(f"\n=== Step {step_i}: Independent Evolution Loop ===")
        pop_list = []
        iteration_new_elites = [] 
        change_Bool = False       
        results_summary = {}
        for strat in target_strategies:
            ctx = STRATEGY_CONTEXTS[strat]
            curr_pop = ctx["pop"]        
            curr_inputs = ctx["inputs"]  
            curr_inputs["step_i"] = step_i
            curr_inputs["diversity_score"] = curr_pop.get_fitness_diversity()
            print(f"\n --- Running Strategy {strat} (Stuck: {curr_inputs['stuck_num']}) ---")
            extra = inputs['default_eqs_groups'] if strat == "B" else None
            strat_name, success, result_data = worker_execution_wrapper(
                strat, curr_inputs, step_i, curr_pop, GLOBAL_ARCHIVE, extra, pyhsics
            )
            if success:
                curr_inputs['stuck_num'] = 0
                curr_inputs['stuck_num1'] = 0
            else:
                curr_inputs['stuck_num'] += 1
                curr_inputs['stuck_num1'] += 1
            results_summary[strat_name] = success
            if isinstance(result_data, str) and "Error" in result_data:
                print(f"  [Error] {strat} failed: {result_data}")
                curr_inputs['stuck_num'] += 1
                results_summary[strat_name] = False
            else:
                returned_store = result_data 
                if hasattr(returned_store, 'pop') and len(returned_store.pop) > 0:
                    STRATEGY_CONTEXTS_1[strat]["pop"] = returned_store.pop
                    print(f"Fast merge: received {len(returned_store.pop)} optimized individuals...")
                    curr_pop.pop.extend(returned_store.pop)       
                    curr_pop.labels.extend(returned_store.labels) 
                    iteration_new_elites.extend(curr_pop.return_best())
                    print("Current population loaded!")
                else:
                    curr_inputs["stuck_num"] += 1
            if len(curr_pop.pop) > 0:
                raw_sorted_pop = sorted(curr_pop.pop, key=lambda x: x[3])[:15]
                sorted_pop = curr_pop.get_pop_str(raw_sorted_pop, simplify_flag=False)
                print("Retrieving the best individual from the current population!")
                list_f = [replace_C_to_one(replace_numbers(curr_pop.str_from(p_str[0][0], simplify_flag=False))) for p_str in sorted_pop]
                print("Sampling complete!")
                list_f_term = list(dict.fromkeys([p_str[1] for p_str in sorted_pop]))[:4]
                list_f = get_unique_random_samples(list_f, 4)
                curr_inputs["fx_list"] = list_f
                curr_inputs["list_f_term"] = list_f_term
            else:
                curr_inputs["fx_list"] = []
                curr_inputs["list_f_term"] = []
            if success or step_i % 20 == 0:
                if curr_inputs["fx_list"]:
                    print(f"  [LLM Analysis] Strategy {strat} requesting independent analysis...")
                    try:
                        result2 = crew_zongjie.kickoff(inputs=curr_inputs)
                        raw_json = extract_json_string(result2.raw)
                        new_suggest = list(json.loads(raw_json).values())
                        curr_inputs["suggest_list1"] = new_suggest
                        curr_inputs["suggest_list"] = new_suggest
                        print(f"  [LLM Analysis] Strategy {strat} received {len(new_suggest)} new suggestions.")
                    except Exception as e:
                        print(f"  [Warning] Strategy {strat} LLM failed: {e}")
                else:
                    print(f"  [LLM Analysis] Strategy {strat} has no equations yet. Skipping.")
        f_global = GLOBAL_ARCHIVE["pop"].get_pop_str(iteration_new_elites, simplify_flag=False)
        print("load over!")
        GLOBAL_ARCHIVE["pop"].append_without_print(
                        [p[0][0] for p in f_global],
                        [p[0][1] for p in f_global],
                        [p[1] for p in f_global],
                        [p[2] for p in f_global],
                        [p[3] for p in f_global],
                        [p[4] for p in f_global]
            )
        print(results_summary)
        break_bool = print_result(GLOBAL_ARCHIVE,STRATEGY_CONTEXTS_1,target_strategies,results_summary,save_path,step_i)
        if break_bool:
            print("Break!")
            break
