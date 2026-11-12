from ast import Dict
import ast
import csv
from datetime import datetime
from itertools import combinations
import itertools
from math import tan
import multiprocessing
import os
import pickle
import random
import re
import signal
import numpy as np
from typing import Type, Tuple, List, Dict 
import pandas as pd
from sklearn.model_selection import train_test_split
import sympy as sp
import threading
from tqdm import tqdm
from crewai.tools import BaseTool, tool
from typing import Type
from pydantic import BaseModel, Field
import torch
import json
import itertools
from fractions import Fraction
from math import lcm
from src.ai_agent_local_llm.tools.aifeynman.S_run_aifeynman import run_aifeynman
from src.ai_agent_local_llm.tools.screen_pretrain_knowledge_eq import load_data_SR_all_following_cars, load_data_SR_following_car, make_pop_from_file, evolve, multi_processing_eval_pop, \
    multi_processing_opt_constant, multi_processing_opt_constant_SR_strategy_12_31, opt_constant_gpu, opt_constant_one_core, opt_constant_one_core_SR, pset_f, pset_g, converter, tournamentSelection, \
    opt_constant_multi_core, multi_processing_opt_constant_SR,load_fish_data_pydaddy,load_cell_data_pydaddy,load_fish_data_ode,load_data_SR_from_csv
import src.ai_agent_local_llm.tools.search_2nd_phase.gp_tools as gp
from sympy import I, symbols, sympify, Add, Mul, Pow, Integer, Float, simplify, preorder_traversal, exp, log, sin, cos, \
    tanh,sqrt
from pmlb import fetch_data, dataset_names
RE_SCI_SMALL = re.compile(r'\d+\.\d+e-\d+')
RE_SCI_LARGE = re.compile(r'\d+\.\d+e\+?\d+')
RE_DECIMAL = re.compile(r'\d+\.\d+')
RE_MAIN_NUM = re.compile(r'(x|\*\*|\^)?(\d+\.?\d*|\.\d+)')
RE_EXP_FLOAT = re.compile(r'\*\*(\d+\.\d+)')  
RE_LOWERCASE = re.compile(r'([a-zA-BD-Z])(?!C)') 
RE_ADD_C_X = re.compile(r'(?<!C\*)\bx(\d+)')   
RE_ADD_C_FUNC = re.compile(r'(?<!C\*)\b(sin|cos|tan|log|abs|exp|tanh|sqrt)\b') 
ALLOWED_EXPONENTS = {'-2', '-1', '0', '1', '2', '3', '4', '5'}
_OBS_INSTANCE = None
import re
from collections import Counter, defaultdict
DERIVED_UNIT_MAP = {
    "Hz": {"s": -1},
    "N": {"kg": 1, "m": 1, "s": -2},
    "Pa": {"kg": 1, "m": -1, "s": -2},
    "J": {"kg": 1, "m": 2, "s": -2},
    "W": {"kg": 1, "m": 2, "s": -3},
    "C": {"A": 1, "s": 1},
    "V": {"kg": 1, "m": 2, "s": -3, "A": -1},
    "Ohm": {"kg": 1, "m": 2, "s": -3, "A": -2},
    "F": {"kg": -1, "m": -2, "s": 4, "A": 2},
    "H": {"kg": 1, "m": 2, "s": -2, "A": -2},
    "T": {"kg": 1, "s": -2, "A": -1},
    "Wb": {"kg": 1, "m": 2, "s": -2, "A": -1},
}
BASE_UNITS = {"kg", "m", "s", "A", "K", "mol", "cd"}
class TimeoutException(Exception):
    pass
class SimplifyTimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Timeout")
def _merge_dim(dim, other, sign=1):
    """合并量纲字典"""
    for k, v in other.items():
        dim[k] += sign * v
def _parse_atomic_unit(unit):
    """解析单个单位（如 V, m, s）"""
    if unit in BASE_UNITS:
        return defaultdict(int, {unit: 1})
    if unit in DERIVED_UNIT_MAP:
        return defaultdict(int, DERIVED_UNIT_MAP[unit])
    if unit == "1":
        return defaultdict(int)
    raise ValueError(f"未知单位: {unit}")
def parse_unit_expr(expr):
    """
    解析单位表达式，如:
    - V/m
    - C*m
    - J*s
    - 1/s
    """
    dim = defaultdict(int)
    tokens = re.split(r'([*/])', expr.replace(" ", ""))
    sign = 1  
    for tok in tokens:
        if tok == "*":
            sign = 1
            continue
        if tok == "/":
            sign = -1
            continue
        if "^" in tok:
            u, p = tok.split("^")
            p = int(p)
        else:
            u, p = tok, 1
        atom_dim = _parse_atomic_unit(u)
        _merge_dim(dim, atom_dim, sign * p)
    return dict(dim)
def dim_to_string(dim):
    """将量纲 dict 转为字符串"""
    if not dim:
        return "1"
    items = []
    for k in sorted(dim.keys()):
        v = dim[k]
        if v == 0:
            continue
        if v == 1:
            items.append(k)
        else:
            items.append(f"{k}^{v}")
    return "*".join(items)
BASE_UNITS = [
    "kg", "mol", "cd", "A", "K", "s", "m"
]
DERIVED_UNITS = [
    "J", "N", "W", "Pa", "Hz", "C", "V", "Ohm", "T"
]
ALL_UNITS = sorted(BASE_UNITS + DERIVED_UNITS, key=len, reverse=True)
def normalize_implicit_mul(unit: str) -> str:
    """
    Js      -> J*s
    kgm     -> kg*m
    Nms^-1  -> N*m*s^-1
    """
    i = 0
    tokens = []
    while i < len(unit):
        matched = False
        for u in ALL_UNITS:
            if unit.startswith(u, i):
                tokens.append(u)
                i += len(u)
                matched = True
                break
        if not matched:
            tokens.append(unit[i])
            i += 1
    return "*".join(tokens)
def convert_unit_list_to_base(unit_list):
    results = []
    for u in unit_list:
        if u in ["-", "", "rad"]:
            results.append("0")
            continue
        if u == "rad/s":
            results.append("s^-1")
            continue
        try:
            dim = parse_unit_expr(u)
            results.append(dim_to_string(dim))
        except Exception as e:
            results.append(f"[ERROR] {u}: {e}")
    return results
def get_equation_complexity(equation_str) -> int:
    """
    计算数学方程字符串的复杂度。
    标准：PySR 风格 (Complexity = 节点总数)
    即：变量、常数、二元运算符、一元运算符、函数调用均各计为 1 点复杂度。
    Args:
        equation_str (str): 方程字符串, 例如 "x0 + sin(x1)"
    Returns:
        int: 复杂度数值。如果解析失败返回 -1 (或你可以改为 float('inf'))
    """
    def _count_nodes(node):
        """递归遍历 AST 节点计算数量"""
        if isinstance(node, ast.BinOp):
            left = _count_nodes(node.left)
            right = _count_nodes(node.right)
            return 1 + left + right
        elif isinstance(node, ast.UnaryOp):
            return 1 + _count_nodes(node.operand)
        elif isinstance(node, ast.Call):
            return 1 + sum(_count_nodes(arg) for arg in node.args)
        elif isinstance(node, (ast.Name, ast.Constant, ast.Num)):
            return 1
        else:
            return 1
    try:
        tree = ast.parse(equation_str, mode='eval')
        return _count_nodes(tree.body)
    except SyntaxError:
        print(f"Error: 无法解析方程 '{equation_str}'，可能是语法错误。")
        return -1
    except Exception as e:
        print(f"Error: 计算复杂度时发生未知错误: {e}")
        return -1
def dimensional_analysis(y_unit: str, x_units: List[str]):
    """
    量纲分析自动求解器
    参数:
        y_unit : 因变量 y 的单位，例如 "kg*m*s^-3*A^-1"
        x_units: 自变量 x0, x1, x2, ... 的单位列表，例如
                  ["1", "m*s^-1*A", "s", "kg*m^2*s^-1", "s^-1", "s^-1"]
    返回:
        包含完整求解结果的字典（与之前 solve_linear_system_sympy 输出格式一致）
    """
    base_units = ['kg', 'm', 's', 'A', 'K', 'mol', 'cd']  
    n = len(x_units)
    symbols = sp.symbols(f'a:{n}')  
    exponents = list(symbols)  
    def parse_unit(unit_str: str) -> Dict[str, sp.Integer]:
        if unit_str.strip() == "1" or unit_str.strip() == "":
            return {base: sp.S(0) for base in base_units}
        unit_str = unit_str.replace('*', ' ').replace('·', ' ')
        terms = unit_str.split()
        result = {base: sp.S(0) for base in base_units}
        i = 0
        while i < len(terms):
            term = terms[i]
            if term in base_units:
                base = term
                exp = sp.S(1)
                i += 1
                if i < len(terms) and terms[i].startswith('^'):
                    exp = sp.S(terms[i][1:])
                    if exp.is_Number and exp > 0:
                        exp = sp.S(exp)
                    elif exp.is_Number:
                        exp = -sp.S(-exp)
                    i += 1
                result[base] += exp
            elif term.startswith('^-'):  
                exp = sp.S(term[1:])
                base = terms[i + 1] if i + 1 < len(terms) else None
                if base in base_units:
                    result[base] -= exp
                    i += 2
                else:
                    i += 1
            elif '^' in term:  
                base_part, exp_part = term.split('^')
                if base_part in base_units:
                    exp = sp.S(exp_part)
                    result[base_part] += exp if exp > 0 else -sp.S(-exp)
                i += 1
            else:
                i += 1
        return result
    y_dim = parse_unit(y_unit)
    equations = []
    y_has_dimension = any(y_dim[base] != 0 for base in base_units)
    for base in base_units:
        lhs = y_dim[base] if y_has_dimension else 0  
        rhs = sp.S(0)
        for i, x_unit in enumerate(x_units):
            x_dim = parse_unit(x_unit)
            rhs += exponents[i] * x_dim.get(base, sp.S(0))
        if lhs != 0 or rhs != 0:
            equations.append(sp.Eq(lhs, rhs))
    vars_list = exponents
    def solve_system(eqns, vars_list):
        if not eqns:
            return {"有解": True, "特解": {v: 0 for v in vars_list}, "基础解系": [], "自由变量": vars_list[:]}
        A, b = sp.linear_eq_to_matrix(eqns, vars_list)
        aug = A.row_join(b)
        rref_mat, pivot_cols = aug.rref()
        rank = len(pivot_cols)
        if rank < aug.rows:
            inconsistent = False
            for i in range(rank, aug.rows):
                if rref_mat[i, -1] != 0:
                    inconsistent = True
                    break
            if inconsistent:
                return {"有解": False, "通解": "方程组无解（量纲不匹配）"}
        free_vars_idx = [i for i in range(len(vars_list)) if i not in pivot_cols]
        free_vars = [vars_list[i] for i in free_vars_idx]
        particular = {}
        subs0 = {v: 0 for v in free_vars}
        for i, var in enumerate(vars_list):
            if i in pivot_cols:
                row = pivot_cols.index(i)
                expr = -rref_mat[row, -1]
                for j in range(i + 1, len(vars_list)):
                    coeff = rref_mat[row, j]
                    if coeff != 0:
                        expr += coeff * vars_list[j]
                particular[var] = expr.subs(subs0)
            else:
                particular[var] = 0
        basis = []
        for k, free_var in enumerate(free_vars):
            subs_k = {v: 0 for v in free_vars}
            subs_k[free_var] = 1
            vec = {}
            for i, var in enumerate(vars_list):
                if i in pivot_cols:
                    row = pivot_cols.index(i)
                    expr = -rref_mat[row, -1]
                    for j in range(i + 1, len(vars_list)):
                        coeff = rref_mat[row, j]
                        if coeff != 0:
                            expr += coeff * vars_list[j]
                    vec[var] = expr.subs(subs_k)
                else:
                    vec[var] = subs_k.get(var, 0)
            basis.append(vec)
        return {
            "有解": True,
            "特解": particular,
            "基础解系": basis,
            "自由变量": free_vars,
            "方程组": eqns,
        }
    def solve_linear_system_sympy(eqns, vars):
        """
        使用 sympy 求解线性方程组，并返回清晰的特解和基础解系。
        参数:
        eqns : sympy.Eq 对象的列表，代表方程组
        vars : sympy.Symbol 对象的列表，代表变量
        返回:
        字典: 包含解的信息
        """
        solution = sp.solve(eqns, vars, dict=True, simplify=False)
        if not solution:
            return {"有解": False, "通解": "方程组无解"}
        sol_dict = solution[0]
        free_vars = []
        for var in vars:
            if var not in sol_dict or sol_dict[var] == var:
                free_vars.append(var)
        particular_solution = {}
        for var in vars:
            if var in sol_dict:
                expr = sol_dict[var]
                for free_var in free_vars:
                    expr = expr.subs(free_var, 0)
                particular_solution[var] = expr
            else:
                particular_solution[var] = 0
        basis_vectors = []
        for free_var in free_vars:
            basis_vec = {}
            substitution = {v: 0 for v in free_vars}
            substitution[free_var] = 1
            for var in vars:
                if var in sol_dict:
                    expr = sol_dict[var]
                    for v, val in substitution.items():
                        expr = expr.subs(v, val)
                    basis_vec[var] = expr - particular_solution[var]
                else:
                    basis_vec[var] = substitution[var]
            basis_vectors.append(basis_vec)
        return {
            "有解": True,
            "特解": particular_solution,
            "基础解系": basis_vectors,
            "自由变量": free_vars
        }
    result = solve_linear_system_sympy(equations, vars_list)
    def format_dimension_from_solution(result_dict, vars_list):
        """
        根据求解器结果（特解 + 基础解系）构造量纲表达式字符串。
        vars_list: exponents 列表（sympy.Symbol，例如 a0,a1,...）
        result_dict: solve_linear_system_sympy 的返回字典
        """
        particular = result_dict.get("特解", {})
        basis = result_dict.get("基础解系", [])  
        n_basis = len(basis)
        k_syms = sp.symbols(f'k1:{n_basis + 1}') if n_basis > 0 else ()
        terms = []
        for var in vars_list:
            exp_expr = sp.S(0)
            if var in particular:
                exp_expr += sp.simplify(particular[var])
            for j in range(n_basis):
                coeff = basis[j].get(var, 0)
                coeff = sp.simplify(coeff)
                if coeff != 0:
                    exp_expr += coeff * k_syms[j]
            exp_expr = sp.simplify(exp_expr)
            if exp_expr == 0:
                continue
            idx_str = var.name[1:] if len(var.name) > 1 else var.name
            if exp_expr == 1:
                terms.append(f"x{idx_str}")
            elif exp_expr == -1:
                terms.append(f"x{idx_str}^-1")
            else:
                terms.append(f"x{idx_str}^({sp.simplify(exp_expr)})")
        if not terms:
            return "∝ 常数（无量纲）"
        else:
            return "*".join(terms)
    def format_dimension_from_solution1111(result_dict, vars_list):
        """
        根据求解器结果构造量纲表达式字符串。
        输出格式：特解项 * (基础解系项1)^k1 * (基础解系项2)^k2 ...
        例如：x0 * (x2*x4)^k1 * (x2*x5)^k2
        """
        particular = result_dict.get("特解", {})
        basis = result_dict.get("基础解系", [])  
        n_basis = len(basis)
        k_syms = sp.symbols(f'k1:{n_basis + 1}') if n_basis > 0 else ()
        parts = []
        p_terms = []
        for var in vars_list:
            val = sp.simplify(particular.get(var, 0))
            if val != 0:
                idx_str = var.name[1:] if len(var.name) > 1 else var.name
                if val == 1:
                    p_terms.append(f"x{idx_str}")
                elif val == -1:
                    p_terms.append(f"x{idx_str}^-1")
                else:
                    p_terms.append(f"x{idx_str}^({val})")
        if p_terms:
            parts.append("*".join(p_terms))
        for j in range(n_basis):
            basis_vec = basis[j]
            group_terms = []
            for var in vars_list:
                coeff = sp.simplify(basis_vec.get(var, 0))
                if coeff != 0:
                    idx_str = var.name[1:] if len(var.name) > 1 else var.name
                    if coeff == 1:
                        group_terms.append(f"x{idx_str}")
                    elif coeff == -1:
                        group_terms.append(f"x{idx_str}^-1")
                    else:
                        group_terms.append(f"x{idx_str}^({coeff})")
            if group_terms:
                inner_expr = "*".join(group_terms)
                k_name = str(k_syms[j])
                parts.append(f"({inner_expr})^({k_name})")
        if not parts:
            return "1 (无量纲)"
        return "+".join(parts)
    def format_dimension_to_list(result_dict, vars_list):
        """
        将量纲求解结果转成 list:
        [
            特解表达式,
            基础解系 inner_expr_1,
            基础解系 inner_expr_2,
            ...
        ]
        """
        if result_dict.get("有解") is False:
            fallback_list = []
            for var in vars_list:
                idx_str = var.name[1:] if len(var.name) > 1 else var.name
                fallback_list.append(f"x{idx_str}") 
            print("无解分解")
            return fallback_list
        particular = result_dict.get("特解", {})
        basis = result_dict.get("基础解系", [])
        expr_list = []
        p_terms = []
        for var in vars_list:
            val = sp.simplify(particular.get(var, 0))
            if val != 0:
                idx_str = var.name[1:] if len(var.name) > 1 else var.name
                if val == 1:
                    p_terms.append(f"x{idx_str}")
                elif val == -1:
                    p_terms.append(f"x{idx_str}^-1")
                else:
                    p_terms.append(f"x{idx_str}^({val})")
        if p_terms:
            expr_list.append("*".join(p_terms))
        else:
            expr_list.append("1")  
        for basis_vec in basis:
            group_terms = []
            for var in vars_list:
                coeff = sp.simplify(basis_vec.get(var, 0))
                if coeff != 0:
                    idx_str = var.name[1:] if len(var.name) > 1 else var.name
                    if coeff == 1:
                        group_terms.append(f"x{idx_str}")
                    elif coeff == -1:
                        group_terms.append(f"x{idx_str}^-1")
                    else:
                        group_terms.append(f"x{idx_str}^({coeff})")
            if group_terms:
                expr_list.append("*".join(group_terms))
        return expr_list
    if result["有解"]:
        result["量纲表达式"] = format_dimension_from_solution1111(result, vars_list)
        result["list"] = format_dimension_to_list(result, vars_list)
    else:
        result["list"] = format_dimension_to_list(result, vars_list)
    return result
def lift_fractional_powers(expr_str):
    """
    若表达式中存在分数幂，则整体提升到最小整数幂
    """
    pattern = r"\^\(([-+]?\d+/\d+)\)"
    fracs = re.findall(pattern, expr_str)
    if not fracs:
        return expr_str, 1  
    fractions = [Fraction(f) for f in fracs]
    denominators = [f.denominator for f in fractions]
    k = lcm(*denominators)
    new_expr = f"({expr_str})^{k}"
    return new_expr, k
def validate_operators(ops, base_ops=None):
    if base_ops is None:
        base_ops = ['+', '-', '*', '/', '^']
    tokens = []
    if ops is None:
        tokens = []
    elif isinstance(ops, (list, tuple)):
        for op in ops:
            tokens.append(str(op).strip())
    elif isinstance(ops, str):
        s = ops.strip()
        tokens = re.findall(r'\^|[+\-*/]|[a-zA-Z_]\w*', s)
    else:
        try:
            tokens = [str(op).strip() for op in ops]
        except Exception:
            tokens = []
    valid_ops = []
    seen = set()
    for tok in tokens:
        if tok and tok not in seen:
            valid_ops.append(tok)
            seen.add(tok)
    for op in base_ops:
        if op not in seen:
            valid_ops.append(op)
    return valid_ops
def process_solution_list(solution):
    temp_list = []
    for i, expr in enumerate(solution["list"]):
        if i >= 0:
            expr, _ = lift_fractional_powers(expr)
        sym_expr = str(sp.sympify(expr))
        temp_list.append([sym_expr.replace("**", "^")])
    processed = []
    LIMIT = 3 
    if len(temp_list) <= LIMIT:
        return temp_list
    else:
        for i in range(LIMIT - 1):
            item = temp_list[i]
            s = item[0] if isinstance(item, list) else item
            processed.append([s])
        rest_parts = []
        for item in temp_list[LIMIT-1:]:
            if isinstance(item, list):
                rest_parts.append(str(item[0]))
            else:
                rest_parts.append(str(item))
        combined_expr = "+".join(rest_parts)
        processed.append([combined_expr])
    return processed
def _parse_worker_new(f_str, g_str, pset_f, pset_g, queue):
    """
    独立进程运行的解析任务
    结果通过 Queue 传回
    """
    try:
        b1 = gp.PrimitiveTree.from_string_sympy_fast(f_str, pset_f)
        b2 = gp.PrimitiveTree.from_string_sympy_fast(g_str, pset_g)
        queue.put({"status": "success", "data": (b1, b2)})
    except Exception as e:
        queue.put({"status": "error", "msg": str(e)})
class Observations:
    def __init__(self):
        self.pop = []
        self.labels = []
    def append1(self, f_eq_str_list, g_eq_str_list, fitness_list):
        if not hasattr(self, 'csv_path'):
            os.makedirs('figs', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_path = f'figs/observation_{timestamp}.csv'
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['index', 'f_eq_str', 'g_eq_str', 'fitness'])
        filtered_count = 0
        for ii in range(len(f_eq_str_list)):
            fitness = fitness_list[ii]
            if fitness > 3:
                print(f' pass fitness > 3 : [%s] %s | %s | %s' %(ii + 1, f_eq_str_list[ii], g_eq_str_list[ii], fitness_list[ii]))
                filtered_count += 1
                continue
            print('Appending to observations: [%s] %s | %s | %s'%(ii+1, f_eq_str_list[ii], g_eq_str_list[ii], fitness_list[ii]))
            f_eq_str = f_eq_str_list[ii]
            g_eq_str = g_eq_str_list[ii]
            b1 = gp.PrimitiveTree.from_string_sympy(f_eq_str, pset_f) 
            b2 = gp.PrimitiveTree.from_string_sympy(g_eq_str, pset_g)
            self.pop.append([(b1, b2), fitness_list[ii]])
            self.labels.append(fitness_list[ii])
            while len(self.pop) > 2000:
                max_index = np.argmax(self.labels)
                del self.pop[max_index]
                del self.labels[max_index]
    def append(self, f_eq_str_list, g_eq_str_list, term_eq_str_list, all_components_list, fitness_list,strategy_name):
        filtered_count = 0
        for ii in range(len(f_eq_str_list)):
            fitness = fitness_list[ii]
            if fitness > 1e9:
                print(f' pass fitness > 1e9 : [{ii + 1}] {f_eq_str_list[ii]} | {g_eq_str_list[ii]} | {term_eq_str_list[ii]} | {all_components_list[ii]} | {fitness_list[ii]} | {strategy_name}')
                filtered_count += 1
                continue
            print(f'Appending to observations: [{ii+1}] {f_eq_str_list[ii]} | {g_eq_str_list[ii]} | {term_eq_str_list[ii]} | {all_components_list[ii]} | {fitness_list[ii]} | {strategy_name}')
            f_eq_str = f_eq_str_list[ii]
            g_eq_str = g_eq_str_list[ii]
            term_eq_str = term_eq_str_list[ii]
            all_components_str = all_components_list[ii]
            strategy_name_str = strategy_name[ii]
            try:
                b1 = gp.PrimitiveTree.from_string_sympy(f_eq_str, pset_f)
                b2 = gp.PrimitiveTree.from_string_sympy(g_eq_str, pset_g)
            except Exception as e:
                print(f"组合失败: fx={f_eq_str}(错误: {str(e)})")
                continue
            self.pop.append([(b1, b2), term_eq_str, all_components_str, fitness, strategy_name_str])
            self.labels.append(fitness)
            while len(self.pop) > 2500:
                max_index = np.argmax(self.labels)
                del self.pop[max_index]
                del self.labels[max_index]    
    def append2(self, f_eq_str_list, g_eq_str_list, term_eq_str_list, fitness_list):
        candidates = []
        for i, fit in enumerate(fitness_list):
            if fit <= 5:
                candidates.append({
                    'type': 'new',
                    'fitness': fit,
                    'data_idx': i  
                })
        if not candidates:
            return
        for i, fit in enumerate(self.labels):
            candidates.append({
                'type': 'old',
                'fitness': fit,
                'data_idx': i 
            })
        candidates.sort(key=lambda x: x['fitness']) 
        survivors = candidates[:2500]
        new_pop = []
        new_labels = []
        for item in survivors:
            if item['type'] == 'old':
                old_index = item['data_idx']
                new_pop.append(self.pop[old_index])
                new_labels.append(self.labels[old_index])
            else:
                idx = item['data_idx']
                f_eq_str = f_eq_str_list[idx]
                g_eq_str = g_eq_str_list[idx]
                term_eq_str = term_eq_str_list[idx]
                fitness = fitness_list[idx]
                print(f'Appending: [{idx+1}] {f_eq_str} | {term_eq_str} | {fitness}')
                try:
                    b1 = gp.PrimitiveTree.from_string_sympy(f_eq_str, pset_f)
                    b2 = gp.PrimitiveTree.from_string_sympy(g_eq_str, pset_g)
                    new_pop.append([(b1, b2), term_eq_str, fitness])
                    new_labels.append(fitness)
                except Exception as e:
                    print(f"解析失败跳过: {e}")
                    continue
        self.pop = new_pop
        self.labels = new_labels
    def append_without_print_1_2_before(self, f_eq_str_list, g_eq_str_list, term_eq_str_list, all_components_list, fitness_list,strategy_name):
        filtered_count = 0
        for ii in range(len(f_eq_str_list)):
            fitness = fitness_list[ii]
            if fitness > 1e10:
                filtered_count += 1
                continue
            f_eq_str = f_eq_str_list[ii]
            g_eq_str = g_eq_str_list[ii]
            term_eq_str = term_eq_str_list[ii]
            all_components_str = all_components_list[ii]
            strategy_name_str = strategy_name[ii]
            try:
                b1 = gp.PrimitiveTree.from_string_sympy_fast(f_eq_str, pset_f)
                b2 = gp.PrimitiveTree.from_string_sympy_fast(g_eq_str, pset_g)
            except Exception as e:
                print(f"组合失败: fx={f_eq_str}(错误: {str(e)})")
                continue
            self.pop.append([(b1, b2), term_eq_str, all_components_str, fitness, strategy_name_str])
            self.labels.append(fitness)
            while len(self.pop) > 2500:
                max_index = np.argmax(self.labels)
                del self.pop[max_index]
                del self.labels[max_index] 
    def append_without_print(self, f_eq_str_list, g_eq_str_list, term_eq_str_list, all_components_list, fitness_list, strategy_name):
        filtered_count = 0
        count_bad_fitness = 0
        count_timeout = 0
        count_parse_error = 0
        valid_indices = []
        for i, fit in enumerate(fitness_list):
            if fit >= 1e9:
                count_bad_fitness += 1
            else:
                valid_indices.append(i)
        for ii in tqdm(valid_indices):
            f_eq_str = f_eq_str_list[ii]
            g_eq_str = g_eq_str_list[ii]
            term_eq_str = term_eq_str_list[ii]
            all_components_str = all_components_list[ii]
            strategy_name_str = strategy_name[ii]
            fitness = fitness_list[ii]
            q = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=_parse_worker_new, 
                args=(f_eq_str, g_eq_str, pset_f, pset_g, q)
            )
            try:
                p.start()
                p.join(timeout=2)
                b1, b2 = None, None
                if p.is_alive():
                    p.terminate() 
                    p.join(timeout=0.1)
                    if p.is_alive():
                        p.kill() 
                        p.join() 
                    count_timeout += 1
                    q.close()
                    q.join_thread()
                    continue 
                else:
                    if not q.empty():
                        try:
                            res = q.get_nowait()
                            if res["status"] == "success":
                                b1, b2 = res["data"]
                            else:
                                count_parse_error += 1
                                q.close()
                                q.join_thread()
                                continue
                        except Exception:
                            count_parse_error += 1
                            q.close()
                            q.join_thread()
                            continue
                    else:
                        count_parse_error += 1
                        q.close()
                        q.join_thread()
                        continue
            except Exception as e:
                print(f"系统级错误: {e}")
                if p.is_alive():
                    p.kill()
                    p.join()
            finally:
                try:
                    q.close()
                    q.join_thread()
                except:
                    pass
            self.pop.append([(b1, b2), term_eq_str, all_components_str, fitness, strategy_name_str, get_equation_complexity(f_eq_str)])
            self.labels.append(fitness)
            if len(self.pop) > 2500:
                max_index = np.argmax(self.labels)
                del self.pop[max_index]
                del self.labels[max_index]
        print(f"成功入库: {len(valid_indices) - count_timeout - count_parse_error}")
    def str_from(self, ind_in_pop_f, simplify_flag=False, timeout_seconds=2):
        raw_str_obj = gp.TreeToDeadStr(ind_in_pop_f, converter)
        raw_str = str(raw_str_obj)
        if simplify_flag:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                simplified_expr = sp.simplify(raw_str)
                return str(simplified_expr)
            except TimeoutException: 
                print(f"[Timeout] 化简超时: {raw_str}")
                return raw_str
            except Exception as e:
                print(f"[Error] 化简出错: {e}")
                return raw_str
            finally:
                signal.alarm(0)
        else:
            return raw_str
    def get_pop_str(self, pop, simplify_flag=False):
        pop_str = []
        for ii in tqdm(range(len(pop))):
            f_eq_str = self.str_from(pop[ii][0][0], simplify_flag=simplify_flag)
            g_eq_str = self.str_from(pop[ii][0][1], simplify_flag=simplify_flag)
            pop_str.append([(f_eq_str, g_eq_str), pop[ii][1], pop[ii][2], pop[ii][3], pop[ii][4], get_equation_complexity(str(f_eq_str))])
        return pop_str
    def return_best(self):
        return [self.pop[np.argmin(self.labels)]]
    def show(self):
        print("+++++++++++++++++++++++++++")
        print("pop:")
        print(self.get_pop_str(self.pop))
        print("labels:")
        print(self.labels)
        print("+++++++++++++++++++++++++++")
    def get_fitness_diversity(self, precision=6, top_k=50):
        """
        计算种群中【表现最好的前 k 个】方程分数的唯一性比率。
        :param precision: 保留小数位。
        :param top_k: 只查看前多少个最好的方程（默认50）。
        :return: 0.0 (精英群体完全一致) ~ 1.0 (精英群体各不相同)
        """
        if not self.labels:
            return 0.0
        sorted_fitness = sorted(self.labels)
        target_fitness = sorted_fitness[:top_k]
        total_count = len(target_fitness)
        if total_count == 0:
            return 0.0
        rounded_fitness = [round(f, precision) for f in target_fitness]
        unique_count = len(set(rounded_fitness))
        diversity_score = unique_count / total_count
        return diversity_score
    def get_population_status(self):
        """
        [高级功能] 获取当前种群的详细状态，重点提取最佳个体的完整信息。
        适配 self.pop = [(trees), term_str, all_comp_str, fitness] 的结构。
        """
        if not self.labels:
            return {
                "diversity_score": 0,
                "best_solution_info": "无数据 (Population Empty)",
                "population_size": 0
            }
        diversity = self.get_fitness_diversity()
        best_loss = min(self.labels)
        best_idx = self.labels.index(best_loss)
        best_individual = self.pop[best_idx]
        best_eq_str = replace_C_to_one(replace_numbers(str(sp.simplify(gp.TreeToDeadStr(best_individual[0][0], converter)))))
        term_str = str(best_individual[1])
        components_str = str(best_individual[2])
        complexity_set = len(best_eq_str)
        best_info_pack = {
            "equation": best_eq_str,       
            "term": term_str,      
            "components":components_str,
            "loss": best_loss,         
            "complexity": complexity_set,  
        }
        return {
            "diversity_score": round(diversity, 6),
            "best_loss": best_loss,
            "best_solution_info": best_info_pack, 
            "population_size": len(self.labels)
        }
class ResultCollector:
    def __init__(self):
        self.new_data = [] 
        self.labels = []   
        self.pop = []
    def append(self, item):
        self.new_data.append(item)
    def str_from(self, ind_in_pop_f, simplify_flag=False):
        if simplify_flag:
            return str(sp.simplify(str(gp.TreeToDeadStr(ind_in_pop_f, converter)),rational=True))
        else:
            return str(gp.TreeToDeadStr(ind_in_pop_f, converter))
    def get_pop_str(self, pop, simplify_flag=False):
        pop_str = []
        for ii in tqdm(range(len(pop))):
            f_eq_str = self.str_from(pop[ii][0][0], simplify_flag=simplify_flag)
            g_eq_str = self.str_from(pop[ii][0][1], simplify_flag=simplify_flag)
            pop_str.append([(f_eq_str, g_eq_str), pop[ii][1], pop[ii][2], pop[ii][3]])
        return pop_str
    def get_fitness_diversity(self):
        return 0 
class Observations_island:
    def __init__(self):
        self.pop = []
        self.labels = []
    def append(self, f_eq_str_list, g_eq_str_list, term_eq_str_list, all_components_list, fitness_list):
        filtered_count = 0
        for ii in range(len(f_eq_str_list)):
            fitness = fitness_list[ii]
            if fitness > 5:
                print(f' pass fitness > 5 : [{ii + 1}] {f_eq_str_list[ii]} | {g_eq_str_list[ii]} | {term_eq_str_list[ii]} | {all_components_list[ii]} | {fitness_list[ii]}')
                filtered_count += 1
                continue
            print(f'Appending to observations: [{ii+1}] {f_eq_str_list[ii]} | {g_eq_str_list[ii]} | {term_eq_str_list[ii]} | {all_components_list[ii]} | {fitness_list[ii]}')
            f_eq_str = f_eq_str_list[ii]
            g_eq_str = g_eq_str_list[ii]
            term_eq_str = term_eq_str_list[ii]
            all_components_str = all_components_list[ii]
            b1 = gp.PrimitiveTree.from_string_sympy(f_eq_str, pset_f)
            b2 = gp.PrimitiveTree.from_string_sympy(g_eq_str, pset_g)
            self.pop.append([(b1, b2), term_eq_str, all_components_str, fitness])
            self.labels.append(fitness)
            while len(self.pop) > 2500:
                max_index = np.argmax(self.labels)
                del self.pop[max_index]
                del self.labels[max_index]    
    def str_from(self, ind_in_pop_f, simplify_flag=False):
        if simplify_flag:
            return str(sp.simplify(str(gp.TreeToDeadStr(ind_in_pop_f, converter)),rational=True))
        else:
            return str(gp.TreeToDeadStr(ind_in_pop_f, converter))
    def get_pop_str(self, pop, simplify_flag=False):
        pop_str = []
        for ii in tqdm(range(len(pop))):
            f_eq_str = self.str_from(pop[ii][0][0], simplify_flag=simplify_flag)
            g_eq_str = self.str_from(pop[ii][0][1], simplify_flag=simplify_flag)
            pop_str.append([(f_eq_str, g_eq_str), pop[ii][1], pop[ii][2], pop[ii][3], pop[ii][4]])
        return pop_str
    def return_best(self):
        return [self.pop[np.argmin(self.labels)]]
    def show(self):
        print("pop:")
        print(self.get_pop_str(self.pop))
        print("labels:")
        print(self.labels)
    def get_fitness_diversity(self, precision=6, top_k=100):
        """
        计算种群中【表现最好的前 k 个】方程分数的唯一性比率。
        :param precision: 保留小数位。
        :param top_k: 只查看前多少个最好的方程（默认50）。
        :return: 0.0 (精英群体完全一致) ~ 1.0 (精英群体各不相同)
        """
        if not self.labels:
            return 0.0
        sorted_fitness = sorted(self.labels)
        target_fitness = sorted_fitness[:top_k]
        total_count = len(target_fitness)
        if total_count == 0:
            return 0.0
        rounded_fitness = [round(f, precision) for f in target_fitness]
        unique_count = len(set(rounded_fitness))
        diversity_score = unique_count / total_count
        return diversity_score
    def get_population_status(self):
        """
        [高级功能] 获取当前种群的详细状态，重点提取最佳个体的完整信息。
        适配 self.pop = [(trees), term_str, all_comp_str, fitness] 的结构。
        """
        if not self.labels:
            return {
                "diversity_score": 0,
                "best_solution_info": "无数据 (Population Empty)",
                "population_size": 0
            }
        diversity = self.get_fitness_diversity()
        best_loss = min(self.labels)
        best_idx = self.labels.index(best_loss)
        best_individual = self.pop[best_idx]
        best_eq_str = replace_C_to_one(replace_numbers(str(sp.simplify(gp.TreeToDeadStr(best_individual[0][0], converter)))))
        term_str = str(best_individual[1])
        components_str = str(best_individual[2])
        complexity_set = len(best_eq_str)
        best_info_pack = {
            "equation": best_eq_str,       
            "term": term_str,      
            "components":components_str,
            "loss": best_loss,         
            "complexity": complexity_set,  
        }
        return {
            "diversity_score": round(diversity, 4),
            "best_loss": best_loss,
            "best_solution_info": best_info_pack, 
            "population_size": len(self.labels)
        }
def replace_numbers(expression): 
    expression = expression.replace("- 1", "+ 1").replace(" ", "")
    expression = re.sub(r'\d+\.\d+e-\d+', '0.00001', expression)
    expression = re.sub(r'\d+\.\d+e\d+', '10', expression)
    expression = re.sub(r'\d+\.\d+e+\d+', '10', expression)
    decimals = re.findall(r'\d+\.\d+', expression)
    for decimal in decimals:
        value = float(decimal)
        if value < 0.001:
            expression = expression.replace(decimal, '0')
    obs = Observations()
    expression = obs.str_from(expression.replace("^", "**"), simplify_flag=False)
    expression = "".join(expression.split())
    expression = add_parentheses_around_denominator(expression)
    result = []
    i = 0
    while i < len(expression):
        if expression[i].isdigit() or (
                expression[i] == '.' and i + 1 < len(expression) and expression[i + 1].isdigit()):
            num_start = i
            while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
                i += 1
            if num_start > 0 and (expression[num_start - 1] == 'x'):
                result.append(expression[num_start:i])
            elif num_start > 0 and (expression[num_start - 1] == '^' or (expression[num_start - 2:num_start] == '**')):
                num_str = expression[num_start:i]
                try:
                    num_value = int(num_str)
                    if num_value in [-2, -1, 0, 1, 2, 3, 4, 5]:
                        result.append(num_str)  
                    else:
                        result.append('C')  
                except ValueError:
                    result.append(num_str)
            else:
                result.append('C')  
        else:
            result.append(expression[i])
            i += 1
    expression = ''.join(result)
    def replace_floats(match):
        num = float(match.group(1))  
        rounded_num = round(num, 1)  
        return f"**{rounded_num}"  
    expression = re.sub(r'\*\*(\d+\.\d+)', replace_floats, expression)
    expression = expression.replace("Ce", "C")
    expression = expression.replace("C**2", "C")
    expression = expression.replace("Ce-C", "0")
    expression = expression.replace("CeC", "1")
    expression = re.sub(r'([a-zA-BD-Z])(?!C)', lambda match: match.group(1).lower(), expression)
    expression = re.sub(r'(?<!C\*)\bx(\d+)', r'C*x\1', expression)
    expression = re.sub(r'(?<!C\*)\b(sin|cos|tan|log|abs|exp|tanh|sqrt)\b', r'C*\1', expression)
    if expression == "0" or expression == "1" or expression == "zoo" or expression == "nan":
       expression = "C"
    return expression
def replace_numbers111(expression):
    global _OBS_INSTANCE
    expression = expression.replace("- 1", "+ 1").replace(" ", "")
    expression = RE_SCI_SMALL.sub('0.00001', expression)
    expression = RE_SCI_LARGE.sub('10', expression)
    has_small_value = False 
    def _check_small_float(match):
        nonlocal has_small_value 
        val = float(match.group(0))
        if val < 0.002:
            has_small_value = True 
            return '0'
        return match.group(0)
    expression = RE_DECIMAL.sub(_check_small_float, expression)
    if _OBS_INSTANCE is None:
        try:
            _OBS_INSTANCE = Observations()
        except NameError:
            pass 
    if _OBS_INSTANCE:
        expression = _OBS_INSTANCE.str_from(expression.replace("^", "**"), simplify_flag=has_small_value)
    else:
        expression = expression.replace("^", "**")
    expression = "".join(expression.split())
    try:
        expression = add_parentheses_around_denominator(expression)
    except NameError:
        pass
    def _process_num(match):
        prefix = match.group(1) 
        num_str = match.group(2) 
        if prefix == 'x':
            return match.group(0)
        if prefix in ('^', '**'):
            if '.' not in num_str and num_str in ALLOWED_EXPONENTS:
                return match.group(0)
            else:
                return prefix + 'C' 
        return 'C'
    expression = RE_MAIN_NUM.sub(_process_num, expression)
    expression = RE_EXP_FLOAT.sub(lambda m: f"**{round(float(m.group(1)), 1)}", expression)
    expression = expression.replace("Ce", "C")\
                           .replace("C**2", "C")\
                           .replace("Ce-C", "0")\
                           .replace("CeC", "1")
    expression = RE_LOWERCASE.sub(lambda m: m.group(1).lower(), expression)
    expression = RE_ADD_C_X.sub(r'C*x\1', expression)       
    expression = RE_ADD_C_FUNC.sub(r'C*\1', expression)     
    if expression in ("0", "1", "zoo", "nan"):
        return "C"
    return expression
def add_parentheses_around_denominator1(expr: str) -> str:
    """
    自动在每个除号 '/' 后的分母部分添加括号（若未被括号包裹）
    """
    result = []
    i = 0
    n = len(expr)
    while i < n:
        if expr[i] == '/':
            result.append('/')
            i += 1
            while i < n and expr[i].isspace():
                result.append(expr[i])
                i += 1
            if i < n and expr[i] == '(':
                paren_count = 1
                result.append('(')
                i += 1
                while i < n and paren_count > 0:
                    if expr[i] == '(':
                        paren_count += 1
                    elif expr[i] == ')':
                        paren_count -= 1
                    result.append(expr[i])
                    i += 1
            else:
                result.append('(')
                start = i
                paren_count = 0
                while i < n:
                    if expr[i] == '(':
                        paren_count += 1
                    elif expr[i] == ')':
                        if paren_count > 0:
                            paren_count -= 1
                        else:
                            break
                    elif paren_count == 0 and expr[i] in '+-*/':
                        if expr[i] == '-' and (i == start or expr[i-1] in '(+*/^'):
                            pass  
                        else:
                            break
                    result.append(expr[i])
                    i += 1
                result.append(')')
        else:
            result.append(expr[i])
            i += 1
    return ''.join(result)
def add_parentheses_around_denominator2(expr: str) -> str:
    """
    自动在每个除号 '/' 后的分母部分添加括号（若未被括号包裹）
    正确处理连续乘法：x0/x1*x2*x3 → x0/(x1*x2*x3)
    """
    result = []
    i = 0
    n = len(expr)
    while i < n:
        if expr[i] == '/':
            result.append('/')
            i += 1
            while i < n and expr[i].isspace():
                result.append(expr[i])
                i += 1
            if i < n and expr[i] == '(':
                paren_count = 1
                result.append('(')
                i += 1
                while i < n and paren_count > 0:
                    if expr[i] == '(':
                        paren_count += 1
                    elif expr[i] == ')':
                        paren_count -= 1
                    result.append(expr[i])
                    i += 1
            else:
                result.append('(')
                start = i
                paren_count = 0
                while i < n:
                    if expr[i] == '(':
                        paren_count += 1
                        result.append(expr[i])
                        i += 1
                    elif expr[i] == ')':
                        if paren_count > 0:
                            paren_count -= 1
                            result.append(expr[i])
                            i += 1
                        else:
                            break
                    elif paren_count == 0:
                        if expr[i] in '+-':
                            if expr[i] == '-' and (i == start or expr[i-1] in '(+*/^'):
                                result.append(expr[i])
                                i += 1
                            else:
                                break
                        elif expr[i] in '*/':
                            result.append(expr[i])
                            i += 1
                        else:
                            result.append(expr[i])
                            i += 1
                    else:
                        result.append(expr[i])
                        i += 1
                result.append(')')
        else:
            result.append(expr[i])
            i += 1
    return ''.join(result)
def add_parentheses_around_denominator(expr: str) -> str:
    """
    自动在每个除号 '/' 后的分母部分添加括号（若未被括号包裹）
    使用更清晰的方法处理复杂情况
    """
    chars = list(expr)
    i = 0
    n = len(chars)
    while i < n:
        if chars[i] == '/':
            denominator_start = i + 1
            while denominator_start < n and chars[denominator_start].isspace():
                denominator_start += 1
            if denominator_start >= n:
                break
            denominator_end = find_denominator_end(chars, denominator_start)
            if denominator_end > denominator_start:
                chars.insert(denominator_end, ')')
                chars.insert(denominator_start, '(')
                n += 2  
                i = denominator_end + 2
            else:
                i += 1
        else:
            i += 1
    return ''.join(chars)
def find_denominator_end(chars, start):
    """
    找到分母的结束位置
    """
    i = start
    n = len(chars)
    paren_count = 0
    bracket_count = 0
    while i < n:
        if chars[i] == '(':
            paren_count += 1
        elif chars[i] == ')':
            if paren_count > 0:
                paren_count -= 1
            else:
                return i
        elif chars[i] == '[':
            bracket_count += 1
        elif chars[i] == ']':
            if bracket_count > 0:
                bracket_count -= 1
            else:
                return i
        if paren_count == 0 and bracket_count == 0:
            if chars[i] in '+-':
                if chars[i] == '-' and (i == start or chars[i-1] in '(+*/^[, '):
                    pass
                else:
                    return i
            elif chars[i] in ',;':
                return i
        i += 1
    return i  
def to_physic(expr,list):
    if '=' in expr:
        expr = expr.split('=', 1)[1].strip()
    mapping = {s: f"x{i-1}" for i, s in enumerate(list[1:], 1)}
    pattern = re.compile(r'\b(' + '|'.join(sorted(mapping, key=len, reverse=True)) + r')\b')
    expr = pattern.sub(lambda m: mapping[m.group()], expr)
    return expr
def replace_C_triangle(input_string):
    input_string = input_string.replace("Ce", "C")
    input_string = input_string.replace("Ce-C", "0")
    input_string = input_string.replace("CeC", "1")
    input_string = input_string.replace("-C**2", "C")
    input_string = input_string.replace("C**2", "C")
    i = 0
    result = ""
    in_trig = False
    trig_stack = 0   
    while i < len(input_string):
        char = input_string[i]
        if input_string[i:i+4] in ("sin(", "cos(", "tan("):
            in_trig = True
            trig_stack = 1
            result += input_string[i:i+4]
            i += 4
            continue
        if in_trig:
            if char == '(':
                trig_stack += 1
            elif char == ')':
                trig_stack -= 1
                if trig_stack == 0:
                    in_trig = False
        if char == 'C':
            if in_trig:
                result += "1"
            else:
                if i > 0 and (input_string[i - 1] == '^' or 
                              (i > 1 and input_string[i - 2:i] == '**')):
                    result += str(random.uniform(-2, 3))
                else:
                    result += str(random.uniform(0.1, 2.5))
        else:
            result += char
        i += 1
    result = add_parentheses_around_denominator(result)
    obs = Observations()
    result = obs.str_from(result, simplify_flag=False)
    return result
def replace_C(input_string):
    input_string = input_string.replace("Ce", "C")
    input_string = input_string.replace("Ce-C", "0")
    input_string = input_string.replace("CeC", "1")
    input_string = input_string.replace("-C**2", "C")
    input_string = input_string.replace("C**2", "C")
    i = 0
    result = ""
    while i < len(input_string):
        char = input_string[i]
        if char == 'C':
            if i > 0 and (input_string[i - 1] == '^' or (i > 1 and input_string[i - 2:i] == '**')):
                result += str(random.uniform(-3, 3))
            else:
                result += str(random.uniform(0.1, 2.5))
        else:
            result += char
        i += 1
    result = add_parentheses_around_denominator(result)
    obs = Observations()
    result = obs.str_from(result, simplify_flag=False)
    return result
def replace_C_to_one(input_string):
    input_string = input_string.replace("Ce", "C")
    input_string = input_string.replace("Ce-C", "0")
    input_string = input_string.replace("C**2", "C")
    input_string = input_string.replace("C**-1", "C")
    input_string = input_string.replace("C**3", "C")
    input_string = input_string.replace("C**4", "C")
    result = ""
    for char in input_string:
        if char == 'C':
            result += str(1)
        else:
            result += char
    result = result.replace("- 1", "+ 1")
    obs = Observations()
    result = add_parentheses_around_denominator(result)
    result = obs.str_from(result, simplify_flag=True)
    result = result.replace("**", "^").replace(" )", ")").replace(" ", "")
    return result
def check_brackets(expression):
    stack = []
    for char in expression:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    return not stack
def operator_combinations(ops):
    base = ['+','*','/','^']
    extra = [op for op in ops if op not in base and op != '-']
    result = []
    for r in range(len(extra)+1):
        for subset in combinations(extra, r):
            combo = base + list(subset)
            result.append(combo)
    return result
def extract_json_string(new_output: str) -> str:
    new_output = new_output.replace(r'\(', '').replace(r'\)', '')
    start = new_output.find('{')
    end = new_output.rfind('}')
    if start == -1 or end == -1:
        return new_output.strip()
    json_str = new_output[start:end + 1].strip()
    json_str = re.sub(r',\s*}', '}', json_str)
    count_open = json_str.count('{')
    count_close = json_str.count('}')
    if count_open > count_close:
        json_str += '}'
    elif count_close > count_open:
        json_str = '{' + json_str
    return json_str
def load_json_structure_agent(json_str):
    json_str = extract_json_string(json_str)
    data = json.loads(json_str)
    global_structure = data["global_structure"]
    a = list(data["sub_expressions"].values())
    return global_structure,a
def load_json_structure_term_agent(json_str):
    json_str = json.loads(extract_json_string(json_str))
    keys = list(json_str.keys())
    values = list(json_str.values())
    return keys,values
def load_json_oper_and_var_agent(json_str):
    json_str = extract_json_string(json_str)
    data = json.loads(json_str)
    desc = data["desc"]
    var = data["vars"]
    ops = data["ops"]
    required_ops = ['+', '-']
    missing_ops = set(required_ops) - set(ops)
    ops.extend(missing_ops)
    valid_ops = ['+', '-', '*', '/', '^', 'exp', 'log', 'sqrt', 'sin', 'cos', 'tan']
    ops = [op for i, op in enumerate(ops) if op in valid_ops and op not in ops[:i]]
    return desc,var,ops
def load_json_f_zuhe(ind_one_new_f, fx_list_all):
    data_f = json.loads(ind_one_new_f)
    common_keys = set(data_f.keys())
    for key in common_keys:
        try:
            fx = data_f[key]
            fx = replace_C(replace_numbers(fx))
            fx_list_all.append(fx)
            if not check_brackets(fx):
                print(f"Invalid brackets in fx: {fx}")
                continue
        except Exception as e:
            print(f"Error processing key {key}: fx={fx}. Error: {str(e)}")
            continue
    return fx_list_all
def load_json_f_to_list(ind_one_new_f, fx_list_all, symbols):
    data_f = json.loads(extract_json_string(ind_one_new_f))
    common_keys = set(data_f.keys())
    for key in common_keys:
        try:
            fx = data_f[key]
            if symbols:
                fx = to_physic(fx,symbols)
            fx = re.sub(r'\|(.*?)\|', r'abs(\1)', fx)
            fx_list_all.append(fx.replace(" ", ""))
            if not check_brackets(fx):
                print(f"Invalid brackets in fx: {fx}")
                continue
        except Exception as e:
            print(f"Error processing key {key}: fx={fx}. Error: {str(e)}")
            continue
    return fx_list_all
def _parse_worker(expr, pset, queue):
    try:
        result = gp.PrimitiveTree.from_string_sympy(expr, pset)
        queue.put(result)
    except Exception as e:
        queue.put(e)
def B1_safe_without_signal(expr_clean, pset_f, timeout=3):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_parse_worker, args=(expr_clean, pset_f, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        print(f"[Timeout] B1_safe 解析超时: {expr_clean}")
        return gp.PrimitiveTree.from_string_sympy_fast(expr_clean, pset_f)
    result = queue.get()
    if isinstance(result, Exception):
        raise result
    return result
def B1_safe(expr_clean, pset_f, timeout=2):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = B1_safe_logic(expr_clean, pset_f) 
        signal.alarm(0)
        return result
    except TimeoutException:
        print(f"[Timeout] 解析超时: {expr_clean}")
        return gp.PrimitiveTree.from_string_sympy_fast(expr_clean, pset_f)
    except Exception as e:
        signal.alarm(0) 
        raise e
    finally:
        signal.alarm(0)
def B1_safe_logic(expr_clean, pset_f):
    return gp.PrimitiveTree.from_string_sympy(expr_clean, pset_f)
def list_to_obs(fx_list_all, term_list, all_components_list):
    pop_new = []
    for idx, fx in enumerate(fx_list_all):
        if not check_brackets(fx):
            continue
        fx = re.sub(r'\|(.*?)\|', r'abs(\1)', fx)
        try:
            fx_processed_nums = replace_numbers(fx)
        except Exception as e:
            print(f"格式错误: fx={fx}(错误: {str(e)})")
            continue
        equations_to_process = []
        fx_normal = replace_C(fx_processed_nums)
        equations_to_process.append(fx_normal)
        if "sin" in fx_normal or "cos" in fx_normal or "tan" in fx_normal:
            try:
                fx_variant = replace_C_triangle(fx_processed_nums)
                equations_to_process.append(fx_variant)
            except Exception as e:
                print(f"生成三角函数变体失败: {str(e)}")
                pass
        for current_fx_str in equations_to_process:
            try:
                b1 = B1_safe(current_fx_str, pset_f)
                if b1 is None:  
                    raise Exception("timeout")
                current_term = term_list[0] if len(term_list) == 1 else term_list[idx]
                current_component = None
                if len(all_components_list) == 1:
                    current_component = all_components_list[0]
                else:
                    current_component = all_components_list[idx]
                pop_new.append([(b1, "0"), current_term, current_component, 1e10])
            except Exception as e:
                print(f"组合失败: fx={current_fx_str}(错误: {str(e)})")
                continue
    if len(pop_new) == 0:
        pop_new.append([("x0", "0"), "T1", "x0", 1e10])
    return pop_new
def expand_formula111(f_term, all_eqs_groups):
    """
    自动将 f_term 中的 TERM_i 替换为 all_eqs_groups 中对应的方程组合，
    同时返回两个 list:
        - expr_list: 所有展开后的表达式（含消融）
        - term_list: 每个表达式对应的骨架（消融时骨架会把某个Ti替换成1）
    """
    term_keys = sorted(set(re.findall(r'T(\d+)', f_term)), key=int)
    n_terms = len(term_keys)
    if len(all_eqs_groups) != n_terms:
        raise ValueError(f"T数量 ({n_terms}) 与 all_eqs_groups 长度 ({len(all_eqs_groups)}) 不一致！")
    expr_list = []
    term_list = []
    for combo in itertools.product(*all_eqs_groups):
        expr = f_term
        for i, val in enumerate(combo):
            expr = expr.replace(f"T{term_keys[i]}", f"({val})")
        expr_list.append(expr)
        term_list.append(f_term)  
    for abl_idx in range(n_terms):  
        for combo in itertools.product(*all_eqs_groups):
            f_term_ablated = f_term
            for i in range(n_terms):
                if i == abl_idx:
                    f_term_ablated = f_term_ablated.replace(f"T{term_keys[i]}", "(1)")
            expr = f_term
            for i, val in enumerate(combo):
                if i == abl_idx:
                    expr = expr.replace(f"T{term_keys[i]}", "(1)")
                else:
                    expr = expr.replace(f"T{term_keys[i]}", f"({val})")
            expr_list.append(expr)
            term_list.append(f_term_ablated)
    return expr_list, term_list
def expand_formula(f_term, all_eqs_groups):
    """
    自动将 f_term 中的 TERM_i 替换为 all_eqs_groups 中对应的方程组合，
    同时返回三个 list:
        - expr_list:       所有展开后的表达式（含消融）
        - term_list:       每个表达式对应的骨架（消融时骨架会把某个Ti替换成1）
        - components_list: 每个表达式对应的具体组件列表 (list[list[str]])
                           例如: [['x', 'sin(x)'], ['1', 'sin(x)']]
    """
    term_keys = sorted(set(re.findall(r'T(\d+)', f_term)), key=int)
    n_terms = len(term_keys)
    if len(all_eqs_groups) != n_terms:
        if n_terms != 0: 
            raise ValueError(f"T数量 ({n_terms}) 与 all_eqs_groups 长度 ({len(all_eqs_groups)}) 不一致！")
    expr_list = []
    term_list = []
    components_list = []  
    if n_terms == 0:
        return [f_term], [f_term], [[]]
    for combo in itertools.product(*all_eqs_groups):
        expr = f_term
        for i, val in enumerate(combo):
            expr = expr.replace(f"T{term_keys[i]}", f"({val})")
        expr_list.append(expr)
        term_list.append(f_term)  
        components_list.append(list(combo)) 
    for abl_idx in range(n_terms):  
        for combo in itertools.product(*all_eqs_groups):
            current_comps = []
            f_term_ablated = f_term
            for i in range(n_terms):
                if i == abl_idx:
                    f_term_ablated = f_term_ablated.replace(f"T{term_keys[i]}", "(1)")
            expr = f_term
            for i, val in enumerate(combo):
                if i == abl_idx:
                    expr = expr.replace(f"T{term_keys[i]}", "(1)")
                    current_comps.append('1')  
                else:
                    expr = expr.replace(f"T{term_keys[i]}", f"({val})")
                    current_comps.append(val)  
            expr_list.append(expr)
            term_list.append(f_term_ablated)
            components_list.append(current_comps) 
    return expr_list, term_list, components_list
def expand_formula_list(f_terms, all_eqs_groups):
    """
    自动展开多个 skeleton (f_terms: list[str])。
    返回:
        all_expr_list:       展开后的所有完整表达式 (list[str])
        all_term_list:       对应的骨架 (list[str])
        all_components_list: 对应的组件列表 (list[list[str]])
                             例如: [['x', 'sin(x)'], ['y', 'sin(x)'], ...]
    """
    all_expr_list = []
    all_term_list = []
    all_components_list = []  
    for f_term in f_terms:
        term_ids = sorted(set(int(x) for x in re.findall(r'T(\d+)', f_term)))
        n_terms = len(term_ids)
        if n_terms == 0:
            all_expr_list.append(f_term)
            all_term_list.append(f_term)
            all_components_list.append([])  
            continue
        eq_groups = []
        for tid in term_ids:
            idx = tid - 1
            if idx >= len(all_eqs_groups):
                raise ValueError(f"骨架中需要 T{tid}，但 all_eqs_groups 长度只有 {len(all_eqs_groups)}")
            eq_groups.append(all_eqs_groups[idx])
        for combo in itertools.product(*eq_groups):
            expr = f_term
            for tid, val in zip(term_ids, combo):
                expr = expr.replace(f"T{tid}", f"({val})")
            all_expr_list.append(expr)
            all_term_list.append(f_term)
            all_components_list.append(list(combo)) 
    return all_expr_list, all_term_list, all_components_list
def expand_formula_list_with_permutations(f_terms, all_eqs_groups):
    """
    自动展开多个 skeleton (f_terms: list[str])，加入 Ti 的排列组合。
    参数:
        f_terms: list[str]，每个是一个骨架，如 'T1 * T2'
        all_eqs_groups: list[list[str]]，如：
                        [
                            ['x0+x1', 'x0*x1'], 
                            ['x2-x3', 'x2+1'], 
                            ['x1*x2', 'x1-x2']
                        ]
    返回:
        expr_list: 所有展开表达式
        term_list: 对应的骨架（消融时把某个Ti替换成1）
    """
    all_expr_list = []
    all_term_list = []
    for f_term in f_terms:
        term_ids = sorted(set(int(x) for x in re.findall(r'T(\d+)', f_term)))
        n_terms = len(term_ids)
        if n_terms == 0:
            all_expr_list.append(f_term)
            all_term_list.append(f_term)
            continue
        if any(tid > len(all_eqs_groups) for tid in term_ids):
            raise ValueError("Ti 超出 all_eqs_groups 范围")
        eq_groups_original = [all_eqs_groups[tid - 1] for tid in term_ids]
        for perm in itertools.permutations(term_ids):
            map_tid = dict(zip(term_ids, perm))
            f_term_perm = f_term
            for old_tid, new_tid in map_tid.items():
                f_term_perm = f_term_perm.replace(f"T{old_tid}", f"T{new_tid}")
            eq_groups = [all_eqs_groups[new_tid - 1] for new_tid in perm]
            for combo in itertools.product(*eq_groups):
                expr = f_term_perm
                for tid, val in zip(perm, combo):
                    expr = expr.replace(f"T{tid}", f"({val})")
                all_expr_list.append(expr)
                all_term_list.append(f_term_perm)
            for abl_tid in perm:
                for combo in itertools.product(*eq_groups):
                    f_term_abl = f_term_perm.replace(f"T{abl_tid}", "(1)")
                    expr = f_term_perm
                    for tid, val in zip(perm, combo):
                        if tid == abl_tid:
                            expr = expr.replace(f"T{tid}", "(1)")
                        else:
                            expr = expr.replace(f"T{tid}", f"({val})")
                    all_expr_list.append(expr)
                    all_term_list.append(f_term_abl)
    return all_expr_list, all_term_list
def load_json_f(ind_one_new_f, T_num):
    list_f = []
    fx_list_all = []
    fx_list_all1 = []
    pop_new = []
    data_f = json.loads(ind_one_new_f)
    common_keys = set(data_f.keys())
    processed_list_f = [replace_numbers(f_expr) for f_expr in list_f]
    for key in common_keys:
        try:
            fx = data_f[key]
            fx_list_all1.append(fx)
            fx = replace_numbers(fx)
            if not check_brackets(fx):
                print(f"Invalid brackets in fx: {fx}")
                continue
            if fx not in processed_list_f:
                fx = replace_C(fx)
                fx_list_all.append(fx)
        except Exception as e:
            print(f"Error processing key {key}: fx={fx}. Error: {str(e)}")
            continue
    i = 0
    for index, fx in enumerate(fx_list_all):
        try:
            b1 = gp.PrimitiveTree.from_string_sympy(fx, pset_f)
            pop_new.append([(b1,"0"), T_num, [fx_list_all1[index]] ,1e10])
            i = i + 1
        except Exception as e:
            print(f"组合失败: fx={fx}(错误: {str(e)})")
            continue
    return pop_new
def multi_constant(generated_init_pop, observations,save_path,step_i = 0):
    bool_change = False
    selected_pop = multi_processing_opt_constant_SR(list(generated_init_pop),
                                                 num_running_each_core=max(1,len(generated_init_pop) // multiprocessing.cpu_count()),
                                                 opt_const_dim=-1)
    selected_pop_str = observations.get_pop_str(selected_pop)
    selected_f_eq_str_list = [p_str[0][0] for p_str in selected_pop_str]
    selected_g_eq_str_list = [p_str[0][1] for p_str in selected_pop_str]
    selected_term_eq_str_list = [p_str[1] for p_str in selected_pop_str]
    sellected_all_components_str = [p_str[2] for p_str in selected_pop_str]
    selected_fitness_list = [p_str[3] for p_str in selected_pop_str]
    result = [selected_fitness_list[i] - selected_fitness_list[0] for i in range(1, len(selected_fitness_list))]
    if len(observations.pop) > 0:
        global_best1 = observations.pop[np.argsort([s_i[3] for s_i in observations.pop])[0]]
    else:
        global_best1 = [("0","0"),"T1",["x0"],1e10]
    observations.append(selected_f_eq_str_list, selected_g_eq_str_list, 
                        selected_term_eq_str_list, sellected_all_components_str, 
                        selected_fitness_list)
    if len(selected_pop_str) == 0:
        print("Warning: selected_pop_str is empty. No elements to process.")
        exit(-1)
    curr_best = selected_pop_str[np.argsort([s_i[3] for s_i in selected_pop_str])[0]]
    global_best = observations.pop[np.argsort([s_i[3] for s_i in observations.pop])[0]]
    if curr_best[3] < global_best1[3]:
        print()
        print("Change!Change!Change!")
        print()
        bool_change = True
    print()
    print('Step %s: curr best is %s | %s | %s | %s, global best is %s | %s | %s | %s'%(step_i,
                                                            str(sp.simplify(curr_best[0][0])),
                                                            curr_best[1],
                                                            curr_best[2],
                                                            curr_best[3],
                                                            str(sp.simplify(gp.TreeToDeadStr(global_best[0][0], converter))),
                                                            global_best[1],
                                                            global_best[2],
                                                            global_best[3]))
    with open(save_path, "a", encoding="utf-8") as fw:
        if curr_best[3] < global_best1[3]:
            fw.write(f"Change! at step {step_i}!")
            fw.write("\n") 
        fw.write(f"Step {step_i} ")
        fw.write(f"curr best is {str(sp.simplify(curr_best[0][0]))} | {str(sp.simplify(curr_best[0][1]))} | {curr_best[1]} | {curr_best[2]} | {curr_best[3]}\n")
        fw.write(f"global best is {str(sp.simplify(gp.TreeToDeadStr(global_best[0][0], converter)))} | {str(sp.simplify(gp.TreeToDeadStr(global_best[0][1], converter)))} | {global_best[1]} | {global_best[2]} | {global_best[3]}\n")
    return bool_change
def multi_constant_parallel(generated_init_pop, local_store, STORED_OBSERVATIONS, save_path,step_i,strategy_name):
    bool_change = False
    selected_pop = multi_processing_opt_constant_SR_strategy_12_31(list(generated_init_pop), strategy_name,
                                                 num_running_each_core=max(1,len(generated_init_pop) // multiprocessing.cpu_count()),
                                                 opt_const_dim=-1)
    selected_pop_str = local_store.get_pop_str(selected_pop)
    print("优化好加载完毕了！")
    selected_f_eq_str_list = [p_str[0][0] for p_str in selected_pop_str]
    selected_g_eq_str_list = [p_str[0][1] for p_str in selected_pop_str]
    selected_term_eq_str_list = [p_str[1] for p_str in selected_pop_str]
    sellected_all_components_str = [p_str[2] for p_str in selected_pop_str]
    selected_fitness_list = [p_str[3] for p_str in selected_pop_str]
    selected_strategy_name_list = [p_str[4] for p_str in selected_pop_str]
    if len(STORED_OBSERVATIONS.pop) > 0:
        global_best1 = STORED_OBSERVATIONS.pop[np.argsort([s_i[3] for s_i in STORED_OBSERVATIONS.pop])[0]]
    else:
        global_best1 = [("0","0"),"T1",["x0"],1e10,'A',10]
    print("存储开始")
    local_store.append_without_print(selected_f_eq_str_list, selected_g_eq_str_list, 
                        selected_term_eq_str_list, sellected_all_components_str, 
                        selected_fitness_list,selected_strategy_name_list)
    print("存储完毕")
    if len(selected_pop_str) == 0:
        print("Warning: selected_pop_str is empty. No elements to process.")
        exit(-1)
    curr_best = selected_pop_str[np.argsort([s_i[3] for s_i in selected_pop_str])[0]]
    if curr_best[3] < global_best1[3]:
        bool_change = True
    return bool_change, local_store
def get_unique_random_samples(population, k=4):
    """
    针对已按性能排序的 population，结合复杂度目标(30-50)挑选 k 个。
    """
    population = [s.replace("- C", "+ C").replace("-C", "C") for s in population]
    unique_population = list(dict.fromkeys(population))
    filtered_population = [s for s in unique_population if s != "1"]
    if not filtered_population:
        return []
    if len(filtered_population) <= k:
        return filtered_population
    scored_population = []
    rank_base = len(filtered_population) 
    for idx, eq in enumerate(filtered_population):
        comp = get_equation_complexity(eq)
        score_from_rank = rank_base - idx + 1
        if 15 <= comp <= 60:
            complexity_multiplier = 1.0  
        elif comp < 15:
            complexity_multiplier = comp / 15.0
        else:
            complexity_multiplier = 15.0 / comp
        final_selection_score = score_from_rank * complexity_multiplier
        scored_population.append({
            'eq': eq, 
            'comp': comp, 
            'score': final_selection_score,
            'original_rank': idx
        })
    scored_population.sort(key=lambda x: x['score'], reverse=True)
    return [item['eq'] for item in scored_population[:k]]
def dict_to_list(data: dict):
    if isinstance(data, str):
        data = json.loads(data)
    return [data["A"], data["B"]]
def load_data_SR(case_name, dim=None, add_str="",train=True,num_training_samples = None):
    """
    智能加载数据函数：
    1. 优先检查本地路径是否存在 pickle 文件。
    2. 如果本地不存在，则检查 case_name 是否在 PMLB 库中。
    3. 如果是 PMLB 数据，symbols 和 symbol_descs 设为 None。
    """
    local_data_path = f"/home/liqifei/SymBOL_Code/data/lsr_transform/test_data_on_{case_name}{add_str}.pickle"
    if os.path.exists(local_data_path):
        with open(local_data_path, "rb") as f:
            batch_data = pickle.load(f)
        batch_data = batch_data[0] 
        X0 = batch_data["X0"].numpy()
        Y = batch_data["obs_state"].numpy()
        symbols = batch_data.get("symbols", None)
        symbol_descs = batch_data.get("symbol_descs", None)
    elif case_name in dataset_names:
        X, y = fetch_data(case_name, return_X_y=True,local_cache_dir='./data/pmlb_cache')
        X0 = X                    
        Y = y.reshape(-1)        
        N_total = X0.shape[0]
        Actual_Dim = X0.shape[1]
        if dim is not None and dim != Actual_Dim:
            Dim = Actual_Dim
        else:
            Dim = Actual_Dim
        rng = np.random.default_rng(seed=1069668)
        N_total = X0.shape[0]
        N_train = int(0.75 * N_total)
        indices = rng.permutation(N_total)
        train_idx = indices[:N_train]
        test_idx = indices[N_train:]
        if train:
            X0 = X0[train_idx]
            Y = Y[train_idx]
            N = N_train
        else:
            X0 = X0[test_idx]
            Y = Y[test_idx]
            N = N_total - N_train
        symbols = None
        symbol_descs = None
    elif case_name in ["SSN"]: 
        file_path = "/home/liqifei/PhysicsRegression-main/data/physics_data/SN_m_tot_V2.0.txt"
        with open(file_path, "r") as fi:
            context = fi.read().split("\n")
        context = [c.split() for c in context][:-1]
        times = np.array([float(c[2]) for c in context])
        nums = np.array([float(c[3]) for c in context])
        c_times = times[1280: 2730]
        c_nums = nums[1280: 2730]
        min_times1 = np.min(c_times)
        max_times1 = np.max(c_times)
        c_times = (c_times - min_times1) / (max_times1 - min_times1) * 8
        X_all = c_times.reshape((-1, 1))
        Y_all = c_nums.reshape(-1)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, Y_all, test_size=0.15, random_state=42
        )
        if train:
            X0 = X_train
            Y = y_train
        else:
            X0 = X_test
            Y = y_test
        N = X0.shape[0]
        symbols = None
        symbol_descs = None
    elif case_name in ["SolarRot"]: 
        file_path = "./data/physics_data/solarDifferentialRotation.csv"
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            df = pd.read_csv("/home/liqifei/PhysicsRegression-main/data/physics_data/solarDifferentialRotation.csv")
        latitude = np.array(list(df["Latitude"])) / 180 * np.pi + np.pi / 2
        speed = np.array(list(df["Two-day_Lag_mean"]))
        X_all = latitude.reshape((-1, 1))
        Y_all = speed.reshape(-1) 
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, Y_all, test_size=0.2, random_state=0
        )
        if train:
            X0 = X_train
            Y = y_train
            if num_training_samples is not None and num_training_samples < X0.shape[0]:
                X0 = X0[:num_training_samples]
                Y = Y[:num_training_samples]
                add_str = f"{add_str}_samples={num_training_samples}"
        else:
            X0 = X_test
            Y = y_test
        N = X0.shape[0]
        symbols = None
        symbol_descs = None
    elif case_name.startswith("ego") and "preceeding" in case_name:
        case_name_list, X0, Y, Dim, N, symbols, symbol_descs =  load_data_SR_all_following_cars()
    elif case_name in ["fish_15","fish_30","fish_60"]:
        case_name_list, X0, Y, Dim, N = load_fish_data_ode(case_name, dim)
        case_name_list, X0, Y, Dim, N = load_fish_data_pydaddy(case_name, dim)
        symbols = None
        symbol_descs = None
    elif case_name in ["cell_15"]:
        case_name_list, X0, Y, Dim, N = load_cell_data_pydaddy(case_name, dim)
        symbols = None
        symbol_descs = None
    elif case_name in ["lichess"]:
        case_name_list, X0, Y, Dim, N = load_data_SR_from_csv()
        symbols = None
        symbol_descs = None
    else:
        raise FileNotFoundError(f"无法加载数据: '{case_name}' \n"
                                f"1. 本地路径不存在: {local_data_path}\n")
    N = X0.shape[0]
    return case_name, X0, Y, Dim, N, symbols, symbol_descs
def input_data_num(case_name: str, dim: int, data_num : int) -> str:
    """加载并处理输入数据，返回前data_num条数据供LLM分析规律，X为一个二维数组，[data_num][dim]对应了数据数量和维度，Y为一维数组，对应了因变量的值."""
    try:
        case_name_list, X0, Y, Dim, N, symbols, symbol_descs = load_data_SR(case_name, dim, add_str="")
        rng = np.random.default_rng() 
        real_sample_num = min(data_num, len(X0))
        idx = rng.choice(len(X0), size=real_sample_num, replace=False)
        X0_sample = X0[idx]
        Y_sample = Y[idx]
        dim = X0_sample.shape[1]
        formatted_lines = []
        for i in range(real_sample_num):
            x_strs = [f"x{j} = {X0_sample[i, j]}" for j in range(dim)]
            x_combined = ", ".join(x_strs)
            y_val = Y_sample[i]
            if hasattr(y_val, 'item'): 
                y_val = y_val.item()
            y_str = f"y = {y_val}"
            formatted_lines.append(f"{x_combined}, {y_str}")
        return "\n".join(formatted_lines)
    except Exception as e:
        return f"数据处理失败: {str(e)}"
@tool("Input_data")
def Input_data(case_name: str, dim: int, data_num : int) -> str:
    """加载并处理输入数据，返回前data_num条数据供LLM分析规律，X为一个二维数组，[data_num][dim]对应了数据数量和维度，Y为一维数组，对应了因变量的值."""
    try:
        case_name_list, X0, Y, Dim, N, symbols, symbol_descs = load_data_SR(case_name, dim, add_str="")
        X0_sample = X0[:data_num]
        Y_sample = Y[:data_num]
        dim = X0_sample.shape[1]
        return (
            f"前{data_num}条x的值为:\n{X0_sample}\n\n"
            f"前{data_num}条y的值为:\n{Y_sample}"
        )
    except Exception as e:
        return f"数据处理失败: {str(e)}"
def interpret_results(results, loss, dim, top_k=5):
    interpretations = []
    for r in results:
        acc = (1 - r['loss']) * (1 - loss)
        method = r['method']
        i = r['i'] 
        j = r['j'] 
        if method.startswith("symmetry_multiply"):
            desc = f"x{i[0]} 和 x{j[0]} 存在乘法关系"
        elif method.startswith("symmetry_divide"):
            desc = f"x{i[0]} 和 x{j[0]} 存在除法关系"
        elif method.startswith("symmetry_minus"):
            desc = f"x{i[0]} 和 x{j[0]} 存在减法关系"
        elif method.startswith("symmetry_plus"):
            desc = f"x{i[0]} 和 x{j[0]} 存在加法关系"
        else:
            desc = f"{method}: 未知关系, i={i}, j={j}"
        used_indices = set()
        group_structure = []
        combined_term = None
        if method.startswith("symmetry_multiply"):
            combined_term = f"x{i[0]}*x{j[0]}"
            used_indices.update([i[0], j[0]])
        elif method.startswith("symmetry_divide"):
            combined_term = f"x{i[0]}/x{j[0]}"
            used_indices.update([i[0], j[0]])
        elif method.startswith("symmetry_minus"):
            combined_term = f"x{i[0]}-x{j[0]}"
            used_indices.update([i[0], j[0]])
        elif method.startswith("symmetry_plus"):
            combined_term = f"x{i[0]}+x{j[0]}"
            used_indices.update([i[0], j[0]])
        if combined_term:
            group_structure.append([combined_term])
        for k in range(dim):
            if k not in used_indices:
                group_structure.append([f"x{k}"])
        interpretations.append({
            "method": method,
            "acc": acc,
            "desc": desc,
            "group": group_structure 
        })
    top_results = interpretations[:top_k]
    descriptions = [r["desc"] for r in top_results]
    groups = [r["group"] for r in top_results]
    return descriptions, groups[0]
def Input_data_dim(case_name: str, dim: int, add_str: str = "") -> str:
    """加载并处理输入数据，调用AI-Feynman工具进行神经网络训练，返回降维后的结果"""
    case_name_list, X0, Y, Dim, N, symbols, symbol_descs = load_data_SR(case_name, dim)
    Y = Y.reshape(-1, 1)
    data = np.hstack([X0, Y]) 
    try:
        os.makedirs(f"/home/liqifei/ai_agent_local_llm/data/{case_name}", exist_ok=True)
        save_path = os.path.join(f"/home/liqifei/ai_agent_local_llm/data/{case_name}", f"{case_name}.txt")
        np.savetxt(save_path, data, fmt="%.6f")
        print(f"保存成功: {save_path}")
    except Exception as e:
        print("保存失败:", e)
    results , loss = run_aifeynman(f"/home/liqifei/ai_agent_local_llm/data/{case_name}/", f"{case_name}.txt", NN_epochs=500)
    top_interpretations = interpret_results(results, loss, dim, top_k=1)
    return top_interpretations
def extract_operators(eq_list):
    func_ops = ["sin", "cos", "tan", "sqrt", "log", "exp"]
    symbol_ops = [r"\*\*", r"\*", r"/", r"\+", r"-", r"\^"]
    found_ops = set()
    for eq in eq_list:
        for f in func_ops:
            if re.search(rf"\b{f}\b", eq):
                found_ops.add(f)
        for s in symbol_ops:
            if re.search(s, eq):
                if s == r"\*\*":
                    found_ops.add("^")
                else:
                    found_ops.add(s.replace("\\", ""))  
    return list(found_ops)
def extract_top_operators(eq_list, top_k=5):
    func_ops = ["sin", "cos", "tan", "sqrt", "log", "exp", "abs"]
    symbol_ops = [r"\*\*", r"\^", r"\*", r"/", r"\+", r"-"]
    func_pattern = r"\b(?:" + "|".join(func_ops) + r")\b"
    symbol_pattern = "|".join(symbol_ops)
    combined_pattern = f"{func_pattern}|{symbol_pattern}"
    total_counter = Counter()
    for eq in eq_list:
        matches = re.findall(combined_pattern, eq)
        for token in matches:
            if token == "**":
                token = "^"
            total_counter[token] += 1
    return [op for op, count in total_counter.most_common(top_k)]
def Init_Pysr(case_name: str, dim: int, add_str: str = "") -> str:
    """加载并数据，根据现有的数据调用Pysr进行初步拟合，为之后LLM进一步优化提供一个初始解."""
    case_name_list, X0, Y, Dim, N, symbols, symbol_descs = load_data_SR(case_name, dim, add_str)
    custom_loss = """
function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    return sum( (1000 .* (prediction .- dataset.y) ) .^ 2) / dataset.n
end
"""
    model = PySRRegressor(
        maxsize=20,
        niterations=15,  
        ncycles_per_iteration=30,
        binary_operators=["+", "*", "-", "/", "^"],
        unary_operators=[
            "sin",
            "cos",
            "exp",
            "sqrt",
            "log",
            "tan",
            "inv(x) = 1/x",
        ],
        verbosity=1,
        constraints={"sin": 10,"cos": 10,"exp": 20,"log": 20,"sqrt": 20,"pow": (-1, 20),"^":(-2,2)},
        nested_constraints={
            "sin": {"sin": 0, "cos": 0},
            "cos": {"sin": 0, "cos": 0},
            "exp": {"exp": 0, "log": 0},
            "log": {"exp": 0, "log": 0},
            "sqrt": {"sqrt": 0},
        },
        progress=True,
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        early_stop_condition=f"f(loss, complexity) = (loss < 1e-6)"
    )
    model.fit(X0, Y)
    eq_df = model.equations_['sympy_format']
    eq_df = model.equations_
    top5 = eq_df.sort_values(by="loss", ascending=True).head(5)
    best5_eqs = top5["sympy_format"].tolist()
    best5_eqs_str = [str(eq) for eq in best5_eqs]
    list_f = [replace_C_to_one(replace_numbers(p_str)) for p_str in best5_eqs_str]
    all_eqs_str = [replace_C(replace_numbers(str(eq))) for eq in eq_df["sympy_format"].tolist()]
    pop_new = []
    for b1 in all_eqs_str:
        b1 = gp.PrimitiveTree.from_string_sympy(b1, pset_f)
        pop_new.append([(b1, "0"), 1e10])
    return list_f, pop_new, extract_top_operators(best5_eqs_str), symbols, symbol_descs
def aaa(str_input):
    if not isinstance(str_input, str):
        str_input = str(str_input)  
    pop_new = []
    b1 = gp.PrimitiveTree.from_string_sympy(str_input, pset_f)
    pop_new.append([(b1, "0"), 1e10])
    return pop_new
def bbb(str_input):
    if not isinstance(str_input, str):
        str_input = str(str_input)  
    pop_new = True
    try:
        gp.PrimitiveTree.from_string_sympy(str_input, pset_f)
    except:
        print("form erro!")
        pop_new = False
    return pop_new
@tool("Init_prompts")
def Init_prompts(case_name: str, dim: int, data_num : int, add_str: str = "") -> str:
    """加载并处理输入数据，返回前data_num条数据供LLM分析规律，X为一个二维数组，[data_num][dim]对应了数据数量和维度，Y为一维数组，对应了因变量的值."""
    try:
        case_name_list, X0, Y, Dim, N, symbols, symbol_descs = load_data_SR(case_name, dim, add_str)
        X0_sample = X0[:data_num]
        Y_sample = Y[:data_num]
        dim = X0_sample.shape[1]
        return (
            f"X的维度为:{dim}\n\n"
            f"前{data_num}条X的值为:\n{X0_sample}\n\n"
            f"前{data_num}条Y的值为:\n{Y_sample}"
        )
    except Exception as e:
        return f"数据处理失败: {str(e)}"
@tool("Make_Prompt_From_Tools")
def Make_Prompt_From_Tools(case_name: str, dim: int, data_num : int, data_analysis_result: str) -> str:
    """
    接收 data_analysis_result 和 pysr_result，
    构造符合要求的提示词，返回字符串。
    """
    pysr_result = Init_Pysr(case_name, dim, data_num)
    prompt = f"""
Below, I will provide you with a list that contains the best equations currently in the population:
{pysr_result}.
To generate equations that are different from those in the given lists (with different operator or variable combinations) and ensure they are not completely identical to any existing equation!
The variables in the equations : [x0, x1]. It is forbidden to generate any variables other than the specified ones.
Operators list : [+, -, *, /, ^, ^(0.5), exp, sqrt, sin, cos, log] (The operators for the equations can only be chosen from the operators list.)
Replace constants with the number 1.
Please analyze the following generation requirements step by step:  
    1. Operator Diversity: Each equation must use at least 3 different operators (chosen from Operators list). Generate an equation no longer than 70 characters!
    2. The generated equations need to satisfy the structure of rational functions, and they should be obtained through the addition, subtraction, multiplication, and division of polynomials.
       Keep the polynomial in factored form. 
    3. Fraction Structure Specifications ("/"): The numerator and denominator must be enclosed in parentheses, and both the numerator and denominator must be in the form of a product or sum of polynomials
{data_analysis_result}
You don't need to output anything extra. Use your imagination as much as possible! Generated equations must be diverse, fully exploring combinations of variables and operators, as well as all possibilities! Please list 5 equations in the following JSON list format:
"""
    return prompt.strip()
def mutate(equations,variables,operators,num):
    def build_transcendental_ops(operator_list):
        unary_map = {
                "sqrt": sqrt,
                "sin": sin,
                "cos": cos,
                "exp": exp,
                "log": log,
                "tan": tan,
                "tanh": tanh,
            }
        return [unary_map[op] for op in operator_list if op in unary_map]
    def parse_operator_list(op_string_or_list):
        if isinstance(op_string_or_list, list):
            return op_string_or_list
        if isinstance(op_string_or_list, str):
            ops = op_string_or_list.strip("[]").replace(" ", "").split(",")
            return [op for op in ops if op]
        raise ValueError("Unsupported operator format")
    operators = parse_operator_list(operators)
    operators = build_transcendental_ops(operators)
    class MutationOptions:
        """
        配置类
        """
        def __init__(self):
            self.maxsize = 25  
            self.maxdepth = 12
            self.alpha = 0.1
            self.temperature = 1.0
            self.deterministic = False
            self.mutation_weights = np.array([0.0, 2.0, 2.0, 3.0, 2.0, 0.05, 1.0, 1.5])
    class PySRMutator:
        def __init__(self, variables, options=None):
            self.variables = variables
            self.symbols = symbols(variables, real=True)
            self.options = options if options else MutationOptions()
            self.binary_ops = [Add, Mul, lambda x, y: x / y, lambda x, y: x - y, Pow]
            self.transcendental_ops = operators
        def _compute_complexity(self, expr):
            return sum(1 for _ in preorder_traversal(expr))
        def _count_depth(self, expr):
            if not expr.args: return 1
            return 1 + max(self._count_depth(arg) for arg in expr.args)
        def _check_constraints(self, expr):
            if expr == sp.nan or expr.has(sp.zoo) or expr.has(sp.nan):
                return False
            if expr.has(I):
                return False
            if expr.is_real is False:
                return False
            size = self._compute_complexity(expr)
            depth = self._count_depth(expr)
            if size > self.options.maxsize: return False
            if depth > self.options.maxdepth: return False
            return True
        def _get_random_leaf(self):
            if random.random() < 0.95:
                return random.choice(self.symbols)
            else:
                return Float(random.uniform(-2, 2), 2)
        def _get_random_exponent(self):
            """
            生成“看起来像物理公式”的指数。
            大概率是整数 (1, 2, 3) 或简单分数 (0.5, -1, -2)。
            """
            choices = [
                Integer(2), Integer(3),  
                Integer(-1), Integer(-2),  
                Float(0.5, 2),  
                Integer(1)  
            ]
            return random.choice(choices)
        def _mutate_operator(self, expr):
            """变异算子: + 变 *, / 变 - 等"""
            nodes = [n for n in preorder_traversal(expr) if n.args and not n.is_Number]
            if not nodes: return expr
            target = random.choice(nodes)
            arity = len(target.args)
            if arity == 2 and self.binary_ops:
                new_op = random.choice(self.binary_ops)
                if new_op == Pow:
                    base = target.args[0]
                    exponent = self._get_random_exponent()
                    new_node = Pow(base, exponent)
                    return expr.subs(target, new_node)
                try:
                    return expr.subs(target, new_op(*target.args))
                except:
                    pass
            return expr
        def _append_prepend_op(self, expr):
            """在根节点处扩展"""
            if not self.binary_ops: return expr
            op = random.choice(self.binary_ops)
            if op == Pow:
                exponent = self._get_random_exponent()
                return Pow(expr, exponent)
            new_leaf = self._get_random_leaf()
            if random.random() < 0.5:
                return op(expr, new_leaf)
            else:
                return op(new_leaf, expr)
        def _insert_op(self, expr):
            """内部插入节点"""
            nodes = list(preorder_traversal(expr))
            target = random.choice(nodes)
            if not self.binary_ops: return expr
            op = random.choice(self.binary_ops)
            if op == Pow:
                exponent = self._get_random_exponent()
                new_subtree = Pow(target, exponent)
                return expr.subs(target, new_subtree)
            new_leaf = self._get_random_leaf()
            if random.random() < 0.5:
                new_subtree = op(target, new_leaf)
            else:
                new_subtree = op(new_leaf, target)
            return expr.subs(target, new_subtree)
        def _delete_op(self, expr):
            """删除子树"""
            nodes = list(preorder_traversal(expr))
            if len(nodes) <= 1: return expr
            target = random.choice(nodes[1:])  
            if target.args:
                new_sub = random.choice(target.args)
            else:
                new_sub = self._get_random_leaf()
            return expr.subs(target, new_sub)
        def _simplify(self, expr):
            return simplify(expr)
        def _regenerate(self, expr):
            op = random.choice(self.binary_ops)
            if op == Pow:
                return Pow(self._get_random_leaf(), self._get_random_exponent())
            return op(self._get_random_leaf(), self._get_random_leaf())
        def _wrap_transcendental_op(self, expr):
            """
            变异逻辑：
            1. 随机选择一个超越函数 (sin, exp, log...)
            2. 随机决定是包裹整个表达式，还是寻找一个分母进行包裹。
            """
            op = random.choice(self.transcendental_ops)
            if random.random() < 0.65:
                return op(expr)
            denominators = [n for n in preorder_traversal(expr) if n.is_Pow and n.exp.is_negative]
            if denominators:
                target = random.choice(denominators)
                base = target.base
                exponent = target.exp
                new_term = Pow(op(base), exponent)
                return expr.subs(target, new_term)
            else:
                return op(expr)
        def next_generation(self, expr_str):
            try:
                prev = sympify(expr_str)
            except:
                return expr_str, "PARSE_ERROR"
            cur_weights = self.options.mutation_weights.copy()
            if not self.transcendental_ops:
                cur_weights[7] = 0.0  
            n = self._compute_complexity(prev)
            depth = self._count_depth(prev)
            if n >= self.options.maxsize or depth >= self.options.maxdepth:
                cur_weights[2] = 0.0  
                cur_weights[3] = 0.0  
                cur_weights[7] = 0.0  
            weight_sum = np.sum(cur_weights)
            if weight_sum == 0:
                return expr_str, "NO_VALID_MUTATION"
            prob_weights = cur_weights / weight_sum
            successful_mutation = False
            attempts = 0
            max_attempts = 15  
            mutated_expr = prev
            mutation_type = ""
            while not successful_mutation and attempts < max_attempts:
                choice = np.random.choice(len(cur_weights), p=prob_weights)
                temp_expr = prev
                if choice == 0:
                    pass
                elif choice == 1:
                    temp_expr = self._mutate_operator(prev)
                    mutation_type = "OPERATOR"
                elif choice == 2:
                    temp_expr = self._append_prepend_op(prev)
                    mutation_type = "APPEND/PREPEND"
                elif choice == 3:
                    temp_expr = self._insert_op(prev)
                    mutation_type = "INSERT"
                elif choice == 4:
                    temp_expr = self._delete_op(prev)
                    mutation_type = "DELETE"
                elif choice == 5:
                    temp_expr = self._simplify(prev)
                    mutation_type = "SIMPLIFY"
                    successful_mutation = True
                elif choice == 6:
                    temp_expr = self._regenerate(prev)
                    mutation_type = "REGENERATE"
                elif choice == 7:
                    temp_expr = self._wrap_transcendental_op(prev)
                    mutation_type = "WRAP_TRANSCENDENTAL"
                if self._check_constraints(temp_expr):
                    if temp_expr != prev:
                        successful_mutation = True
                        mutated_expr = temp_expr
                attempts += 1
            if not successful_mutation:
                return str(prev).replace(" ", ""), "NO_CHANGE"
            return str(mutated_expr).replace(" ", ""), mutation_type
    mutator = PySRMutator(variables)
    results = []
    for eq in equations:
        for _ in range(num):  
            new_eq, m_type = mutator.next_generation(eq)
            results.append(new_eq)
    return results