import ast
import glob
import os
from sklearn.model_selection import train_test_split
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import copy
import multiprocessing
import threading
import torch
import warnings
warnings.filterwarnings("ignore")
import math
import random
import time
from functools import partial
import numpy as np
import sympy as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from tqdm import tqdm
import func_timeout
from func_timeout import func_set_timeout
import pandas as pd
import csv
import pickle
import src.ai_agent_local_llm.tools.search_2nd_phase.gp_tools as gp
import sympytorch
import signal
import queue 
import argparse
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import numpy as np
from pmlb import fetch_data
def get_parser():
    parser = argparse.ArgumentParser(description='Search_for_2ndPhase')
    parser.add_argument('--case_name', help='case_name', default="lichess") 
    parser.add_argument('--dim', help='dim, default:3', default=3)
    parser.add_argument('--opt_flag', help='opt_flag, default:0', default=0)
    parser.add_argument('--add_str', help='add_str, default:""', default="")
    return parser
args = get_parser().parse_args()
case_name = args.case_name
dim = int(args.dim)
opt_flag = int(args.opt_flag)
add_str = args.add_str
class TimeoutException(Exception):
    pass
def handler(signum, frame):
    raise TimeoutException()
signal.signal(signal.SIGALRM, handler)
def load_data_SR(case_name, dim, add_str=""):
    test_data_set_path = 'data/id/test_data_on_%s%s.pickle' % (case_name, add_str)
    with open(test_data_set_path, 'rb') as f:
        batch_data = pickle.load(f)
    batch_data = batch_data[0]
    case_name_list = ["%s_dim=%s" % (case_name, i) for i in range(dim)]
    X0 = batch_data['X0'].numpy()
    Y = batch_data['obs_state'].numpy()
    noise_ratio = 0  
    Y = Y + noise_ratio * np.std(Y, axis=0, keepdims=True) * np.random.randn(*Y.shape)
    Dim = X0.shape[1]
    N = X0.shape[0]
    return case_name_list, X0, Y, Dim, N
def load_data_SR_from_csv(csv_path="data/pure_empirical_density.csv"):
    """
    从 CSV 加载阵发性概率密度数据，并输出为适配符号回归(SR)框架的张量格式。
    参数:
        csv_path (str): CSV 数据集的路径。
    返回:
        case_name_list (list): 变量名称列表。
        X (np.ndarray): 自变量矩阵 (tau_X)，形状为 (N, Dim)，即 (样本数, 1)。
        Y (np.ndarray): 因变量向量 (probability_density_Y)，形状为 (N,)。
        Dim (int): 特征维度 (此处固定为 1)。
        N (int): 样本数量 (即对数分箱后的数据行数)。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}。请确保文件路径正确。")
    df = pd.read_csv(csv_path)
    df = df.head(100)  
    X = df[['tau_X']].values
    Y = df['probability_density_Y'].values
    N = X.shape[0]       
    Dim = X.shape[1]     
    case_name_list = ["tau_X"]
    return case_name_list, X, Y, Dim, N
def load_data(case_name, dim, add_str=""):
    test_data_set_path = 'data/test_data_on_%s%s.pickle' % (case_name, add_str)
    with open(test_data_set_path, 'rb') as f:
        batch_data = pickle.load(f)
    batch_data = batch_data[1]
    case_name_list = ["%s_dim=%s"%(case_name, i) for i in range(dim)]
    sparse_A = np.array(batch_data['adj'].numpy(), dtype=int)
    X0 = batch_data['X0'].numpy()[2:-2,:]
    t_range = batch_data['t_target'].numpy()
    Mask = batch_data['obs_mask'].numpy()[2:-2,:,:]
    Y = batch_data['obs_state'].numpy()
    time_steps_to_keep = 60
    t_range = t_range[:time_steps_to_keep]
    Mask = Mask[:time_steps_to_keep][2:-2,:,:]
    Y = Y[:time_steps_to_keep]
    Y, diff_Y = gp.finite_difference(Y, 0.001)
    Dim = X0.shape[1]
    N = X0.shape[0]
    return case_name_list, sparse_A, X0, t_range, Mask, Y, diff_Y, Dim, N
def load_data_discrete(case_name, dim, add_str=""):
    test_data_set_path = 'data/test_data_on_%s%s.pickle' % (case_name, add_str)
    with open(test_data_set_path, 'rb') as f:
        batch_data = pickle.load(f)
    case_name_list = ["%s_dim=%s"%(case_name, i) for i in range(dim)]
    sparse_A = np.array(batch_data['adj'].numpy(), dtype=float)
    X0 = batch_data['X0'].numpy()
    t_range = batch_data['t_target'].numpy()
    Mask = batch_data['obs_mask'].numpy()
    Y = batch_data['obs_state'].numpy()
    Y1, diff_Y = gp.finite_difference(Y, 1)
    Dim = X0.shape[1]
    N = X0.shape[0]
    period = 45
    limit = 100
    country_select_index = []
    time_select_index = []
    for country_index in range(Y.shape[1]):
        first_no0_index = (np.nonzero(Y[:, country_index])[0])[0]
        try:
            if len(Y[first_no0_index:, country_index]) >= period and (Y[first_no0_index:, country_index])[
                period - 1] >= limit:
                country_select_index.append(country_index)
                time_select_index.append([first_no0_index, first_no0_index + period - 1])
        except Exception as e:
            continue
    X0 = np.zeros(Y.shape[1], dtype=np.float32)
    for idx, country_index in enumerate(country_select_index):
        start_time = time_select_index[idx][0]
        X0[country_index] = Y[start_time, country_index]
    X0 = X0.reshape(-1, 1)  
    data_list = []
    for idx, (country, (start, end)) in enumerate(zip(country_select_index, time_select_index)):
        data_segment = Y[start:end + 1, country]  
        data_list.append(data_segment.reshape(-1, 1))  
    true = np.stack(data_list, axis=1)  
    return case_name_list, sparse_A, X0, t_range, Mask, Y, diff_Y, Dim, N,true[:, :9], country_select_index[:9], time_select_index
def load_data_SR_following_car(case_name, dim, add_str=""):
    """
    加载 NGSIM 数据，并返回 SR 任务所需的输入、输出及物理含义描述。
    返回:
        case_name_list: 特征名称列表 (简写+符号)
        X0: 输入状态矩阵 (numpy array)
        Y:  目标加速度 (numpy array)
        Dim: 特征数量
        N:   样本数量
        symbols: 数学符号列表 (例如 ['s', 'v', 'dv'])
        symbol_descs: 详细物理含义描述列表
    """
    file_path = '/data/NGSIM_dataNpy/trajectories-0750am-0805am/%s.npy' % (case_name)
    filter_dt = 0.5  
    raw_dt = 0.1     
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    data_dict = np.load(file_path, allow_pickle=True).item()
    ids = list(data_dict.keys())
    if data_dict[ids[0]]['x'][-1] > data_dict[ids[1]]['x'][-1]:
        lead_id, follow_id = ids[0], ids[1]
    else:
        lead_id, follow_id = ids[1], ids[0]
    car0_data = data_dict[lead_id]   
    car1_data = data_dict[follow_id] 
    preceeding_length = car0_data['veh'][0]
    s_raw = car0_data['x'] - car1_data['x'] - preceeding_length 
    v_raw = car1_data['v']                                      
    dv_raw = car1_data['v'] - car0_data['v']                    
    filter_length = int(filter_dt / raw_dt)
    length = int(s_raw.shape[0] / filter_length)
    valid_len = length * filter_length
    def downsample(arr):
        return np.mean(arr[:valid_len].reshape(-1, filter_length), axis=1)
    s_smooth = downsample(s_raw)
    v_smooth = downsample(v_raw)
    dv_smooth = downsample(dv_raw)
    a_follow = (v_smooth[1:] - v_smooth[:-1]) / filter_dt
    v0_val = 26.8  
    s0_val = 1.6   
    T_val  = 1.1   
    n_samples = s_smooth.shape[0] - 1
    v0_col = np.full((n_samples, ), v0_val)
    s0_col = np.full((n_samples, ), s0_val)
    T_col  = np.full((n_samples, ), T_val)
    X_full = np.stack((
        s_smooth[:-1], 
        v_smooth[:-1], 
        dv_smooth[:-1], 
        v0_col, 
        s0_col, 
        T_col
    ), axis=1)
    Y = a_follow[:, np.newaxis].reshape(-1)
    full_symbols = ["a", "s", "v", "dv", "v0", "s0", "T"]
    full_symbol_descs = [
        "Acceleration of ego vehicle (m/s^2)",          
        "Gap distance to leading vehicle (m)",          
        "Velocity of ego vehicle (m/s)",                
        "Relative velocity (v_ego - v_lead) (m/s)",     
        "Desired velocity (m/s)",                       
        "Minimum gap at standstill (m)",                
        "Safe time headway (s)"                         
    ]
    full_case_names = [
        "Acceleration(a)", 
        "Gap(s)", 
        "Velocity(v)", 
        "Delta_V(dv)", 
        "Desired_Vel(v0)", 
        "Min_Gap(s0)", 
        "Time_Headway(T)"
    ]
    X0 = X_full
    Dim = X0.shape[1]
    symbols = full_symbols
    symbol_descs = full_symbol_descs
    case_name_list = full_case_names
    N = X0.shape[0]
    print(X0)
    print(X0.shape)
    return case_name_list, X0, Y, Dim, N, symbols, symbol_descs
def load_data_SR_all_following_cars(
    file_folder_list=[
        "data/NGSIM_dataNpy/trajectories-0750am-0805am",
        "data/NGSIM_dataNpy/trajectories-0805am-0820am"
        "data/NGSIM_dataNpy/trajectories-0820am-0835am"
    ],
    need_first_K=-1,  
    dim=3
):
    """
    批量加载指定文件夹列表下的 NGSIM 跟车数据，并拼接。
    参数:
        file_folder_list: 包含 .npy 文件的文件夹路径列表
        need_first_K: 每个文件夹下只读取前 K 个文件 (-1 代表读取所有)
        dim: 特征维度 (默认为3，对应 s, v, dv)
    """
    filter_dt = 0.5  
    raw_dt = 0.1     
    X_collector = []
    Y_collector = []
    total_files_processed = 0
    for folder_path in file_folder_list:
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在，跳过 - {folder_path}")
            continue
        all_files = glob.glob(os.path.join(folder_path, "*.npy"))
        if need_first_K != -1:
            target_files = all_files[:need_first_K]
        else:
            target_files = all_files
        print(f"正在处理文件夹: {os.path.basename(folder_path)} (选中 {len(target_files)}/{len(all_files)} 个文件)")
        for file_path in target_files:
            try:
                data_dict = np.load(file_path, allow_pickle=True).item()
                keys = list(data_dict.keys())
                keys.sort(key=lambda x: int(x) if x.isdigit() else x)
                if len(keys) < 2:
                    continue
                for i in range(len(keys) - 1):
                    id_a = keys[i]
                    id_b = keys[i+1]
                    pos_a = data_dict[id_a]['x'][-1]
                    pos_b = data_dict[id_b]['x'][-1]
                    if pos_a > pos_b:
                        lead_id, follow_id = id_a, id_b
                    else:
                        lead_id, follow_id = id_b, id_a
                    car0_data = data_dict[lead_id]   
                    car1_data = data_dict[follow_id] 
                    preceeding_length = car0_data['veh'][0]
                    s_raw = car0_data['x'] - car1_data['x'] - preceeding_length
                    v_raw = car1_data['v']
                    dv_raw = car1_data['v'] - car0_data['v']
                    filter_length = int(filter_dt / raw_dt)
                    length = int(s_raw.shape[0] / filter_length)
                    valid_len = length * filter_length
                    def downsample(arr):
                        return np.mean(arr[:valid_len].reshape(-1, filter_length), axis=1)
                    s_smooth = downsample(s_raw)
                    v_smooth = downsample(v_raw)
                    dv_smooth = downsample(dv_raw)
                    a_follow = (v_smooth[1:] - v_smooth[:-1]) / filter_dt
                    v0_val = 20  
                    s0_val = 1.6   
                    T_val  = 1.1   
                    n_samples = s_smooth.shape[0] - 1
                    v0_col = np.full((n_samples, ), v0_val)
                    s0_col = np.full((n_samples, ), s0_val)
                    T_col  = np.full((n_samples, ), T_val)
                    X_partial = np.stack((
                        s_smooth[:-1], 
                        v_smooth[:-1], 
                        dv_smooth[:-1], 
                        v0_col, 
                        s0_col, 
                        T_col
                    ), axis=1)
                    Y = a_follow[:, np.newaxis].reshape(-1)
                    X_collector.append(X_partial)
                    Y_collector.append(a_follow)
                total_files_processed += 1
            except Exception as e:
                print(f"Error processing {os.path.basename(file_path)}: {e}")
                continue
    if len(X_collector) == 0:
        print("未提取到任何数据，请检查路径或参数。")
        return [], np.array([]), np.array([]), 0, 0, [], []
    X0 = np.concatenate(X_collector, axis=0) 
    Y = np.concatenate(Y_collector, axis=0)  
    Y = Y.reshape(-1) 
    full_symbols = ["a", "s", "v", "dv", "v0", "s0", "T"]
    full_symbol_descs = [
        "Acceleration of ego vehicle (m/s^2)",          
        "Gap distance to leading vehicle (m)",          
        "Velocity of ego vehicle (m/s)",                
        "Relative velocity (v_ego - v_lead) (m/s)",     
        "Desired velocity (m/s)",                       
        "Minimum gap at standstill (m)",                
        "Safe time headway (s)"                         
    ]
    full_case_names = [
        "Acceleration(a)", 
        "Gap(s)", 
        "Velocity(v)", 
        "Delta_V(dv)", 
        "Desired_Vel(v0)", 
        "Min_Gap(s0)", 
        "Time_Headway(T)"
    ]
    symbols = full_symbols
    symbol_descs = full_symbol_descs
    case_name_list = full_case_names
    Dim = X0.shape[1]
    N = X0.shape[0]
    return case_name_list, X0, Y, Dim, N, symbols, symbol_descs
print("case_name:", case_name)
if case_name.startswith("ego") and "preceeding" in case_name:
    case_name_list, X0, Y, Dim, N, symbols, symbol_descs = load_data_SR_all_following_cars()
elif case_name in ["lichess"]:
    case_name_list, X0, Y, Dim, N = load_data_SR_from_csv()
else:
    case_name_list, X0, Y, Dim, N = load_data_SR(case_name, dim)
min_ = 1
max_ = 5
tol_err = 5e-3   
termination_condition_err = 1e-4
timeout_second = 30
num_choose = 20
min_const = -3.0
max_const = 3.0
sampling_const = partial(random.uniform, min_const, max_const)
def build_psets(dim=1):
    real_pow = lambda x, y: np.where(
        (x < 0) & (np.floor(y) != y), np.nan, np.power(x, y)
    )
    pset_f = gp.PrimitiveSet("F", dim, prefix="x")
    pset_f.addPrimitive(np.add, 2, name="Add")
    pset_f.addPrimitive(np.subtract, 2, name="Sub")
    pset_f.addPrimitive(np.multiply, 2, name="Mul")
    pset_f.addPrimitive(np.divide, 2, name="Div")
    pset_f.addPrimitive(np.exp, 1, name="exp")
    pset_f.addPrimitive(np.sin, 1, name="sin")
    pset_f.addPrimitive(np.cos, 1, name="cos")
    pset_f.addPrimitive(np.tan, 1, name="tan")
    pset_f.addPrimitive(np.abs, 1, name="Abs")
    pset_f.addPrimitive(np.log, 1, name="log")
    pset_f.addPrimitive(real_pow, 2, name="Pow")  
    pset_f.addPrimitive(np.tanh, 1, name="tanh")
    pset_f.addPrimitive(np.sinh, 1, name="sinh")
    pset_f.addPrimitive(np.cosh, 1, name="cosh")
    pset_f.addPrimitive(lambda x: 1 / np.tan(x), 1, name="cot")
    pset_f.addPrimitive(lambda x: 1 / np.cos(x), 1, name="sec")
    pset_f.addPrimitive(lambda x: 1 / np.sin(x), 1, name="csc")
    pset_f.addPrimitive(np.arctan, 1, name="arctan")
    pset_f.addPrimitive(np.arctan, 1, name="atan")
    pset_f.addPrimitive(np.arcsin, 1, name="arcsin")
    pset_f.addPrimitive(np.arccos, 1, name="arccos")
    pset_f.addTerminal(np.e, name="e")
    pset_f.addTerminal(np.pi, name="pi")
    pset_f.addEphemeralConstant("C", sampling_const)
    pset_g = gp.PrimitiveSet("G", int(dim * 2), prefix="x")
    pset_g.addPrimitive(np.add, 2, name="Add")
    pset_g.addPrimitive(np.subtract, 2, name="Sub")
    pset_g.addPrimitive(np.multiply, 2, name="Mul")
    pset_g.addPrimitive(np.divide, 2, name="Div")
    pset_g.addPrimitive(np.exp, 1, name="exp")
    pset_g.addPrimitive(np.sin, 1, name="sin")
    pset_g.addPrimitive(np.cos, 1, name="cos")
    pset_g.addPrimitive(np.tan, 1, name="tan")
    pset_g.addPrimitive(np.abs, 1, name="Abs")
    pset_g.addPrimitive(np.log, 1, name="log")
    pset_g.addPrimitive(real_pow, 2, name="Pow")  
    pset_g.addPrimitive(np.tanh, 1, name="tanh")
    pset_g.addPrimitive(np.sinh, 1, name="sinh")
    pset_g.addPrimitive(np.cosh, 1, name="cosh")
    pset_g.addTerminal(np.e, name="e")
    pset_g.addTerminal(np.pi, name="pi")
    pset_g.addEphemeralConstant("C", sampling_const)
    pset = (pset_f, pset_g)
    return pset_f, pset_g, pset
pset_f, pset_g, pset = build_psets(Dim)
converter = {
    'Sub': lambda x, y: x - y,
    'Div': lambda x, y: x / y,
    'Mul': lambda x, y: x * y,
    'Add': lambda x, y: x + y,
    'Neg': lambda x: -x,
    'Pow': lambda x, y: x ** y,
    'Inv': lambda x: x ** (-1),
    'Sqrt': lambda x: sp.sqrt(x),
    'exp': lambda x: sp.exp(x),
    'sin': lambda x: sp.sin(x),
    'cos': lambda x: sp.cos(x),
    'tan': lambda x: sp.tan(x),
    'log': lambda x: sp.log(x),
    'Abs': lambda x: sp.Abs(x)
}
def stat_complex(pop):
    Stat_Complex = {}
    for p_i in pop:
        complex_of_eq = len(p_i[0][0]) + len(p_i[0][1])
        if complex_of_eq not in Stat_Complex.keys():
            Stat_Complex[complex_of_eq] = 1
        else:
            Stat_Complex[complex_of_eq] += 1
    return Stat_Complex
def eval_func(ind_i_1_2, pset, X0, sparse_A, Y, Mask, Stat_Complex=None):
    pset_f_, pset_g_ = pset
    def compile_expr(exprs, pset):
        compiled = gp.compile(exprs, pset)
        if opt_flag is not None and Dim is not None:
            zero_func = gp.compile(gp.PrimitiveTree.from_string_sympy("x0", pset), pset)
            result = [zero_func for _ in range(Dim)]
            result[opt_flag] = compiled
            return result
        else:
            if isinstance(exprs, list):
                return [gp.compile(e, pset) for e in exprs]
            else:
                return [compiled for _ in range(Dim)]
    def compile_expr1(exprs, pset):
        if isinstance(exprs, list):
            return [gp.compile(e, pset) for e in exprs]
        else:
            return [gp.compile(exprs, pset)]
    if Dim == 1:
        eval_func_f_list = gp.compile(ind_i_1_2[0], pset_f_)
        eval_func_g_list = gp.compile(ind_i_1_2[1], pset_g_)
    else:
        eval_func_f_list = compile_expr(ind_i_1_2[0], pset_f_)
        eval_func_g_list = compile_expr(ind_i_1_2[1], pset_g_)
    soluation_Y = gp.solve_ivp_diff(
        eval_func_f_list,
        eval_func_g_list,
        X0, sparse_A, Y, diff_Y,
        t_start=0, t_end=1, t_inc=0.01
    )
    dim = soluation_Y.shape[-1]  
    loss_list = []
    for d in range(dim):
        if d == opt_flag:
            pred_d = soluation_Y[..., d]
            true_d = diff_Y[..., d]
            mask_d = Mask[..., d]
            mse_d = np.mean((pred_d[mask_d == 2] - true_d[mask_d == 2]) ** 2)
            denom_d = np.var(true_d[mask_d == 2])
            loss_d = mse_d / denom_d if denom_d != 0 else 1e30
            loss_list.append(loss_d)
    if Dim == 1:
        total_loss = loss_list[0]
    else:
       total_loss = np.mean(loss_list)
    if Stat_Complex is not None:
        complex_of_eq = sum(len(e) for e in (ind_i_1_2[0] if isinstance(ind_i_1_2[0], list) else [ind_i_1_2[0]])) \
                        + sum(len(e) for e in (ind_i_1_2[1] if isinstance(ind_i_1_2[1], list) else [ind_i_1_2[1]]))
        total_loss *= np.exp(float(Stat_Complex[complex_of_eq] / np.sum(list(Stat_Complex.values()))))
    return np.nan_to_num(total_loss, nan=1e30)
def eval_func_NMSE_discrete(ind_i_1_2, pset, X0, sparse_A, Y, Mask,t_start=0, t_end=10, t_inc=0.01,Stat_Complex=None):
    pset_f_, pset_g_ = pset
    def compile_expr(exprs, pset):
        compiled = gp.compile(exprs, pset)
        if opt_flag is not None and Dim is not None:
            zero_func = gp.compile(gp.PrimitiveTree.from_string_sympy("x0", pset), pset)
            result = [zero_func for _ in range(Dim)]
            result[opt_flag] = compiled
            return result
        else:
            if isinstance(exprs, list):
                return [gp.compile(e, pset) for e in exprs]
            else:
                return [compiled for _ in range(Dim)]
    def compile_expr1(exprs, pset):
        if isinstance(exprs, list):
            return [gp.compile(e, pset) for e in exprs]
        else:
            return [gp.compile(exprs, pset)]
    if Dim == 1:
        eval_func_f_list = gp.compile(ind_i_1_2[0], pset_f_)
        eval_func_g_list = gp.compile(ind_i_1_2[1], pset_g_)
    else:
        eval_func_f_list = compile_expr(ind_i_1_2[0], pset_f_)
        eval_func_g_list = compile_expr(ind_i_1_2[1], pset_g_)
    soluation_Y = gp.solve_ivp_diff_modified_local(eval_func_f_list,
        eval_func_g_list,
        Y, country_select_index,time_select_index,sparse_A)
    dim = soluation_Y.shape[-1]  
    loss_list = []
    for d in range(dim):
        if d == opt_flag:
            pred_d = soluation_Y[..., d]
            true = true_data[..., d]
            X0 = X0.reshape(-1)  
            cumulative_pred = np.vstack([X0, X0 + np.cumsum(pred_d, axis=0)])  
            sub_matrix = cumulative_pred[:, country_select_index]
            mse_per_country = np.mean((true - sub_matrix) ** 2, axis=0)
            var_per_country = np.var(true, axis=0)
            loss_d = np.max(mse_per_country/var_per_country)
            if loss_d > 10 or loss_d == 1e30:
                loss_d = random.uniform(3.8, 3.99)
            loss_list.append(loss_d)
    if Dim == 1:
        total_loss = loss_list[0]
    else:
       total_loss = np.mean(loss_list)
    if Stat_Complex is not None:
        complex_of_eq = sum(len(e) for e in (ind_i_1_2[0] if isinstance(ind_i_1_2[0], list) else [ind_i_1_2[0]])) \
                        + sum(len(e) for e in (ind_i_1_2[1] if isinstance(ind_i_1_2[1], list) else [ind_i_1_2[1]]))
        total_loss *= np.exp(float(Stat_Complex[complex_of_eq] / np.sum(list(Stat_Complex.values()))))
    return np.nan_to_num(total_loss, nan=1e30)
def evaluate_gp_function(eval_func, x):
    """
    x: 输入数据 (N, x_dim)
    eval_func: 编译后的GP函数
    """
    if Y.ndim == 1:
        diff_f = np.array(
            eval_func(*[x[:, m].reshape(-1, 1) for m in range(x.shape[1])]),
            dtype=np.float32
        )
        if diff_f.ndim < 2:
            diff_f = diff_f.reshape(-1, 1)
        return diff_f
    else:
        """
        x: 输入数据 (N, 2), x[:, 0] 是 mx, x[:, 1] 是 my
        Y: 目标数据 (N,) 或 (N, 2)
        eval_func: 编译后的标量 GP 函数 F(x0, x1)
        """
        m_x = x[:, 0]
        m_y = x[:, 1]
        m_extra = x[:, 2]
        res_fx = eval_func(m_x, m_extra)
        res_fy = eval_func(m_y, m_extra)
        f_x_arr = np.array(res_fx, dtype=np.float32).flatten()
        f_y_arr = np.array(res_fy, dtype=np.float32).flatten()
        diff_f = np.stack([f_x_arr, f_y_arr], axis=1)
        return diff_f
def diff_func_ode_symmetric(eval_func, x):
    if Y.ndim == 2:
        m_x = x[:, 0]
        m_y = x[:, 1]
        m_extra = x[:, 2]
        res_fx = eval_func(m_x, m_extra)
        res_fy = eval_func(m_y, m_extra)
        f_x_arr = np.array(res_fx, dtype=np.float32).flatten()
        f_y_arr = np.array(res_fy, dtype=np.float32).flatten()
        if f_x_arr.size == 1:
            f_x_arr = np.full_like(m_x, fill_value=f_x_arr[0], dtype=np.float32)
        if f_y_arr.size == 1:
            f_y_arr = np.full_like(m_y, fill_value=f_y_arr[0], dtype=np.float32)
        dX_pred = np.stack([f_x_arr, f_y_arr], axis=1)
        return dX_pred
    else:
        x_dim = x.shape[-1]
        x_flat = x.reshape(-1, x_dim)
        f_inputs = [x_flat[:, m] for m in range(x_dim)]
        diff_f = np.array(eval_func(*f_inputs), dtype=np.float32)
        if np.isscalar(diff_f) or diff_f.size == 1:
            diff_f = np.full_like(x_flat[:, 0], fill_value=float(diff_f), dtype=np.float32)
        dX_pred = diff_f.reshape(-1, 1)
        return dX_pred
def eval_func_SR(ind_i_1_2, pset, X0, Y, Stat_Complex=None):
        pset_f_, pset_g_ = pset
        def compile_expr(exprs, pset):
            compiled = gp.compile(exprs, pset)
            if opt_flag is not None and Dim is not None:
                zero_func = gp.compile(gp.PrimitiveTree.from_string_sympy("x0", pset), pset)
                result = [zero_func for _ in range(Dim)]
                result[opt_flag] = compiled
                return result
            else:
                if isinstance(exprs, list):
                    return [gp.compile(e, pset) for e in exprs]
                else:
                    return [compiled for _ in range(Dim)]
        def compile_expr1(exprs, pset):
            if isinstance(exprs, list):
                return [gp.compile(e, pset) for e in exprs]
            else:
                return [gp.compile(exprs, pset)]
        if Dim == 1:
            eval_func_f_list = [gp.compile(ind_i_1_2[0], pset_f_)]
        else:
            eval_func_f_list = compile_expr(ind_i_1_2[0], pset_f_)
        if case_name in ["fish_15","fish_30","fish_60"]:
            soluation_Y = diff_func_ode_symmetric(eval_func_f_list[0],X0)
        else:
            soluation_Y = evaluate_gp_function(eval_func_f_list[0],X0)
        dim = soluation_Y.shape[-1]  
        loss_list = []
        for d in range(dim):
            if case_name in ["fish_15", "fish_30", "fish_60"]:
                pass
            else:
                if d != opt_flag:
                    continue
            pred_d = soluation_Y[..., d]
            true_d = Y if Y.ndim == 1 else Y[:, d] 
            log_pred = np.log10(np.abs(pred_d))
            log_true = np.log10(np.abs(true_d))
            log_mse_d = np.mean((log_pred - log_true) ** 2)
            log_var_d = np.var(log_true)
            log_nmse_d = log_mse_d / log_var_d if log_var_d != 0 else 0
            mse_d = np.mean((pred_d - true_d) ** 2)
            var_d = np.var(true_d)  
            nmse_d = mse_d / var_d if var_d != 0 else 1e30
            loss_list.append(log_nmse_d)
        if Dim == 1:
            total_loss = loss_list[0]
        else:
            total_loss = np.mean(loss_list)
        if Stat_Complex is not None:
            complex_of_eq = sum(len(e) for e in (ind_i_1_2[0] if isinstance(ind_i_1_2[0], list) else [ind_i_1_2[0]])) \
                            + sum(len(e) for e in (ind_i_1_2[1] if isinstance(ind_i_1_2[1], list) else [ind_i_1_2[1]]))
            total_loss *= np.exp(float(Stat_Complex[complex_of_eq] / np.sum(list(Stat_Complex.values()))))
        return np.nan_to_num(total_loss, nan=1e30)
def optimize_in_scipy(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options):
    minimizer_kwargs = {"method": method}
    tol = 1e-10
    res = opt.minimize(fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback, options)
    return res
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
def optimize_constants_gpu(expressions_list, X_data, Y_data, device=None, 
                           epochs=2000, lr=5e-2, patience=40):
    """
    使用 GPU 并行优化一组 SymPy 表达式中的常数。
    Args:
        expressions_list (list): SymPy 表达式或字符串列表 (e.g., ["1.0 * x0 + 0.5", ...])
        X_data (np.ndarray): 输入数据，形状 [n_samples, n_features]
        Y_data (np.ndarray): 目标数据，形状 [n_samples, 1] 或 [n_samples, n_outputs]
        device (torch.device): 'cuda' or 'cpu'
        epochs (int): 最大训练轮数
        lr (float): 学习率
        patience (int): 早停耐心值
    Returns:
        optimized_exprs (list): 优化后的 SymPy 表达式字符串列表
        final_losses (np.ndarray): 每个表达式对应的最终 loss
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device} 进行并行常数优化...")
    if isinstance(X_data, np.ndarray):
        X_tensor = torch.from_numpy(X_data).float().to(device)
    else:
        X_tensor = X_data.float().to(device)
    if isinstance(Y_data, np.ndarray):
        Y_tensor = torch.from_numpy(Y_data).float().to(device)
    else:
        Y_tensor = Y_data.float().to(device)
    if len(Y_tensor.shape) == 1:
        Y_tensor = Y_tensor.unsqueeze(1)
    input_params = {}
    for dim_i in range(X_tensor.size(-1)):
        input_params[f'x{dim_i+1}'] = X_tensor[:, dim_i] 
    try:
        mod = sympytorch.SymPyModule(expressions=expressions_list).to(device)
        print(mod)
    except Exception as e:
        print(f"转换 SymPyModule 失败: {e}")
        return expressions_list, np.full(len(expressions_list), np.inf)
    optimizer = torch.optim.Adam(mod.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=False)
    early_stopping = EarlyStopping(patience=patience, verbose=False, delta=1e-5)
    pbar = tqdm(range(epochs), desc="GPU Optimizing", leave=False)
    for epoch in pbar:
        mod.train()
        optimizer.zero_grad()
        try:
            out = mod(**input_params)
        except Exception as e:
            print(f"前向传播计算错误 (可能是定义域问题): {e}")
            break
        mse_per_expression = torch.mean((out - Y_tensor) ** 2, dim=0)
        mse_per_expression = torch.nan_to_num(mse_per_expression, nan=1e10)
        total_loss = torch.mean(mse_per_expression) 
        total_loss.backward()
        optimizer.step()
        current_loss_val = total_loss.item()
        scheduler.step(current_loss_val)
        early_stopping(current_loss_val, mod)
        pbar.set_postfix({'loss': f"{current_loss_val:.5f}"})
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    optimized_strs = [str(expr) for expr in mod.sympy()]
    mod.eval()
    with torch.no_grad():
        final_out = mod(**input_params)
        final_losses = torch.mean((final_out - Y_tensor) ** 2, dim=0)
        final_losses = torch.nan_to_num(final_losses, nan=1e10)
    return optimized_strs, final_losses.cpu().numpy()
def optimize_in_scipy_with_timeout_check(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol,
                                         callback, options):
    return optimize_in_scipy(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints, tol, callback,
                                options)
def multistart(on_core_i, fun, x0, nrestart=1, full_output=False, args=(), method=None, jac=None, hess=None, hessp=None,
               bounds=None, constraints=(), tol=None, callback=None, options = None):
    options={'xatol': 1e-10, 'fatol': 1e-10}
    res_list = np.empty(nrestart, dtype=object)
    res = optimize_in_scipy_with_timeout_check(on_core_i, fun, x0, args, method, jac, hess, hessp, bounds, constraints,
                                               tol, callback, options)
    res_list[0] = res
    for i in range(nrestart - 1):
        new_x0 = x0 + np.array([random.gauss(0, 0.1) * 1. for _ in range(x0.shape[0])])
        res = optimize_in_scipy_with_timeout_check(on_core_i, fun, new_x0, args, method, jac, hess, hessp, bounds,
                                                   constraints, tol, callback, options)
        res_list[i + 1] = res
        print(res)
    res_fun_list = []
    for res in res_list:
        if res is None:
            res_fun_list.append(1e30)
        else:
            res_fun_list.append(res.fun)
    sort_res_list = res_list[np.argsort(res_fun_list)]
    if full_output:
        return sort_res_list[0], sort_res_list
    else:
        return sort_res_list[0]
def extract_optimizable_constants(expr):
    consts = []
    for i, node in enumerate(expr):
        if isinstance(node, gp.Terminal) and not isinstance(node.value, str):
            if i > 0 and str(expr[i - 1]) == 'Pow':
                continue
            if np.isclose(node.value, 0.5):
                continue
            consts.append(node.value)
    return np.array(consts)
def optimizeConstants_SR(individual, X0, Y, on_core_i, opt_const_dim=-1):
    constants_f = extract_optimizable_constants(individual[0])
    constants_g = extract_optimizable_constants(individual[1])
    if opt_const_dim > 0:
        constants_f = constants_f[:opt_const_dim]
    num_constants_f = len(constants_f)
    if opt_const_dim > 0:
        constants_g = constants_g[:opt_const_dim]
    num_constants_g = len(constants_g)
    constants = constants_f
    if constants.size > 0:
        def setConstants1(individual, constants):
            optIndividual = individual
            c = 0
            for i in range(0, len(optIndividual)):
                if isinstance(optIndividual[i], gp.Terminal) and not isinstance(optIndividual[i].value, str) and c < len(constants):
                    optIndividual[i].value = np.nan_to_num(constants[c])
                    optIndividual[i].name = str(optIndividual[i].value)
                    c += 1
            return optIndividual
        def setConstants(individual, constants):
            """
            将优化后的 constants 按顺序写回 individual 中的常数节点，
            跳过 Pow 的幂参数以及值为 0.5 的结构常数。
            """
            optIndividual = individual
            c = 0
            for i, node in enumerate(optIndividual):
                if isinstance(node, gp.Terminal) and not isinstance(node.value, str):
                    if i > 0 and str(optIndividual[i - 1]) == 'Pow':
                        continue
                    if np.isclose(node.value, 0.5):
                        continue
                    if c < len(constants):
                        node.value = np.nan_to_num(constants[c])
                        node.name = str(node.value)
                        c += 1
            return optIndividual
        def evaluate(constants, individual):
            individual = setConstants(individual, constants)
            individual_f_ = setConstants(individual[0], constants[:num_constants_f])
            individual_g_ = setConstants(individual[1], constants[num_constants_f:])
            return eval_func_SR((individual_f_, individual_g_), pset, X0,Y)
        def evaluateLM(constants):
            return evaluate(constants, individual)
        res = multistart(on_core_i, evaluateLM, constants, nrestart=1,method='Nelder-Mead')
        if res is not None:
            individual_f = setConstants(individual[0], res.x[:num_constants_f])
            individual_g = setConstants(individual[1], res.x[num_constants_f:])
        else:
            individual_f = setConstants(individual[0], np.array([sampling_const() for _ in range(num_constants_f)]))
            individual_g = setConstants(individual[1], np.array([sampling_const() for _ in range(num_constants_g)]))
        return (individual_f, individual_g)
    else:
        return individual
def optimizeConstants(individual, X0, sparse_A, Y, Mask, on_core_i, opt_const_dim=-1):
    constants_f = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[0]))))
    if opt_const_dim > 0:
        constants_f = constants_f[:opt_const_dim]
    num_constants_f = len(constants_f)
    constants_g = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[1]))))
    if opt_const_dim > 0:
        constants_g = constants_g[:opt_const_dim]
    num_constants_g = len(constants_g)
    constants = np.concatenate([constants_f, constants_g], axis=-1)
    if constants.size > 0:
        def setConstants(individual, constants):
            optIndividual = individual
            c = 0
            for i in range(0, len(optIndividual)):
                if isinstance(optIndividual[i], gp.Terminal) and not isinstance(optIndividual[i].value, str) and c < len(constants):
                    optIndividual[i].value = np.nan_to_num(constants[c])
                    optIndividual[i].name = str(optIndividual[i].value)
                    c += 1
            return optIndividual
        def evaluate(constants, individual):
            individual_f_ = setConstants(individual[0], constants[:num_constants_f])
            individual_g_ = setConstants(individual[1], constants[num_constants_f:])
            return eval_func((individual_f_, individual_g_), pset, X0, sparse_A, Y, Mask)
        def evaluateLM(constants):
            return evaluate(constants, individual)
        res = multistart(on_core_i, evaluateLM, constants, nrestart=1, method='Nelder-Mead')
        if res is not None:
            individual_f = setConstants(individual[0], res.x[:num_constants_f])
            individual_g = setConstants(individual[1], res.x[num_constants_f:])
        else:
            individual_f = setConstants(individual[0], np.array([sampling_const() for _ in range(num_constants_f)]))
            individual_g = setConstants(individual[1], np.array([sampling_const() for _ in range(num_constants_g)]))
        return (individual_f, individual_g)
    else:
        return individual
def optimizeConstants_discrete(individual, X0, sparse_A, Y, Mask, on_core_i, opt_const_dim=-1):
    constants_f = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[0]))))
    if opt_const_dim > 0:
        constants_f = constants_f[:opt_const_dim]
    num_constants_f = len(constants_f)
    constants_g = np.array(list(map(lambda n: n.value,
                                    filter(lambda n: isinstance(n, gp.Terminal) and not isinstance(n.value, str),
                                           individual[1]))))
    if opt_const_dim > 0:
        constants_g = constants_g[:opt_const_dim]
    num_constants_g = len(constants_g)
    constants = np.concatenate([constants_f, constants_g], axis=-1)
    if constants.size > 0:
        def setConstants(individual, constants):
            optIndividual = individual
            c = 0
            for i in range(0, len(optIndividual)):
                if isinstance(optIndividual[i], gp.Terminal) and not isinstance(optIndividual[i].value, str) and c < len(constants):
                    optIndividual[i].value = np.nan_to_num(constants[c])
                    optIndividual[i].name = str(optIndividual[i].value)
                    c += 1
            return optIndividual
        def evaluate(constants, individual):
            individual_f_ = setConstants(individual[0], constants[:num_constants_f])
            individual_g_ = setConstants(individual[1], constants[num_constants_f:])
            return eval_func_NMSE_discrete((individual_f_, individual_g_), pset, X0, sparse_A, Y, Mask)
        def evaluateLM(constants):
            return evaluate(constants, individual)
        res = multistart(on_core_i, evaluateLM, constants, nrestart=1, method='Nelder-Mead')
        if res is not None:
            individual_f = setConstants(individual[0], res.x[:num_constants_f])
            individual_g = setConstants(individual[1], res.x[num_constants_f:])
        else:
            individual_f = setConstants(individual[0], np.array([sampling_const() for _ in range(num_constants_f)]))
            individual_g = setConstants(individual[1], np.array([sampling_const() for _ in range(num_constants_g)]))
        return (individual_f, individual_g)
    else:
        return individual
def tournamentSelection(P, num=12, prob=0.9):
    Q = random.sample(P, num)
    while len(Q) > 1:
        sort_index = np.argsort([ii[1] for ii in Q])
        E = Q[sort_index[0]]
        if random.uniform(0, 1) < prob:
            break
        Q.remove(E)
    return E
def eval_pop(pop):
    pop_eval = []
    for idx in tqdm(range(len(pop))):
        ind_i_1 = pop[idx][0][0]
        ind_i_2 = pop[idx][0][1]
        ind_i_fitness = eval_func((ind_i_1, ind_i_2), pset, X0, sparse_A, Y, Mask)
        pop_eval.append([(ind_i_1, ind_i_2), ind_i_fitness])
    return pop_eval
def multi_processing_eval_pop(pop, num_running_each_core=10):
    store_all = []
    store_all2 = []
    with multiprocessing.Pool() as p:
        for i in range(math.ceil(len(pop) / num_running_each_core)):
            pop_eval = p.apply_async(eval_pop, args=(
                pop[i * num_running_each_core:i * num_running_each_core + num_running_each_core],))
            store_all.append(pop_eval)
        p.close()
        p.join()
        for i in store_all:
            store_all2.append(i.get())
    new_pop = []
    for store_i in store_all2:
        new_pop += store_i
    return new_pop
def evolve(pop, num_evo):
    pop_new = []
    for _ in tqdm(range(num_evo)):
        ind_one = tournamentSelection(pop, num=12)
        ind_one_new = gp.Mutations_f_g(ind_one[0], pset, sampling_const, converter, min_=min_, max_=max_)
        print(ind_one_new)
        pop_new.append([ind_one_new, 1e30])
    return pop_new
def choose(pop, num_choose):
    Stat_Complex = stat_complex(pop)
    fitness_list = []
    for p_i in pop:
        complex_of_eq = len(p_i[0][0]) + len(p_i[0][1])
        fitness_list.append(p_i[1] * np.exp(float(Stat_Complex[complex_of_eq] / np.sum(list(Stat_Complex.values())))))
    pop_choose = np.array(pop)[np.argsort(fitness_list)[:num_choose]].tolist()
    print([len(p_i[0][0])+len(p_i[0][1]) for p_i in pop_choose])
    return pop_choose
def choose_best(pop, num_choose):
    if num_choose == -1:
        pop_choose = np.array(pop)[np.argsort([p_i[1] for p_i in pop])].tolist()
    else:
        pop_choose = np.array(pop)[np.argsort([p_i[1] for p_i in pop])[:num_choose]].tolist()
    return pop_choose
def choose_diversity(pop, num_choose):
    pop_choose = get_return_pop(pop, filter=False)
    if num_choose is None:
        return pop_choose
    return pop_choose[:num_choose]
def opt_constant_one_core(pop, core_i, opt_const_dim=-1):
    pop_opt = []
    opt_flag_one_core = True
    for i in tqdm(range(len(pop))):
        ind_i = copy.deepcopy(pop[i][0])
        if opt_flag_one_core:
            ind_i = optimizeConstants(ind_i, X0, sparse_A, Y, Mask, core_i, opt_const_dim=opt_const_dim)
        ind_i_fitness = eval_func(ind_i, pset, X0, sparse_A, Y, Mask)
        print('ind_i_fitness = %s'%ind_i_fitness)
        if ind_i_fitness <= termination_condition_err:
            opt_flag_one_core = False
            print("\n==== Found opt that meets the termination condition on core [%s]\n" % (core_i))
        pop_opt.append([ind_i, ind_i_fitness])
    return pop_opt
def opt_constant_one_core_discrete(pop, core_i, opt_const_dim=-1):
    pop_opt = []
    opt_flag_one_core = True
    for i in tqdm(range(len(pop))):
        ind_i = copy.deepcopy(pop[i][0])
        if opt_flag_one_core:
            ind_i = optimizeConstants_discrete(ind_i, X0, sparse_A, Y, Mask, core_i, opt_const_dim=opt_const_dim)
        ind_i_fitness = eval_func_NMSE_discrete(ind_i, pset, X0, sparse_A, Y, Mask)
        print('ind_i_fitness = %s'%ind_i_fitness)
        if ind_i_fitness <= termination_condition_err:
            opt_flag_one_core = False
            print("\n==== Found opt that meets the termination condition on core [%s]\n" % (core_i))
        pop_opt.append([ind_i, ind_i_fitness])
    return pop_opt
def opt_constant_one_core_SR(pop, core_i, opt_const_dim=-1):
    pop_opt = []
    opt_flag_one_core = True
    for i in range(len(pop)):
        ind_i = copy.deepcopy(pop[i][0])
        if opt_flag_one_core:
            ind_i = optimizeConstants_SR(ind_i, X0, Y, core_i, opt_const_dim=opt_const_dim)
        ind_i_fitness = eval_func_SR(ind_i, pset, X0, Y)
        print('ind_i_fitness = %s'%ind_i_fitness)
        if ind_i_fitness <= termination_condition_err:
            opt_flag_one_core = False
            print("\n==== Found opt that meets the termination condition on core [%s]\n" % (core_i))
        pop_opt.append([ind_i, pop[i][1], pop[i][2], ind_i_fitness])
    return pop_opt
def opt_constant_one_core_SR_fast(pop, core_i, opt_const_dim=-1):
    fitness_list = []
    opt_flag_one_core = True
    for i in range(len(pop)):
        ind_i = pop[i][0]
        if opt_flag_one_core:
            ind_i = optimizeConstants_SR(ind_i, X0, Y, core_i, opt_const_dim=opt_const_dim)
        ind_i_fitness = float(eval_func_SR(ind_i, pset, X0, Y))
        print('nmse = %s'%ind_i_fitness)
        complexity = len(ind_i[0])
        penalty = (0.5 * ind_i_fitness) * np.exp(-50 / complexity)
        total_fitness = ind_i_fitness + penalty
        print('ind_i_fitness = %s'% total_fitness)
        if ind_i_fitness <= termination_condition_err:
            opt_flag_one_core = False
            print("\n==== Found opt that meets the termination condition on core [%s]\n" % (core_i))
        fitness_list.append(total_fitness)
    return fitness_list
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
def opt_constant_one_core_SR_strategy(pop_chunk, core_i, opt_const_dim=-1):
    """
    pop_chunk: [(global_idx, pop_item), ...]
    return: [(global_idx, optimized_ind, total_fitness), ...]
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    results = []
    opt_flag_one_core = True
    for global_idx, item in pop_chunk:
        ind_i = item[0]
        if opt_flag_one_core:
            try:
                ind_i = optimizeConstants_SR(
                    ind_i, X0, Y, core_i, opt_const_dim=opt_const_dim
                )
            except Exception:
                ind_i = item[0]
                fitness = 1e10
                total_fitness = 1e10
                results.append((global_idx, ind_i, total_fitness))
                continue
        try:
            fitness = float(eval_func_SR(ind_i, pset, X0, Y))
            if not np.isfinite(fitness):
                fitness = 1e10
        except Exception:
            fitness = 1e10
        try:
            complexity = len(ind_i[0])
            penalty = 2*fitness * np.exp(-50 / complexity)
            total_fitness = fitness + penalty
        except Exception:
            total_fitness = 1e10
        if fitness <= termination_condition_err:
            opt_flag_one_core = False
            print('NMSE = %s'%fitness, flush=True)
            print(f"\n==== Found opt that meets termination condition on core [{core_i}] ====\n")
        results.append((global_idx, ind_i, total_fitness))
    return results
def optimize_and_eval_one(global_idx, item, opt_const_dim=-1):
    """
    返回 (global_idx, optimized_ind, total_fitness)
    """
    ind_i = item[0]
    try:
        ind_i = optimizeConstants_SR(ind_i, X0, Y, core_i=global_idx, opt_const_dim=opt_const_dim)
    except Exception:
        ind_i = item[0]
    try:
        fitness = float(eval_func_SR(ind_i, pset, X0, Y))
        if not np.isfinite(fitness) or abs(fitness) > 1e20:
            fitness = 1e10
    except Exception:
        fitness = 1e10
    print('nmse = %s'%fitness)
    try:
        complexity = max(len(ind_i[0]), 1)
        penalty = fitness * np.exp(-50 / complexity)
        penalty = min(penalty, 1e6)
        total_fitness = fitness + penalty
    except Exception:
        total_fitness = 1e10
    return (global_idx, ind_i, total_fitness)
def opt_constant_gpu1(pop,device=None, epochs=2000, lr=0.05, termination_condition_err=1e-5):
    """
    GPU 并行常数优化函数 (稳定版)
    Args:
        pop: 种群列表 [[ind, meta, fit], ...]
        X_input: 输入数据 (numpy)
        Y_input: 标签数据 (numpy)
        device: torch device
    Returns:
        pop_opt: 优化后的种群列表 [[str_eq, meta, new_fit], ...]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X0, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    if Y_tensor.ndim == 1:
        Y_tensor = Y_tensor.unsqueeze(1)
    sympy_exprs = []       
    meta_data_list = []    
    original_strs = []     
    for i, item in enumerate(pop):
        raw_ind = str(gp.TreeToDeadStr(item[0][0], converter))
        print(raw_ind)
        if isinstance(raw_ind, (tuple, list)) and len(raw_ind) > 0:
            ind_obj = raw_ind[0]
        else:
            ind_obj = raw_ind
        meta_data = item[1]
        s_str = str(ind_obj).replace("^", "**")
        try:
            expr_obj = sp.sympify(s_str, evaluate=False)
            sympy_exprs.append(expr_obj)
            meta_data_list.append(meta_data)
            original_strs.append(s_str) 
        except Exception as e:
            continue
    if not sympy_exprs:
        print("Warning: No valid expressions to optimize.")
        return []
    try:
        mod = sympytorch.SymPyModule(expressions=sympy_exprs).to(device)
    except Exception as e:
        print(f"SymPyModule Build Error: {e}")
        return []
    input_params = {}
    for dim_i in range(X_tensor.shape[1]):
        input_params[f'x{dim_i}'] = X_tensor[:, dim_i]
    optimizer = torch.optim.Adam(mod.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
    pbar = tqdm(range(epochs), desc="GPU Optimizing", leave=False)
    for epoch in pbar:
        mod.train()
        optimizer.zero_grad()
        try:
            out = mod(**input_params) 
            diff_sq = (out - Y_tensor) ** 2
            mse_per_eq = torch.mean(diff_sq, dim=0) 
            loss_each = torch.log(mse_per_eq + 1.0)
            loss_each = torch.nan_to_num(loss_each, nan=100.0, posinf=100.0, neginf=100.0)
            loss_scalar = torch.mean(loss_each)
            loss_scalar.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), max_norm=1.0)
            optimizer.step()
            loss_val = loss_scalar.item()
            scheduler.step(loss_val)
            real_min_mse = torch.min(mse_per_eq).item()
            pbar.set_postfix({'BestMSE': f"{real_min_mse:.6f}"})
            if real_min_mse < termination_condition_err:
                pass 
        except Exception as e:
            print(f"Optimization Loop Error: {e}")
            break
    optimized_exprs_obj = mod.sympy() 
    mod.eval()
    with torch.no_grad():
        final_out = mod(**input_params)
        final_diff = (final_out - Y_tensor) ** 2
        final_mse = torch.mean(final_diff, dim=0)
        final_mse = torch.nan_to_num(final_mse, nan=1e10, posinf=1e10).cpu().numpy()
    pop_opt = []
    print("\nBatch Optimization Finished. Processing results...")
    for i in range(len(optimized_exprs_obj)):
        opt_str = str(optimized_exprs_obj[i])
        fitness = final_mse[i]
        meta = meta_data_list[i]
        is_failed = False
        if fitness >= 1e9 or np.isnan(fitness):
            is_failed = True
        if 'nan' in opt_str.lower():
            is_failed = True
        if is_failed:
            final_res_str = original_strs[i]
            final_res_fit = 1e10 
        else:
            final_res_str = opt_str
            final_res_fit = fitness
            if final_res_fit <= termination_condition_err:
                print(f"==== Found opt: {final_res_fit:.6f} ====")
        pop_opt.append([final_res_str, meta, final_res_fit])
    return pop_opt
def opt_constant_gpu(pop, device=None, 
                     epochs=2000, lr=0.1, termination_condition_err=1e-5):
    """
    GPU 并行常数优化函数
    替代原来的 opt_constant_one_core_SR
    """
    pop_opt = []
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Batch Optimization on {device}...")
    X_tensor = torch.tensor(X0, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    if Y_tensor.ndim == 1:
        Y_tensor = Y_tensor.unsqueeze(1) 
    sympy_exprs = []       
    meta_data_list = []    
    valid_indices = []     
    input_params = {}
    for dim_i in range(X_tensor.shape[1]):
        input_params[f'x{dim_i}'] = X_tensor[:, dim_i]
    for i, item in enumerate(pop):
        ind = str(gp.TreeToDeadStr(item[0][0], converter))
        meta_data = item[1]
        s_str = str(ind).replace("^", "**")
        try:
            expr_obj = sp.sympify(s_str, evaluate=False)
            sympy_exprs.append(expr_obj)
            meta_data_list.append(meta_data)
            valid_indices.append(i)
        except Exception as e:
            print(f"Skipping invalid equation [{i}]: {s_str} | Error: {e}")
            continue
    if not sympy_exprs:
        print("Error: No valid SymPy expressions found.")
        return []
    print()
    mod = sympytorch.SymPyModule(expressions=sympy_exprs).to(device)
    optim = torch.optim.Adam(mod.parameters(), lr=5e-2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.9, patience=10)
    num_epochs = 500
    patience = 20
    early_stopping = EarlyStopping(patience=patience, verbose=False, delta=1e-4)
    loss_func = torch.nn.MSELoss()
    aa = time.time()
    y_var = torch.var(Y_tensor, unbiased=False) + 1e-15
    for epoch in tqdm(range(num_epochs)):
        mod.train()
        optim.zero_grad()
        out = mod(**input_params) 
        mse_each = torch.mean((out - Y_tensor) ** 2, dim=0)/y_var
        with torch.no_grad(): 
            valid_mask = ~torch.isnan(mse_each) & ~torch.isinf(mse_each)
        if valid_mask.sum() == 0:
            print(f"Epoch {epoch}: All equations resulted in NaN/Inf. Stopping optimization.")
            break
        loss_each = torch.log(mse_each + 1.)
        loss = torch.mean(loss_each[valid_mask])
        loss.backward()
        optim.step()
        current_loss_val = loss.item()
        lr_scheduler.step(current_loss_val)
        early_stopping(current_loss_val, mod)
        if early_stopping.early_stop or epoch >= num_epochs - 1:
            print('epoch 
                  (epoch, current_loss_val, valid_mask.sum(), len(mse_each), time.time() - aa))
            if early_stopping.early_stop:
                print("Early stopping")
            return [str(expr_i) for expr_i in mod.sympy()], loss_each.detach().exp().cpu().numpy() - 1.
def process_individual_wrapper(i, pop, core_i, opt_const_dim, X0, sparse_A, Y, Mask, termination_condition_err):
    return process_individual(i, pop, core_i, opt_const_dim, X0, sparse_A, Y, Mask, termination_condition_err)
def process_individual(i, pop, core_i, opt_const_dim, X0, sparse_A, Y, Mask, termination_condition_err):
    ind_i = copy.deepcopy(pop[i][0])
    ind_i = optimizeConstants(ind_i, X0, sparse_A, Y, Mask, core_i, opt_const_dim=opt_const_dim)
    ind_i_fitness = eval_func(ind_i, pset, X0, sparse_A, Y, Mask)
    print(f'ind_i_fitness = {ind_i_fitness}')
    return ind_i, ind_i_fitness
def opt_constant_multi_core(pop, core_i, opt_const_dim=-1, num_cores=None):
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    pop_opt = []
    opt_flag_one_core = True
    process_partial = partial(
        process_individual,
        pop=pop,
        core_i=core_i,
        opt_const_dim=opt_const_dim,
        X0=X0,
        sparse_A=sparse_A,
        Y=Y,
        Mask=Mask,
        termination_condition_err=termination_condition_err
    )
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(
            process_partial,  
            range(len(pop))),
            total=len(pop),
            desc=f"Optimizing on core [{core_i}]"
        ))
    for i, (ind_i, ind_i_fitness) in enumerate(results):
        print(f'ind_i_fitness = {ind_i_fitness}')
        if ind_i_fitness <= termination_condition_err and opt_flag_one_core:
            opt_flag_one_core = False
            print(f"\n==== Found opt that meets the termination condition on core [{core_i}]\n")
        pop_opt.append([ind_i, ind_i_fitness])
    return pop_opt
def multi_processing_opt_constant_SR(pop, strategy_name,num_running_each_core=10, opt_const_dim=-1):
    num_cores = multiprocessing.cpu_count()
    total_individuals = len(pop)
    chunk_size = (total_individuals + num_cores - 1) // num_cores
    chunks = [
        pop[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_cores)
        if i * chunk_size < total_individuals  
    ]
    with multiprocessing.Pool(num_cores) as p:
        args = [(chunk, i + 1, opt_const_dim) for i, chunk in enumerate(chunks)]
        results = p.starmap(opt_constant_one_core_SR, args)
    new_pop = [item for sublist in results for item in sublist]
    print(f"优化后的种群长度: {len(new_pop)}")
    return new_pop
def multi_processing_opt_constant_SR_strategy_12_27(pop, strategy_name,num_running_each_core=10,opt_const_dim=-1):
    assert isinstance(strategy_name, str)
    total_individuals = len(pop)
    if total_individuals == 0:
        return []
    indexed_pop = list(enumerate(pop))
    MAX_CORES = 64
    num_cores = min(MAX_CORES, multiprocessing.cpu_count(), total_individuals)
    chunk_size = (total_individuals + num_cores - 1) // num_cores
    chunks = [
        indexed_pop[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_cores)
        if i * chunk_size < total_individuals
    ]
    MAX_CORES = 32
    num_workers = min(MAX_CORES, len(chunks))
    results = []
    with multiprocessing.Pool(processes=num_workers) as p:
        async_results = []
        for i, chunk in enumerate(chunks):
            ar = p.apply_async(
                opt_constant_one_core_SR_strategy,
                args=(chunk, i + 1, opt_const_dim)
            )
            async_results.append(ar)
        for i, ar in enumerate(async_results):
            try:
                sublist = ar.get(timeout=300)  
            except TimeoutError:
                print(f"[WARN] Core-{i+1} timeout, drop this chunk")
                sublist = []   
            except Exception as e:
                print(f"[ERROR] Core-{i+1} failed: {e}")
                sublist = []
            results.append(sublist)
    result_map = {}
    for sublist in results:
        for idx, ind_i, fit in sublist:
            result_map[idx] = (ind_i, fit)
    new_pop = []
    for i in range(total_individuals):
        old_item = pop[i].copy()
        new_ind, new_fit = result_map[i]
        old_item[0] = new_ind      
        old_item[3] = new_fit      
        if len(old_item) < 5:
            old_item.append(strategy_name)
        else:
            old_item[4] = strategy_name
        new_pop.append(old_item)
    print(f"优化后的种群长度: {len(new_pop)}")
    return new_pop
def worker_process_wrapper(result_queue, chunk, core_i, opt_const_dim):
    """
    子进程执行的包装函数。
    负责运行具体的优化逻辑，并将结果 put 到队列中。
    """
    try:
        result = opt_constant_one_core_SR_strategy(chunk, core_i, opt_const_dim)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)
def multi_processing_opt_constant_SR_strategy_12_31(pop, strategy_name, num_running_each_core=10, opt_const_dim=-1):
    assert isinstance(strategy_name, str)
    total_individuals = len(pop)
    if total_individuals == 0:
        return []
    indexed_pop = list(enumerate(pop))
    MAX_CORES = 40
    num_cores = min(MAX_CORES, multiprocessing.cpu_count(), total_individuals)
    chunk_size = (total_individuals + num_cores - 1) // num_cores
    chunks = [
        indexed_pop[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_cores)
        if i * chunk_size < total_individuals
    ]
    print(f"Starting manual multi-processing: {len(chunks)} chunks on {num_cores} cores.")
    pending_chunks = list(enumerate(chunks))
    pending_chunks.reverse()
    running_processes = {}
    collected_results = []
    CHUNK_TIMEOUT = 60
    pbar = tqdm(total=total_individuals, desc="Optimizing", unit="eq")
    try:
        while len(pending_chunks) > 0 or len(running_processes) > 0:
            while len(running_processes) < num_cores and len(pending_chunks) > 0:
                c_idx, c_data = pending_chunks.pop()
                q = multiprocessing.Queue()
                p = multiprocessing.Process(
                    target=worker_process_wrapper, 
                    args=(q, c_data, c_idx + 1, opt_const_dim)
                )
                p.daemon = True
                p.start()
                running_processes[c_idx] = {
                    'proc': p,
                    'queue': q,
                    'start_time': time.time(),
                    'data': c_data
                }
            active_indices = list(running_processes.keys())
            for c_idx in active_indices:
                info = running_processes[c_idx]
                proc = info['proc']
                q = info['queue']
                start_t = info['start_time']
                chunk_data = info['data']
                is_timeout = (time.time() - start_t) > CHUNK_TIMEOUT
                is_dead = not proc.is_alive()
                result_sublist = None
                error_msg = None
                if is_timeout:
                    tqdm.write(f"[WARN] Worker-{c_idx+1} TIMEOUT. Killing...")
                    if proc.is_alive():
                        proc.terminate()
                        time.sleep(0.1)
                        if proc.is_alive():
                            proc.kill()
                    proc.join()
                    error_msg = "TIMEOUT"
                elif is_dead:
                    try:
                        res = q.get_nowait()
                        if isinstance(res, Exception):
                            tqdm.write(f"[ERROR] Worker-{c_idx+1} Exception: {res}")
                            error_msg = "EXCEPTION"
                        else:
                            result_sublist = res
                    except queue.Empty:
                        tqdm.write(f"[ERROR] Worker-{c_idx+1} Crashed.")
                        error_msg = "CRASH"
                    proc.join()
                if result_sublist is not None:
                    collected_results.append(result_sublist)
                    del running_processes[c_idx]
                    pbar.update(len(result_sublist))
                elif error_msg is not None:
                    fallback_sublist = []
                    for global_idx, item in chunk_data:
                        ind = item[0]
                        fallback_sublist.append((global_idx, ind, 1e10))
                    collected_results.append(fallback_sublist)
                    del running_processes[c_idx]
                    pbar.update(len(fallback_sublist))
            time.sleep(0.1)
    finally:
        pbar.close()
    flat_results = [item for sublist in collected_results for item in sublist]
    result_map = {}
    for idx, ind_i, fit in flat_results:
        result_map[idx] = (ind_i, fit)
    new_pop = []
    for i in range(total_individuals):
        if i not in result_map:
            tqdm.write(f"[CRITICAL] Index {i} missing! Using original.")
            new_ind = pop[i][0]
            new_fit = 1e10
        else:
            new_ind, new_fit = result_map[i]
        old_item = pop[i].copy()
        old_item[0] = new_ind
        old_item[3] = new_fit
        if len(old_item) < 5:
            old_item.append(strategy_name)
        else:
            old_item[4] = strategy_name
        new_pop.append(old_item)
    print(f"优化后的种群长度: {len(new_pop)}")
    return new_pop
def multi_processing_opt_constant_discrete(pop, num_running_each_core=10, opt_const_dim=-1):
    num_cores = multiprocessing.cpu_count()
    total_individuals = len(pop)
    chunk_size = (total_individuals + num_cores - 1) // num_cores
    chunks = [
        pop[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_cores)
        if i * chunk_size < total_individuals  
    ]
    with multiprocessing.Pool(num_cores) as p:
        args = [(chunk, i + 1, opt_const_dim) for i, chunk in enumerate(chunks)]
        results = p.starmap(opt_constant_one_core_discrete, args)
    new_pop = [item for sublist in results for item in sublist]
    print(f"优化后的种群长度: {len(new_pop)}")
    return new_pop
def multi_processing_opt_constant(pop, num_running_each_core=10, opt_const_dim=-1):
    num_cores = multiprocessing.cpu_count()
    total_individuals = len(pop)
    chunk_size = (total_individuals + num_cores - 1) // num_cores
    chunks = [
        pop[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_cores)
        if i * chunk_size < total_individuals  
    ]
    with multiprocessing.Pool(num_cores) as p:
        args = [(chunk, i + 1, opt_const_dim) for i, chunk in enumerate(chunks)]
        results = p.starmap(opt_constant_one_core, args)
    new_pop = [item for sublist in results for item in sublist]
    print(f"优化后的种群长度: {len(new_pop)}")
    return new_pop
def get_return_pop(pop, filter=True):
    complex_of_eq_list = []
    for idx in range(len(pop)):
        complex_of_eq = len(pop[idx][0][0]) + len(pop[idx][0][1])
        pop[idx].append(complex_of_eq)
        complex_of_eq_list.append(complex_of_eq)
    complex_of_eq_list_sort = sorted(list(set(complex_of_eq_list)))
    best_ind_list = []
    for complex_i in complex_of_eq_list_sort:
        best_ind_complex_i = None
        for p_i in pop:
            if p_i[-1] == complex_i:
                if best_ind_complex_i is None or p_i[1] < best_ind_complex_i[1]:
                    best_ind_complex_i = p_i
        best_ind_list.append(best_ind_complex_i)
    if filter:
        best_ind_list_filter = []
        for ind in best_ind_list:
            if ind[1] <= tol_err:
                best_ind_list_filter.append(ind)
    else:
        best_ind_list_filter = best_ind_list
    return best_ind_list_filter
def make_pop_from_file(init_pop_str):
    POP = []
    for ii in tqdm(range(len(init_pop_str))):
        f_eq_str = init_pop_str.iloc[ii]['f_eq_str']
        g_eq_str = init_pop_str.iloc[ii]['g_eq_str']
        b1 = gp.PrimitiveTree.from_string_sympy(f_eq_str, pset_f)
        b2 = gp.PrimitiveTree.from_string_sympy(g_eq_str, pset_g)
        POP.append([(b1, b2), 1e30])
    return POP
