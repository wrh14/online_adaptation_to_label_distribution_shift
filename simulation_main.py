import numpy as np
from derivative import dxdt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import os
from copy import deepcopy
import time

import argparse

parser = argparse.ArgumentParser(description='Online label shifting-multi class')
parser.add_argument('--algo', type=str, default='const', help='[const, history_dist, fixed_history_dist, ogd, ogd_{lr}]')
parser.add_argument('--shift_process', type=str, default='constant_shift')
parser.add_argument('--dataset', type=str, default='cifar10', help='[cifar10, svhn]')
parser.add_argument('--model', type=str, default='resnet18', help='[resnet18, resnet50]')
parser.add_argument('--cal_stat', type=str, default='cal', help='[cal, no_cal]')
parser.add_argument('--T', type=int, default=100000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--smooth_k', type=int, default=3)
parser.add_argument('--grad_N', type=int, default=20)
parser.add_argument('--delta', type=float, default=0.01)
parser.add_argument('--conf_type', type=str, default="zero_one")
args = parser.parse_args()
print(args)

def load_val_preds_y_conf_mat(dataset, model, cal_stat="cal"):
    checkpoint = torch.load("data_preparation/val_{}_{}_{}.pt".format(dataset, model, cal_stat))
    val_preds = checkpoint["val_preds"]
    val_y = checkpoint["val_y"]
    val_conf_mat = checkpoint["conf_mat"].numpy()
    return val_preds, val_y, val_conf_mat

def smooth(y, k):
    box = np.ones(2 * k + 1)/ (2 * k + 1)
    y = np.concatenate([np.ones(k)*y[0], y, np.ones(k)*y[-1]])
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def load_numerical_grad_func(numerical_loss):
    def compute_grad(i_p_N, delta=args.delta):
        (i, p, q, N, numerical_loss) = i_p_N
        ps = np.concatenate([np.expand_dims(p, axis=0) for j in range(N)], axis=0)
        p_i_s = p[i] + (np.arange(N) - int(N/2)) * delta
        ps[:, i] = p_i_s
        func_vals = np.asarray([numerical_loss(grid_p, q) for grid_p in ps])
        func_vals = smooth(func_vals, k=args.smooth_k)
        grads_i = dxdt(func_vals[np.logical_and(p_i_s >= 0, p_i_s <= 1)], p_i_s[np.logical_and(p_i_s >= 0, p_i_s <= 1)], kind="finite_difference", k=3)
        return grads_i[p_i_s[np.logical_and(p_i_s >= 0, p_i_s <= 1)] == p[i]]

    def numerical_loss_grad(p, q, N=args.grad_N):
        i_p_q_N_list = [(i, deepcopy(p), deepcopy(q), N, deepcopy(numerical_loss)) for i in range(len(p))]
        results = [compute_grad(i_p_q_N) for i_p_q_N in i_p_q_N_list]
        grad = np.asarray(results).squeeze()
        return grad
    return numerical_loss_grad

def load_numerical_grad_func_fast(val_preds, val_y, p_train):
    val_preds_div_p_train = val_preds / p_train
    def numerical_loss_grad(p, q):
        grad = np.zeros(num_classes)
        for i in range(num_classes):
            if_i = (val_y == i)
            b = val_preds_div_p_train[if_i] 
            ai = val_preds_div_p_train[if_i, i]
            sum_bi_pi = (b * p).sum(1)
            grad += -( b.transpose() / (sum_bi_pi ** 2) * ai * p[i]).mean(1) * q[i]
            grad[i] += (ai / sum_bi_pi).mean() * q[i]
        return -grad
    return numerical_loss_grad

def load_numerical_loss_func(val_preds, val_y, p_train):
    val_preds_div_p_train = val_preds / p_train
    def numerical_loss(p, q):
        adjust_preds = val_preds_div_p_train * p
        conf_diag = np.zeros(len(p))
        conf_diag = np.asarray([(np.argmax(adjust_preds[val_y == i], axis=1) == i).mean() for i in range(len(p))])
        return (1 - conf_diag).dot(q)
    return numerical_loss

def history_dist_algo(base_pred_list, p_cur):
    q_pred_hist = np.asarray([(base_pred_list == i).mean() for i in range(d)])
    p_next = q_pred_hist.dot(inv_val_conf_mat)
    return p_next

def const_algo(base_pred_list, p_cur):
    p_next = p_cur
    return p_next

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def gd_constructor(lr, d, inv_val_conf_mat, eps_cube=None):
    def gd(base_pred_list, p_cur):
        q_vec = np.zeros([d])
        q_vec[base_pred_list[-1]] = 1
        mean_q = q_vec.dot(inv_val_conf_mat)

        grad = loss_grad_func(p_cur, mean_q)
        p_next = p_cur - lr * grad
        if eps_cube is None:
            return projection_simplex_sort(p_next)
        else:
            p_next_new = projection_simplex_sort(np.clip(p_next, eps_cube, 1 - eps_cube))
            while np.linalg.norm(p_next_new - p_next) >= 1e-4:
                p_next = p_next_new
                p_next_new = projection_simplex_sort(np.clip(p_next, eps_cube, 1 - eps_cube))
            return p_next_new
    return gd

def exp_gd_constructor(lr, d, inv_val_conf_mat, eps_cube=None):
    def exp_gd(base_pred_list, p_cur):
        q_vec = np.zeros([d])
        q_vec[base_pred_list[-1]] = 1
        mean_q = q_vec.dot(inv_val_conf_mat)
        grad = loss_grad_func(p_cur, mean_q)
        p_next = p_cur * np.exp(-lr * grad)
        p_next = p_next / p_next.sum()
        return p_next
    return exp_gd

def fix_length_history_constructor(window_size, d, inv_val_conf_mat, eps_cube=None):
    def fix_length_history_dist_algo(base_pred_list, p_cur):
        q_pred_hist = np.asarray([(base_pred_list[-window_size:] == i).mean() for i in range(d)])
        p_next = q_pred_hist.dot(inv_val_conf_mat)
        return p_next
    return fix_length_history_dist_algo
    
def follow_the_leader_constructor(lr, steps):
    def follow_the_leader(base_pred_list, p_cur):
        q_pred_hist = np.asarray([(base_pred_list == i).mean() for i in range(d)])
        q_hist = q_pred_hist.dot(inv_val_conf_mat)
        q_hist = np.ones(d) / d
        p_next = p_cur
        for step in range(steps):
            p_next = projection_simplex_sort(p_next - lr * loss_grad_func(p_next, q_hist))
            print(f"step: {step}; loss: {loss_func(p_next, q_hist)}")
        return p_next
    return follow_the_leader

def generate_test_probs_and_ys(q_all):
    np.random.seed(args.seed)
    ys = np.squeeze(np.asarray([np.random.choice(10, 1, p=q) for q in q_all]))
    num_test = len(ys)
    probs = np.zeros([num_test, num_classes])
    for i in range(num_classes):
        num_i = (ys == i).sum()
        if num_i == 0:
            continue
        num_test_i = (test_y == i).sum()
        sampled_indices = np.random.randint(0, num_test_i, num_i)
        probs[ys == i] = test_preds[test_y == i][sampled_indices]
    np.random.seed(int(time.time()))
    return probs, ys

def find_opt(q, init_p, lr=1e-1, max_T=40):
    p_cur = init_p
    grad = loss_grad_func(p_cur, q)
    p_next = projection_simplex_sort(p_cur - lr * grad)
    
    min_loss = loss_func(p_next, q)
    opt_p = p_next
    for i in range(max_T):
        print(loss_func(p_cur, q), np.linalg.norm(grad))
        p_cur = p_next
        grad = loss_grad_func(p_cur, q)
        p_next = projection_simplex_sort(p_cur - lr * grad)
        if loss_func(p_next, q) < min_loss:
            min_loss = loss_func(p_next, q)
            opt_p = p_next
        if i >= 2/3*max_T:
            lr=lr / 10
    return opt_p

def online_process(probs, ys, algos, p_train, seed=0, verbose=False, checkpoint=None):
    T = len(ys)
    base_pred_y = np.argmax(probs, axis=1)
    
    algo_name = [args.algo, "const"]
    
    if checkpoint is None:
        loss = np.zeros([len(algos), T])
        p_vec = []
        for i in range(len(algos)):
            if algo_name[i] == "opt_const":
                p_vec.append(np.asarray(q_all).mean(0))
            if algo_name[i] == "true_opt_const":
                checkpoint_path = "checkpoint/opt_p{}_{}_{}_{}_{}_{}.pt".format(pre_name, args.dataset, args.model, args.cal_stat, args.T, args.shift_process)
                if os.path.exists(checkpoint_path):
                    opt_p = torch.load(checkpoint_path)
                else:
                    opt_p = find_opt(np.asarray(q_all).mean(0), init_p=np.asarray(q_all).mean(0), lr=1e-2)
                    torch.save(opt_p, checkpoint_path)
                p_vec.append(opt_p)
            else:
                p_vec.append(p_train)
        if verbose:
            p_hist = [p_vec]
        start_T=0
    else:
        start_T = checkpoint["t"] + 1
        p_hist = checkpoint["p_hist"]
        loss = checkpoint["loss"]
        p_vec = p_hist[-1]
    
    pbar = tqdm(range(start_T, T), total=T-start_T)
    for t in pbar:
        for i, algo in enumerate(algos):
            adjust_prob = probs[t] * p_vec[i] / p_train
            pred = np.argmax(adjust_prob)
            base_pred_y[t] = np.argmax(probs[t])
            loss[i, t] = (pred != ys[t]).astype(np.float)
            p_vec[i] = algo(base_pred_y[:t+1], p_vec[i])
            
        if verbose:
            p_hist.append(deepcopy(p_vec))
        if t%1000== 0 or t == T-1:
            pbar.set_description(" ".join(["Alg {}: {:.4f}".format(algo_name[i], loss[i, :t+1].mean()) for i in range(len(algos))]))
            checkpoint = {}
            checkpoint["t"] = t
            checkpoint["p_hist"] = p_hist
            checkpoint["loss"] = loss
            
            torch.save(checkpoint, "checkpoint/online_checkpoint_{}_{}_{}_{}_{}_{}{}.pt".format(dataset, model, cal_stat, T, args.shift_process, args.algo, pre_name))
            
    if verbose:
        return loss.mean(1), loss, p_hist
    else:
        return loss.mean(1)
    
def constant_shift_constructor(q):
    def constant_shift(T):
        return np.concatenate([np.expand_dims(q, axis=0) for t in range(T)], axis=0)
    return constant_shift

def monotone_shift_constructor(q1, q2):
    def monotone_shift(T):
        lamb = 1.0 / (T-1)
        return np.concatenate([np.expand_dims(q1 * (1 - lamb * t) + q2 * lamb * t, axis=0) for t in range(T)], axis=0)
    return monotone_shift

def period_shift_constructor(q1, q2, T_p):
    def period_shift(T):
        return np.concatenate([np.expand_dims(q1 * (1 - int(int(t / T_p)%2 > 0)) + q2 * int(int(t / T_p)%2 > 0), axis=0) for t in range(T)], axis=0)
    return period_shift

def period_continuous_shift_constructor(q1, q2, T_p):
    def period_continuous_shift(T):
        return np.concatenate([np.expand_dims((q1 * (1 - float(t%T_p) / T_p) + q2 * (float(t%T_p) / T_p)) * (1 - int(int(t / T_p)%2 > 0)) + (q2 * (1 - float(t%T_p) / T_p) + q1 * (float(t%T_p) / T_p)) * int(int(t / T_p)%2 > 0), axis=0) for t in range(T)], axis=0)
    return period_continuous_shift

def exp_period_shift_constructor(q1, q2, k=2):
    def exp_period_shift(T):
        return np.concatenate([np.expand_dims(q1 * (1 - int(int(np.log(t+1) / np.log(k))%2 > 0)) + q2 * int(int(np.log(t+1) / np.log(k))%2 > 0), axis=0) for t in range(T)], axis=0)
    return exp_period_shift

def uniform_sample_from_simplex(d):
    x = np.random.rand(d - 1)
    x = x[np.argsort(x)]
    return np.concatenate([x, np.ones(1)]) - np.concatenate([np.zeros(1), x])
  
if not os.path.exists("./checkpoint"):
    os.makedirs("checkpoint")   
pre_name = ""
if (args.smooth_k != 3) or (args.grad_N != 20) or (args.delta != 0.01):
    pre_name = f"_{args.smooth_k}_{args.grad_N}_{args.delta}"
if args.conf_type != "zero_one":
    pre_name = pre_name + f"_{args.conf_type}"
    eps_cube = 0.01
else:
    eps_cube = None
if args.seed is not None:
    pre_name = pre_name + f"_{args.seed}"  

dataset = args.dataset
model = args.model
T = args.T
cal_stat = args.cal_stat
val_preds, val_y, val_conf_mat = load_val_preds_y_conf_mat(dataset, model, cal_stat)
val_preds = nn.Softmax()(torch.from_numpy(val_preds)).numpy()
d = val_conf_mat.shape[0]

checkpoint = torch.load("data_preparation/test_{}_{}_{}.pt".format(dataset, model, cal_stat))
test_preds = checkpoint["test_preds"]
test_y = checkpoint["y"]
p_train = checkpoint["p_train"]
test_preds = nn.Softmax()(torch.from_numpy(test_preds)).numpy()
val_preds_div_p_train = val_preds / p_train

num_classes = np.max(test_y) + 1

loss_func = load_numerical_loss_func(val_preds, val_y, p_train)
if args.conf_type == "zero_one":
    loss_grad_func = load_numerical_grad_func(loss_func)
elif args.conf_type == "prob":
    loss_grad_func = load_numerical_grad_func_fast(val_preds, val_y, p_train)
inv_val_conf_mat = np.linalg.inv(val_conf_mat)

if (not os.path.exists("checkpoint/max_M{}_{}_{}_{}_{}.pt".format(pre_name, eps_cube, dataset, model, cal_stat))) and (args.algo == "ogd"):
    M = 0
    max_p = None
    max_j = None
    sample_num = 100
    for i1 in tqdm(range(d)):
        if eps_cube is not None:
            p = np.ones(d) * eps_cube
            p[i1] = 1 - (d -1) * eps_cube
            p = p / p.sum()
        else:
            p = np.zeros(d)
            p[i1] = 1
        for j in range(d):
            q_hat = inv_val_conf_mat[j]
            grad_norm = np.linalg.norm(loss_grad_func(p, q_hat))
            if grad_norm > M:
                M = grad_norm
                max_p = p
                max_j = j

    np.random.seed(0)
    rand_P_list = [uniform_sample_from_simplex(d) for _ in range(sample_num)]
    for p in tqdm([np.ones(d)/d] + rand_P_list):
        if eps_cube is not None:
            p_proj = p
            p_proj_new = projection_simplex_sort(np.clip(p, eps_cube, 1 - eps_cube))
            while np.linalg.norm(p_proj_new - p_proj) >= 1e-4:
                p_proj = p_proj_new
                p_proj_new = projection_simplex_sort(np.clip(p_proj, eps_cube, 1 - eps_cube))
            p = p_proj_new
        for j in range(d):
            pred_vec = np.zeros(d)
            pred_vec[j] = 1
            q_hat = pred_vec.dot(inv_val_conf_mat)
            grad_norm = np.linalg.norm(loss_grad_func(p, q_hat))
            if grad_norm > M:
                M = grad_norm
                max_p = p
                max_j = j
    np.random.seed(int(time.time()))
    torch.save({"M" : M, "max_p": max_p, "max_j": max_j}, "checkpoint/max_M{}_{}_{}_{}_{}.pt".format(pre_name, eps_cube, dataset, model, cal_stat))

if args.shift_process == "constant_shift":
    maj_class = 3
    q_const = np.ones(num_classes) * 0.05
    q_const[maj_class] = 1 - (q_const.sum() - q_const[maj_class])
    shift_proccess = constant_shift_constructor(q_const)
elif args.shift_process == "monotone_shift":
    maj_class_1 = 3
    maj_class_2 = 5
    q1 = np.ones(num_classes) * 0.05
    q1[maj_class_1] = 1 - (q1.sum() - q1[maj_class_1])
    q2 = np.ones(num_classes) * 0.05
    q2[maj_class_2] = 1 - (q2.sum() - q2[maj_class_2])
    shift_proccess = monotone_shift_constructor(q1, q2)
elif args.shift_process.startswith("exp_period_shift"):
    maj_class_1 = 3
    maj_class_2 = 5
    q1 = np.ones(num_classes) * 0.05
    q1[maj_class_1] = 1 - (q1.sum() - q1[maj_class_1])
    q2 = np.ones(num_classes) * 0.05
    q2[maj_class_2] = 1 - (q2.sum() - q2[maj_class_2])
    shift_proccess = exp_period_shift_constructor(q1, q2, k=float(args.shift_process.split("_")[-1]))
elif args.shift_process.startswith("period_shift"):
    maj_class_1 = 3
    maj_class_2 = 5
    q1 = np.ones(num_classes) * 0.05
    q1[maj_class_1] = 1 - (q1.sum() - q1[maj_class_1])
    q2 = np.ones(num_classes) * 0.05
    q2[maj_class_2] = 1 - (q2.sum() - q2[maj_class_2])
    shift_proccess = period_shift_constructor(q1, q2, T_p=int(args.shift_process.split("_")[-1]))


q_all = shift_proccess(T)
probs, ys = generate_test_probs_and_ys(q_all)
q_dist = np.asarray(q_all).mean(0)

if args.algo == "fth":
    algo = history_dist_algo
elif args.algo.startswith("ftfwh"):
    algo = fix_length_history_constructor(window_size=int(args.algo.split("_")[-1]), d=d, inv_val_conf_mat=inv_val_conf_mat, eps_cube=eps_cube)
elif args.algo == "ogd":
    M = torch.load("checkpoint/max_M{}_{}_{}_{}_{}.pt".format(pre_name, eps_cube, dataset, model, cal_stat))["M"]
    print("lr: {}".format(1 / (np.sqrt(T / 2) * M)))
    algo = gd_constructor(lr=1 / (np.sqrt(T / 2) * M), d=d, inv_val_conf_mat=inv_val_conf_mat, eps_cube=eps_cube)
elif args.algo.startswith("ogd"):
    lr = float(args.algo.split("_")[1])
    algo = gd_constructor(lr=lr, d=d, inv_val_conf_mat=inv_val_conf_mat, eps_cube=eps_cube)
elif args.algo == "const":
    algo = const_algo
elif args.algo == "opt_const":
    algo = const_algo

checkpoint = None
checkpoint_path = "checkpoint/online_checkpoint_{}_{}_{}_{}_{}_{}{}.pt".format(dataset, model, cal_stat, T, args.shift_process, args.algo, pre_name)
if os.path.exists(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
    except:
        checkpoint = None
algos = [algo, const_algo]
loss_mean, loss, p_hist = online_process(probs, ys, algos, p_train, verbose=True, checkpoint=checkpoint)