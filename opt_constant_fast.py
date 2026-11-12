import time
from tqdm import tqdm
import numpy as np

import torch
import sympy as sp
import sympytorch


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
        # torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def opt_constant_batch(expressions_batch, data, device):
    X = data['x'].to(device)   # [#points, dim]
    Y = data['y'].to(device)   # [#points, #eqs]
    ##
    input_params = {}
    for dim_i in range(X.size(-1)):
        input_params['x_%s' % (dim_i+1)] = X[:, dim_i]
    ##

    # safe_sympy_exprs = []
    # for expr in expressions_batch:
    #     # 替换 x**0.5 为 abs(x)**0.5 (防止底数为负导致 NaN)
    #     # 替换 log(x) 为 log(abs(x))
    #     safe_expr = expr.replace(
    #         lambda e: e.is_Pow and abs(float(e.exp) - 0.5) < 1e-6,
    #         lambda e: sp.Pow(sp.Abs(e.base), e.exp) # +1e-6 防止0梯度爆炸
    #     )
    #     safe_sympy_exprs.append(safe_expr)
    mod = sympytorch.SymPyModule(expressions=expressions_batch).to(device)
    
    print(mod)
    # # ==================== [新增逻辑: 冻结特定常数] ====================
    # print("Freezing constants (0.5)...")
    # frozen_count = 0
    # for param in mod.parameters():
    #     # 检测参数值是否接近 0.5
    #     if abs(param.item() - 0.5) < 1e-6:
    #         param.requires_grad = False
    #         frozen_count += 1
            
    # print(f"Frozen {frozen_count} parameters equal to 0.5")
    # # ===============================================================

    # 注意：这里传入 mod.parameters() 时，PyTorch 优化器会自动忽略 requires_grad=False 的参数
    # 或者为了更严谨，你可以只传 filter(lambda p: p.requires_grad, mod.parameters())
    # optim = torch.optim.Adam(filter(lambda p: p.requires_grad, mod.parameters()), lr=5e-2)
    #for p in mod.parameters():
    #    print(p)
    
    optim = torch.optim.Adam(mod.parameters(), lr=5e-2)
    # optim = torch.optim.SGD(mod.parameters(), lr=1e-1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.9, patience=10)

    num_epochs = 2000
    patience = 20
    early_stopping = EarlyStopping(patience=patience, verbose=False, delta=1e-4)

    loss_func = torch.nn.MSELoss()

    aa = time.time()
    y_var = torch.var(Y, unbiased=False) + 1e-15

    for epoch in tqdm(range(num_epochs)):
        mod.train()

        optim.zero_grad()

        out = mod(**input_params)  # out has shape (200, 2)
        # if epoch == 0:
        #     print("--- DEBUG INFO ---")
        #     print(f"Keys in input_params: {list(input_params.keys())}")
        #     # print(f"Model expects symbols: {mod.expressions[0].free_symbols}")
        #     print(f"Output mean: {out.mean().item()}, Output std: {out.std().item()}")
        #     print(f"Y_tensor mean: {Y.mean().item()}")
            # 如果 out 全是 0，说明 input_params 没传进去
        #print(out)


        loss_each = torch.log(torch.mean((out - Y) ** 2, dim=0)/y_var + 1.)
        loss = torch.mean(loss_each)

        # loss = torch.sum(torch.mean((out - Y) ** 2, dim=0))
        # loss = torch.sum(torch.mean(torch.abs(out - Y)/torch.abs(Y), dim=0))

        # loss = loss_func(out, Y)
        # loss = torch.sum(loss_each)

        # print(loss_each)
        
        loss.backward()

        optim.step()  ## update paramters

        lr_scheduler.step(loss.item())

        early_stopping(loss.item(), mod)

        if early_stopping.early_stop or epoch >= num_epochs - 1:
            print('epoch #%s, loss = %.5f, time cost = %.2f' % (epoch, loss.item(), time.time() - aa))
            if early_stopping.early_stop:
                print("Early stopping")
            return [str(expr_i) for expr_i in mod.sympy()], loss_each.detach().exp().cpu().numpy() - 1.


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #device = torch.device("cpu")

    # print(device)

    def sample_uniform(min_=1, max_=5):
        return np.random.rand() * (max_ - min_) + min_


    expressions = []
    for _ in range(1):
        expressions.extend(["%.4f * cos(%.4f*x_1) + %.4f + %.4f * cos(x_2 + %.4f) + %.4f " % (sample_uniform(), sample_uniform(),sample_uniform(), sample_uniform(),sample_uniform(), sample_uniform()),
                   "%.4f * sin(x_1) + %.4f + %.4f * sin(%.4f*x_2) + %.4f" % (sample_uniform(), sample_uniform(),sample_uniform(), sample_uniform(), sample_uniform()),
                   "%.4f * x_1**2 + %.4f * x_1 + %.4f + %.4f * x_2**2 + %.4f * x_2 + %.4f" % (sample_uniform(), sample_uniform(), sample_uniform(),sample_uniform(), sample_uniform(), sample_uniform()),
                   "%.4f * sqrt(%.4f*x_1) + %.4f + %.4f * x_2**2 + %.4f*x_1*x_2" % (sample_uniform(), sample_uniform(),sample_uniform(), sample_uniform(), sample_uniform()),
                   ])
    expressions = [sp.simplify(e) for e in expressions]

    print(expressions)
    print(len(expressions))


    # # 统一安全处理
    # safe_expressions = []
    # for expr in expressions:
    #     # 确保是 SymPy 对象
    #     if isinstance(expr, str):
    #         expr = sp.sympify(expr)
            
    #     # 自动查找所有开根号操作，并给底数加上绝对值
    #     safe_expr = expr.replace(
    #         lambda e: e.is_Pow and abs(float(e.exp) - 0.5) < 1e-6,
    #         lambda e: sp.Pow(sp.Abs(e.base), e.exp)
    #     )
    #     # 1. 修复 Log: log(x) -> log(|x| + 1e-6)
    #         # 解决: log(-5) -> nan 和 log(0) -> -inf 的问题
    #     safe_expr = safe_expr.replace(
    #         lambda e: isinstance(e, sp.log),
    #         lambda e: sp.log(sp.Abs(e.args[0]) + 1e-16)
    #     )

        # 2. 修复小数次幂: x**0.5 -> (|x| + 1e-6)**0.5
        # 解决: (-5)**0.5 -> nan 的问题
        # safe_expr = expr.replace(
        #     lambda e: e.is_Pow and not e.exp.is_integer,
        #     lambda e: sp.Pow(sp.Abs(e.base), e.exp)
        # )
    #     safe_expressions.append(safe_expr)

    # expressions = safe_expressions

    support_max = 10.0
    support_min = 9.0

    x_ = torch.rand(2000, 2) * (support_max - support_min) + support_min

    # ground truth
    y_truth = torch.cat([x_.cos().sum(-1).unsqueeze(-1),
                         -x_.sin().sum(-1).unsqueeze(-1),
                         (-0.5 * x_ ** 2 - x_).sum(-1).unsqueeze(-1),
                         (-0.5 * x_ ** 2 + x_).sum(-1).unsqueeze(-1),
                         ] * 1, dim=-1)

    data = {
        'x': x_,  # [#points, dim]
        'y': y_truth  # [#points, #eqs]
    }

    opt_sympy = opt_constant_batch(expressions, data, device)

    print(opt_sympy)
