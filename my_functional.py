import torch.nn.functional as f
import torch


def my_identity(x):
    return x


def my_one(x):    
    return torch.ones_like(x)


def my_relu(x):
    y = x.clone()
    # y = x
    y[y < 0] = 0
    return y


def my_relu_dot(x):
    y = x.clone()
    # y = x
    y[y <= 0] = 0
    y[y > 0] = 1
    return y


def my_leaky_relu_dot(x):
    y = x.clone()
    y[y <= 0] = 0.01
    y[y > 0] = 1
    return y


def my_leaky_relu_inv(x):
    y = x.clone()
    y[y < 0] = 100 * y[y < 0]
    return y


def my_softplus(x):
    y = x.clone()
    y[y < 88] = torch.log(0.8 + torch.exp(y[y < 88]))
    return y


def my_softplus_inv(x):
    const = torch.log(torch.tensor(0.8)) + 10 ** (-7)
    y = x.clone()
    y[y < const] = const
    y[y < 88] = torch.log(torch.exp(y[y < 88]) - 0.8)
    return y


def my_softplus_dot(x):
    y = x.clone()
    y[y > 88] = 88
    y = torch.exp(y) / (0.8 + torch.exp(y))
    return y


def inv_fun(fun):
    inv_f = my_identity
    if fun == f.relu:
        inv_f = f.relu
    elif fun == f.leaky_relu:
        inv_f = my_leaky_relu_inv
    elif fun == my_leaky_relu_inv:
        inv_f = f.leaky_relu
    elif fun == my_softplus:
        inv_f = my_softplus_inv
    elif fun == my_softplus_inv:
        inv_f = my_softplus
    return inv_f


def fun_cut(val, func):
    val_c = val.clone()
    if func == f.relu:
        val_c[val_c < 0] = 0
    elif func == my_softplus:
        const = torch.log(torch.tensor(0.8))
        val_c[val_c < const] = const
    return val_c


def derivative_fun(fun):
    dot_f = my_one
    if fun == f.relu:
        dot_f = my_relu_dot
    elif fun == f.leaky_relu:
        dot_f = my_leaky_relu_dot
    elif fun == my_softplus:
        dot_f = my_softplus_dot
    return dot_f


def fun_max_derivative(fun):
    max_slope = 1
    if fun == f.relu or fun == f.leaky_relu or fun == my_softplus:
        max_slope = 1
    elif fun == torch.sigmoid or fun == f.sigmoid:
        max_slope = 0.25
    return max_slope
