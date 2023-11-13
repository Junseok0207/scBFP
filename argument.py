import argparse

def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='baron_mouse')
    parser.add_argument('--eval_clustering', type=str2bool, default=True)   
    parser.add_argument('--drop_rate', type=float, default=0.0)

    # Preprocessing
    parser.add_argument('--filter', type=str2bool, default=True)
    parser.add_argument('--cell_min', type=int, default=1)
    parser.add_argument('--gene_min', type=int, default=1)
    parser.add_argument('--hvg', type=str2bool, default=True)
    parser.add_argument('--n_hvg', type=int, default=2000)
    parser.add_argument('--sf', type=str2bool, default=True)
    parser.add_argument('--log', type=str2bool, default=True)

    # graph construction
    parser.add_argument('--cell_k', type=int, default=10)
    parser.add_argument('--gene_k', type=int, default=10)
    parser.add_argument('--knnfast', type=str2bool, default=False)
    parser.add_argument('--fb', type=int, default=10000)

    # diffusion
    parser.add_argument('--gene_iter', type=int, default=10)
    parser.add_argument('--cell_iter', type=int, default=40)

    # setting
    parser.add_argument('--gpu', type=str2bool, default=True)
    parser.add_argument('--device', type=int, default=0)

    return parser.parse_known_args()

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals

def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        st_ = "{} <- {} / ".format(name, val)
        st += st_

    return st[:-1]

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['device']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]


