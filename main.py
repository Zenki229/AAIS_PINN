from libs import *

dtp = torch.float64

try:
    os.mkdir('./results')
except FileExistsError:
    pass


def main():
    parser = argparse.ArgumentParser(description='Training')
    # system and basic setting
    parser.add_argument('--cuda_dev', type=str, default='0',
                        help='choose the cuda device, default "0"(cuda:0)')
    parser.add_argument('--dirname', type=str, default="debug",
                        help='name of current saving folder in ./results, '
                             'default: "pde_name"+"domain_name"+"strategy_name"+debug')
    # pde setting
    parser.add_argument('--pde', type=str, default='Poisson2D1Peak',
                        help='pde type: default is Poisson2D1Peak. Others please see in libs')
    # net and optimizer
    parser.add_argument('--NeuralShape', nargs='+', type=int, default=[20, 7],
                        help='the  hidden shape of PINNs, [hiden_size, depth] all must be integer. default is [20,7]')
    parser.add_argument('--lr', nargs='+', type=float, default=[1e-4, 1.0],
                        help='adam learning rate and lbfgs (default: [1e-4, 1.0])')
    parser.add_argument('--epoch', nargs='+', type=int, default=[10, 10, 10, 10],
                        help='number of epochs, [adam_pretrain, lbfgs_pretrain, adam, lebfgs] the pre-training use adam of pretrain_epoch+lbfgs_epoch, if lbfgs_epoch=0, means no lbfgs in training. ')
    # adaptive sample setting
    parser.add_argument('--strategy', type=str, default='RAD_resample',
                        help='adaptive strategy: combination=SampleMethod_NodeCombineMethod, SampleMethod has "Uni", "AAIS_g", "AAIS_t", "RAD", NodeCombineMethod has "resample')
    parser.add_argument('--num_sample', nargs='+', type=int, default=[100, 100, 200],
                        help='num sampled in the domain, num[0] means the number of points uniformly sampled in the domain during the pretrain, num[1] means the number of points sampled on the boundary(including initial hypersurface), num[2] means the resampled(or added) points in the domain by different sampling methods.')
    parser.add_argument('--num_search', type=int, default=1000,
                        help='num of nosed used to search in the domain.')
    parser.add_argument('--weighted_sample', type=int, default=1,
                        help='weighted_sampling or not for mixture, 0 for False, 1 for True')
    # train set
    parser.add_argument('--max_iter', type=int, default=100,
                        help='max iteration for retrain, default is 100')
    args = parser.parse_args()
    device = torch.device('cuda:'+args.cuda_dev if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing {device}\n")
    try:
        os.mkdir('./results/' + f'{args.pde}/')
    except FileExistsError:
        pass
    path_father = './results/' + f'{args.pde}/' + args.pde + '_' + args.strategy + '_' + args.dirname
    try:
        os.mkdir(path_father)
    except FileExistsError:
        for root, dirs, files in os.walk(path_father, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    try:
        os.mkdir(path_father+'/img')
        os.mkdir(path_father+'/net')
        os.mkdir(path_father+'/train')
        os.mkdir(path_father+'/test')
    except FileExistsError:
        pass
    logger = log_gen(path=path_father)
    with open(path_father+"/inform.txt", "w") as f:
        f.write(str(args)+"\n")
    configs = load_yaml(r'./configs.yml', key=args.pde)
    with open(path_father + "/inform.txt", "a") as f:
        f.write(str(configs) + "\n")
    if 'Poisson2D1Peak' in args.pde:
        from libs.Poisson import Poisson2D1Peak
        pde = Poisson2D1Peak(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1], **configs)
    elif 'Poisson2D9Peak' in args.pde:
        from libs.Poisson import Poisson2D9Peak
        pde = Poisson2D9Peak(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1], **configs)
    elif 'Poisson2DLshape' in args.pde:
        from libs.Poisson import Poisson2DLshape
        pde = Poisson2DLshape(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1], **configs)
    elif 'Burgers2D' in args.pde:
        from libs.Burgers import Burgers2D
        pde = Burgers2D(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1], **configs)
    elif 'AllenCahn2D' in args.pde:
        from libs.AllenCahn import AllenCahn2D
        pde = AllenCahn2D(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1],
                          **configs)
    elif 'KdV2D' in args.pde:
        from libs.KdV import KdV1D
        pde = KdV1D(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1],
                    **configs)
    else:
        raise NotImplementedError
    net = DeepNeuralNet(input_size=pde.input_size,
                        hidden_size=args.NeuralShape[0],
                        output_size=pde.output_size,
                        depth=args.NeuralShape[1]
                        )
    print(f"\nTraining for {pde.__class__.__name__} with {get_num_params(net)} params under {args.strategy}\n")
    net.to(device=device, dtype=dtp)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr[0])
    if args.epoch[1] >= 1:
        lbfgs_pretrain = torch.optim.LBFGS(
            net.parameters(),
            lr=args.lr[1],
            max_iter=args.epoch[1],
            max_eval=args.epoch[1],
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # better numerical stability
        )
    else:
        lbfgs_pretrain = None
    if args.epoch[3] >= 1:
        lbfgs = torch.optim.LBFGS(
            net.parameters(),
            lr=args.lr[1],
            max_iter=args.epoch[3],
            max_eval=args.epoch[3],
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # better numerical stability
        )
    else:
        lbfgs = None
    if 'AAIS_t' in args.strategy:
        if 'highess' in args.strategy:
            configs = load_yaml(r'./configs.yml', key='AAIS_t_highess')
        else:
            configs = load_yaml(r'./configs.yml', key='AAIS_t')
        with open(path_father + "/inform.txt", "a") as f:
            f.write(str(configs) + "\n")
        sample_method_choose = AAISt(dim=pde.dim, log=logger, weighted_sample=args.weighted_sample, **configs)
    elif 'AAIS_g' in args.strategy:
        if 'highess' in args.strategy:
            configs = load_yaml(r'./configs.yml', key='AAIS_g_highess')
        else:
            configs = load_yaml(r'./configs.yml', key='AAIS_g')
        with open(path_father + "/inform.txt", "a") as f:
            f.write(str(configs) + "\n")
        sample_method_choose = AAISGaussian(dim=pde.dim, log=logger, weighted_sample=args.weighted_sample, **configs)
    elif 'RAD' in args.strategy:
        with open(path_father + "/inform.txt", "a") as f:
            f.write('RAD' + "\n")
        sample_method_choose = RAD(dim=pde.dim, log=logger)
    elif 'Uni' in args.strategy:
        with open(path_father + "/inform.txt", "a") as f:
            f.write('Uni_resample' + "\n")
        sample_method_choose = Uni(sample=pde.sample)
    else:
        raise NotImplementedError
    params = {
        'net': net,
        'pde': pde,
        'dev': device,
        'optimizer': optimizer,
        'lbfgs_pretrain': lbfgs_pretrain,
        'lbfgs': lbfgs,
        'optim_epoch': args.epoch,
        'file_path': path_father,
        'num_add': args.num_sample[2],
        'num_search': args.num_search,
        'logger': logger,
        'sample_method': sample_method_choose,
        'max_iter': args.max_iter,
    }
    if 'resample' in args.strategy:
        go_train = TrainResample(**params)
    elif 'add' in args.strategy:
        go_train = TrainAdd(**params)
    else:
        raise NotImplementedError
    go_train.forward()
    loss_err_plot(path_father=path_father)
    shape_ess_plot(path_father=path_father)


if __name__ == "__main__":
    main()
