from libs import *

get_seed(229)
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
    parser.add_argument('--pde', type=str, default='LdC2D',
                        help='pde type: default is Poisson2D1Peak. Others please see in libs')
    # net and optimizer
    parser.add_argument('--NeuralShape', nargs='+', type=int, default=[20, 7],
                        help='the  hidden shape of PINNs, [hiden_size, depth] all must be integer. default is [20,7]')
    parser.add_argument('--lr', nargs='+', type=float, default=[1e-4, 1.0],
                        help='adam learning rate and lbfgs (default: [1e-4, 1.0])')
    parser.add_argument('--epoch', nargs='+', type=int, default=[10, 10, 10],
                        help='number of epochs, [pretraining, adam, lebfgs] the pre-training use adam of pretrain_epoch+lbfgs_epoch, if lbfgs_epoch=0, means no lbfgs in training. ')
    # adaptive sample setting
    parser.add_argument('--strategy', type=str, default='Uni_add',
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
    parser.add_argument('--loss_tol', type=int, default=0,
                        help='loss tolerance for model')
    args = parser.parse_args()
    device = torch.device('cuda:'+args.cuda_dev if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing {device}\n")
    path_father = './results/' + args.pde + '_' + args.strategy + '_' + args.dirname
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
    elif 'Heat3D' in args.pde:
        from libs.Heat import Heat2D
        pde = Heat2D(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1], **configs)
    elif 'Burgers2D' in args.pde:
        from libs.Burgers import Burgers2D
        pde = Burgers2D(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1], **configs)
    elif 'Burgers3D' in args.pde:
        from libs.Burgers import Burgers3D
        pde = Burgers3D(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1],
                        **configs)
    elif 'NSCylinder' in args.pde:
        from libs.NavierStokes import NSCylinder
        pde = NSCylinder(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1], **configs)
    elif 'NSwake' in args.pde:
        from libs.NavierStokes import NSWake
        pde = NSWake(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1], **configs)
    elif 'AllenCahn2D' in args.pde:
        from libs.AllenCahn import AllenCahn2D
        pde = AllenCahn2D(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1],
                          **configs)
    elif 'AC2DHC' in args.pde:
        from libs.AllenCahn import AC2DHC
        pde = AC2DHC(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1],
                     **configs)
    elif 'KdV2D' in args.pde:
        from libs.KdV import KdV1D
        pde = KdV1D(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1],
                    **configs)
    elif 'Wave3D' in args.pde:
        from libs.Wave import Wave2D
        pde = Wave2D(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1],
                     **configs)
    elif 'LdC2D' in args.pde:
        from libs.NavierStokes import LdC2D
        pde = LdC2D(dev=device, dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1],
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
    if args.epoch[2] >= 1:
        lbfgs = torch.optim.LBFGS(
            net.parameters(),
            lr=args.lr[1],
            max_iter=args.epoch[2],
            max_eval=args.epoch[2],
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",  # better numerical stability
        )
    else:
        lbfgs = None
    IS_sign = True
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
        sample_method_choose = Uni
        IS_sign = False
    else:
        raise NotImplementedError
    params = {
        'net': net,
        'pde': pde,
        'dev': device,
        'optimizer': optimizer,
        'scheduler': None,
        'lbfgs': lbfgs,
        'optim_epoch': args.epoch[0:2],
        'file_path': path_father,
        'num_add': args.num_sample[2],
        'num_search': args.num_search,
        'logger': logger,
        'sample_method': sample_method_choose,
        'IS_sign': IS_sign,
        'max_iter': args.max_iter,
        'loss_tol': args.loss_tol
    }
    if 'resample' in args.strategy:
        go_train = train_resample(**params)
    elif 'add' in args.strategy:
        go_train = train_add(**params)
    else:
        raise NotImplementedError
    go_train.forward()
    loss_err_plot(path_father=path_father)
    if IS_sign:
        shape_ess_plot(path_father=path_father)


if __name__ == "__main__":
    main()
