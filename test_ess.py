from libs import *

dtp = torch.float64

try:
    os.mkdir('./results')
except FileExistsError:
    pass


def main():
    parser = argparse.ArgumentParser(description='Training')
    # system and basic setting
    parser.add_argument('--dirname', type=str, default="debug",
                        help='name of current saving folder in ./results, '
                             'default: "pde_name"+"domain_name"+"strategy_name"+debug')
    # function setting
    parser.add_argument('--func', type=str, default='Poisson9Peaks_testESS',
                        help='pde type: default is Poisson9Peaks. Others please see in libs')
    # adaptive sample setting
    parser.add_argument('--strategy', type=str, default='AAIS_t_resample',
                        help='adaptive strategy: combination=SampleMethod_NodeCombineMethod, SampleMethod has "Uni", "AAIS_g", "AAIS_t", "RAD", NodeCombineMethod has "resample')
    parser.add_argument('--num_sample', nargs='+', type=int, default=[200],
                        help='number of samples from the sample method')
    parser.add_argument('--num_search', type=int, default=1000,
                        help='num of nosed used to search in the domain.')
    parser.add_argument('--weighted_sample', type=int, default=1,
                        help='weighted_sampling or not for mixture, 0 for False, 1 for True')
    # train set
    args = parser.parse_args()
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
    configs = load_yaml(r'./configs.yml', key=args.func)
    with open(path_father + "/inform.txt", "a") as f:
        f.write(str(configs) + "\n")
    if 'Poisson9Peaks' in args.func:
        from libs.TestESS import TestESS
        pde = TestESS(dtp=dtp, num_in=args.num_sample[0], num_bd=args.num_sample[1], **configs)
    else:
        raise NotImplementedError
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
        'func': pde,
        'optim_epoch': args.epoch,
        'file_path': path_father,
        'num_add': args.num_sample[2],
        'num_search': args.num_search,
        'logger': logger,
        'sample_method': sample_method_choose,
        'max_iter': args.max_iter,
    }
    go_train = ESSResample(**params)
    go_train.forward()
    loss_err_plot(path_father=path_father)
    shape_ess_plot(path_father=path_father)

class ESSResample:
    def __init__(self, func, file_path, logger: logging, num_add, num_search, sample_method):
        self.func = func
        self.file_path = file_path
        self.logger = logger
        self.num_resap = num_add
        self.sample_method = sample_method
        self.num_search = num_search

    def forward(self):
        def target(node_cpu):
            # node:cpu
            return np.where(self.func.is_node_in(node_cpu),
                            self.func.form(node_cpu), 0)
        ess = np.zeros((0, 1))
        shape = np.zeros((0, 1))
        np.save(self.file_path+'/train/'+'ess.npy', ess)
        np.save(self.file_path + '/train/' + 'shape.npy', shape)
        log = self.logger
        node_domain = {}
        node_search = self.func.sample(self.num_search, 'in')
        node = node_domain.copy()
        t1 = time.time()
        node_loss, proposal = self.sample_method.sample(target, node_search, self.func.is_node_in, self.num_resap, path=self.file_path)
        self.func.target_node_plot_together(loss=target,
                                           node_add=node_loss,
                                           node_domain=node_domain,
                                           proposal=proposal,
                                           path=self.file_path + '/img',
                                           num=0)
        t2 = time.time() - t1
        log.info('=' * 3 + 'End sample time' + time.strftime("%H:%M:%S", time.gmtime(t2)) + '=' * 10)
