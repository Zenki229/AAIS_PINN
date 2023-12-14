from matplotlib.ticker import MaxNLocator
try:
    from libs.utils import *
except FileNotFoundError:
    from utils import *
import time


class TrainResample:
    def __init__(self, pde, net, dev, optimizer, lbfgs_pretrain, lbfgs, optim_epoch, file_path, logger: logging, num_add, num_search, max_iter, sample_method):
        self.net = net
        self.pde = pde
        self.dev = dev
        self.epoch_init = optim_epoch[0]
        self.epoch = optim_epoch[2]
        self.optimizer = optimizer
        self.lbfgs_pretrain = lbfgs_pretrain
        self.lbfgs = lbfgs
        self.file_path = file_path
        self.logger = logger
        self.num_resap = num_add
        self.sample_method = sample_method
        self.max_iter = max_iter
        self.num_search = num_search

    def forward(self):
        def target(node_cpu):
            # node:cpu
            node_gpu = torch.from_numpy(node_cpu).to(device=self.dev)
            return np.where(self.pde.is_node_in(node_gpu).detach().cpu().numpy(),
                            torch.pow(input=self.pde.residual(node=node_gpu, net=self.net, cls="ele", mode="in"), exponent=2).detach().cpu().numpy(), 0)
        ess = np.zeros((0, 1))
        shape = np.zeros((0, 1))
        np.save(self.file_path+'/train/'+'ess.npy', ess)
        np.save(self.file_path + '/train/' + 'shape.npy', shape)
        log = self.logger
        loss_save = 100000.0
        node_domain = {}
        for state in self.pde.physics:
            node_domain[state] = self.pde.sample(self.pde.size[state], state)
        self.logger.info('=' * 3 + f' First Training with inside node shape {node_domain["in"].shape[0]}' + '=' * 10)
        t1 = time.time()
        rec, loss_save = run_train(self.net, self.pde, node_domain, self.epoch_init, self.optimizer,  self.lbfgs_pretrain, self.logger, self.file_path, loss_save)
        t_train = time.time() - t1
        self.logger.info('=' * 3 + f'Train Done, time ' + time.strftime("%H:%M:%S", time.gmtime(t_train)) + '=' * 10)
        with open(self.file_path + "/train" + f"/rec_0.pkl", "wb") as f:
            pickle.dump(rec, f)
        self.pde.test_err_plot(self.net, self.file_path + '/test', 'pre')
        for count in range(100):
            node_search = self.pde.sample(self.num_search, 'in').detach().cpu().numpy()
            node = node_domain.copy()
            log.info('=' * 3 + f'{count}-th ' + f'{self.sample_method.__class__.__name__}' + f' with num {node_search.shape[0]}' + '=' * 10)
            t1 = time.time()
            node_loss, proposal = self.sample_method.sample(target, node_search, self.pde.is_node_in, self.num_resap, path=self.file_path)
            node_loss = torch.from_numpy(node_loss).to(device=self.dev)
            self.pde.target_node_plot_together(loss=target,
                                               node_add=node_loss,
                                               node_domain=node_domain,
                                               proposal=proposal,
                                               path=self.file_path + '/img',
                                               num=count)
            t2 = time.time() - t1
            log.info('=' * 3 + 'End sample time' + time.strftime("%H:%M:%S", time.gmtime(t2)) + '=' * 10)
            node['in'] = torch.cat((node['in'].detach(), node_loss), dim=0)
            self.logger.info('=' * 3 + f'{count}-th Training with node shape {node["in"].shape[0]}' + '=' * 10)
            t1 = time.time()
            rec, loss_save = run_train(self.net, self.pde, node, self.epoch, self.optimizer, self.lbfgs, self.logger, self.file_path, loss_save)
            t_train = time.time() - t1
            self.logger.info(
                '=' * 3 + f'Train Done, time ' + time.strftime("%H:%M:%S", time.gmtime(t_train)) + '=' * 10)
            with open(self.file_path + "/train" + f"/rec_{count}.pkl", "wb") as f:
                pickle.dump(rec, f)
            for state in self.pde.physics:
                if state == 'in':
                    continue
                node_domain[state] = self.pde.sample(self.pde.size[state], state)
            node_aux = torch.cat([node_domain['in'], node_loss], dim=0)
            ind = np.random.choice(a=len(node_aux), size=self.pde.size['in'], replace=False)
            node_domain['in'] = node_aux[ind, :]
            self.pde.test_err_plot(self.net, self.file_path + '/test', count)
        net = torch.load(self.file_path + f"/net/net_bestloss.pkl")
        err_save = self.pde.test_err(net)
        self.logger.info('=' * 3 + f'the best solution err is {round(err_save,4)}')
        self.pde.test_err_plot(net, self.file_path + '/test', 'best')


class TrainAdd:
    def __init__(self, pde, net, dev, optimizer, lbfgs_pretrain, lbfgs, optim_epoch, file_path, logger: logging, num_add, num_search, max_iter, loss_tol, sample_method, IS_sign
                 ):
        self.net = net
        self.pde = pde
        self.dev = dev
        self.epoch_init = optim_epoch[0]
        self.epoch = optim_epoch[2]
        self.optimizer = optimizer
        self.lbfgs_pretrain = lbfgs_pretrain
        self.lbfgs = lbfgs
        self.file_path = file_path
        self.logger = logger
        self.num_resap = num_add
        self.sample_method = sample_method
        self.max_iter = max_iter
        self.loss_tol = loss_tol
        self.num_search = num_search
        self.IS_sign = IS_sign

    def forward(self):
        def target(node):
            # node:cpu
            node = torch.from_numpy(node).to(device=self.dev)
            return np.where(self.pde.is_node_in(node).detach().cpu().numpy(),
                            torch.pow(input=self.pde.residual(node=node, net=self.net, cls="ele", mode="in"),
                                      exponent=2).detach().cpu().numpy(), 0)
        if self.IS_sign:
            ess = np.zeros((0,1))
            shape = np.zeros((0,1))
            np.save(self.file_path+'/train/'+'ess.npy', ess)
            np.save(self.file_path + '/train/' + 'shape.npy', shape)
            node_search = self.pde.sample(self.num_search, 'in').detach().cpu().numpy()
        log = self.logger
        loss_save = 100000.0
        node_domain = {}
        for state in self.pde.physics:
            node_domain[state] = self.pde.sample(self.pde.size[state], state)
        self.logger.info('=' * 3 + f' First Training with inside node shape {node_domain["in"].shape[0]}' + '=' * 10)
        t1 = time.time()
        rec, loss_save = run_train(self.net, self.pde, node_domain, self.epoch_init,
                        self.optimizer, self.lbfgs_pretrain, self.logger, self.file_path, loss_save)
        t_train = time.time() - t1
        self.logger.info('=' * 3 + f'Train Done, time ' + time.strftime("%H:%M:%S", time.gmtime(t_train)) + '=' * 10)
        with open(self.file_path + "/train" + f"/rec_0.pkl", "wb") as f:
            pickle.dump(rec, f)
        self.pde.test_err_plot(self.net, self.file_path + '/test', 0)
        count = 1
        node = node_domain.copy()
        while (loss_save > self.loss_tol) & (count <= self.max_iter):
            if self.IS_sign:
                log.info('=' * 3 + f'{count}-th ' + f'{self.sample_method.__class__.__name__}' +f' with num {node_search.shape[0]}' + '=' * 10)
            else:
                log.info(
                    '=' * 3 + f'{count}-th ' + f'{self.sample_method.__class__.__name__}' + '=' * 10)
            t1 = time.time()
            if self.IS_sign:
                node_add, proposal = self.sample_method.sample(target, node_search,
                                                                self.pde.is_node_in, self.num_resap,
                                                                path=self.file_path)
                node_add = torch.from_numpy(node_add).to(self.dev)
            else:
                node_add = self.pde.sample(self.num_resap, 'in')
                proposal = None
            t2 = time.time() - t1
            log.info('=' * 3 + 'End sample time' + time.strftime("%H:%M:%S", time.gmtime(t2)) + '=' * 10)
            self.pde.target_node_plot_together(loss=target,
                                               node_add=node_add,
                                               node_domain=node,
                                               IS_sign=self.IS_sign,
                                               proposal=proposal,
                                               path=self.file_path + '/img',
                                               num=count)
            node['in'] = torch.cat((node['in'].detach(), node_add), dim=0)
            self.logger.info('=' * 3 + f'{count}-th Training with node shape {node["in"].shape[0]}' + '=' * 10)
            t1 = time.time()
            rec, loss_save = run_train(self.net, self.pde, node, self.epoch,
                            self.optimizer, self.lbfgs,
                            self.logger, self.file_path, loss_save)
            t_train = time.time() - t1
            self.logger.info(
                '=' * 3 + f'Train Done, time ' + time.strftime("%H:%M:%S", time.gmtime(t_train)) + '=' * 10)
            with open(self.file_path + "/train" + f"/rec_{count}.pkl", "wb") as f:
                pickle.dump(rec, f)
            for state in self.pde.physics:
                if state == 'in':
                    pass
                else:
                    node[state] = self.pde.sample(self.pde.size[state], state)
            self.pde.test_err_plot(self.net, self.file_path + '/test', count)
            count += 1
        net = torch.load(self.file_path + f"/net/net_bestloss.pkl")
        err_save = self.pde.test_err(net)
        self.logger.info('='*3+f'the best loss model out put the relative error  is {round(err_save,4)}')
        self.pde.test_err_plot(net, self.file_path + '/test', 'best')


def run_train(net, pde, node, epoch, optimizer, lbfgs, log, file_path, loss_save):
    iter_count = 1
    loss_count = []
    err_count = []
    net.train()
    columns = ["epoch", "name", "lr", "loss", "err"]
    file_path = file_path
    loss_save = loss_save

    def closure():
        nonlocal iter_count, loss_count, columns, err_count, file_path, loss_save
        optimizer.zero_grad()
        if lbfgs:
            lbfgs.zero_grad()
        loss_dict = {}
        loss = torch.zeros((1,)).to(pde.dev)
        for state in pde.physics:
            loss_dict['state'] = pde.weight[state] * pde.residual(node[state], net, cls="loss", mode=state)
            loss += loss_dict['state']
        loss_count.append(loss.item())
        if loss.item() < loss_save:
            torch.save(net, file_path + f"/net/net_bestloss.pkl")
            loss_save = loss.item()
        loss.backward()
        if iter_count <= epoch:
            name = optimizer.__class__.__name__
            lr = optimizer.param_groups[0]['lr']
        elif lbfgs:
            name = "lbfgs"
            lr = lbfgs.param_groups[0]['lr']
        err = pde.test_err(net)
        err_count.append(err)
        values = [iter_count, name, lr, loss.item(), err]
        table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="5.4e")
        if iter_count % 500 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        if iter_count % 100 == 0:
            print(table)
        if iter_count % 200 == 0:
            log.info(table)
        iter_count += 1
        return loss.item()

    for e in range(epoch):
        optimizer.step(closure)
        # if scheduler:
        #     scheduler.step(closure)
    if lbfgs:
        lbfgs.step(closure)
    return {"count": iter_count, "loss": loss_count, "err": err_count}, loss_save


def loss_err_plot(path_father):
    count = 1
    loss = {}
    err = {}
    while os.path.exists(path_father+'/train'+f'/rec_{count}.pkl'):
        with open(path_father+'/train'+f'/rec_{count}.pkl', 'rb') as f:
            data = pickle.load(f)
            loss[f"{count}"] = np.array(data["loss"])
            err[f"{count}"] = np.array(data["err"])
        count = count + 1
    # plot loss
    loss_all = np.concatenate(list(loss.values()))
    err_all = np.concatenate(list(err.values()))
    fig, ax = plt.subplots(layout='constrained', figsize=(19.2, 4.8))
    ax.semilogy(np.array(loss_all))
    # ax.set_ylim(np.min(loss_all), 1)
    plt.savefig(path_father+'/img'+'/loss_plot.jpg', dpi=150)
    plt.close()
    fig, ax = plt.subplots(layout='constrained', figsize=(19.2, 4.8))
    ax.semilogy(np.array(err_all))
    # ax.set_ylim(np.min(err_all), 1)
    plt.savefig(path_father + '/img' + '/err_plot.jpg', dpi=150)
    plt.close()
    loss_err = np.stack([loss_all, err_all], axis=1)
    np.save(path_father+'/train'+'/loss_err.npy', loss_err)


def shape_ess_plot(path_father):
    ess = np.load(path_father+'/train/'+'ess.npy').flatten()
    shape = np.load(path_father+'/train/'+'shape.npy').flatten()
    max_iter = len(ess)
    x = np.arange(start=0, stop=max_iter, step=1).astype(dtype=str)
    fig, ax = plt.subplots(layout='constrained')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(x, ess)
    plt.savefig(path_father+'/img/'+'ess_plot.jpg')
    plt.close()
    fig, ax = plt.subplots(layout='constrained')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(x, shape)
    plt.savefig(path_father + '/img/' + 'shape_plot.jpg')
    plt.close()
    ess_shape = np.stack([ess, shape], axis=1)
    np.save(path_father + '/train' + '/ess_shape.npy', ess_shape)


if __name__ == "__main__":
    # path = input("input path_father:")
    # loss_err_plot("../results/Poisson2D1Peak/Poisson2D1Peak_AAIS_g_resample_19076082_500e-pp1")
    # loss_err_plot("../results/Poisson2D1Peak/Poisson2D1Peak_AAIS_g_add_19076082_500e-pp1")
    # loss_err_plot("../results/Poisson2D1Peak/Poisson2D1Peak_AAIS_g_resample_19076082_1000e-pp1")
    # loss_err_plot("../results/Poisson2D1Peak/Poisson2D1Peak_AAIS_g_add_19076082_1000e-pp1")
    shape_ess_plot("../results/Burgers2D/Burgers2D_AAIS_t_resample_500e")
    shape_ess_plot("../results/Burgers2D/Burgers2D_AAIS_g_resample_500e")
    # shape_ess_plot("../results/Burgers1D_v1Uni_resample_1116-aft-1")

