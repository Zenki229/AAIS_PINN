from libs import *

dtp = torch.float64

try:
    os.mkdir('./results')
except FileExistsError:
    pass

num_search = np.array([0, 1, 2, 3, 4, 5, 6, 7])
num_search = np.power(2, num_search)*1000
num_sample = 500
func = 'Poisson9Peaks'
try:
    os.mkdir('./results/' + f'{func}/')
except FileExistsError:
    pass
path_father = './results/' + f'{func}/' + args.pde + '_' + args.strategy + '_' + args.dirname
try:
    os.mkdir(path_father)
except FileExistsError:
    for root, dirs, files in os.walk(path_father, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
try:
    os.mkdir(path_father + '/img')
    os.mkdir(path_father + '/net')
    os.mkdir(path_father + '/train')
    os.mkdir(path_father + '/test')
except FileExistsError:
    pass
logger = log_gen(path=path_father)
with open(path_father + "/inform.txt", "w") as f:
    f.write(str(args) + "\n")
configs = load_yaml(r'./configs.yml', key=args.func)
with open(path_father + "/inform.txt", "a") as f:
    f.write(str(configs) + "\n")


class Poisson9Peaks:
    def __init__(self, dev, dtp, weight, xlim, ylim, num_in, num_bd, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.ylim, self.input_size, self.output_size = input_size, dev, dtp, weight, xlim, ylim, input_size, output_size
        self.criterion = torch.nn.MSELoss()
        self.physics = ['in', 'bd']
        self.size = {'in': num_in, 'bd': num_bd}
        grid = np.array([-0.5, 0, 0.5])
        x, y = np.meshgrid(grid, grid)
        self.center = np.stack([x.flatten(), y.flatten()], axis=1)

    def sample(self, size, mode):
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        x_len, y_len = xe - xs, ye-ys
        if mode == 'in':
            node_in = torch.cat(
                (torch.rand([size, 1]) * x_len + torch.ones(size=[size, 1]) * xs,
                 torch.rand([size, 1]) * y_len + torch.ones(size=[size, 1]) * ys), dim=1)
            return node_in.to(device=self.dev, dtype=self.dtp)
        if mode == 'bd':
            bd_num = torch.randint(low=0, high=4, size=(size,))
            node_bd = list(range(4))
            for i in range(4):
                ind = bd_num[bd_num == i]
                num = bd_num[ind].shape[0]
                if i == 0:
                    node_bd[i] = torch.cat([
                        torch.rand([num, 1]) * x_len + torch.ones([num, 1]) * xs,
                        torch.ones([num, 1]) * ys], dim=1)
                elif i == 1:
                    node_bd[i] = torch.cat([
                        torch.ones([num, 1]) * xs,
                        torch.rand([num, 1]) * y_len + torch.ones([num, 1]) * ys], dim=1)
                elif i == 2:
                    node_bd[i] = torch.cat([
                        torch.rand([num, 1]) * x_len + torch.ones([num, 1]) * xs,
                        torch.ones([num, 1]) * ye], dim=1)
                else:
                    node_bd[i] = torch.cat([
                        torch.ones([num, 1]) * xe,
                        torch.rand([num, 1]) * y_len + torch.ones([num, 1]) * ys], dim=1)
            return torch.cat(node_bd, dim=0).to(device=self.dev, dtype=self.dtp)

    def solve(self, mode, node):
        if mode == 'in':
            val_in = torch.zeros_like(node[:, 0])
            for i in range(self.center.shape[0]):
                val_in += torch.exp(-1000 * (
                                (node[:, 0] - self.center[i, 0]) ** 2 + (node[:, 1] - self.center[i, 1]) ** 2))
            return val_in
        elif mode == "bd":
            val_bd = torch.zeros_like(node[:, 0])
            for i in range(self.center.shape[0]):
                val_bd += torch.exp(-1000 * (
                        (node[:, 0] - self.center[i, 0]) ** 2 + (node[:, 1] - self.center[i, 1]) ** 2)
                                    )
            return val_bd
        else:
            raise ValueError('Invalid mode')

    def residual(self, node, net, cls, mode):
        pred = self.solve(mode=mode, node=node)
        if mode == 'in':
            x = node
            x.requires_grad = True
            d = torch.autograd.grad(outputs=net(x),
                                    inputs=x,
                                    grad_outputs=torch.ones_like(net(x)),
                                    retain_graph=True,
                                    create_graph=True)[0]
            dx = d[:, 0].reshape(-1, 1)
            dy = d[:, 1].reshape(-1, 1)
            dxx = torch.autograd.grad(inputs=x,
                                      outputs=dx,
                                      grad_outputs=torch.ones_like(dx),
                                      retain_graph=True,
                                      create_graph=True)[0][:, 0].flatten()
            dyy = torch.autograd.grad(dy, x,
                                      grad_outputs=torch.ones_like(dy),
                                      retain_graph=True,
                                      create_graph=True)[0][:, 1].flatten()
            if cls == 'loss':
                pde_res = self.criterion(-dyy - dxx, pred)
            elif cls == 'ele':
                pde_res = pred
            else:
                raise ValueError('Invalid cls')
            return pde_res
        elif mode == 'bd':
            bd_res = self.criterion(net(node).flatten(), pred)
            return bd_res
        else:
            raise ValueError('Invalid mode')

    def grid(self, size):
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        inter_x = np.linspace(start=xs, stop=xe, num=size + 2)
        inter_y = np.linspace(start=ys, stop=ye, num=size + 2)
        mesh_x, mesh_y = np.meshgrid(inter_x, inter_y)
        return mesh_x, mesh_y

    def is_node_in(self, node):
        return ((self.xlim[0] < node[:, 0]) & (node[:, 0] < self.xlim[1])
                & (self.ylim[0] < node[:, 1]) & (node[:, 1] < self.ylim[1]))

    def test_err(self, net):
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        node_aux = torch.from_numpy(node).to(device=self.dev)
        val = net(node_aux).detach().cpu().numpy().flatten()
        exact = np.zeros_like(node[:, 0])
        for i in range(self.center.shape[0]):
            exact += np.exp(-1000 * ((node[:, 0] - self.center[i, 0]) ** 2 + (node[:, 1] - self.center[i, 1]) ** 2))
        err = np.sqrt(np.sum(np.power(val - exact, 2)) / np.sum(np.power(exact, 2)))
        return err

    def target_node_plot_together(self, node_domain, node_add, loss, proposal, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['bd'].detach()])
        node_all = node_all.cpu().numpy()
        node_add = node_add.detach().cpu().numpy()
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        val = loss(node).reshape(mesh_x.shape)
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        # plot loss
        plot = ax.pcolormesh(mesh_x, mesh_y, val, shading='gouraud', cmap='jet', vmin=0, vmax=np.max(val))
        fig.colorbar(plot, ax=ax, format="%1.1e")
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        fig.savefig(path + f'/{num}_loss.png', dpi=300)
        plt.close(fig)
        # plot node
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        ax.set_xlim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.20)
        ax.set_ylim(ys - (ye - ys) * 0.05, ye + (ye - ys) * 0.20)
        ax.scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.', s=np.ones_like(node_all[:, 0]), alpha=0.3, label=f'$\\mathcal{{S}}_{{{num}}}$')
        ax.scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.', s=np.ones_like(node_add[:, 0]), alpha=1.0, label=f'$\\mathcal{{D}}$')
        ax.legend(loc='upper right', fontsize=12)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        fig.savefig(path + f'/{num}_node.png', dpi=300)
        plt.close(fig)
        if proposal:
            val_prop = proposal(node).reshape(mesh_x.shape)
            fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
            # plot proposal
            plot = ax.pcolormesh(mesh_x, mesh_y, val_prop, shading='gouraud', cmap='jet', vmin=0, vmax=np.max(val_prop))
            fig.colorbar(plot, ax=ax, format="%1.1e")
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            plt.savefig(path + f'/{num}_proposal.png', dpi=300)
            plt.close()

    def test_err_plot(self, net, path, num):
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        node_aux = torch.from_numpy(node).to(device=self.dev)
        val = net(node_aux).detach().cpu().numpy().flatten()
        exact = np.zeros_like(node[:, 0])
        for i in range(self.center.shape[0]):
            exact += np.exp(-1000 * ((node[:, 0] - self.center[i, 0]) ** 2 + (node[:, 1] - self.center[i, 1]) ** 2))
        err = np.sqrt(np.sum(np.power(val - exact, 2)) / np.sum(np.power(exact, 2)))
        err_plt = np.abs(val - exact)
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        # err plot
        plot = ax.pcolormesh(mesh_x, mesh_y, err_plt.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=0, vmax=np.max(err_plt))
        fig.colorbar(plot, ax=ax, format="%1.1e")
        ax.set_title(f'$e_r(u_{{{num}}}(\\cdot;\\theta))={round(err, 4)}$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        fig.savefig(path + f'/{num}_abs.png', dpi=300)
        plt.close(fig)
        # val plot
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        plot = ax.pcolormesh(mesh_x, mesh_y, val.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=np.min(val), vmax=np.max(val))
        fig.colorbar(plot, ax=ax, format="%1.1e")
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        fig.savefig(path + f'/{num}_sol.png', dpi=300)
        plt.close(fig)
        # exact plot
        if num == 1:
            fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
            plot = ax.pcolormesh(mesh_x, mesh_y, exact.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=0, vmax=1)
            fig.colorbar(plot, ax=ax, format="%1.1e")
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            fig.savefig(path + f'/{num}_exact.png', dpi=300)
            plt.close(fig)