from .utils import *


class Poisson2D1Peak:
    def __init__(self, dev, dtp, weight, xlim, ylim, num_in, num_bd, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.ylim, self.input_size, self.output_size = 2, dev, dtp, weight, xlim, ylim, input_size, output_size
        self.criterion = torch.nn.MSELoss()
        self.physics = ['in', 'bd']
        self.size = {'in': num_in, 'bd': num_bd}

    def sample(self, size, mode):
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        x_len, y_len = xe - xs, ye - ys
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

    @staticmethod
    def solve(mode, node):
        if mode == 'in':
            val_in = (-(torch.exp(-1000 * ((node[:, 0] - 0.5) ** 2 + (node[:, 1] - 0.5) ** 2))
                        * (torch.pow((-2 * 1000) * (node[:, 0] - 0.5), 2) + (-2 * 1000)))
                      - (torch.exp(-1000 * ((node[:, 0] - 0.5) ** 2 + (node[:, 1] - 0.5) ** 2))
                         * (torch.pow((-2 * 1000) * (node[:, 1] - 0.5), 2) + (-2 * 1000))))
            return val_in
        elif mode == "bd":
            val_bd = torch.exp(-1000 * ((node[:, 0] - 0.5) ** 2 + (node[:, 1] - 0.5) ** 2))
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
                pde_res = -dyy - dxx - pred
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
        inter_x = np.linspace(start=xs, stop=xe, num=size + 1)
        inter_y = np.linspace(start=ys, stop=ye, num=size + 1)
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
        exact = np.exp(-1000 * ((node[:, 0] - 0.5) ** 2 + (node[:, 1] - 0.5) ** 2))
        err = np.sqrt(np.sum(np.power(val - exact, 2)) / np.sum(np.power(exact, 2)))
        return err

    def target_node_plot_together(self, node_domain, node_add, loss, proposal, IS_sign, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['bd'].detach()])
        node_all = node_all.cpu().numpy()
        node_add = node_add.detach().cpu().numpy()
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        val = loss(node).reshape(mesh_x.shape)
        if (IS_sign == 0) | (proposal is None):
            fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(12.8, 4.8))
            # plot loss
            plot = ax[0].pcolormesh(mesh_x, mesh_y, val, shading='gouraud', cmap='jet', vmin=0, vmax=np.max(val))
            fig.colorbar(plot, ax=ax[0], format="%1.1e")
            ax[0].set_title(f'residual $\\mathcal{{Q}}_{{{num}}}$')
            ax[0].set_xlabel('$x$')
            ax[0].set_ylabel('$y$')
            # plot node
            ax[1].set_xlim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.15)
            ax[1].set_ylim(ys - (ye - ys) * 0.05, ye + (ye - ys) * 0.15)
            ax[1].scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.',
                          s=np.ones_like(node_all[:, 0]), alpha=0.5, label=f'$\\mathcal{{S}}_{num}$')
            ax[1].scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.',
                          s=np.ones_like(node_add[:, 0]), alpha=1.0, label=f'$\\mathcal{{D}}$')
            ax[1].legend(loc='upper right', fontsize=12)
            ax[1].set_title(f'nodes')
            ax[1].set_xlabel('$x$')
            ax[1].set_ylabel('$y$')
            plt.savefig(path + f'/{num}_loss.png', dpi=300)
            plt.close()
        if proposal:
            val_prop = proposal(node).reshape(mesh_x.shape)
            fig, ax = plt.subplots(1, 3, layout='constrained', figsize=(19.2, 4.8))
            # plot loss
            plot = ax[0].pcolormesh(mesh_x, mesh_y, val, shading='gouraud',
                                    cmap='jet', vmin=0, vmax=np.max(val))
            fig.colorbar(plot, ax=ax[0], format="%1.1e")
            ax[0].set_title(f'residual $\\mathcal{{Q}}_{{{num}}}$')
            ax[0].set_xlabel('$x$')
            ax[0].set_ylabel('$y$')
            # plot proposal
            plot = ax[1].pcolormesh(mesh_x, mesh_y, val_prop, shading='gouraud',
                                    cmap='jet', vmin=0, vmax=np.max(val_prop))
            fig.colorbar(plot, ax=ax[1], format="%1.1e")
            ax[1].set_title('proposal')
            ax[1].set_xlabel('$x$')
            ax[1].set_ylabel('$y$')
            # plot node
            ax[2].set_xlim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.15)
            ax[2].set_ylim(ys - (ye - ys) * 0.05, ye + (ye - ys) * 0.15)
            ax[2].scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.',
                          s=np.ones_like(node_all[:, 0]), alpha=0.5, label=f'$\\mathcal{{S}}_{{{num}}}$')
            ax[2].scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.',
                          s=np.ones_like(node_add[:, 0]), alpha=1.0, label=f'$\\mathcal{{D}}$')
            ax[2].legend(loc='upper right', fontsize=12)
            ax[2].set_title('nodes')
            ax[2].set_xlabel('$x$')
            ax[2].set_ylabel('$y$')
            plt.savefig(path + f'/{num}_loss.png', dpi=300)
            plt.close()

    def test_err_plot(self, net, path, num):
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        node_aux = torch.from_numpy(node).to(device=self.dev)
        val = net(node_aux).detach().cpu().numpy().flatten()
        exact = np.exp(-1000 * ((node[:, 0] - 0.5) ** 2 + (node[:, 1] - 0.5) ** 2))
        err = np.sqrt(np.sum(np.power(val - exact, 2)) / np.sum(np.power(exact, 2)))
        err_plt = np.abs(val - exact)
        fig, ax = plt.subplots(1, 3, layout='constrained', figsize=(19.2, 4.8))
        # err plot
        plot = ax[0].pcolormesh(mesh_x, mesh_y, err_plt.reshape(mesh_x.shape), shading='gouraud',
                                cmap='jet', vmin=0, vmax=np.max(err_plt))
        fig.colorbar(plot, ax=ax[0], format="%1.1e")
        ax[0].set_title(f'$e_r(u^\\theta_{{{num}}})={round(err, 4)}$')
        ax[0].set_xlabel('$x$')
        ax[0].set_ylabel('$y$')
        # val plot
        plot = ax[1].pcolormesh(mesh_x, mesh_y, val.reshape(mesh_x.shape), shading='gouraud',
                                cmap='jet', vmin=np.min(val), vmax=np.max(val))
        fig.colorbar(plot,  ax=ax[1], format="%1.1e")
        ax[1].set_title(f'$u^\\theta_{{{num}}}$')
        ax[1].set_xlabel('$x$')
        ax[1].set_ylabel('$y$')
        # exact plot
        plot = ax[2].pcolormesh(mesh_x, mesh_y, exact.reshape(mesh_x.shape), shading='gouraud',
                                cmap='jet', vmin=0, vmax=1)
        fig.colorbar(plot, ax=ax[2], format="%1.1e")
        ax[2].set_title(f'$u^*$')
        ax[2].set_xlabel('$x$')
        ax[2].set_ylabel('$y$')
        plt.savefig(path + f'/{num}_sol.png', dpi=300)
        plt.close()


class Poisson2D9Peak:
    def __init__(self, dev, dtp, weight, xlim, ylim, num_in, num_bd, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.ylim, self.input_size, self.output_size = 2, dev, dtp, weight, xlim, ylim, input_size, output_size
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
                val_in += (
                        -(torch.exp(-1000 * (
                                (node[:, 0] - self.center[i, 0]) ** 2 + (node[:, 1] - self.center[i, 1]) ** 2)
                                    ) * (torch.pow((-2 * 1000) * (node[:, 0] - self.center[i, 0]), 2) + (-2 * 1000)))
                        - (torch.exp(-1000 * (
                                (node[:, 0] - self.center[i, 0]) ** 2 + (node[:, 1] - self.center[i, 1]) ** 2)
                                    ) * (torch.pow((-2 * 1000) * (node[:, 1] - self.center[i, 1]), 2) + (-2 * 1000)))
                        )
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
                pde_res = -dyy - dxx - pred
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

    def target_node_plot_together(self, node_domain, node_add, loss, proposal, IS_sign, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['bd'].detach()])
        node_all = node_all.cpu().numpy()
        node_add = node_add.detach().cpu().numpy()
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        val = loss(node).reshape(mesh_x.shape)
        if (IS_sign == 0) | (proposal is None):
            fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(12.8, 4.8))
            # plot loss
            plot = ax[0].pcolormesh(mesh_x, mesh_y, val, shading='gouraud', cmap='jet', vmin=0, vmax=np.max(val))
            fig.colorbar(plot, ax=ax[0], format="%1.1e")
            ax[0].set_title(f'residual $\\mathcal{{Q}}_{{{num}}}$')
            ax[0].set_xlabel('$x$')
            ax[0].set_ylabel('$y$')
            # plot node
            ax[1].set_xlim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.15)
            ax[1].set_ylim(ys - (ye - ys) * 0.05, ye + (ye - ys) * 0.15)
            ax[1].scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.',
                          s=np.ones_like(node_all[:, 0]), alpha=0.5, label=f'$\\mathcal{{S}}_{{{num}}}$')
            ax[1].scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.',
                          s=np.ones_like(node_add[:, 0]), alpha=1.0, label=f'$\\mathcal{{D}}$')
            ax[1].legend(loc='upper right', fontsize=12)
            ax[1].set_title(f'nodes')
            ax[1].set_xlabel('$x$')
            ax[1].set_ylabel('$y$')
            plt.savefig(path + f'/{num}_loss.png', dpi=300)
            plt.close()
        if proposal:
            val_prop = proposal(node).reshape(mesh_x.shape)
            fig, ax = plt.subplots(1, 3, layout='constrained', figsize=(19.2, 4.8))
            # plot loss
            plot = ax[0].pcolormesh(mesh_x, mesh_y, val, shading='gouraud',
                                    cmap='jet', vmin=0, vmax=np.max(val))
            fig.colorbar(plot, ax=ax[0], format="%1.1e")
            ax[0].set_title(f'residual $\\mathcal{{Q}}_{{{num}}}$')
            ax[0].set_xlabel('$x$')
            ax[0].set_ylabel('$y$')
            # plot proposal
            plot = ax[1].pcolormesh(mesh_x, mesh_y, val_prop, shading='gouraud',
                                    cmap='jet', vmin=0, vmax=np.max(val_prop))
            fig.colorbar(plot, ax=ax[1], format="%1.1e")
            ax[1].set_title('proposal')
            ax[1].set_xlabel('$x$')
            ax[1].set_ylabel('$y$')
            # plot node
            ax[2].set_xlim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.15)
            ax[2].set_ylim(ys - (ye - ys) * 0.05, ye + (ye - ys) * 0.15)
            ax[2].scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.',
                          s=np.ones_like(node_all[:, 0]), alpha=0.5, label=f'mathcal{{S}}_{{{num}}}$')
            ax[2].scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.',
                          s=np.ones_like(node_add[:, 0]), alpha=1.0, label=f'$\\mathcal{{D}}$')
            ax[2].legend(loc='upper right', fontsize=12)
            ax[2].set_title('nodes')
            ax[2].set_xlabel('$x$')
            ax[2].set_ylabel('$y$')
            plt.savefig(path + f'/{num}_loss.png', dpi=300)
            plt.close()

    def test_err_plot(self, net, path, num):
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        node_aux = torch.from_numpy(node).to(device=self.dev)
        val = net(node_aux).detach().cpu().numpy().flatten()
        exact = np.exp(-1000 * ((node[:, 0] - 0.5) ** 2 + (node[:, 1] - 0.5) ** 2))
        err = np.sqrt(np.sum(np.power(val - exact, 2)) / np.sum(np.power(exact, 2)))
        err_plt = np.abs(val - exact)
        fig, ax = plt.subplots(1, 3, layout='constrained', figsize=(19.2, 4.8))
        # err plot
        plot = ax[0].pcolormesh(mesh_x, mesh_y, err_plt.reshape(mesh_x.shape), shading='gouraud',
                                cmap='jet', vmin=0, vmax=np.max(err_plt))
        fig.colorbar(plot, ax=ax[0], format="%1.1e")
        ax[0].set_title(f'$e_r(u^\\theta_{{{num}}})={round(err, 4)}$')
        ax[0].set_xlabel('$x$')
        ax[0].set_ylabel('$y$')
        # val plot
        plot = ax[1].pcolormesh(mesh_x, mesh_y, val.reshape(mesh_x.shape), shading='gouraud',
                                cmap='jet', vmin=np.min(val), vmax=np.max(val))
        fig.colorbar(plot, ax=ax[1], format="%1.1e")
        ax[1].set_title(f'$u^\\theta_{{{num}}}$')
        ax[1].set_xlabel('$x$')
        ax[1].set_ylabel('$y$')
        # exact plot
        plot = ax[2].pcolormesh(mesh_x, mesh_y, exact.reshape(mesh_x.shape), shading='gouraud',
                                cmap='jet', vmin=0, vmax=1)
        fig.colorbar(plot, ax=ax[2], format="%1.1e")
        ax[2].set_title(f'$u^*$')
        ax[2].set_xlabel('$x$')
        ax[2].set_ylabel('$y$')
        plt.savefig(path + f'/{num}_sol.png', dpi=300)
        plt.close()


class Poisson2DLshape:
    def __init__(self, dev, dtp, weight, xlim, ylim, x2lim, y2lim, num_in, num_bd, input_size, output_size):
        (self.dim, self.dev, self.dtp, self.weight, self.xlim, self.ylim, self.x2lim, self.y2lim, self.input_size,
         self.output_size) = 2, dev, dtp, weight, xlim, ylim, x2lim, y2lim, input_size, output_size
        self.criterion = torch.nn.MSELoss()
        self.physics = ['in', 'bd']
        self.size = {'in': num_in, 'bd': num_bd}

    def is_node_in(self, node):
        return ((self.xlim[0] < node[:, 0]) & (node[:, 0] < self.xlim[1])
                & (self.ylim[0] < node[:, 1]) & (node[:, 1] < self.ylim[1])
                & ~self.is_node_cube(node))

    def is_node_cube(self, node):
        return ((self.x2lim[0] < node[:, 0]) & (node[:, 0] < self.x2lim[1])
                & (self.y2lim[0] < node[:, 1]) & (node[:, 1] < self.y2lim[1]))

    def sample(self, size, mode):
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        x_len, y_len = xe - xs, ye - ys
        if mode == 'in':
            node_in = torch.empty((0, self.dim))
            res = size
            while res >= 1:
                node_aux = torch.cat(
                    (torch.rand([res, 1]) * x_len + torch.ones(size=[res, 1]) * xs,
                     torch.rand([res, 1]) * y_len + torch.ones(size=[res, 1]) * ys), dim=1)
                node_in = torch.cat([node_in, node_aux[~self.is_node_cube(node_aux)]], dim=0)
                res = node_aux[self.is_node_cube(node_aux)].shape[0]
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
                    node_bd[i][node_bd[i][:, 0] > self.x2lim[0], 1] = self.y2lim[0]
                else:
                    node_bd[i] = torch.cat([
                        torch.ones([num, 1]) * xe,
                        torch.rand([num, 1]) * y_len + torch.ones([num, 1]) * ys], dim=1)
                    node_bd[i][node_bd[i][:, 1] > self.y2lim[0], 0] = self.x2lim[0]
            return torch.cat(node_bd, dim=0).to(device=self.dev, dtype=self.dtp)

    @staticmethod
    def solve(mode, node):
        if mode == 'in':
            val_in = torch.ones_like(node[:, 0])
            return val_in
        elif mode == "bd":
            val_bd = torch.zeros_like(node[:, 0])
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
                pde_res = -dyy - dxx - pred
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
        inter_x = np.linspace(start=xs, stop=xe, num=size + 1)
        inter_y = np.linspace(start=ys, stop=ye, num=size + 1)
        mesh_x, mesh_y = np.meshgrid(inter_x, inter_y)
        return mesh_x, mesh_y

    def test_err(self, net):
        data = np.load('./data/Poisson_Lshape.npz')
        node, exact = data['X_test'], data['y_ref'].flatten()
        ind = np.where(~np.isnan(exact))[0]
        node = node[ind, :]
        exact = exact[ind]
        node_aux = torch.from_numpy(node).to(device=self.dev)
        val = net(node_aux).detach().cpu().numpy().flatten()
        err = np.sqrt(np.sum(np.power(val - exact, 2)) / np.sum(np.power(exact, 2)))
        return err

    def target_node_plot_together(self, node_domain, node_add, loss, proposal, IS_sign, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['bd'].detach()])
        node_all = node_all.cpu().numpy()
        node_add = node_add.detach().cpu().numpy()
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        data = np.load('./data/Poisson_Lshape.npz')
        node, exact = data['X_test'], data['y_ref'].flatten()
        ind = np.where(np.isnan(exact))[0]
        val = loss(node)
        val[ind] = np.nan
        mesh_x, mesh_y = node[:, 0].reshape(161, 161), node[:, 1].reshape(161, 161)
        if IS_sign == 0:
            fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(12.8, 4.8))
            # plot loss
            plot = ax[0].pcolormesh(mesh_x, mesh_y, val.reshape(161, 161), shading='gouraud', cmap='jet')
            fig.colorbar(plot, ax=ax[0], format="%1.1e")
            ax[0].set_title(f'residual $\\mathcal{{Q}}_{{{num}}}$')
            ax[0].set_xlabel('$x$')
            ax[0].set_ylabel('$y$')
            # plot node
            ax[1].set_xlim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.15)
            ax[1].set_ylim(ys - (ye - ys) * 0.05, ye + (ye - ys) * 0.15)
            ax[1].scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.',
                          s=np.ones_like(node_all[:, 0]), alpha=0.5, label=f'$\\mathcal{{S}}_{{{num}}}$')
            ax[1].scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.',
                          s=np.ones_like(node_add[:, 0]), alpha=1.0, label=f'$\\mathcal{{D}}$')
            ax[1].legend(loc='upper right', fontsize=12)
            ax[1].set_title('nodes')
            ax[1].set_xlabel('$x$')
            ax[1].set_ylabel('$y$')
            plt.savefig(path + f'/{num}_loss.png', dpi=300)
            plt.close()
        if IS_sign == 1:

            fig, ax = plt.subplots(1, 3, layout='constrained', figsize=(19.2, 4.8))
            # plot loss
            plot = ax[0].pcolormesh(mesh_x, mesh_y, val.reshape(161, 161), shading='gouraud', cmap='jet')
            fig.colorbar(plot, ax=ax[0], format="%1.1e")
            ax[0].set_title(f'residual $\\mathcal{{Q}}_{{{num}}}$')
            ax[0].set_xlabel('$x$')
            ax[0].set_ylabel('$y$')
            # plot node
            ax[2].set_xlim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.15)
            ax[2].set_ylim(ys - (ye - ys) * 0.05, ye + (ye - ys) * 0.15)
            ax[2].scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.',
                          s=np.ones_like(node_all[:, 0]), alpha=0.5, label=f'$\\mathcal{{S}}_{{{num}}}$')
            ax[2].scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.',
                          s=np.ones_like(node_add[:, 0]), alpha=1.0, label=f'$\\mathcal{{D}}$')
            ax[2].legend(loc='upper right')
            ax[2].set_title('nodes')
            ax[2].set_xlabel('$x$')
            ax[2].set_ylabel('$y$')
            # plot proposal
            val_prop = proposal(node)
            val_prop[ind] = np.nan
            plot = ax[1].pcolormesh(mesh_x, mesh_y, val_prop.reshape(mesh_x.shape), cmap='jet')
            fig.colorbar(plot, ax=ax[1], format="%1.1e")
            ax[1].set_title('proposal')
            ax[1].set_xlabel('$x$')
            ax[1].set_ylabel('$y$')
            plt.savefig(path + f'/{num}_loss.png', dpi=300)
            plt.close()

    def test_err_plot(self, net, path, num):
        data = np.load('./data/Poisson_Lshape.npz')
        node, exact = data['X_test'], data['y_ref'].flatten()
        ind = np.where(~np.isnan(exact))[0]
        node_aux = torch.from_numpy(node[ind, :]).to(device=self.dev)
        val = net(node_aux).detach().cpu().numpy().flatten()
        mesh_x, mesh_y = node[:, 0].reshape(161, 161), node[:, 1].reshape(161, 161)
        err = np.sqrt(np.sum(np.power(val - exact[ind], 2)) / np.sum(np.power(exact[ind], 2)))
        err_plt = np.empty_like(exact)
        err_plt[:] = np.nan
        err_plt[ind] = np.abs(val - exact[ind])
        val_plt = np.empty_like(exact)
        val_plt[:] = np.nan
        val_plt[ind] = val
        fig, ax = plt.subplots(1, 3, layout='constrained', figsize=(19.2, 4.8))
        # err plot
        plot = ax[0].pcolormesh(mesh_x, mesh_y, err_plt.reshape(mesh_x.shape), shading='gouraud', cmap='jet')
        fig.colorbar(plot, ax=ax[0], format="%1.1e")
        ax[0].set_title(f'$e_r(u^\\theta_{{{num}}})={round(err, 4)}$')
        ax[0].set_xlabel('$x$')
        ax[0].set_ylabel('$y$')
        # val plot
        plot = ax[1].pcolormesh(mesh_x, mesh_y, val_plt.reshape(mesh_x.shape), shading='gouraud', cmap='jet')
        fig.colorbar(plot,  ax=ax[1], format="%1.1e")
        ax[1].set_title(f'$u^\\theta_{{{num}}}$')
        ax[1].set_xlabel('$x$')
        ax[1].set_ylabel('$y$')
        # exact plot
        plot = ax[2].pcolormesh(mesh_x, mesh_y, exact.reshape(mesh_x.shape), shading='gouraud', cmap='jet')
        fig.colorbar(plot, ax=ax[2], format="%1.1e")
        ax[2].set_title(f'$u^*$')
        ax[2].set_xlabel('$x$')
        ax[2].set_ylabel('$y$')
        plt.savefig(path + f'/{num}_sol.png', dpi=300)
        plt.close()
