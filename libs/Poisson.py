from .utils import *
import scipy.stats as ss

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
        fig.savefig(path+f'/{num}_loss.png', dpi=300)
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
            fig.savefig(path + f'/{num}_proposal.png', dpi=300)
            plt.close(fig)

    def test_err_plot(self, net, path, num):
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        node_aux = torch.from_numpy(node).to(device=self.dev)
        val = net(node_aux).detach().cpu().numpy().flatten()
        exact = np.exp(-1000 * ((node[:, 0] - 0.5) ** 2 + (node[:, 1] - 0.5) ** 2))
        err = np.sqrt(np.sum(np.power(val - exact, 2)) / np.sum(np.power(exact, 2)))
        err_plt = np.abs(val - exact)
        # err plot
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        plot = ax.pcolormesh(mesh_x, mesh_y, err_plt.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=0, vmax=np.max(err_plt))
        fig.colorbar(plot, ax=ax, format="%1.1e")
        ax.set_title(f'$e_r(u_{{{num}}}(\\cdot;\\theta))={round(err, 4)}$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        fig.savefig(path+f'/{num}_abs.png', dpi=300)
        plt.close(fig)
        # val plot
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        plot = ax.pcolormesh(mesh_x, mesh_y, val.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=np.min(val), vmax=np.max(val))
        fig.colorbar(plot,  ax=ax, format="%1.1e")
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        fig.savefig(path+f'/{num}_sol.png', dpi=300)
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


class Poisson3DPeaks:
    def __init__(self, dev, dtp, weight, xlim, ylim, zlim, num_in, num_bd, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.ylim, self.zlim, self.input_size, self.output_size = 3, dev, dtp, weight, xlim, ylim, zlim, input_size, output_size
        self.criterion = torch.nn.MSELoss()
        self.physics = ['in', 'bd']
        self.size = {'in': num_in, 'bd': num_bd}
        #grid = np.array([-0.5, 0, 0.5])
        #grid2 = np.array([-0.5, 0, 0.5])
        #x, y, z = np.meshgrid(grid, grid2, grid2)
        #self.center = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        self.center = np.array([[-0.5,-0.5,-0.5], [-0.5,-0.5,0],[0,-0.5,-0.5],[0,-0.5,0]])
        self.K = 1000
        #self.center = np.array([[-0.5, -0.5, 0], [0, 0, 0]])


    def sample(self, size, mode):
        xs, xe, ys, ye, zs, ze = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1], self.zlim[0], self.zlim[1]
        x_len, y_len, z_len = xe - xs, ye-ys, ze-zs
        if mode == 'in':
            size_in = size
            node_in = torch.cat(
                (
                    torch.rand([size_in, 1]) * x_len + torch.ones(size=[size_in, 1]) * xs,
                    torch.rand([size_in, 1]) * y_len + torch.ones(size=[size_in, 1]) * ys,
                    torch.rand([size_in, 1]) * z_len + torch.ones(size=[size_in, 1]) * zs,
                ), dim=1)
            return node_in.to(device=self.dev, dtype=self.dtp)
        if mode == 'bd':
            size_bd = size
            bd_num = torch.randint(low=0, high=6, size=(size_bd,))
            node_bd = list(range(6))
            for i in range(6):
                ind = bd_num[bd_num == i]
                num = bd_num[ind].shape[0]
                if i == 0:
                    node_bd[i] = torch.cat([
                        torch.rand([num, 1]) * x_len + torch.ones([num, 1]) * xs,
                        torch.rand([num, 1]) * y_len + torch.ones([num, 1]) * ys,
                        torch.ones([num, 1]) * zs], dim=1)
                elif i == 1:
                    node_bd[i] = torch.cat([
                        torch.rand([num, 1]) * x_len + torch.ones([num, 1]) * xs,
                        torch.rand([num, 1]) * y_len + torch.ones([num, 1]) * ys,
                        torch.ones([num, 1]) * ze], dim=1)
                elif i == 2:
                    node_bd[i] = torch.cat([
                        torch.rand([num, 1]) * x_len + torch.ones([num, 1]) * xs,
                        torch.ones([num, 1]) * ys,
                        torch.rand([num, 1]) * z_len + torch.ones([num, 1]) * zs], dim=1)
                elif i == 3:
                    node_bd[i] = torch.cat([
                        torch.rand([num, 1]) * x_len + torch.ones([num, 1]) * xs,
                        torch.ones([num, 1]) * ye,
                        torch.rand([num, 1]) * z_len + torch.ones([num, 1]) * zs], dim=1)
                elif i == 4:
                    node_bd[i] = torch.cat([
                        torch.ones([num, 1]) * xs,
                        torch.rand([num, 1]) * y_len + torch.ones([num, 1]) * ys,
                        torch.rand([num, 1]) * z_len + torch.ones([num, 1]) * zs], dim=1)
                else:
                    node_bd[i] = torch.cat([
                        torch.ones([num, 1]) * xe,
                        torch.rand([num, 1]) * y_len + torch.ones([num, 1]) * ys,
                        torch.rand([num, 1]) * z_len + torch.ones([num, 1]) * zs], dim=1)
            return torch.cat(node_bd, dim=0).to(device=self.dev, dtype=self.dtp)

    def solve(self, mode, node):
        if mode == 'in':
            val_in = torch.zeros_like(node[:, 0])
            for i in range(self.center.shape[0]):
                val_in += -torch.exp(-self.K * ((node[:, 0] - self.center[i, 0]) ** 2 + (node[:, 1] - self.center[i, 1]) ** 2 + (node[:, 2] - self.center[i, 2]) ** 2)) * (
                        torch.pow((-2 * self.K) * (node[:, 0] - self.center[i, 0]), 2) + (-2 * self.K)
                      + torch.pow((-2 * self.K) * (node[:, 1] - self.center[i, 1]), 2) + (-2 * self.K)
                      + torch.pow((-2 * self.K) * (node[:, 2] - self.center[i, 2]), 2) + (-2 * self.K))
            return val_in
        elif mode == "bd":
            val_bd = torch.zeros_like(node[:, 0])
            for i in range(self.center.shape[0]):
                val_bd += torch.exp(-self.K * (
                        (node[:, 0] - self.center[i, 0]) ** 2 + (node[:, 1] - self.center[i, 1]) ** 2 + (node[:, 2] - self.center[i, 2]) ** 2)
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
            dz = d[:, 2].reshape(-1, 1)
            dxx = torch.autograd.grad(inputs=x,
                                      outputs=dx,
                                      grad_outputs=torch.ones_like(dx),
                                      retain_graph=True,
                                      create_graph=True)[0][:, 0].flatten()
            dyy = torch.autograd.grad(dy, x,
                                      grad_outputs=torch.ones_like(dy),
                                      retain_graph=True,
                                      create_graph=True)[0][:, 1].flatten()
            dzz = torch.autograd.grad(dz, x,
                                      grad_outputs=torch.ones_like(dz),
                                      retain_graph=True,
                                      create_graph=True)[0][:, 2].flatten()
            if cls == 'loss':
                pde_res = self.criterion(-dyy - dxx - dzz, pred)
            elif cls == 'ele':
                pde_res = -dyy - dxx - dzz - pred
            else:
                raise ValueError('Invalid cls')
            return pde_res
        elif mode == 'bd':
            bd_res = self.criterion(net(node).flatten(), pred)
            return bd_res
        else:
            raise ValueError('Invalid mode')

    def grid(self, size):
        xs, xe, ys, ye, zs, ze = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1], self.zlim[0], self.zlim[1]
        inter_x = np.linspace(start=xs, stop=xe, num=size + 1)
        inter_y = np.linspace(start=ys, stop=ye, num=size + 1)
        inter_z = np.linspace(start=zs, stop=ze, num=size + 1)
        mesh_x, mesh_y, mesh_z = np.meshgrid(inter_x, inter_y, inter_z)
        return mesh_x, mesh_y, mesh_z

    def is_node_in(self, node):
        return ((self.xlim[0] < node[:, 0]) & (node[:, 0] < self.xlim[1])
                & (self.ylim[0] < node[:, 1]) & (node[:, 1] < self.ylim[1])
                & (self.zlim[0] < node[:, 2]) & (node[:, 2] < self.zlim[1]))

    def exact(self, node):
        exact = np.zeros_like(node[:, 0])
        for i in range(self.center.shape[0]):
            exact += np.exp(-self.K * ((node[:, 0] - self.center[i, 0]) ** 2 + (node[:, 1] - self.center[i, 1]) ** 2 + (node[:, 2] - self.center[i, 2]) ** 2))
        return exact

    def test_err(self, net):
        mesh_x, mesh_y, mesh_z = self.grid(size=10)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()), axis=1)
        node_aux = torch.from_numpy(node).to(device=self.dev)
        val = net(node_aux).detach().cpu().numpy().flatten()
        exact = self.exact(node)
        err = np.sqrt(np.sum(np.power(val - exact, 2)) / np.sum(np.power(exact, 2)))
        return err

    def target_node_plot_together(self, loss, node_add, node_domain, proposal, path, num):
        #node_all = torch.cat([node_domain['in'].detach(),
                              #node_domain['bd'].detach()])
        node_all = node_domain['in'].detach()
        node_add = node_add.detach().cpu().numpy()
        node_all = node_all.cpu().numpy()
        xs, xe, ys, ye, zs, ze = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1], self.zlim[0], self.zlim[1]
        mesh_x, mesh_y, mesh_z = self.grid(size=50)
        node = np.stack([mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()], axis=1)
        val = loss(node).flatten()
        ####### use plotly ################
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.35 / 1.1, y=-1.95 / 1.1, z=1.45 / 1.1)
        )
        # plot loss
        self.plot_vol(node, val, None, path + f'/{num}_loss.png')
        # plot node
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(name=f'$S_{{{num}}}$', x=node_all[:, 0], y=node_all[:, 1], z=node_all[:, 2], mode='markers', marker=dict(size=1, opacity=0.08, color='blue')))
        fig.add_trace(go.Scatter3d(name='$\\mathcal{D}$', x=node_add[:, 0], y=node_add[:, 1], z=node_add[:, 2], mode='markers', marker=dict(size=1, opacity=1.0, color='red')))
        fig.update_layout(scene=dict(
            xaxis=dict(range=[xs, xe], title='x'),
            yaxis=dict(range=[ys, ye], title='y'),
            zaxis=dict(range=[zs, ze], title='z'),
        ),
            scene_camera=camera,
            width=640,
            height=480,
            margin=dict(l=20, r=20, b=50, t=20))
        fig.write_image(path+f'/{num}_node.png')
        ###### use matplotlib
        # # plot loss
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # for center in self.center:
        #     # plot XOY slice
        #     xx = mesh_x[:, :, 1].squeeze()
        #     yy = mesh_y[:, :, 1].squeeze()
        #     zz = np.ones_like(xx)*center[2]
        #     node_aux = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        #     val_aux = loss(node_aux).flatten()
        #     cmap = plt.get_cmap('jet')
        #     norm = plt.Normalize(val_aux.min(), val_aux.max())
        #     colors = cmap(norm(val_aux))
        #     ax.plot_surface(xx, yy, zz, facecolors=colors)
        #     # plot YOZ slice
        #     yy = mesh_y[:, :, 1].squeeze()
        #     zz = mesh_z[:, :, 1].squeeze()
        #     xx = np.ones_like(yy) * center[0]
        #     node_aux = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        #     val_aux = loss(node_aux).flatten()
        #     cmap = plt.get_cmap('jet')
        #     norm = plt.Normalize(val_aux.min(), val_aux.max())
        #     colors = cmap(norm(val_aux))
        #     ax.plot_surface(xx, yy, zz, facecolors=colors)
        if proposal:
            # plot proposal
            val = proposal(node).flatten()
            self.plot_vol(node, val, None, path + f'/{num}_proposal.png')

    def test_err_plot(self, net, path, num):
        mesh_x, mesh_y, mesh_z = self.grid(60)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()), axis=1)
        sol = self.exact(node)
        val = net(torch.from_numpy(node).to(device=self.dev)).detach().cpu().numpy().flatten()
        err = np.sqrt(np.sum(np.power(val - sol, 2)) / np.sum(np.power(sol, 2)))
        err_plt = val - sol
        # plot absolute error
        self.plot_vol(node, np.abs(err_plt), f'$e_r(u_{{{num}}}(\\cdot;\\theta))={round(err, 4)}$', path + f'/{num}_abs.png')
        # plot solution
        self.plot_vol(node, val, None, path + f'/{num}_sol.png')
        # plot exact
        if num == 1:
            self.plot_vol(node, sol.flatten(), None, path + f'/exact.png')
        # plot slice
        xs, xe, zs, ze = self.xlim[0], self.xlim[1], self.zlim[0], self.zlim[1]
        inter_x = np.linspace(start=xs, stop=xe, num=256 + 1)
        inter_z = np.linspace(start=zs, stop=ze, num=256 + 1)
        mesh_x, mesh_z= np.meshgrid(inter_x, inter_z)
        node = np.stack([mesh_x.flatten(), mesh_z.flatten()], axis=1)
        node = np.stack([node[:, 0], np.ones_like(node[:,0])*(-0.5), node[:, 1]], axis=1)
        val = net(torch.from_numpy(node).to(device=self.dev)).detach().cpu().numpy().flatten()
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        plot = ax.pcolormesh(mesh_x, mesh_z, val.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=np.min(val), vmax=np.max(val))
        fig.colorbar(plot, ax=ax, format="%1.1e")
        ax.set_xlabel('$x$')
        ax.set_ylabel('$z$')
        fig.savefig(path + f'/{num}_sol_xz.png', dpi=300)
        plt.close(fig)
    
    @staticmethod
    def plot_vol(node, val, title, fname):
        fig = go.Figure(data=go.Volume(
            x=node[:, 0],
            y=node[:, 1],
            z=node[:, 2],
            value=val,
            isomin=np.min(val),
            isomax=np.max(val),
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=21,  # needs to be a large number for good volume renderingNone
            colorscale='jet'
        ))
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.35 / 1.1, y=-1.95 / 1.1, z=1.45 / 1.1)
        )
        fig.update_layout(
            title=dict(text=title, x=0.5, y=0.9, xanchor='center', yanchor='top'),
            scene=dict(
            xaxis=dict(title='x'),
            yaxis=dict(title='y'),
            zaxis=dict(title='z'),
        ),
            scene_camera=camera,
            width=640,
            height=480,
            margin=dict(l=20, r=20, b=50, t=20))
        fig.update_traces(colorbar=dict(tickformat='.1e'))
        if 'html' in fname:
            fig.write_html(fname, include_mathjax='cdn')
        else:
            fig.write_image(fname)


class Poisson9DPeaks:
    def __init__(self, dev, dtp, weight, axeslim, num_in, num_bd, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.axeslim, self.input_size, self.output_size = 5, dev, dtp, weight, np.array(axeslim), input_size, output_size
        self.criterion = torch.nn.MSELoss()
        self.physics = ['in', 'bd']
        self.size = {'in': num_in, 'bd': num_bd}
        # grid = np.array([-0.5, 0, 0.5])
        # grid2 = np.array([-0.5, 0.5])
        # x, y, z = np.meshgrid(grid, grid2, grid2)
        # self.center = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        self.K = 100
        self.center = np.stack([np.zeros((self.dim, )), np.zeros((self.dim,))], axis=0)
        self.center[0, 0], self.center[0, 1] = -0.5, -0.5
        self.center[1, 0], self.center[1, 1] = 0.5, 0.5

    def sample(self, size, mode):
        if mode == 'in':
            node_in = torch.zeros((size, self.dim))
            for i in range(self.dim):
                leftlim = self.axeslim[i, 0]
                rightlim = self.axeslim[i, 1]
                length = rightlim - leftlim
                node_in[:, i] = torch.rand((size, ))*length+torch.ones((size, )) * leftlim
            return node_in.to(device=self.dev, dtype=self.dtp)
        if mode == 'bd':
            size_bd = size
            bd_num = torch.randint(low=0, high=2*self.dim, size=(size_bd,))
            node_bd = list(range(2*self.dim))
            for i in range(2*self.dim):
                ind = bd_num[bd_num == i]
                num = bd_num[ind].shape[0]
                m, n = i//2, i % 2
                node_bd[i] = torch.empty((num, 0))
                for j in range(self.dim):
                    if j != m:
                        node_bd[i] = torch.cat([node_bd[i],
                                                torch.rand([num, 1])*(self.axeslim[j, 1] - self.axeslim[j, 0])+torch.ones([num, 1])*self.axeslim[j, 0]], dim=1)
                    else:
                        node_bd[i] = torch.cat([node_bd[i],
                                                torch.ones([num, 1]) * self.axeslim[m, n]], dim=1)
            return torch.cat(node_bd, dim=0).to(device=self.dev, dtype=self.dtp)

    def solve(self, mode, node):
        if mode == 'in':
            val_in = torch.zeros_like(node[:, 0])
            for i in range(self.center.shape[0]):
                node_exp = torch.zeros_like(node[:, 0])
                val_aux = torch.ones_like(node[:, 0])
                for j in range(self.dim):
                    val_aux *= torch.exp(-self.K*(torch.pow((node[:, j]-self.center[i, j]), 2)))
                node_exp += val_aux
                val_aux = torch.zeros_like(node[:, 0])
                for j in range(self.dim):
                    val_aux += -node_exp * (
                        torch.pow((-2 * self.K) * (node[:, j] - self.center[i, j]), 2) + (-2 * self.K))
                val_in += val_aux
            return val_in
        elif mode == "bd":
            node_exp = torch.zeros_like(node[:, 0])
            for i in range(self.center.shape[0]):
                val_aux = torch.ones_like(node[:, 0])
                for j in range(self.dim):
                    val_aux *= torch.exp(-self.K * (torch.pow((node[:, j] - self.center[i, j]), 2)))
                node_exp += val_aux
            return node_exp
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
            dd = torch.zeros_like(pred)
            for i in range(self.dim):
                di = d[:, i].reshape(-1, 1)
                dii = torch.autograd.grad(inputs=x,
                                          outputs=di,
                                          grad_outputs=torch.ones_like(di),
                                          retain_graph=True,
                                          create_graph=True)[0][:, i].flatten()
                dd += dii
            if cls == 'loss':
                pde_res = self.criterion(-dd, pred)
            elif cls == 'ele':
                pde_res = -dd - pred
            else:
                raise ValueError('Invalid cls')
            return pde_res
        elif mode == 'bd':
            bd_res = self.criterion(net(node).flatten(), pred)
            return bd_res
        else:
            raise ValueError('Invalid mode')

    def grid(self, size):
        # project on the first two dims
        xs, xe, ys, ye= self.axeslim[0, 0], self.axeslim[0, 1], self.axeslim[1, 0], self.axeslim[1, 1]
        inter_x = np.linspace(start=xs, stop=xe, num=size + 1)
        inter_y = np.linspace(start=ys, stop=ye, num=size + 1)
        mesh_x, mesh_y = np.meshgrid(inter_x, inter_y)
        return mesh_x, mesh_y

    def is_node_in(self, node):
        node_aux = np.copy(node)
        aux = np.full(node_aux[:, 0].shape, True)
        for i in range(self.dim):
            aux = aux & (self.axeslim[i, 0]<node_aux[:, i]) & (node_aux[:, i]<self.axeslim[i, 1])
        return aux
        
    def exact(self, node):
        node_exp = np.zeros_like(node[:, 0])
        for i in range(self.center.shape[0]):
            val_aux = np.ones_like(node[:, 0])
            for j in range(self.dim):
                val_aux *= np.exp(-self.K * (np.power((node[:, j] - self.center[i, j]), 2)))
            node_exp += val_aux
        return node_exp

    def test_err(self, net):
        node = self.sample(1500, 'in').detach().cpu().numpy()
        node_aux = np.empty((0, self.dim))
        for i in range(self.center.shape[0]):
            node_aux = np.concatenate([node_aux, ss.multivariate_normal.rvs(mean=self.center[i, :], cov=np.diag(np.ones((self.dim,))*(1/(self.K*2))), size=5000)], axis=0)
        node = np.concatenate([node, node_aux], axis=0)
        node_aux = torch.from_numpy(node).to(device=self.dev)
        val = net(node_aux).detach().cpu().numpy().flatten()
        exact = self.exact(node)
        err = np.sqrt(np.sum(np.power(val - exact, 2)) / np.sum(np.power(exact, 2)))
        return err

    def target_node_plot_together(self, loss, node_add, node_domain, proposal, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['bd'].detach()])
        node_add = node_add.detach().cpu().numpy()
        node_all = node_all.cpu().numpy()
        xs, xe, ys, ye = self.axeslim[0, 0], self.axeslim[0, 1], self.axeslim[1, 0], self.axeslim[1,1]
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack([mesh_x.flatten(), mesh_y.flatten()], axis=1)
        for i in range(self.dim - 2):
            node = np.concatenate([node, np.ones_like(node[:, 0]).reshape(-1, 1) * self.center[0, i + 2]], axis=1)
        val = loss(node).flatten()
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        # plot loss
        plot = ax.pcolormesh(mesh_x, mesh_y, val.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=0, vmax=np.max(val))
        fig.colorbar(plot, ax=ax, format="%1.1e")
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        fig.savefig(path + f'/{num}_loss.png', dpi=300)
        plt.close(fig)
        # plot node
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        ax.set_xlim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.20)
        ax.set_ylim(ys - (ye - ys) * 0.05, ye + (ye - ys) * 0.20)
        ax.scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.', s=np.ones_like(node_all[:, 0]), alpha=0.1,
                   label=f'$\\mathcal{{S}}_{{{num}}}$')
        ax.scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.', s=np.ones_like(node_add[:, 0]), alpha=1.0,
                   label=f'$\\mathcal{{D}}$')
        ax.legend(loc='upper right', fontsize=12)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        fig.savefig(path + f'/{num}_node.png', dpi=300)
        plt.close(fig)
        if proposal:
            val_prop = proposal(node).reshape(mesh_x.shape)
            fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
            # plot proposal
            plot = ax.pcolormesh(mesh_x, mesh_y, val_prop, shading='gouraud', cmap='jet', vmin=0, vmax=np.max(val_prop))
            fig.colorbar(plot, ax=ax, format="%1.1e")
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            plt.savefig(path + f'/{num}_proposal.png', dpi=300)
            plt.close()

    def test_err_plot(self, net, path, num):
        node = self.sample(150000, 'in').detach().cpu().numpy()
        node_aux = np.empty((0, self.dim))
        for i in range(self.center.shape[0]):
            node_aux = np.concatenate([node_aux, ss.multivariate_normal.rvs(mean=self.center[i, :], cov=np.diag(np.ones((self.dim,))*(1/(self.K*2))), size=5000)], axis=0)
        node = np.concatenate([node, node_aux], axis=0)
        node_aux = torch.from_numpy(node).to(device=self.dev)
        val = net(node_aux).detach().cpu().numpy().flatten()
        exact = self.exact(node)
        err_show = np.sqrt(np.sum(np.power(val - exact, 2)) / np.sum(np.power(exact, 2)))
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        for i in range(self.dim-2):
            node = np.concatenate([node, np.ones_like(node[:, 0]).reshape(-1,1)*self.center[0, i+2]], axis=1)
        node_aux = torch.from_numpy(node).to(device=self.dev)
        val = net(node_aux).detach().cpu().numpy().flatten()
        exact = self.exact(node)
        err = np.sqrt(np.sum(np.power(val - exact, 2)) / np.sum(np.power(exact, 2)))
        err_plt = np.abs(val - exact)
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        # err plot
        plot = ax.pcolormesh(mesh_x, mesh_y, err_plt.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=0,
                             vmax=np.max(err_plt))
        fig.colorbar(plot, ax=ax, format="%1.1e")
        ax.set_title(f'$e_r(u_{{{num}}}(\\cdot;\\theta))={round(err_show, 4)}$')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        fig.savefig(path + f'/{num}_abs.png', dpi=300)
        plt.close(fig)
        # val plot
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        plot = ax.pcolormesh(mesh_x, mesh_y, val.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=np.min(val),
                             vmax=np.max(val))
        fig.colorbar(plot, ax=ax, format="%1.1e")
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        fig.savefig(path + f'/{num}_sol.png', dpi=300)
        plt.close(fig)
        # exact plot
        if num == 1:
            fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
            plot = ax.pcolormesh(mesh_x, mesh_y, exact.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=0, vmax=1)
            fig.colorbar(plot, ax=ax, format="%1.1e")
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            fig.savefig(path + f'/{num}_exact.png', dpi=300)
            plt.close(fig)


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


