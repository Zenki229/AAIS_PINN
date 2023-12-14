from .utils import *


class AllenCahn2D:

    def __init__(self,  dev, dtp, weight, xlim, tlim, num_in, num_bd, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.tlim, self.input_size, self.output_size \
            = 2, dev, dtp, weight, xlim, tlim, input_size, output_size
        self.criterion = torch.nn.MSELoss()
        self.physics = ['in', 'bd', 'init']
        self.size = {'in': num_in, 'bd': num_bd, 'init': num_bd}

    def sample(self, size, mode):
        xs, xe, ts, te = self.xlim[0], self.xlim[1], self.tlim[0], self.tlim[1]
        x_len, t_len = xe - xs, te-ts
        if mode == 'in':
            node_in = torch.cat(
                (
                        torch.rand([size, 1]) * t_len + torch.ones(size=[size, 1]) * ts,
                        torch.rand([size, 1]) * x_len + torch.ones(size=[size, 1]) * xs
                        ), dim=1)
            return node_in.to(device=self.dev, dtype=self.dtp)
        if mode == 'bd':
            bd_num = torch.randint(low=0, high=2, size=(size,))
            node_bd = list(range(2))
            for i in range(2):
                ind = bd_num[bd_num == i]
                num = bd_num[ind].shape[0]
                if i == 0:
                    node_bd[i] = torch.cat(
                        [
                                torch.rand([num, 1]) * t_len + torch.ones([num, 1]) * ts,
                                torch.ones([num, 1]) * xs,
                                ], dim=1)
                else:
                    node_bd[i] = torch.cat(
                        [
                            torch.rand([num, 1]) * t_len + torch.ones([num, 1]) * ts,
                            torch.ones([num, 1]) * xe,
                            ], dim=1)
            return torch.cat(node_bd, dim=0).to(device=self.dev, dtype=self.dtp)
        if mode == 'init':
            size_init = size
            node_init = torch.cat(
                        [
                                torch.ones([size_init, 1]) * ts,
                                torch.rand([size_init, 1]) * x_len + torch.ones([size_init, 1]) * xs
                                ], dim=1)
            return node_init.to(device=self.dev, dtype=self.dtp)

    @staticmethod
    def solve(mode, node):
        if mode == "in":
            val_in = torch.zeros_like(node[:, 0])
            return val_in
        elif mode == "bd":
            val_bd = -torch.ones_like(node[:, 0])
            return val_bd
        elif mode == "init":
            val_init = node[:, 1]**2 * torch.cos(torch.pi * node[:, 1])
            return val_init
        else:
            raise ValueError("invalid mode")

    def residual(self, node, net, cls, mode):
        pred = self.solve(mode, node)
        if mode == "in":
            x = node
            x.requires_grad = True
            val = net(x)
            d = torch.autograd.grad(outputs=val,
                                    inputs=x,
                                    grad_outputs=torch.ones_like(val),
                                    retain_graph=True,
                                    create_graph=True)[0]
            dt = d[:, 0].reshape(-1, 1)
            dx = d[:, 1].reshape(-1, 1)
            dxx = torch.autograd.grad(inputs=x,
                                      outputs=dx,
                                      grad_outputs=torch.ones_like(dx),
                                      retain_graph=True,
                                      create_graph=True)[0][:, 1].flatten()
            if cls == "loss":
                pde_res = self.criterion(dt.flatten()-0.001*dxx-5*(val.flatten()-torch.pow(val.flatten(), 3)), pred)
            elif cls == "ele":
                pde_res = dt.flatten()-0.001*dxx-5*(val.flatten()-torch.pow(val.flatten(), 3)) - pred
            else:
                raise ValueError("Invalid cls")
            return pde_res
        elif mode == "bd":
            bd_res = self.criterion(net(node).flatten(), pred)
            return bd_res
        elif mode == "init":
            init_res = self.criterion(net(node).flatten(), pred)
            return init_res
        else:
            raise ValueError("Invalid mode")

    def grid(self, size):
        ts, te, xs, xe = self.tlim[0], self.tlim[1], self.xlim[0], self.xlim[1]
        inter_t = np.linspace(start=ts, stop=te, num=size + 1)
        inter_x = np.linspace(start=xs, stop=xe, num=size + 1)
        mesh_t, mesh_x = np.meshgrid(inter_t, inter_x)
        return mesh_t, mesh_x

    def is_node_in(self, node):
        return ((self.tlim[0] < node[:, 0]) & (node[:, 0] < self.tlim[1])
                & (self.xlim[0] < node[:, 1]) & (node[:, 1] < self.xlim[1]))

    def test_err(self, net):
        data = loadmat('./data/AllenCahn.mat')
        t, x, exact = data['t'], data['x'], data['u']
        mesh_t,  mesh_x = np.meshgrid(t, x, indexing='ij')
        node = np.stack((mesh_t.flatten(), mesh_x.flatten()), axis=1)
        node = torch.from_numpy(node).to(device=self.dev, dtype=self.dtp)
        val = net(node).detach().cpu().numpy().flatten()
        sol = exact.flatten()
        err = np.sqrt(np.sum(np.power(val - sol, 2)) / np.sum(np.power(sol, 2)))
        return err

    def target_node_plot_together(self, loss, node_add, node_domain, proposal, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['bd'].detach(),
                              node_domain['init'].detach()], dim=0)
        node_all = node_all.cpu().numpy()
        node_add = node_add.detach().cpu().numpy()
        ts, te, xs, xe = self.tlim[0], self.tlim[1], self.xlim[0], self.xlim[1]
        mesh_t, mesh_x = self.grid(size=256)
        node = np.stack((mesh_t.flatten(), mesh_x.flatten()), axis=1)
        val = loss(node).reshape(mesh_t.shape)
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        # plot loss
        plot = ax.pcolormesh(mesh_t, mesh_x, val, shading='gouraud', cmap='jet', vmin=0, vmax=np.max(val))
        fig.colorbar(plot, ax=ax, format="%1.1e")
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        fig.savefig(path + f'/{num}_loss.png', dpi=300)
        plt.close(fig)
        # plot node
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        ax.set_xlim(ts - (te - ts) * 0.05, te + (te - ts) * 0.20)
        ax.set_ylim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.20)
        ax.scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.', s=np.ones_like(node_all[:, 0]), alpha=0.3, label=f'$\\mathcal{{S}}_{{{num}}}$')
        ax.scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.', s=np.ones_like(node_add[:, 0]), alpha=1.0, label=f'$\\mathcal{{D}}$')
        ax.legend(loc='upper right')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        fig.savefig(path + f'/{num}_node.png', dpi=300)
        plt.close(fig)
        if proposal:
            val_prop = proposal(node).reshape(mesh_x.shape)
            fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
            # plot proposal
            plot = ax.pcolormesh(mesh_t, mesh_x, val_prop, shading='gouraud',
                                    cmap='jet', vmin=0, vmax=np.max(val_prop))
            fig.colorbar(plot, ax=ax, format="%1.1e")
            ax.set_xlabel('$t$')
            ax.set_ylabel('$x$')
            plt.savefig(path + f'/{num}_proposal.png', dpi=300)
            plt.close()

    def test_err_plot(self, net, path, num):
        data = loadmat('./data/AllenCahn.mat')
        t, x, exact = data['t'].flatten(), data['x'].flatten(), data['u']
        mesh_t, mesh_x = np.meshgrid(t, x, indexing='ij')
        node = np.stack((mesh_t.flatten(), mesh_x.flatten()), axis=1)
        node = torch.from_numpy(node).to(device=self.dev, dtype=self.dtp)
        val = net(node).detach().cpu().numpy().flatten()
        sol = exact.flatten()
        err = np.sqrt(np.sum(np.power(val - sol, 2)) / np.sum(np.power(sol, 2)))
        err_plt = np.abs(val - sol)
        fig, axes = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        # plot absolute error
        plot = axes.pcolormesh(mesh_t, mesh_x, err_plt.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=0, vmax=np.max(err_plt))
        fig.colorbar(plot, ax=axes, format="%1.1e")
        axes.set_title(f'$e_r(u_{{{num}}}(\\cdot;\\theta))={round(err, 4)}$')
        axes.set_xlabel('$t$')
        axes.set_ylabel('$x$')
        fig.savefig(path + f'/{num}_abs.png', dpi=300)
        plt.close(fig)
        # plot predict
        fig, axes = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        plot = axes.pcolormesh(mesh_t, mesh_x, val.reshape(mesh_x.shape), shading='gouraud', cmap='jet', vmin=-1.05, vmax=1.05)
        fig.colorbar(plot, ax=axes, format="%1.1e")
        axes.set_xlabel('$t$')
        axes.set_ylabel('$x$')
        fig.savefig(path + f'/{num}_sol.png', dpi=300)
        plt.close(fig)
        # plot exact
        if num == 1:
            fig, axes = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
            plot = axes.pcolormesh(mesh_t, mesh_x, exact, shading='gouraud', cmap='jet', vmin=-1.05, vmax=1.05)
            fig.colorbar(plot, ax=axes, format="%1.1e")
            axes.set_xlabel('$t$')
            axes.set_ylabel('$x$')
            fig.savefig(path + f'/{num}_exact.png', dpi=300)
            plt.close(fig)
        # plot at t
        # t_plt = [0.10, 0.50, 0.90]
        # for enum, t_now in enumerate(t_plt):
        #     ind = np.where(t == t_now)[0]
        #     sol_t = exact[ind, :]
        #     node = np.stack((np.ones_like(x) * t[ind], x), axis=1)
        #     node = torch.from_numpy(node).to(device=self.dev, dtype=self.dtp)
        #     val_t = net(node).detach().cpu().numpy().flatten()
        #     axes[1, enum].plot(x, sol_t.flatten(), 'r', label=f'$u^*({t_now}, x)$')
        #     axes[1, enum].plot(x, val_t, 'b--', label=f'$u^\\theta_{{{num}}}({t_now}, x)$')
        #     err = np.sqrt(np.sum(np.power(val_t - sol_t.flatten(), 2)) / np.sum(np.power(sol_t.flatten(), 2)))
        #     axes[1, enum].set_title(f'$e_t(u^\\theta_{{{num}}},{t_now})={round(err, 4)}$')
        #     axes[1, enum].legend(loc='upper right')
        # plt.savefig(path + f'/{num}_sol.png', dpi=300)
        # plt.close()
