import torch

from .utils import *
from scipy.stats import beta
from scipy.special import gamma
class FDF2D:

    def __init__(self,  dev, dtp, weight, xlim, tlim, num_in, num_bd, input_size, output_size, MC_size, eps, alpha):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.tlim, self.input_size, self.output_size = 2, dev, dtp, weight, xlim, tlim, input_size, output_size
        self.criterion = torch.nn.MSELoss()
        self.eps = eval(eps)
        self.physics = ['in', 'bd', 'init']
        self.size = {'in': num_in, 'bd': num_bd, 'init': num_bd}
        self.MC_size = MC_size
        self.alpha = alpha

    def sample(self, size, mode):
        xs, xe, ts, te = self.xlim[0], self.xlim[1], self.tlim[0], self.tlim[1]
        x_len, t_len = xe - xs, te - ts
        if mode == 'in':
            size_in = size
            node_in = torch.cat(
                (
                        torch.rand([size_in, 1]) * t_len + torch.ones(size=[size_in, 1]) * ts, torch.rand([size_in, 1]) * x_len + torch.ones(size=[size_in, 1]) * xs
                        ), dim=1)
            return node_in.to(device=self.dev, dtype=self.dtp)
        if mode == 'bd':
            size_bd = size
            bd_num = torch.randint(low=0, high=2, size=(size_bd,))
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
            node_in = node
            val_in = torch.zeros_like(node_in[:, 0])
            return val_in
        elif mode == "bd":
            node_bd = node
            val_bd = torch.zeros_like(node_bd[:, 0])
            return val_bd
        elif mode == "init":
            node_init = node
            val_init = torch.sin(torch.pi * node_init[:, 1])
            return val_init
        else:
            raise ValueError("invalid mode")

    def FracD_MC(self, net, node):
        val = net(node)
        node_0 = torch.clone(node)
        node_0[:, 0] = 0
        val_0 = net(node_0)
        endtime = node[:, 0].detach().cpu().numpy()
        space = node[:, 1:].detach().cpu().numpy().reshape(-1, self.dim)
        sample_monte_carlo = beta(1-self.alpha, 1, size=self.MC_size)
        rI_eps = np.where(sample_monte_carlo<self.eps, self.eps, sample_monte_carlo)
        mesh_x, mesh_y = np.meshgrid(endtime, rI_eps, indexing='ij')
        timelist_aux = mesh_y*mesh_x
        timelist = mesh_x-mesh_y*mesh_x
        node_MC = np.concatenate((np.expand_dims(timelist, -1), np.expand_dims(space, 1).repeat(self.MC_size, axis=1)), axis=-1)
        shape_node_MC = node_MC.shape
        val_MC = net(torch.from_numpy(node_MC.reshape(-1, self.dim)).to(device=self.dev)).reshape(shape_node_MC[:-1])
        expect = np.mean((val.reshape((-1, 1))-val_MC)/timelist)
        frac_der = torch.zeros_like(node[:, 0])
        for i, node_aux in enumerate(node_MC):
                val_aux = net(torch.from_numpy(node_aux).to(device=self.dev))
                expect = torch.mean((val[i]-val_aux)/(timelist_aux[i, :]))
                frac_der[i] = self.alpha/gamma(2-self.alpha)*node[i, 0]**(1-self.alpha)*expect+(val[i]-val_0[i])/(node[i, 0]**self.alpha*gamma(1-self.alpha))
        return frac_der

    def residual(self, node, net, cls="loss", mode="in"):
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
            dt = self.FracD_MC(net, node)
            dx = d[:, 1].reshape(-1, 1)
            dxx = torch.autograd.grad(inputs=x,
                                      outputs=dx,
                                      grad_outputs=torch.ones_like(dx),
                                      retain_graph=True,
                                      create_graph=True)[0][:, 1].flatten()
            if cls == "loss":
                pde_res = self.criterion(dt.flatten()-dxx, pred)
            elif cls == "ele":
                pde_res = dt.flatten()-dxx-pred
            else:
                raise ValueError("Invalid mode")
            return pde_res
        elif mode == "bd":
            node_bd = node
            bd_res = self.criterion(net(node_bd).flatten(), pred)
            return bd_res
        elif mode == "init":
            node_init = node
            init_res = self.criterion(net(node_init).flatten(), pred)
            return init_res
        else:
            raise ValueError("Invalid mode")

    def grid(self, size):
        xs, xe, ts, te = self.xlim[0], self.xlim[1], self.tlim[0], self.tlim[1]
        inter_t = np.linspace(start=ts, stop=te, num=size + 1)
        inter_x = np.linspace(start=xs, stop=xe, num=size + 1)
        mesh_t, mesh_x = np.meshgrid(inter_t, inter_x)
        return mesh_t, mesh_x

    def is_node_in(self, node):
        return ((self.tlim[0] < node[:, 0]) & (node[:, 0] < self.tlim[1])
                & (self.xlim[0] < node[:, 1]) & (node[:, 1] < self.xlim[1]))

    def test_err(self, net):
        data = np.load("./data/FracDff.mat")
        t, x, exact = data["t"], data["x"], data["exact"]
        mesh_t, mesh_x = np.meshgrid(t, x)
        node = np.stack((mesh_t.flatten(), mesh_x.flatten()), axis=1)
        node = torch.from_numpy(node).to(device=self.dev)
        val = net(node).detach().cpu().numpy().flatten()
        sol = exact.reshape(-1, 1).flatten()
        err = np.sqrt(np.sum(np.power(val - sol, 2)) / np.sum(np.power(sol, 2)))
        return err

    def target_node_plot_together(self, loss, node_add, node_domain, proposal, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['bd'].detach(),
                              node_domain['init'].detach()])
        node_all = node_all.cpu().numpy()
        node_add = node_add.detach().cpu().numpy()
        ts, te = self.tlim[0], self.tlim[1]
        xs, xe = self.xlim[0], self.xlim[1]
        mesh_t, mesh_x = self.grid(size=256)
        node = np.stack((mesh_t.flatten(), mesh_x.flatten()), axis=1)
        val = loss(node).reshape(mesh_t.shape)
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        # plot loss
        plot = ax.pcolormesh(mesh_t, mesh_x, val, shading='gouraud', cmap='jet', vmin=0, vmax=np.max(val))
        fig.colorbar(plot, ax=ax, format="%1.1e")
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        fig.savefig(path + f'/{num}_loss.png', dpi=300)
        plt.close(fig)
            # plot node
        fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        ax.set_xlim(ts - (te - ts) * 0.05, te + (te - ts) * 0.20)
        ax.set_ylim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.20)
        ax.scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.', s=np.ones_like(node_all[:, 0]), alpha=0.3)
        ax.scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.', s=np.ones_like(node_add[:, 0]), alpha=1.0)
        ax.legend(loc='upper right')
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        fig.savefig(path + f'/{num}_node.png', dpi=300)
        plt.close(fig)
        if proposal:
            val_prop = proposal(node).reshape(mesh_x.shape)
            fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
            # plot proposal
            plot = ax.pcolormesh(mesh_t, mesh_x, val_prop, shading='gouraud',
                                 cmap='jet', vmin=0, vmax=np.max(val_prop))
            fig.colorbar(plot, ax=ax, format="%1.1e")
            ax.set_xlabel('t')
            ax.set_ylabel('x')
            plt.savefig(path + f'/{num}_proposal.png', dpi=300)
            plt.close()

    def test_err_plot(self, net, path, num):
        data = np.load("./data/FracDff.mat")
        t, x, exact = data["t"], data["x"], data["exact"]
        X, Y = np.meshgrid(t, x)
        node = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
        node = torch.from_numpy(node).to(device=self.dev)
        val = net(node).detach().cpu().numpy().flatten()
        sol = exact.reshape(-1, 1).flatten()
        err = np.sqrt(np.sum(np.power(val - sol, 2)) / np.sum(np.power(sol, 2)))
        err_plt = np.abs(val - sol)
        # plot absolute error
        fig, axes = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        plot = axes.pcolormesh(X, Y, err_plt.reshape(X.shape), shading='gouraud',
                                     cmap='jet', vmin=0, vmax=np.max(err_plt))
        fig.colorbar(plot, ax=axes, format="%1.1e")
        axes.set_title(f'$e_r(u_{{{num}}}(\\cdot;\\theta))={round(err, 4)}$')
        axes.set_xlabel('$t$')
        axes.set_ylabel('$x$')
        fig.savefig(path + f'/{num}_abs.png', dpi=300)
        plt.close(fig)
        # plot predict
        fig, axes = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
        plot = axes.pcolormesh(X, Y, val.reshape(X.shape), shading='gouraud',
                                     cmap='jet', vmin=-1.05, vmax=1.05)
        fig.colorbar(plot, ax=axes, format="%1.1e")
        axes.set_xlabel('$t$')
        axes.set_ylabel('$x$')
        fig.savefig(path + f'/{num}_sol.png', dpi=300)
        plt.close(fig)
        # plot exact
        if num == 1:
            fig, axes = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
            plot = axes.pcolormesh(X, Y, exact, shading='gouraud',
                                         cmap='jet', vmin=-1.05, vmax=1.05)
            fig.colorbar(plot, ax=axes, format="%1.1e")
            axes.set_xlabel('$t$')
            axes.set_ylabel('$x$')
            fig.savefig(path + f'/{num}_exact.png', dpi=300)
            plt.close(fig)
        # plot at t
        # t_plt = [0.10, 0.50, 0.90]
        # for count, t_now in enumerate(t_plt):
        #     ind = np.where(t == t_now)[0]
        #     sol_t = exact[ind, :]
        #     node = np.concatenate((np.ones_like(x) * t[ind], x), axis=1)
        #     node = torch.from_numpy(node).to(device=self.dev)
        #     val_t = net(node).detach().cpu().numpy().flatten()
        #     axes[1, count].plot(x, sol_t.flatten(), 'r', label=f'$u^*({t_now}, x)$')
        #     axes[1, count].plot(x, val_t, 'b--', label=f'$u^\\theta_{{{num}}}({t_now}, x)$')
        #     err = np.sqrt(np.sum(np.power(val_t - sol_t, 2)) / np.sum(np.power(sol_t, 2)))
        #     axes[1, count].set_title(f'$e_t(u^\\theta_{{{num}}},{t_now})={round(err, 4)}$')
        #     axes[1, count].legend(loc='upper right')
        #     axes[1, count].set_xlabel('$x$')
        #     count += 1
        # plt.savefig(path + f'/{num}_sol.png', dpi=300)
        # plt.close()