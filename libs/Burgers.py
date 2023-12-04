from .utils import *


class Burgers2D:

    def __init__(self,  dev, dtp, weight, xlim, tlim, viscosity, num_in, num_bd, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.tlim, self.input_size, self.output_size = 2, dev, dtp, weight, xlim, tlim, input_size, output_size
        self.viscosity = eval(viscosity)
        self.criterion = torch.nn.MSELoss()
        self.physics = ['in', 'bd', 'init']
        self.size = {'in': num_in, 'bd': num_bd, 'init': num_bd}

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
            val_init = -torch.sin(torch.pi * node_init[:, 1])
            return val_init
        else:
            raise ValueError("invalid mode")

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
            dt = d[:, 0].reshape(-1, 1)
            dx = d[:, 1].reshape(-1, 1)
            dxx = torch.autograd.grad(inputs=x,
                                      outputs=dx,
                                      grad_outputs=torch.ones_like(dx),
                                      retain_graph=True,
                                      create_graph=True)[0][:, 1].flatten()
            if cls == "loss":
                pde_res = self.criterion(dt.flatten()+val.flatten()*dx.flatten()-0.01/torch.pi*dxx, pred)
            elif cls == "ele":
                pde_res = dt.flatten()+val.flatten()*dx.flatten()-self.viscosity*dxx-pred
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
        data = np.load("./data/Burgers1D.npz")
        t, x, exact = data["t"], data["x"], data["usol"].T
        mesh_t, mesh_x = np.meshgrid(t, x, indexing='ij')
        node = np.stack((mesh_t.flatten(), mesh_x.flatten()), axis=1)
        node = torch.from_numpy(node).to(device=self.dev)
        val = net(node).detach().cpu().numpy().flatten()
        sol = exact.reshape(-1, 1).flatten()
        err = np.sqrt(np.sum(np.power(val - sol, 2)) / np.sum(np.power(sol, 2)))
        return err

    def target_node_plot_together(self, loss, node_add, node_domain, IS_sign, proposal, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['bd'].detach(),
                              node_domain['init'].detach()])
        node_all = node_all.cpu().numpy()
        node_add = node_add.detach().cpu().numpy()
        xs, xe = self.tlim[0], self.tlim[1]
        ys, ye = self.xlim[0], self.xlim[1]
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        val = loss(node).reshape(mesh_x.shape)
        if (IS_sign == 0) | (proposal is None):
            fig, ax = plt.subplots(1, 2, layout='constrained', figsize=(12.8, 4.8))
            # plot loss
            plot = ax[0].pcolormesh(mesh_x, mesh_y, val, shading='gouraud', cmap='jet', vmin=0, vmax=np.max(val))
            fig.colorbar(plot, ax=ax[0], format="%1.1e")
            ax[0].set_title(f'residual $\\mathcal{{Q}}_{{{num}}}$')
            ax[0].set_xlabel('$t$')
            ax[0].set_ylabel('$x$')
            # plot node
            ax[1].set_xlim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.15)
            ax[1].set_ylim(ys - (ye - ys) * 0.05, ye + (ye - ys) * 0.15)
            ax[1].scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.',
                          s=np.ones_like(node_all[:, 0]), alpha=0.5, label='before')
            ax[1].scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.',
                          s=np.ones_like(node_add[:, 0]), alpha=1.0, label='now')
            ax[1].legend(loc='upper right')
            ax[1].set_title('nodes')
            ax[1].set_xlabel('$t$')
            ax[1].set_ylabel('$x$')
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
            ax[0].set_xlabel('$t$')
            ax[0].set_ylabel('$x$')
            # plot proposal
            plot = ax[1].pcolormesh(mesh_x, mesh_y, val_prop, shading='gouraud',
                                    cmap='jet', vmin=0, vmax=np.max(val_prop))
            fig.colorbar(plot, ax=ax[1], format="%1.1e")
            ax[1].set_title('proposal')
            ax[1].set_xlabel('$t$')
            ax[1].set_ylabel('$x$')
            # plot node
            ax[2].set_xlim(xs - (xe - xs) * 0.05, xe + (xe - xs) * 0.15)
            ax[2].set_ylim(ys - (ye - ys) * 0.05, ye + (ye - ys) * 0.15)
            ax[2].scatter(node_all[:, 0], node_all[:, 1], c='b', marker='.',
                          s=np.ones_like(node_all[:, 0]), alpha=0.5, label=f'$\\mathcal{{S}}_{{{num}}}$')
            ax[2].scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.',
                          s=np.ones_like(node_add[:, 0]), alpha=1.0, label=f'$\\mathcal{{D}}$')
            ax[2].legend(loc='upper right')
            ax[2].set_title('nodes')
            ax[2].set_xlabel('$t$')
            ax[2].set_ylabel('$x$')
            plt.savefig(path + f'/{num}_loss.png', dpi=300)
            plt.close()

    def test_err_plot(self, net, path, num):
        data = np.load("./data/Burgers1D.npz")
        t, x, exact = data["t"], data["x"], data["usol"].T
        X, Y = np.meshgrid(t, x, indexing='ij')
        node = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
        node = torch.from_numpy(node).to(device=self.dev)
        val = net(node).detach().cpu().numpy().flatten()
        sol = exact.reshape(-1, 1).flatten()
        err = np.sqrt(np.sum(np.power(val - sol, 2)) / np.sum(np.power(sol, 2)))
        err_plt = np.abs(val - sol)
        fig, axes = plt.subplots(2, 3, layout='constrained', figsize=(19.2, 9.6))
        # plot absolute error
        plot = axes[0, 0].pcolormesh(X, Y, err_plt.reshape(X.shape), shading='gouraud',
                                     cmap='jet', vmin=0, vmax=np.max(err_plt))
        fig.colorbar(plot, ax=axes[0, 0], format="%1.1e")
        axes[0, 0].set_title(f'$e_r(u^\\theta_{{{num}}})$={round(err, 4)}')
        axes[0, 0].set_xlabel('$t$')
        axes[0, 0].set_ylabel('$x$')
        # plot predict
        plot = axes[0, 1].pcolormesh(X, Y, val.reshape(X.shape), shading='gouraud',
                                     cmap='jet', vmin=-1.05, vmax=1.05)
        fig.colorbar(plot, ax=axes[0, 1], format="%1.1e")
        axes[0, 1].set_title(f'$u^\\theta_{{{num}}}$')
        axes[0, 1].set_xlabel('$t$')
        axes[0, 1].set_ylabel('$x$')
        # plot exact
        plot = axes[0, 2].pcolormesh(X, Y, exact, shading='gouraud',
                                     cmap='jet', vmin=-1.05, vmax=1.05)
        fig.colorbar(plot, ax=axes[0, 2], format="%1.1e")
        axes[0, 2].set_title(f'$u^*$')
        axes[0, 2].set_xlabel('$t$')
        axes[0, 2].set_ylabel('$x$')
        # plot at t
        t_plt = [0.10, 0.50, 0.90]
        for count, t_now in enumerate(t_plt):
            ind = np.where(t == t_now)[0]
            sol_t = exact[ind, :]
            node = np.concatenate((np.ones_like(x) * t[ind], x), axis=1)
            node = torch.from_numpy(node).to(device=self.dev)
            val_t = net(node).detach().cpu().numpy().flatten()
            axes[1, count].plot(x, sol_t.flatten(), 'b--', label=f'$u^*({t_now}, x)$')
            axes[1, count].plot(x, val_t, 'r', label=f'$u^\\theta_{{{num}}}({t_now}, x)$')
            err = np.sqrt(np.sum(np.power(val_t - sol_t, 2)) / np.sum(np.power(sol_t, 2)))
            axes[1, count].set_title(f'$e_t(u^\\theta_{{{num}}},{t_now})={round(err, 4)}$')
            axes[1, count].legend(loc='upper right')
            axes[1, count].set_xlabel('$x$')
            count += 1
        plt.savefig(path + f'/{num}_sol.png')
        plt.close()


class Burgers3D:

    def __init__(self, dev, dtp, weight, Renold,
                 xlim, tlim, ylim, num_in, num_bd, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.tlim, self.ylim, self.input_size, self.output_size = 3, dev, dtp, weight, xlim, tlim, ylim, input_size, output_size
        self.criterion = torch.nn.MSELoss()
        self.physics = ['in', 'bd', 'init']
        self.size = {'in': num_in, 'bd': num_bd, 'init': num_bd}
        self.viscosity = float(1 / Renold)

    def sample(self, size, mode):
        ts, te, xs, xe, ys, ye = self.tlim[0], self.tlim[1], self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        t_len, x_len, y_len = te-ts, xe - xs, ye - ys
        if mode == 'in':
            size_in = size
            node_in = torch.cat(
                (
                    torch.rand([size_in, 1]) * t_len + torch.ones(size=[size_in, 1]) * ts,
                    torch.rand([size_in, 1]) * x_len + torch.ones(size=[size_in, 1]) * xs,
                    torch.rand([size_in, 1]) * y_len + torch.ones(size=[size_in, 1]) * ys,
                ), dim=1)
            return node_in.to(device=self.dev, dtype=self.dtp)
        if mode == 'bd':
            size_bd = size
            bd_num = torch.randint(low=0, high=4, size=(size_bd,))
            node_bd = list(range(4))
            for i in range(4):
                ind = bd_num[bd_num == i]
                num = bd_num[ind].shape[0]
                if i == 0:
                    node_bd[i] = torch.cat([
                        torch.rand([num, 1]) * t_len + torch.ones([num, 1]) * ts,
                        torch.rand([num, 1]) * x_len + torch.ones([num, 1]) * xs,
                        torch.ones([num, 1]) * ys], dim=1)
                elif i == 1:
                    node_bd[i] = torch.cat([
                        torch.rand([num, 1]) * t_len + torch.ones([num, 1]) * ts,
                        torch.ones([num, 1]) * xs,
                        torch.rand([num, 1]) * y_len + torch.ones([num, 1]) * ys], dim=1)
                elif i == 2:
                    node_bd[i] = torch.cat([
                        torch.rand([num, 1]) * t_len + torch.ones([num, 1]) * ts,
                        torch.rand([num, 1]) * x_len + torch.ones([num, 1]) * xs,
                        torch.ones([num, 1]) * ye], dim=1)
                else:
                    node_bd[i] = torch.cat([
                        torch.rand([num, 1]) * t_len + torch.ones([num, 1]) * ts,
                        torch.ones([num, 1]) * xe,
                        torch.rand([num, 1]) * y_len + torch.ones([num, 1]) * ys], dim=1)
            return torch.cat(node_bd, dim=0).to(device=self.dev, dtype=self.dtp)
        if mode == 'init':
            size_init = size
            node_init = torch.cat(
                [
                    torch.ones([size_init, 1]) * ts,
                    torch.rand([size_init, 1]) * x_len + torch.ones([size_init, 1]) * xs,
                    torch.rand([size_init, 1]) * y_len + torch.ones([size_init, 1]) * ys
                ], dim=1)
            return node_init.to(device=self.dev, dtype=self.dtp)

    def solve(self, mode, node):
        if mode == "in":
            node_in = node
            val_in = torch.zeros_like(node_in[:, 0])

            return torch.stack([val_in]*2, dim=1)
        elif mode == "bd":
            u = 3 / 4 - 1 / (4 * (1 + torch.exp((-4 * node[:, 1] + 4 * node[:, 2]-node[:, 0]) / (32 * self.viscosity))))
            v = 3 / 4 + 1 / (4 * (1 + torch.exp((-4 * node[:, 1] + 4 * node[:, 2]-node[:, 0]) / (32 * self.viscosity))))
            return torch.stack([u, v], dim=1)
        elif mode == "init":
            u_init = 3/4 - 1/(4*(1+torch.exp((-4*node[:, 1]+4*node[:, 2])/(32*self.viscosity))))
            v_init = 3/4 + 1/(4*(1+torch.exp((-4*node[:, 1]+4*node[:, 2])/(32*self.viscosity))))
            return torch.stack([u_init, v_init], dim=1)
        else:
            raise ValueError("invalid mode")

    def exact(self, node):
        u = 3 / 4 - 1 / (4 * (1 + np.exp((-4 * node[:, 1] + 4 * node[:, 2]-node[:, 0]) / (32 * self.viscosity))))
        v = 3 / 4 + 1 / (4 * (1 + np.exp((-4 * node[:, 1] + 4 * node[:, 2]-node[:, 0]) / (32 * self.viscosity))))
        return np.stack([u, v], axis=1)

    def residual(self, node, net, cls="loss", mode="in"):
        pred = self.solve(mode, node)
        if mode == "in":
            x = node
            x.requires_grad = True
            val = net(x)
            u = val[:, 0]
            v = val[:, 1]
            ud = torch.autograd.grad(outputs=u,
                                     inputs=x,
                                     grad_outputs=torch.ones_like(u),
                                     retain_graph=True,
                                     create_graph=True)[0]
            udt = ud[:, 0].reshape(-1, 1)
            udx = ud[:, 1].reshape(-1, 1)
            udy = ud[:, 2].reshape(-1, 1)
            udxx = torch.autograd.grad(inputs=x,
                                       outputs=udx,
                                       grad_outputs=torch.ones_like(udx),
                                       retain_graph=True,
                                       create_graph=True)[0][:, 1].flatten()
            udyy = torch.autograd.grad(inputs=x,
                                       outputs=udy,
                                       grad_outputs=torch.ones_like(udy),
                                       retain_graph=True,
                                       create_graph=True)[0][:, 2].flatten()
            vd = torch.autograd.grad(outputs=v,
                                     inputs=x,
                                     grad_outputs=torch.ones_like(v),
                                     retain_graph=True,
                                     create_graph=True)[0]
            vdt = vd[:, 0].reshape(-1, 1)
            vdx = vd[:, 1].reshape(-1, 1)
            vdy = vd[:, 2].reshape(-1, 1)
            vdxx = torch.autograd.grad(inputs=x,
                                       outputs=vdx,
                                       grad_outputs=torch.ones_like(vdx),
                                       retain_graph=True,
                                       create_graph=True)[0][:, 1].flatten()
            vdyy = torch.autograd.grad(inputs=x,
                                       outputs=vdy,
                                       grad_outputs=torch.ones_like(vdy),
                                       retain_graph=True,
                                       create_graph=True)[0][:, 2].flatten()
            if cls == "loss":
                pde_res = (self.criterion(udt.flatten()+u.flatten()*udx.flatten()+v.flatten()*udy.flatten()-self.viscosity*(udxx + udyy), pred[:, 0])
                           + self.criterion(vdt.flatten()+u.flatten()*vdx.flatten()+v.flatten()*vdy.flatten()-self.viscosity*(vdxx + vdyy), pred[:, 1]))
            elif cls == "ele":
                pde_res = (udt.flatten()+u.flatten()*udx.flatten()+v.flatten()*udy.flatten()-self.viscosity*(udxx + udyy)-pred[:, 0]
                           + vdt.flatten()+u.flatten()*vdx.flatten()+v.flatten()*vdy.flatten()-self.viscosity*(vdxx + vdyy)-pred[:, 1])
            else:
                raise ValueError("Invalid cls")
            return pde_res
        elif mode == "bd":
            node_bd = node
            bd_res = self.criterion(net(node_bd)[:, 0], pred[:, 0])+self.criterion(net(node_bd)[:, 1], pred[:, 1])
            return bd_res
        elif mode == "init":
            node_init = node
            init_res = self.criterion(net(node_init)[:, 0], pred[:, 0])+self.criterion(net(node_init)[:, 1], pred[:, 1])
            return init_res
        else:
            raise ValueError("Invalid mode")

    def grid(self, size):
        ts, te, xs, xe, ys, ye = self.tlim[0], self.tlim[1], self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        inter_t = np.linspace(start=ts, stop=te, num=size + 1)
        inter_x = np.linspace(start=xs, stop=xe, num=size + 1)
        inter_y = np.linspace(start=ys, stop=ye, num=size + 1)
        mesh_t, mesh_x, mesh_y = np.meshgrid(inter_t, inter_x, inter_y)
        return mesh_t, mesh_x, mesh_y

    def is_node_in(self, node):
        return (
                (self.tlim[0] < node[:, 0]) & (node[:, 0] < self.tlim[1])
                & (self.xlim[0] < node[:, 1]) & (node[:, 1] < self.xlim[1])
                & (self.ylim[0] < node[:, 2]) & (node[:, 2] < self.ylim[1]))

    def test_err(self, net):
        mesh_t, mesh_x, mesh_y = self.grid(40)
        node = np.stack((mesh_t.flatten(), mesh_x.flatten(), mesh_y.flatten()), axis=1)
        sol = self.exact(node)
        val = net(torch.from_numpy(node).to(device=self.dev)).detach().cpu().numpy()
        err = np.mean(np.sqrt(np.sum(np.power(val - sol, 2), axis=0) / np.sum(np.power(sol, 2), axis=0)))
        return err

    def target_node_plot_together(self, loss, node_add, node_domain, IS_sign, proposal, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['bd'].detach(),
                              node_domain['init'].detach()])
        node_add = node_add.detach().cpu().numpy()
        node_all = node_all.cpu().numpy()
        ts, te, xs, xe, ys, ye = self.tlim[0], self.tlim[1], self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        mesh_t, mesh_x, mesh_y = self.grid(size=40)
        node = np.stack([mesh_t.flatten(), mesh_x.flatten(), mesh_y.flatten()], axis=1)
        val = loss(node).flatten()
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.35 / 1.1, y=-1.95 / 1.1, z=1.45 / 1.1)
        )
        # plot loss
        self.plot_vol(node, val, None, path + f'/{num}_loss.html')
        # plot node
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(name='before', x=node_all[:, 1], y=node_all[:, 2], z=node_all[:, 0],
                                   mode='markers',
                                   marker=dict(
                                       size=1,
                                       opacity=0.2,
                                       color='blue'
                                   )))
        fig.add_trace(go.Scatter3d(name='now', x=node_add[:, 1], y=node_add[:, 2], z=node_add[:, 0],
                                   mode='markers',
                                   marker=dict(
                                       size=1,
                                       opacity=1.0,
                                       color='red'
                                   )))
        fig.update_layout(scene=dict(
            xaxis=dict(range=[xs, xe], title='x'),
            yaxis=dict(range=[ys, ye], title='y'),
            zaxis=dict(range=[ts, te], title='t'),
        ),
            scene_camera=camera,
            width=640,
            height=480,
            margin=dict(l=20, r=20, b=50, t=20))
        fig.write_html(path + f'/{num}_node.html')
        if IS_sign == 1:
            # plot proposal
            val = proposal(node).flatten()
            self.plot_vol(node, val, None, path + f'/{num}_proposal.html')

    def test_err_plot(self, net, path, num):
        mesh_t, mesh_x, mesh_y = self.grid(30)
        node = np.stack((mesh_t.flatten(), mesh_x.flatten(), mesh_y.flatten()), axis=1)
        sol = self.exact(node)
        val = net(torch.from_numpy(node).to(device=self.dev)).detach().cpu().numpy()
        err_u = np.sqrt(np.sum(np.power(val[:, 0] - sol[:, 0], 2), axis=0) / np.sum(np.power(sol[:, 0], 2), axis=0))
        err_v = np.sqrt(np.sum(np.power(val[:, 1] - sol[:, 1], 2), axis=0) / np.sum(np.power(sol[:, 1], 2), axis=0))
        err_u_plt = val[:, 0] - sol[:, 0]
        err_v_plt = val[:, 1] - sol[:, 1]
        self.plot_vol(node, err_u_plt.flatten(), f'RE of u = {round(err_u, 4)}', path + f'/{num}_err_u.html')
        self.plot_vol(node, err_v_plt.flatten(), f'RE of v = {round(err_v, 4)}', path + f'/{num}_err_v.html')
        self.plot_vol(node, val[:, 0], None, path + f'/{num}_u.html')
        self.plot_vol(node, val[:, 1], None, path + f'/{num}_v.html')
        # plot exact
        if num == 0:
            self.plot_vol(node, sol.flatten(), None, path + f'/exact.html')
        # plot at t
        fig_u, axes_u = plt.subplots(3, 3, layout='constrained', figsize=(12, 12))
        fig_u.suptitle(f'RE of u = {round(err_u, 4)}')
        fig_v, axes_v = plt.subplots(3, 3, layout='constrained', figsize=(12, 12))
        fig_v.suptitle(f'RE of v = {round(err_v, 4)}')
        t_plt = [0.10, 0.50, 0.90]
        mesh_x, mesh_y = np.meshgrid(
            np.linspace(self.xlim[0], self.xlim[1], 256),
            np.linspace(self.ylim[0], self.ylim[1], 256))
        for enum, t_now in enumerate(t_plt):
            node = np.stack((np.ones_like(mesh_x.flatten()) * t_now, mesh_x.flatten(), mesh_y.flatten()), axis=1)
            node = torch.from_numpy(node).to(device=self.dev)
            val_t = net(node).detach().cpu().numpy()
            sol_t = self.exact(node)
            err_u_t = np.sqrt(np.sum(np.power(val_t[:, 0]-sol_t[:, 0], 2))/np.sum(np.power(sol_t[:, 0], 2)))
            err_v_t = np.sqrt(np.sum(np.power(val_t[:, 1] - sol_t[:, 1], 2)) / np.sum(np.power(sol_t[:, 1], 2)))
            # plot u
            plot = axes_u[enum, 0].pcolormesh(mesh_x, mesh_y, val_t[:, 0].reshape(mesh_x.shape), shading='gouraud', cmap='jet')
            axes_u[enum, 0].set_title(f'predict of u at t = {t_now}')
            fig_u.colorbar(plot, ax=axes_u[enum, 0], format="%1.1e")
            plot = axes_u[enum, 1].pcolormesh(mesh_x, mesh_y, sol_t[:, 0].reshape(mesh_x.shape), shading='gouraud', cmap='jet')
            axes_u[enum, 1].set_title(f'exact u')
            fig_u.colorbar(plot, ax=axes_u[enum, 1], format="%1.1e")
            plot = axes_u[enum, 2].pcolormesh(mesh_x, mesh_y, (val_t[:, 0]-sol_t[:, 0]).reshape(mesh_x.shape), shading='gouraud', cmap='jet')
            axes_u[enum, 2].set_title(f'RET of u = {round(err_u_t, 4)}')
            fig_u.colorbar(plot, ax=axes_u[enum, 2], format="%1.1e")
            # plot v
            plot = axes_v[enum, 0].pcolormesh(mesh_x, mesh_y, val_t[:, 1].reshape(mesh_x.shape), shading='gouraud', cmap='jet')
            axes_v[enum, 0].set_title(f'predict of v at t = {t_now}')
            fig_u.colorbar(plot, ax=axes_v[enum, 0], format="%1.1e")
            plot = axes_v[enum, 1].pcolormesh(mesh_x, mesh_y, sol_t[:, 1].reshape(mesh_x.shape), shading='gouraud', cmap='jet')
            axes_v[enum, 1].set_title(f'exact v')
            fig_u.colorbar(plot, ax=axes_v[enum, 1], format="%1.1e")
            plot = axes_v[enum, 2].pcolormesh(mesh_x, mesh_y, (val_t[:, 1] - sol_t[:, 1]).reshape(mesh_x.shape), shading='gouraud', cmap='jet')
            axes_v[enum, 2].set_title(f'RET of v = {round(err_v_t, 4)}')
            fig_u.colorbar(plot, ax=axes_v[enum, 2], format="%1.1e")
        fig_u.tight_layout()
        fig_u.savefig(path+f'/{num}_u_slice.png')
        plt.close(fig_u)
        fig_v.tight_layout()
        fig_v.savefig(path + f'/{num}_u_slice.png')
        plt.close(fig_v)

    @staticmethod
    def plot_vol(node, val, title, fname):
        fig = go.Figure(data=go.Volume(
            x=node[:, 1],
            y=node[:, 2],
            z=node[:, 0],
            value=val,
            isomin=np.min(val),
            isomax=np.max(val),
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=21,
            colorscale='jet'# needs to be a large number for good volume rendering
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
                zaxis=dict(title='t'),
            ),
            scene_camera=camera,
            width=640,
            height=480,
            margin=dict(l=20, r=20, b=50, t=20))
        fig.update_traces(colorbar=dict(tickformat='.1e'))
        if 'html' in fname:
            fig.write_html(fname)
        else:
            fig.write_image(fname)
