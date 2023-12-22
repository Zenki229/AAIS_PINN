import torch

from .utils import *


class Wave3D:

    def __init__(self,  dev, dtp, weight,
                 xlim, tlim, ylim,  num_in, num_bd, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.tlim, self.ylim, self.input_size, self.output_size \
            = 3, dev, dtp, weight, xlim, tlim, ylim, input_size, output_size
        self.criterion = torch.nn.MSELoss()
        self.physics = ['in', 'bd', 'init']
        self.size = {'in': num_in, 'bd': num_bd, 'init': num_bd}

    def sample(self, size, mode):
        xs = self.xlim[0]
        xe = self.xlim[1]
        x_len = xe - xs
        ts = self.tlim[0]
        te = self.tlim[1]
        t_len = te - ts
        ys = self.ylim[0]
        ye = self.ylim[1]
        y_len = ye-ys
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
                        torch.rand([num, 1]) * t_len + torch.ones([num, 1])*ts,
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
        """
        node is a dict{"in": node_in, "bd":node_bd}
        mode is a string that can be "rhs", "bd" or "exact"
        return a dict{"in": val_in, "bd": val_bd} or a tensor depending on the mode
        """
        if mode == "in":
            node_in = node
            val_in = torch.zeros_like(node_in[:, 0])
            return val_in
        elif mode == "bd":
            node_bd = node
            val_bd = torch.zeros_like(node_bd[:, 0])
            return val_bd
        elif mode == "init":
            PI = np.pi
            u = torch.sin(2 * PI * node[:, 1]) * torch.sin(2 * PI * node[:, 2]) * torch.cos(4*PI*node[:, 0])
            du = torch.zeros_like(u)
            return torch.stack([u, du], dim=1)

        else:
            raise ValueError("invalid mode")

    def exact(self, node):
        PI = np.pi
        #return (np.sin(2*PI*node[:, 1])*np.sin(2*PI*node[:, 2])*(np.cos(4*PI*node[:, 0])+np.cos(4*PI*node[:, 0]))
        #        + 0.6*np.sin(3*PI*node[:, 1])*np.sin(3*PI*node[:, 2])*(np.cos(6*PI*node[:, 0])+np.cos(6*PI*node[:, 0])))
        return (np.sin(2 * PI * node[:, 1]) * np.sin(2 * PI * node[:, 2]) * np.cos(4*PI*node[:, 0]))
    def residual(self, node, net, cls="loss", mode="in"):
        """
        compute the pde residual in the domain,
        mode: "loss" compute the MSELoss from pde
              "ele" elementwise residual
        cls is a string that can be "in", "bd" computing value in the domain or on the boundary
        """
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
            dy = d[:, 2].reshape(-1, 1)
            dtt = torch.autograd.grad(inputs=x,
                                      outputs=dt,
                                      grad_outputs=torch.ones_like(dt),
                                      retain_graph=True,
                                      create_graph=True)[0][:, 0].flatten()
            dxx = torch.autograd.grad(inputs=x,
                                      outputs=dx,
                                      grad_outputs=torch.ones_like(dx),
                                      retain_graph=True,
                                      create_graph=True)[0][:, 1].flatten()
            dyy = torch.autograd.grad(inputs=x,
                                      outputs=dy,
                                      grad_outputs=torch.ones_like(dy),
                                      retain_graph=True,
                                      create_graph=True)[0][:, 2].flatten()
            if cls == "loss":
                pde_res = self.criterion(dtt-2*(dxx+dyy), pred)
            elif cls == "ele":
                pde_res = dtt-2*(dxx+dyy)-pred
            else:
                raise ValueError("Invalid mode")
            return pde_res
        elif mode == "bd":
            node_bd = node
            bd_res = self.criterion(net(node_bd).flatten(), pred)
            return bd_res
        elif mode == "init":
            x = node
            x.requires_grad = True
            val = net(x)
            d = torch.autograd.grad(outputs=val,
                                    inputs=x,
                                    grad_outputs=torch.ones_like(val),
                                    retain_graph=True,
                                    create_graph=True)[0]
            dt = d[:, 0].reshape(-1, 1)
            init_res = self.criterion(val.flatten(), pred[:, 0]) + self.criterion(dt.flatten(), pred[:, 1])
            return init_res
        else:
            raise ValueError("Invalid mode")

    def grid(self, size):
        ts = self.tlim[0]
        te = self.tlim[1]
        xs = self.xlim[0]
        xe = self.xlim[1]
        ys = self.ylim[0]
        ye = self.ylim[1]
        inter_t = np.linspace(start=ts, stop=te, num=size + 2)
        inter_x = np.linspace(start=xs, stop=xe, num=size + 2)
        inter_y = np.linspace(start=ys, stop=ye, num=size + 2)
        T, X, Y = np.meshgrid(inter_t, inter_x, inter_y)
        return T, X, Y

    def is_node_in(self, node):
        return (
                (self.tlim[0] < node[:, 0]) & (node[:, 0] < self.tlim[1])
              & (self.xlim[0] < node[:, 1]) & (node[:, 1] < self.xlim[1])
              & (self.ylim[0] < node[:, 2]) & (node[:, 2] < self.ylim[1]))

    def test_err(self, net):
        T, X, Y = self.grid(30)
        node = np.stack((T.flatten(), X.flatten(), Y.flatten()), axis=1)
        sol = self.exact(node)
        val = net(torch.from_numpy(node).to(device=self.dev)).detach().cpu().numpy().flatten()
        err = np.sqrt(np.sum(np.power(val - sol, 2)) / np.sum(np.power(sol, 2)))
        return err

    def scat_node_dict(self, node,  fname):
        # node: dict in cuda
        node_all = np.empty((0, self.dim))
        for key, values in node.items():
            node_all = np.concatenate((node_all,values.detach().cpu().numpy()))
        ts = self.tlim[0]
        te = self.tlim[1]
        xs = self.tlim[0]
        xe = self.tlim[1]
        ys = self.xlim[0]
        ye = self.xlim[1]
        fig = go.Figure(data=[go.Scatter3d(x=node_all[:, 1], y=node_all[:, 2], z=node_all[:, 0],
                                           mode='markers',
                                           marker=dict(
                                               size=1,
                                               opacity=0.5,
                                               color='blue'
                                           ))])
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.35/1.1, y=-1.95/1.1, z=1.45/1.1)
        )
        fig.update_layout(scene=dict(
            xaxis=dict(range=[xs,xe], title='x'),
            yaxis=dict(range=[ys,ye], title='y'),
            zaxis=dict(range=[ts,te], title='t'),
            ),
            scene_camera=camera,
            width=640,
            height=480,
            margin=dict(l=20, r=20, b=50, t=20))
        fig.write_image(fname)

    def target_node_plot_together(self, loss, node_add, node_domain, proposal, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['bd'].detach(),
                              node_domain['init'].detach()])
        node_add = node_add.detach().cpu().numpy()
        node_all = node_all.cpu().numpy()
        ts, te, xs, xe, ys, ye = self.tlim[0], self.tlim[1], self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        mesh_t, mesh_x, mesh_y = self.grid(size=30)
        node = np.stack([mesh_t.flatten(), mesh_x.flatten(), mesh_y.flatten()], axis=1)
        val = loss(node).flatten()
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.35 / 1.1, y=-1.95 / 1.1, z=1.45 / 1.1)
        )
        # plot loss
        self.plot_vol(node, val, None, path + f'/{num}_loss.png')
        # plot node
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(name=f'$\\mathcal{{S}}_{{{num}}}$', x=node_all[:, 1], y=node_all[:, 2], z=node_all[:, 0], mode='markers', marker=dict(size=1, opacity=0.3, color='blue')))
        fig.add_trace(go.Scatter3d(name='$\\mathcal{D}$', x=node_add[:, 1], y=node_add[:, 2], z=node_add[:, 0], mode='markers', marker=dict(size=1, opacity=1.0, color='red')))
        fig.update_layout(scene=dict(
            xaxis=dict(range=[xs, xe], title='x'),
            yaxis=dict(range=[ys, ye], title='y'),
            zaxis=dict(range=[ts, te], title='t'),
        ),
            scene_camera=camera,
            width=640,
            height=480,
            margin=dict(l=20, r=20, b=50, t=20))
        fig.write_image(path+f'/{num}_node.png')
        if proposal:
            # plot proposal
            val = proposal(node).flatten()
            self.plot_vol(node, val, None, path + f'/{num}_proposal.png')

    def test_err_plot(self, net, path, num):
        mesh_t, mesh_x, mesh_y = self.grid(30)
        node = np.stack((mesh_t.flatten(), mesh_x.flatten(), mesh_y.flatten()), axis=1)
        sol = self.exact(node)
        val = net(torch.from_numpy(node).to(device=self.dev)).detach().cpu().numpy().flatten()
        err = np.sqrt(np.sum(np.power(val - sol, 2)) / np.sum(np.power(sol, 2)))
        err_plt = val-sol
        self.plot_vol(node, err_plt, f'$e_r(u_{{{num}}}(\\cdot;\\theta))={round(err, 4)}$', path+f'/{num}_abs.png')
        self.plot_vol(node, val, None, path + f'/{num}_sol.png')
        # plot exact
        if num == 1:
            self.plot_vol(node, sol.flatten(), None, path+f'/exact.png')
        # plot at t
        # fig, axes = plt.subplots(4, 3, figsize=(12.8, 12.8))
        # fig.suptitle(f'relative error {round(err, 6)}')
        # t_plt = [0.00, 0.25, 0.75, 1.00]
        # mesh_x, mesh_y = np.meshgrid(
        #     np.linspace(self.xlim[0], self.xlim[1], 100),
        #     np.linspace(self.ylim[0], self.ylim[1], 100)
        # )
        # for enum, t_now in enumerate(t_plt):
        #     node = np.stack((np.ones_like(mesh_x.flatten()) * t_now, mesh_x.flatten(), mesh_y.flatten()), axis=1)
        #     node_aux = torch.from_numpy(node).to(device=self.dev)
        #     val_t = net(node_aux).detach().cpu().numpy().flatten()
        #     exact_t = self.exact(node).flatten()
        #     err_t = np.sqrt(np.sum(np.power(val_t - exact_t, 2)) / np.sum(np.power(exact_t, 2)))
        #     err_t_plt = np.abs(val_t-exact_t)
        #     plot = axes[enum, 0].pcolormesh(mesh_x, mesh_y, val_t.reshape(mesh_x.shape), shading='gouraud', cmap='jet')
        #     axes[enum, 0].set_title(f'time {t_now}')
        #     fig.colorbar(plot, ax=axes[enum, 0], format="%1.1e")
        #     plot = axes[enum, 1].pcolormesh(mesh_x, mesh_y, exact_t.reshape(mesh_x.shape), shading='gouraud', cmap='jet')
        #     axes[enum, 1].set_title(f'exact')
        #     fig.colorbar(plot, ax=axes[enum,1], format="%1.1e")
        #     plot = axes[enum, 2].pcolormesh(mesh_x, mesh_y, err_t_plt.reshape(mesh_x.shape), shading='gouraud', cmap='jet')
        #     axes[enum, 2].set_title(f'RE {round(err_t, 6)}')
        #     fig.colorbar(plot, ax=axes[enum,2], format="%1.1e")
        # fig.tight_layout()
        # fig.savefig(path + f'/{num}_slice.png')
        # plt.close(fig)

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
                zaxis=dict(title='t'),
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
