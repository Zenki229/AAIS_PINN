import torch

from .utils import *


class NSCylinder:

    def __init__(self, dev, dtp, weight, viscosity, density, center, radius,
                 xlim, tlim, ylim, num_in, num_bd, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.tlim, self.ylim, self.input_size, self.output_size \
            = 3, dev, dtp, weight, xlim, tlim, ylim, input_size, output_size
        self.viscosity = viscosity
        self.density = density
        self.criterion = torch.nn.MSELoss()
        self.physics = ['in', 'wall', 'inflow', 'outflow', 'init']
        self.size = {'in': num_in}
        len_wall_1 = len_wall_2 = xlim[1] - xlim[0]
        len_inflow_3 = len_outflow_4 = ylim[1] - ylim[0]
        len_circ_5 = np.pi * 2 * radius
        len_sum = len_wall_1 * 2 + len_inflow_3 * 2 + len_circ_5
        self.prob_sample = np.array([len_wall_1, len_wall_2, len_circ_5]) / (len_wall_1 + len_wall_2 + len_circ_5)
        num_bd_ratio = np.array([len_wall_1, len_wall_2, len_inflow_3, len_outflow_4, len_circ_5]) / len_sum
        self.size['wall'] = int(np.floor(num_bd * (num_bd_ratio[0]+num_bd_ratio[1]+num_bd_ratio[4])))
        self.size['inflow'] = int(np.floor(num_bd * num_bd_ratio[2]))
        self.size['outflow'] = num_bd-self.size['wall']-self.size['inflow']
        self.size['init'] = num_bd
        self.center = center
        self.radius = radius

    def is_node_circ(self, node):
        return (node[:, 1]-self.center[0])**2+(node[:, 2]-self.center[1])**2 <= self.radius*self.radius

    def is_node_in(self, node):
        return (
                (self.tlim[0] < node[:, 0]) & (node[:, 0] < self.tlim[1])
                & (self.xlim[0] < node[:, 1]) & (node[:, 1] < self.xlim[1])
                & (self.ylim[0] < node[:, 2]) & (node[:, 2] < self.ylim[1])
                & (~ self.is_node_circ(node))
        )

    def sample(self, size, mode):
        xs, xe, ts, te, ys, ye = self.xlim[0], self.xlim[1], self.tlim[0], self.tlim[1], self.ylim[0], self.ylim[1]
        x_len = xe - xs
        t_len = te - ts
        y_len = ye - ys
        if mode == 'in':
            size_in = size
            node_in = torch.cat(
                (
                    torch.rand([size_in, 1]) * t_len + torch.ones(size=[size_in, 1]) * ts,
                    torch.rand([size_in, 1]) * x_len + torch.ones(size=[size_in, 1]) * xs,
                    torch.rand([size_in, 1]) * y_len + torch.ones(size=[size_in, 1]) * ys,
                ), dim=1)
            res = node_in[self.is_node_circ(node_in)].shape[0]
            while res >= 1:
                node_in = torch.cat((node_in[~ self.is_node_circ(node_in)],
                                     torch.cat(
                                         (
                                             torch.rand([res, 1]) * t_len + torch.ones(size=[res, 1]) * ts,
                                             torch.rand([res, 1]) * x_len + torch.ones(size=[res, 1]) * xs,
                                             torch.rand([res, 1]) * y_len + torch.ones(size=[res, 1]) * ys,
                                         ), dim=1)
                                     ), dim=0)
                res = node_in[self.is_node_circ(node_in)].shape[0]
            return node_in.to(device=self.dev, dtype=self.dtp)
        if mode == 'wall':
            size_bd = size
            bd_num = np.random.choice(3, size_bd, p=self.prob_sample)
            node_bd = list(range(3))
            for i in range(3):
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
                        torch.rand([num, 1]) * x_len + torch.ones([num, 1]) * xs,
                        torch.ones([num, 1]) * ye], dim=1)
                else:
                    theta = torch.rand(num) * 2 * math.pi
                    node_bd[i] = torch.stack([torch.rand(num) * t_len + torch.ones(num) * ts,
                                              self.radius*torch.cos(theta)+self.center[0], self.radius*torch.sin(theta)
                                              + self.center[1]], dim=1)
            return torch.cat(node_bd, dim=0).to(device=self.dev, dtype=self.dtp)
        if mode == 'inflow':
            node = torch.cat([
                torch.rand([size, 1]) * t_len + torch.ones([size, 1]) * ts,
                torch.ones([size, 1]) * xs,
                torch.rand([size, 1]) * y_len + torch.ones([size, 1]) * ys], dim=1)
            return node.to(device=self.dev, dtype=self.dtp)
        if mode == 'outflow':
            node = torch.cat([
                torch.rand([size, 1]) * t_len + torch.ones([size, 1]) * ts,
                torch.ones([size, 1]) * xe,
                torch.rand([size, 1]) * y_len + torch.ones([size, 1]) * ys], dim=1)
            return node.to(device=self.dev, dtype=self.dtp)
        if mode == 'init':
            size_init = size
            node_init = torch.cat(
                [
                    torch.ones([size_init, 1]) * ts,
                    torch.rand([size_init, 1]) * x_len + torch.ones([size_init, 1]) * xs,
                    torch.rand([size_init, 1]) * y_len + torch.ones([size_init, 1]) * ys
                ], dim=1)
            res = node_init[self.is_node_circ(node_init)].shape[0]
            while res >= 1:
                node_init = torch.cat(
                    [
                        node_init[~ self.is_node_circ(node_init)],
                        torch.cat(
                            [
                                torch.ones([res, 1]) * ts,
                                torch.rand([res, 1]) * x_len + torch.ones([res, 1]) * xs,
                                torch.rand([res, 1]) * y_len + torch.ones([res, 1]) * ys
                            ], dim=1)
                    ]
                )
                res = node_init[self.is_node_circ(node_init)].shape[0]
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
            return torch.stack([val_in]*3, dim=1).to(device=self.dev)
        elif mode == "wall":
            u = torch.zeros_like(node[:, 0])
            v = torch.zeros_like(node[:, 0])
            return torch.stack([u, v], dim=1).to(device=self.dev)
        elif mode == "inflow":
            v = torch.zeros_like(node[:, 0])
            u = (4 * 1.5 * torch.sin(torch.pi*node[:, 0]/8)*node[:, 2]*(0.41-node[:, 2]))/(0.41*0.41)
            return torch.stack([u, v], dim=1).to(device=self.dev)
        elif mode == "outflow":
            return torch.zeros_like(node[:, 0]).to(device=self.dev)
        elif mode == "init":
            u = torch.zeros_like(node[:, 0])
            v = torch.zeros_like(node[:, 0])
            return torch.stack([u, v], dim=1).to(device=self.dev)
        else:
            raise ValueError("invalid mode")

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
            u = val[:, 0]
            v = val[:, 1]
            p = val[:, 2]
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
            dp = torch.autograd.grad(outputs=p,
                                     inputs=x,
                                     grad_outputs=torch.ones_like(p),
                                     retain_graph=True,
                                     create_graph=True)[0]
            pdx = dp[:, 1].reshape(-1, 1)
            pdy = dp[:, 2].reshape(-1, 1)
            if cls == "loss":
                pde_res = (self.criterion(udt.flatten()+u.flatten()*udx.flatten()+v.flatten()*udy.flatten()
                                          - self.viscosity*(udxx + udyy)+pdx.flatten(), pred[:, 0])
                           + self.criterion(vdt.flatten()+u.flatten()*vdx.flatten()+v.flatten()*vdy.flatten()
                                            - self.viscosity*(vdxx + vdyy)+pdy.flatten(), pred[:, 1])
                           + self.criterion(udx.flatten()+vdy.flatten(), pred[:, 2]))
            elif cls == "ele":
                pde_res = (udt.flatten()+u.flatten()*udx.flatten()+v.flatten()*udy.flatten()
                           - self.viscosity*(udxx + udyy)-pred[:, 0]
                           + vdt.flatten()+u.flatten()*vdx.flatten()+v.flatten()*vdy.flatten()
                           - self.viscosity*(vdxx + vdyy)-pred[:, 1]
                           + udx.flatten()+vdy.flatten()-pred[:, 2])
            else:
                raise ValueError("Invalid mode")
            return pde_res
        elif mode == "wall":
            node_bd = node
            bd_res = self.criterion(net(node_bd)[:, 0], pred[:, 0])+self.criterion(net(node_bd)[:, 1], pred[:, 1])
            return bd_res
        elif mode == "inflow":
            node_bd = node
            bd_res = self.criterion(net(node_bd)[:, 0], pred[:, 0])+self.criterion(net(node_bd)[:, 1], pred[:, 1])
            return bd_res
        elif mode == "outflow":
            node_bd = node
            bd_res = self.criterion(net(node_bd)[:, 2], pred)
            return bd_res
        elif mode == "init":
            node_init = node
            init_res = self.criterion(net(node_init)[:, 0], pred[:, 0])+self.criterion(net(node_init)[:, 1], pred[:, 1])
            return init_res
        else:
            raise ValueError("Invalid mode")

    def grid(self, size):
        xs, xe, ts, te, ys, ye = self.xlim[0], self.xlim[1], self.tlim[0], self.tlim[1], self.ylim[0], self.ylim[1]
        inter_t = np.linspace(start=ts, stop=te, num=size + 2)
        inter_x = np.linspace(start=xs, stop=xe, num=size + 2)
        inter_y = np.linspace(start=ys, stop=ye, num=size + 2)
        mesh_t, mesh_x, mesh_y = np.meshgrid(inter_t, inter_x, inter_y)
        return mesh_t, mesh_x, mesh_y

    def test_err(self, net):
        data_dict = np.load("./data/NSClinder.npz")
        data = data_dict['data']
        ind = np.where(~np.isnan(data[:, 3]))[0]
        node = data[ind, 0:3]
        node = torch.from_numpy(node).to(device=self.dev)
        val = net(node).detach().cpu().numpy()
        sol = data[ind, 3:]
        err = np.sqrt(np.sum(np.power(val - sol, 2), axis=0) / np.sum(np.power(sol, 2), axis=0))
        return np.mean(err.flatten())

    def target_node_plot_together(self, loss, node_add, node_domain, IS_sign, proposal, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['wall'].detach(),
                              node_domain['inflow'].detach(),
                              node_domain['outflow'].detach(),
                              node_domain['init'].detach()])
        node_add = node_add.detach().cpu().numpy()
        node_all = node_all.cpu().numpy()
        ts, te, xs, xe, ys, ye = self.tlim[0], self.tlim[1], self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        mesh_t, mesh_x, mesh_y = self.grid(size=30)
        node = np.stack([mesh_t.flatten(), mesh_x.flatten(), mesh_y.flatten()], axis=1)
        # plot loss
        val = loss(node).reshape(mesh_x.shape)
        mask = np.sqrt((mesh_x - self.center[0]) ** 2 + (mesh_y - self.center[1]) ** 2) <= self.radius
        val[mask] = np.nan
        self.plot_vol(node, val.flatten(), None, path + f'/{num}_loss.png')
        # plot node
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.55 / 1.1, y=-1.75 / 1.1, z=1.55 / 1.1)
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(name='before', x=node_all[:, 1], y=node_all[:, 2], z=node_all[:, 0],
                                   mode='markers',
                                   marker=dict(
                                       size=1,
                                       opacity=0.2,
                                       color='blue'
                                   )))
        fig.add_trace(go.Scatter3d(name='add', x=node_add[:, 1], y=node_add[:, 2], z=node_add[:, 0],
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
        fig.write_image(path + f'/{num}_node.png')
        if IS_sign == 1:
            # plot proposal
            val = proposal(node).reshape(mesh_x.shape)
            val[mask] = np.nan
            self.plot_vol(node, val.flatten(), None, path + f'/{num}_proposal.png')

    def test_err_plot(self, net, path, num):
        data_dict = np.load("./data/NSClinder.npz")
        t, mesh_x, mesh_y, data = data_dict['t'], data_dict['mesh_x'], data_dict['mesh_y'], data_dict['data']
        ind = np.where(~np.isnan(data[:, 3]))[0]
        node = data[ind, 0:3]
        node = torch.from_numpy(node).to(device=self.dev)
        val = net(node).detach().cpu().numpy()
        sol = data[ind, 3:]
        err = np.sqrt(np.sum(np.power(val - sol, 2), axis=0) / np.sum(np.power(sol, 2), axis=0))
        err = np.mean(err.flatten())
        node = data[:, 0:3].copy()
        # if num == 0:
        #     u = data[:, 3]
        #     self.plot_vol(node, u, 'exact_u', path + '/exact_u.png')
        #     v = data[:, 4]
        #     self.plot_vol(node, v, 'exact_v', path + '/exact_v.png')
        #     p = data[:, 5]
        #     self.plot_vol(node, p, 'exact_p', path + '/exact_p.png')
        #     t_plt = [4.00, 5.04, 6.00]
        #     for enum, t_now in enumerate(t_plt):
        #plot at t
        t_plt = [4.00, 5.04, 6.00]
        gridx = np.arange(start=self.xlim[0], stop=self.xlim[1]+0.01, step=0.01)
        gridy = np.arange(start=self.ylim[0], stop=self.ylim[1]+0.01, step=0.01)
        X, Y = np.meshgrid(gridx, gridy)
        r = np.sqrt((X-self.center[0])**2+(Y-self.center[1])**2)
        mask = r < self.radius
        fig, axes = plt.subplots(2, 3, figsize=(30.0, 8.0))
        for enum, t_now in enumerate(t_plt):
            node = np.stack((np.ones_like(X.flatten()) * t_now, X.flatten(), Y.flatten()), axis=1)
            node = torch.from_numpy(node).to(device=self.dev)
            val_t = net(node).detach().cpu().numpy()
            u = val_t[:, 0].reshape((X.shape[0], X.shape[1]))
            u[mask] = np.nan
            plot = axes[0, enum].pcolormesh(X, Y, u, shading='gouraud', cmap='jet')
            axes[0, enum].set_title(f'solution u at time {t_now}')
            fig.colorbar(plot, ax=axes[0, enum], format="%1.1e")
            v = val_t[:, 1].reshape((X.shape[0], X.shape[1]))
            v[mask] = np.nan
            plot = axes[1, enum].pcolormesh(X, Y, v, shading='gouraud', cmap='jet')
            axes[1, enum].set_title(f'solution v at time {t_now}')
            fig.colorbar(plot, ax=axes[1, enum], format="%1.1e")
        plt.savefig(path  + f'/{num}_uv.png')
        plt.close()

    def plot_vol(self, node, val, title, fname):
        ts = self.tlim[0]
        te = self.tlim[1]
        xs = self.xlim[0]
        xe = self.xlim[1]
        ys = self.ylim[0]
        ye = self.ylim[1]
        fig = go.Figure(data=go.Volume(
            x=node[:, 1],
            y=node[:, 2],
            z=node[:, 0],
            value=val,
            isomin=np.min(val),
            isomax=np.max(val),
            opacity=0.2,
            surface_count=21,
            colorscale='jet'
        ))
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.55 / 1.1, y=-1.75 / 1.1, z=1.55 / 1.1)
        )
        fig.update_layout(
            title=dict(text=title, x=0.5, y=0.9, xanchor='center', yanchor='top'),
            scene=dict(
                xaxis=dict(range=[xs, xe], title='x'),
                yaxis=dict(range=[ys, ye], title='y'),
                zaxis=dict(range=[ts, te], title='t'),
            ),
            scene_camera=camera,
            width=640,
            height=480,
            margin=dict(l=20, r=20, b=50, t=20))
        fig.update_traces(colorbar=dict(tickformat='.1e'))
        fig.write_image(fname)


class NSWake:

    def __init__(self, dev, dtp, weight, viscosity,
                 xlim, tlim, ylim, num_in, num_bd, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.tlim, self.ylim, self.input_size, self.output_size \
            = 3, dev, dtp, weight, xlim, tlim, ylim, input_size, output_size
        self.viscosity = viscosity
        self.criterion = torch.nn.MSELoss()
        self.physics = ['in', 'bd', 'init']
        self.size = {'in': num_in, 'bd': num_bd, 'init': num_bd}

    def is_node_in(self, node):
        return (
                (self.tlim[0] < node[:, 0]) & (node[:, 0] < self.tlim[1])
                & (self.xlim[0] < node[:, 1]) & (node[:, 1] < self.xlim[1])
                & (self.ylim[0] < node[:, 2]) & (node[:, 2] < self.ylim[1])
        )

    def sample(self, size, mode):
        xs, xe, ts, te, ys, ye = self.xlim[0], self.xlim[1], self.tlim[0], self.tlim[1], self.ylim[0], self.ylim[1]
        x_len = xe - xs
        t_len = te - ts
        y_len = ye - ys
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
            data = np.load('./data/NSwake.npz')
            node_bd = data['data_bd'][:, 0:3]
            idx = np.random.choice(node_bd.shape[0], size=size, replace=False)
            return torch.from_numpy(node_bd[idx, :]).to(device=self.dev, dtype=self.dtp)
        if mode == 'init':
            data = np.load('./data/NSwake.npz')
            node_init = data['data_init'][:, 0:3]
            idx = np.random.choice(node_init.shape[0], size=size, replace=False)
            return torch.from_numpy(node_init[idx, :]).to(device=self.dev, dtype=self.dtp)

    def solve(self, mode, node):
        """
        node is a dict{"in": node_in, "bd":node_bd}
        mode is a string that can be "rhs", "bd" or "exact"
        return a dict{"in": val_in, "bd": val_bd} or a tensor depending on the mode
        """
        if mode == "in":
            node_in = node
            val_in = torch.zeros_like(node_in[:, 0])
            return torch.stack([val_in]*3, dim=1).to(device=self.dev)
        elif mode == 'bd':
            node = node.detach().cpu().numpy()
            data = np.load('./data/NSwake.npz')
            node_bd = data['data_bd'][:, 0:3]
            idx = []
            for i in range(node.shape[0]):
                position = np.where((node_bd == node[i, :]).all(axis=1))[0]
                idx.append(position.item())
            return torch.from_numpy(data['data_bd'][idx, 3:5]).to(device=self.dev, dtype=self.dtp)
        elif mode == "init":
            node = node.detach().cpu().numpy()
            data = np.load('./data/NSwake.npz')
            node_init = data['data_init'][:, 0:3]
            idx = []
            for i in range(node.shape[0]):
                position = np.where((node_init == node[i, :]).all(axis=1))[0]
                idx.append(position.item())
            return torch.from_numpy(data['data_init'][idx, 3:5]).to(device=self.dev, dtype=self.dtp)
        else:
            raise ValueError("invalid mode")

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
            u = val[:, 0]
            v = val[:, 1]
            p = val[:, 2]
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
            dp = torch.autograd.grad(outputs=p,
                                     inputs=x,
                                     grad_outputs=torch.ones_like(p),
                                     retain_graph=True,
                                     create_graph=True)[0]
            pdx = dp[:, 1].reshape(-1, 1)
            pdy = dp[:, 2].reshape(-1, 1)
            if cls == "loss":
                pde_res = (self.criterion(udt.flatten()+u.flatten()*udx.flatten()+v.flatten()*udy.flatten()
                                          - self.viscosity*(udxx + udyy)+pdx.flatten(), pred[:, 0])
                           + self.criterion(vdt.flatten()+u.flatten()*vdx.flatten()+v.flatten()*vdy.flatten()
                                            - self.viscosity*(vdxx + vdyy)+pdy.flatten(), pred[:, 1])
                           + self.criterion(udx.flatten()+vdy.flatten(), pred[:, 2]))
            elif cls == "ele":
                pde_res = (udt.flatten()+u.flatten()*udx.flatten()+v.flatten()*udy.flatten()
                           - self.viscosity*(udxx + udyy)-pred[:, 0]
                           + vdt.flatten()+u.flatten()*vdx.flatten()+v.flatten()*vdy.flatten()
                           - self.viscosity*(vdxx + vdyy)-pred[:, 1]
                           + udx.flatten()+vdy.flatten()-pred[:, 2])
            else:
                raise ValueError("Invalid mode")
            return pde_res
        elif mode == "bd":
            node_bd = node
            bd_res = self.criterion(net(node_bd)[:, 0], pred[:, 0]) + self.criterion(net(node_bd)[:, 1], pred[:, 1])
            return bd_res
        elif mode == "init":
            node_init = node
            init_res = self.criterion(net(node_init)[:, 0], pred[:, 0]) + self.criterion(net(node_init)[:, 1],
                                                                                         pred[:, 1])
            return init_res

        else:
            raise ValueError("Invalid mode")

    def grid(self, size):
        xs, xe, ts, te, ys, ye = self.xlim[0], self.xlim[1], self.tlim[0], self.tlim[1], self.ylim[0], self.ylim[1]
        inter_t = np.linspace(start=ts, stop=te, num=size + 2)
        inter_x = np.linspace(start=xs, stop=xe, num=size + 2)
        inter_y = np.linspace(start=ys, stop=ye, num=size + 2)
        mesh_t, mesh_x, mesh_y = np.meshgrid(inter_t, inter_x, inter_y)
        return mesh_t, mesh_x, mesh_y

    def test_err(self, net):
        # data_dict = np.load("./data/NSwake.npz")
        # data_test = data_dict['data_test']
        # node = data_test[:, 0:3]
        # node = torch.from_numpy(node).to(device=self.dev)
        # val = net(node).detach().cpu().numpy()
        # sol = data_test[:, 3:]
        # err = np.sqrt(np.sum(np.power(val - sol, 2), axis=0) / np.sum(np.power(sol, 2), axis=0))
        # return np.mean(err.flatten())
        return 0
    def target_node_plot_together(self, loss, node_add, node_domain, IS_sign, proposal, path, num):
        node_all = torch.cat([node_domain['in'].detach(),
                              node_domain['bd'].detach(),
                              node_domain['init'].detach()])
        node_add = node_add.detach().cpu().numpy()
        node_all = node_all.cpu().numpy()
        ts, te, xs, xe, ys, ye = self.tlim[0], self.tlim[1], self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        mesh_t, mesh_x, mesh_y = self.grid(size=30)
        node = np.stack([mesh_t.flatten(), mesh_x.flatten(), mesh_y.flatten()], axis=1)
        # plot loss
        val = loss(node).reshape(mesh_x.shape)
        self.plot_vol(node, val.flatten(), None, path + f'/{num}_loss.png')
        # plot node
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.55 / 1.1, y=-1.75 / 1.1, z=1.55 / 1.1)
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(name='before', x=node_all[:, 1], y=node_all[:, 2], z=node_all[:, 0],
                                   mode='markers',
                                   marker=dict(
                                       size=1,
                                       opacity=0.1,
                                       color='blue'
                                   )))
        fig.add_trace(go.Scatter3d(name='add', x=node_add[:, 1], y=node_add[:, 2], z=node_add[:, 0],
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
        fig.write_image(path + f'/{num}_node.png')
        if IS_sign == 1:
            # plot proposal
            val = proposal(node).reshape(mesh_x.shape)
            self.plot_vol(node, val.flatten(), None, path + f'/{num}_proposal.png')

    def test_err_plot(self, net, path, num):
        data_dict = np.load("./data/NSwake.npz")
        data_test = data_dict['data_test']
        node = data_test[:, 0:3]
        node = torch.from_numpy(node).to(device=self.dev)
        val = net(node).detach().cpu().numpy()
        sol = data_test[:, 3:]
        err = np.sqrt(np.sum(np.power(val - sol, 2), axis=0) / np.sum(np.power(sol, 2), axis=0))
        node = data_test[:, 0:3]
        if num == 0:
            u = data_test[:, 3]
            self.plot_vol(node, u, 'exact_u', path + '/exact_u.png')
            v = data_test[:, 4]
            self.plot_vol(node, v, 'exact_v', path + '/exact_v.png')
            p = data_test[:, 5]
            self.plot_vol(node, p, 'exact_p', path + '/exact_p.png')
        # plot at t
        t_plt = [0.0, 2.3, 4.6, 7.0]
        mesh_x, mesh_y = data_dict['mesh_x'], data_dict['mesh_y']
        fig_u, axes_u = plt.subplots(4, 3, figsize=(12.8, 9.6))
        fig_u.suptitle(f'relative error of u {round(err[0], 6)}')
        fig_v, axes_v = plt.subplots(4, 3)
        fig_u.suptitle(f'relative error of u {round(err[1], 6)}')
        fig_p, axes_p = plt.subplots(4, 3)
        fig_u.suptitle(f'relative error of u {round(err[2], 6)}')
        for enum, t_now in enumerate(t_plt):
            node = np.stack((np.ones_like(mesh_x.flatten()) * t_now, mesh_x.flatten(), mesh_y.flatten()), axis=1)
            node = torch.from_numpy(node).to(device=self.dev)
            val_t = net(node).detach().cpu().numpy()
            ind = np.abs(data_test[:, 0]-t_now) < 0.05
            # plot u
            u = val_t[:, 0].reshape(mesh_x.shape)
            exact_u = data_test[ind, 3].reshape(mesh_x.shape)
            err_now = np.sqrt(np.sum(np.power(u - exact_u, 2)) / np.sum(np.power(exact_u, 2)))
            plot = axes_u[enum, 0].pcolormesh(mesh_x, mesh_y, u, shading='gouraud', cmap='jet')
            axes_u[enum, 0].set_title(f'at time {t_now}')
            fig_u.colorbar(plot, ax=axes_u[enum, 0], format="%1.1e")
            plot = axes_u[enum, 1].pcolormesh(mesh_x, mesh_y, exact_u, shading='gouraud', cmap='jet')
            axes_u[enum, 1].set_title(f'exact u')
            fig_u.colorbar(plot, ax=axes_u[enum, 1], format="%1.1e")
            plot = axes_u[enum, 2].pcolormesh(mesh_x, mesh_y, np.abs(exact_u-u), shading='gouraud', cmap='jet')
            fig_u.colorbar(plot, ax=axes_u[enum, 2], format="%1.1e")
            axes_u[enum, 2].set_title(f'temporal RE {round(err_now, 6)}')
            # plot v
            v = val_t[:, 1].reshape(mesh_x.shape)
            exact_v = data_test[ind, 4].reshape(mesh_x.shape)
            err_now = np.sqrt(np.sum(np.power(v - exact_v, 2)) / np.sum(np.power(exact_v, 2)))
            plot = axes_v[enum, 0].pcolormesh(mesh_x, mesh_y, v, shading='gouraud', cmap='jet')
            axes_v[enum, 0].set_title(f'at time {t_now}')
            fig_v.colorbar(plot, ax=axes_v[enum, 0], format="%1.1e")
            plot = axes_v[enum, 1].pcolormesh(mesh_x, mesh_y, exact_v, shading='gouraud', cmap='jet')
            axes_v[enum, 1].set_title(f'exact v')
            fig_v.colorbar(plot, ax=axes_v[enum, 1], format="%1.1e")
            plot = axes_v[enum, 2].pcolormesh(mesh_x, mesh_y, np.abs(exact_v - v), shading='gouraud', cmap='jet')
            fig_v.colorbar(plot, ax=axes_v[enum, 2], format="%1.1e")
            axes_v[enum, 2].set_title(f'temporal RE {round(err_now, 6)}')
            # plot p
            v = val_t[:, 2].reshape(mesh_x.shape)
            exact_v = data_test[ind, 5].reshape(mesh_x.shape)
            err_now = np.sqrt(np.sum(np.power(v - exact_v, 2)) / np.sum(np.power(exact_v, 2)))
            plot = axes_p[enum, 0].pcolormesh(mesh_x, mesh_y, v, shading='gouraud', cmap='jet')
            axes_p[enum, 0].set_title(f'at time {t_now}')
            fig_p.colorbar(plot, ax=axes_p[enum, 0], format="%1.1e")
            plot = axes_p[enum, 1].pcolormesh(mesh_x, mesh_y, exact_v, shading='gouraud', cmap='jet')
            axes_p[enum, 1].set_title(f'exact p')
            fig_p.colorbar(plot, ax=axes_p[enum, 1], format="%1.1e")
            plot = axes_p[enum, 2].pcolormesh(mesh_x, mesh_y, np.abs(exact_v - v), shading='gouraud', cmap='jet')
            fig_p.colorbar(plot, ax=axes_p[enum, 2], format="%1.1e")
            axes_p[enum, 2].set_title(f'temporal RE {round(err_now, 6)}')
        fig_u.tight_layout()
        fig_u.savefig(path + f'/{num}_u.png')
        plt.close(fig_u)
        fig_v.tight_layout()
        fig_v.savefig(path + f'/{num}_v.png')
        plt.close(fig_v)
        fig_p.tight_layout()
        fig_p.savefig(path + f'/{num}_p.png')
        plt.close(fig_p)

    def plot_vol(self, node, val, title, fname):
        ts = self.tlim[0]
        te = self.tlim[1]
        xs = self.xlim[0]
        xe = self.xlim[1]
        ys = self.ylim[0]
        ye = self.ylim[1]
        fig = go.Figure(data=go.Volume(
            x=node[:, 1],
            y=node[:, 2],
            z=node[:, 0],
            value=val,
            isomin=np.min(val),
            isomax=np.max(val),
            opacity=0.1,
            surface_count=21,
            colorscale='jet'
        ))
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.55 / 1.1, y=-1.75 / 1.1, z=1.55 / 1.1)
        )
        fig.update_layout(
            title=dict(text=title, x=0.5, y=0.9, xanchor='center', yanchor='top'),
            scene=dict(
                xaxis=dict(range=[xs, xe], title='x'),
                yaxis=dict(range=[ys, ye], title='y'),
                zaxis=dict(range=[ts, te], title='t'),
            ),
            scene_camera=camera,
            width=640,
            height=480,
            margin=dict(l=20, r=20, b=50, t=20))
        fig.update_traces(colorbar=dict(tickformat='.1e'))
        fig.write_image(fname)
