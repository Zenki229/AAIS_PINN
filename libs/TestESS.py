from .utils import *
import scipy.stats as ss


class Poisson2D9Peaks:
    def __init__(self, dev, dtp, weight, xlim, ylim, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.ylim, self.input_size, self.output_size = input_size, dev, dtp, weight, xlim, ylim, input_size, output_size
        grid = np.array([-0.5, 0, 0.5])
        x, y = np.meshgrid(grid, grid)
        self.center = np.stack([x.flatten(), y.flatten()], axis=1)

    def sample(self, size):
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        x_len, y_len = xe - xs, ye-ys
        node_in = torch.cat(
                (torch.rand([size, 1]) * x_len + torch.ones(size=[size, 1]) * xs,
                 torch.rand([size, 1]) * y_len + torch.ones(size=[size, 1]) * ys), dim=1)
        return node_in.numpy()

    def solve(self, node):
        val_in = np.zeros_like(node[:, 0])
        for i in range(self.center.shape[0]):
            val_in += np.exp(-1000 * (
                                (node[:, 0] - self.center[i, 0]) ** 2 + (node[:, 1] - self.center[i, 1]) ** 2))
        return val_in

    def grid(self, size):
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        inter_x = np.linspace(start=xs, stop=xe, num=size + 2)
        inter_y = np.linspace(start=ys, stop=ye, num=size + 2)
        mesh_x, mesh_y = np.meshgrid(inter_x, inter_y)
        return mesh_x, mesh_y

    def is_node_in(self, node):
        return ((self.xlim[0] < node[:, 0]) & (node[:, 0] < self.xlim[1])
                & (self.ylim[0] < node[:, 1]) & (node[:, 1] < self.ylim[1]))

    def exact_int(self):
        return 0.05605*9

    def ess_plot(self, node_search, node_add, loss, proposal, path, num):
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        val = loss(node).reshape(mesh_x.shape)
        integration = np.mean(loss(node_add))
        exact_int = self.exact_int()
        err = np.abs(exact_int-integration)/np.abs(exact_int)
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
        ax.scatter(node_search[:, 0], node_search[:, 1], c='b', marker='.', s=np.ones_like(node_search  [:, 0]), alpha=0.3, label=f'$\\mathcal{{S}}_{{{num}}}$')
        ax.scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.', s=np.ones_like(node_add[:, 0]), alpha=1.0, label=f'$\\mathcal{{D}}$')
        ax.set_title(f'$Err={round(err, 4)}$')
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


class Poisson2D1Peak:
    def __init__(self, dev, dtp, weight, xlim, ylim, input_size, output_size):
        self.dim, self.dev, self.dtp, self.weight, self.xlim, self.ylim, self.input_size, self.output_size = input_size, dev, dtp, weight, xlim, ylim, input_size, output_size
        # grid = np.array([-0.5, 0, 0.5])
        # x, y = np.meshgrid(grid, grid)
        self.center = np.array([[0.5, 0.5]])

    def sample(self, size):
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        x_len, y_len = xe - xs, ye-ys
        node_in = torch.cat(
                (torch.rand([size, 1]) * x_len + torch.ones(size=[size, 1]) * xs,
                 torch.rand([size, 1]) * y_len + torch.ones(size=[size, 1]) * ys), dim=1)
        return node_in.numpy()

    def solve(self, node):
        val_in = np.zeros_like(node[:, 0])
        for i in range(self.center.shape[0]):
            val_in += np.exp(-1000 * (
                                (node[:, 0] - self.center[i, 0]) ** 2 + (node[:, 1] - self.center[i, 1]) ** 2))
        return val_in

    def grid(self, size):
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        inter_x = np.linspace(start=xs, stop=xe, num=size + 2)
        inter_y = np.linspace(start=ys, stop=ye, num=size + 2)
        mesh_x, mesh_y = np.meshgrid(inter_x, inter_y)
        return mesh_x, mesh_y

    def is_node_in(self, node):
        return ((self.xlim[0] < node[:, 0]) & (node[:, 0] < self.xlim[1])
                & (self.ylim[0] < node[:, 1]) & (node[:, 1] < self.ylim[1]))

    def exact_int(self):
        return 0.05605*9

    def ess_plot(self, node_search, node_add, loss, proposal, path, num):
        xs, xe, ys, ye = self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]
        mesh_x, mesh_y = self.grid(size=256)
        node = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=1)
        val = loss(node).reshape(mesh_x.shape)
        # integration = np.mean(loss(node_add))
        # exact_int = self.exact_int()
        # err = np.abs(exact_int-integration)/np.abs(exact_int)
        answer = np.var(node_add)
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
        ax.scatter(node_search[:, 0], node_search[:, 1], c='b', marker='.', s=np.ones_like(node_search  [:, 0]), alpha=0.3, label=f'$\\mathcal{{S}}_{{{num}}}$')
        ax.scatter(node_add[:, 0], node_add[:, 1], c='r', marker='.', s=np.ones_like(node_add[:, 0]), alpha=1.0, label=f'$\\mathcal{{D}}$')
        ax.set_title(f'$Err={round(answer, 4)}$')
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
