import numpy as np
import scipy.stats as ss


class MixGaussian:
    '''
    params.shape = [N,1+dim+dim^2]
    '''

    def __init__(self, params, dim):
        self.params = params
        self.dim = dim
        self.cov_init = 0.1

    def sampling(self, num):
        params = self.params
        dim = self.dim
        weights = params[:, 0]
        mixture_idx = np.random.choice(params.shape[0], size=num, replace=True, p=weights)
        counts = np.bincount(mixture_idx)
        pnts = np.empty((num, dim))
        for i in range(params.shape[0]):
            if len(np.where(mixture_idx == i)[0]) == 0:
                continue
            w, m, c = np.split(params[i], [1, 1 + dim])
            m = np.reshape(m, (dim,))
            c = np.reshape(c, (dim, dim))
            pnts[mixture_idx == i] = ss.multivariate_normal.rvs(size=counts[i], mean=m, cov=c)
        return pnts

    def pdf(self, node):
        params = self.params
        dim = self.dim
        prob = np.zeros_like(node.shape[0])
        for i in range(params.shape[0]):
            try:
                prob = prob + params[i, 0] * ss.multivariate_normal.pdf(x=node,
                                                                        mean=params[i, 1:1 + dim].reshape((dim,)),
                                                                        cov=params[i, 1 + dim:].reshape((dim, dim)))
            except:
                cov = np.matmul(params[i, 1 + dim:].reshape((dim, dim)), np.diag(np.ones((dim,)) * np.log10(dim)))
                prob = prob + params[i, 0] * ss.multivariate_normal.pdf(x=node,
                                                                        mean=params[i, 1:1 + dim].reshape((dim,)),
                                                                        cov=cov)
                print('Warning: nonPD cov')
                self.params[i, 1 + dim:] = cov.reshape(1, -1)
        return prob

    def IS_w(self, target, node):
        val = target(node)
        w = np.zeros_like(val)
        prob = self.pdf(node)
        ind = prob >= 1e-16
        w[ind] = val[ind] / prob[ind]
        if np.sum(w) == 0:
            return w
        return w / np.sum(w)

    def ess(self, IS_w, n):
        if np.sum(IS_w) <= 0.1:
            ess = 0
        else:
            ess = 1 / (n * np.sum(IS_w ** 2))
        return ess

    def EM_alg(self, node, IS_w, del_ts):
        params = self.params
        dim = self.dim
        params_next = np.zeros_like(params)
        prob_mass = self.pdf(node)
        ind_prob_mass = prob_mass >= 1e-16
        for i in range(params.shape[0]):
            post_prob_i = np.zeros_like(prob_mass)
            mean = params[i, 1:1 + dim].reshape((dim,))
            cov = params[i, 1 + dim:].reshape((dim, dim))
            post_prob_i[ind_prob_mass] = params[i, 0] * ss.multivariate_normal.pdf(
                x=node[ind_prob_mass], mean=mean, cov=cov) / prob_mass[ind_prob_mass]
            params_next[i, 0] = np.sum(post_prob_i * IS_w)
            if params_next[i, 0] <= del_ts:
                continue
            params_next[i, 1:1 + dim] = np.sum(IS_w.reshape(-1, 1) * post_prob_i.reshape(-1, 1) * node, axis=0) / \
                                        params_next[i, 0]
            cov = np.cov(node.transpose(), ddof=0, aweights=IS_w * post_prob_i)
            eig = np.linalg.eig(cov)[0]
            eig_min = np.min(eig)
            if eig_min < 1e-10:
                cov = cov + np.diag(np.ones(dim) * (np.abs(eig_min) * 2+1e-10))
            params_next[i, 1 + dim:] = cov.reshape(1, -1)
        if params.shape[0] == 1:
          params_next[0, 0] = 1
        return params_next


    def resort(self):
        self.params = np.array(sorted(self.params, key=lambda x: x[0], reverse=True))

    # Initial guess
    def Init(self, target, node, num, ts, count_add):
        ind = np.argsort(target(node))[::-1]
        pnt = node[ind][0, :]
        mean = pnt.copy()
        cov = np.diag(np.ones(self.dim) * self.cov_init)
        p = MixGaussian(np.concatenate([np.ones(1).reshape(1, 1), mean.reshape(1, -1), cov.reshape(1, -1)], axis=1),
                          dim=self.dim)
        node_p = p.sampling(num)
        IS_w_p = p.IS_w(target, node_p)
        ess_p = p.ess(IS_w_p, num)
        count_p = 1
        while ess_p <= ts:
            node_p = p.sampling(num)
            IS_w_p = p.IS_w(target, node_p)
            p.params = p.EM_alg(node_p, IS_w_p, 0.00001)
            ess_p = p.ess(IS_w_p, num)
            count_p += 1
            if count_p >= count_add:
                break
        self.params = p.params


    # Merge
    def merge(self, num, ts):
        dim = self.dim
        for i in range(self.params.shape[0]):
            if self.params[i, 0] == 0:
                continue
            p = MixGaussian(self.params[i, :].reshape(1, -1).copy(), dim=dim)
            p.params[0, 0] = 1
            node_p = p.sampling(num)
            ess_merge = np.zeros(self.params.shape[0])
            for j in range(self.params.shape[0]):
                if self.params[j, 0] == 0:
                    continue
                aux = MixGaussian(self.params[j, :].reshape(1, -1).copy(), dim=dim)
                aux.params[0, 0] = 1
                IS_w_i = p.IS_w(aux.pdf, node_p)
                ess_merge[j] = p.ess(IS_w_i, num)
            ess_merge[i] = 0
            if any(ess_merge >= ts):
                ind_merge = np.where(ess_merge >= ts)[0]
                ess_merge_ts = ess_merge[ind_merge] * self.params[ind_merge, 0] / (
                    np.sum(self.params[ind_merge, 0]))
                ind_max = ind_merge[np.argmax(ess_merge_ts)]
                if self.params[ind_max, 0] == 0:
                    break
                weight = self.params[i, 0] + self.params[ind_max, 0]
                self.params[i, 1:] = (self.params[i, 0] * self.params[i, 1:] + self.params[ind_max, 0] * self.params[
                                                                                                         ind_max,
                                                                                                         1:]) / weight
                self.params[i, 0] = weight
                self.params[ind_max] = 0
            else:
                continue
            print(f'merge self at  {i} with {ind_max} ')

    def merge_one(self, component, ind, weight):
        self.params[:, 0] *= weight
        params_1 = self.params[ind, :].reshape(1, -1)
        params_2 = component.params
        params_2[:, 0] = 1 - weight
        params_merge = np.zeros_like(params_1)
        params_merge[:, 0] = np.sum(np.concatenate([params_1[:, 0], params_2[:, 0]], axis=0))
        params_merge[:, 1:] = np.average(np.concatenate([params_1[:, 1:], params_2[:, 1:]], axis=0), axis=0,
                                         weights=np.concatenate([params_1[:, 0], params_2[:, 0]], axis=0))
        self.params[ind, :] = params_merge

    # update
    def update_new(self, p, node_p, merge_ts, num):
        ess_merge = np.zeros(self.params.shape[0])
        for i in range(self.params.shape[0]):
            aux = MixGaussian(self.params[i, :].reshape(1, -1).copy(), dim=self.dim)
            aux.params[0, 0] = 1
            IS_w_i = p.IS_w(aux.pdf, node_p)
            ess_merge[i] = p.ess(IS_w_i, num)
        if any(ess_merge >= merge_ts):
            ind_merge = np.where(ess_merge >= merge_ts)[0]
            ess_merge_ts = ess_merge[ind_merge] * self.params[ind_merge, 0] / (np.sum(self.params[ind_merge, 0]))
            ind_max = ind_merge[np.argmax(ess_merge_ts)]
            weight = ess_merge[ind_max] / (ess_merge[ind_max] + 0.1)
            self.merge_one(p, ind=ind_max, weight=weight)
        else:
            weight = np.max(ess_merge) / (0.1 + np.max(ess_merge))
            self.add_params(p, weight=0.5)

    def add_params(self, component, weight):
        # weight = 0.5
        params_1 = self.params
        params_1[:, 0] *= weight
        params_2 = component.params
        params_2[:, 0] *= (1 - weight)
        self.params = np.concatenate([params_1, params_2], axis=0)

    def delete(self, ts):
        stay_ind = np.where(self.params[:, 0] > ts)[0]
        params_stay = self.params[stay_ind, :]
        params_stay[:, 0] = params_stay[:, 0] / np.sum(params_stay[:, 0])
        self.params = params_stay


class Mixt:
    def __init__(self, params, dim, df):
        self.params = params
        self.dim = dim
        self.df = df
        self.cov_init = 0.1

    def sampling(self, num):
        params = self.params
        dim = self.dim
        weights = params[:, 0]
        mixture_idx = np.random.choice(params.shape[0], size=num, replace=True, p=weights)
        counts = np.bincount(mixture_idx)
        pnts = np.empty((num, dim))
        for i in range(params.shape[0]):
            if len(np.where(mixture_idx == i)[0]) == 0:
                continue
            w, m, c = np.split(params[i], [1, 1 + dim])
            m = np.reshape(m, (dim,))
            c = np.reshape(c, (dim, dim))
            pnts[mixture_idx == i] = ss.multivariate_t.rvs(size=counts[i], loc=m, shape=c, df=self.df)
        return pnts

    def pdf(self, node):
        params = self.params
        dim = self.dim
        prob = np.zeros_like(node.shape[0])
        for i in range(params.shape[0]):
            try:
                prob = prob + params[i, 0] * ss.multivariate_t.pdf(x=node,
                                                                        loc=params[i, 1:1 + dim].reshape((dim,)),
                                                                        shape=params[i, 1 + dim:].reshape((dim, dim)),
                                                                        df=self.df)
            except:
                cov = np.matmul(params[i, 1 + dim:].reshape((dim, dim)), np.diag(np.ones((dim,)) * np.log10(dim)))
                prob = prob + params[i, 0] * ss.multivariate_t.pdf(x=node,
                                                                        loc=params[i, 1:1 + dim].reshape((dim,)),
                                                                        shape=cov,
                                                                        df=self.df)
                print('Warning: nonPD cov')
                self.params[i, 1 + dim:] = cov.reshape(1, -1)
        return prob

    def IS_w(self, target, node):
        val = target(node)
        w = np.zeros_like(val)
        prob = self.pdf(node)
        ind = prob >= 1e-16
        w[ind] = val[ind] / prob[ind]
        if np.sum(w) == 0:
            return w
        return w / np.sum(w)

    def ess(self, IS_w, n):
        if np.sum(IS_w) <= 0.1:
            ess = 0
        else:
            ess = 1 / (n * np.sum(IS_w ** 2))
        return ess

    def EM_alg(self, node, IS_w, del_ts):
        params = self.params
        dim = self.dim
        params_next = np.zeros_like(params)
        prob_mass = self.pdf(node)
        ind_prob_mass = prob_mass >= 1e-16
        gam = np.zeros_like(prob_mass)
        for i in range(params.shape[0]):
            post_prob_i = np.zeros_like(prob_mass)
            mean = params[i, 1:1 + dim].reshape((dim,))
            cov = params[i, 1 + dim:].reshape((dim, dim))
            aux = node - np.tile(mean,(node.shape[0],1))
            gam = np.einsum('ij,ij->i',aux,np.matmul(np.linalg.inv(cov), aux.T).T)

            gam = (self.df+dim)/(self.df+gam)
            # for j in range(node.shape[0]):
            #     aux = np.matmul((node[j, :]-mean).T, np.linalg.lstsq(cov, node[j,:]-mean)[0])
            #     gam[j] = (self.df+dim)/(self.df+aux)
            post_prob_i[ind_prob_mass] = params[i, 0] * ss.multivariate_t.pdf(
                x=node[ind_prob_mass], loc=mean, shape=cov, df=self.df) / prob_mass[ind_prob_mass]
            params_next[i, 0] = np.sum(post_prob_i * IS_w)
            if params_next[i, 0] <= del_ts:
                continue
            params_next[i, 1:1 + dim] = np.sum(IS_w.reshape(-1, 1) * post_prob_i.reshape(-1, 1) * gam.reshape(-1,1)* node, axis=0) / np.sum(IS_w * post_prob_i * gam)
            cov = np.cov(node.transpose(), ddof=0, aweights=IS_w * post_prob_i*gam)*np.sum(IS_w * post_prob_i * gam)/np.sum(IS_w*post_prob_i)
            eig = np.linalg.eig(cov)[0]
            eig_min = np.min(eig)
            if eig_min < 1e-10:
                cov = cov + np.diag(np.ones(dim) * (np.abs(eig_min) * 2+1e-8))
            params_next[i, 1 + dim:] = cov.reshape(1, -1)
        return params_next
    def resort(self):
        self.params = np.array(sorted(self.params, key=lambda x: x[0], reverse=True))

    # Initial guess
    def Init(self, target, node, num, ess_add, count_add):
        ind = np.argsort(target(node))[::-1]
        pnt = node[ind][0, :]
        mean = pnt.copy()
        cov = np.diag(np.ones(self.dim) * self.cov_init)
        p = Mixt(np.concatenate([np.ones(1).reshape(1, 1), mean.reshape(1, -1), cov.reshape(1, -1)], axis=1),
                          dim=self.dim, df=self.df)
        node_p = p.sampling(num)
        IS_w_p = p.IS_w(target, node_p)
        ess_p = p.ess(IS_w_p, num)
        count_p = 1
        while ess_p <=ess_add:
            node_p = p.sampling(num)
            IS_w_p = p.IS_w(target, node_p)
            p.params = p.EM_alg(node_p, IS_w_p, 0.00001)
            ess_p = p.ess(IS_w_p, num)
            count_p += 1
            if count_p >= count_add:
                break
        self.params = p.params


    # Merge
    def merge(self, num, ts):
        dim = self.dim
        for i in range(self.params.shape[0]):
            if self.params[i, 0] == 0:
                continue
            p = AAIS_Gaussian(self.params[i, :].reshape(1, -1).copy(), dim=dim)
            p.params[0, 0] = 1
            node_p = p.sampling(num)
            ess_merge = np.zeros(self.params.shape[0])
            for j in range(self.params.shape[0]):
                if self.params[j, 0] == 0:
                    continue
                aux = AAIS_Gaussian(self.params[j, :].reshape(1, -1).copy(), dim=dim)
                aux.params[0, 0] = 1
                IS_w_i = p.IS_w(aux.pdf, node_p)
                ess_merge[j] = p.ess(IS_w_i, num)
            ess_merge[i] = 0
            if any(ess_merge >= ts):
                ind_merge = np.where(ess_merge >= ts)[0]
                ess_merge_ts = ess_merge[ind_merge] * self.params[ind_merge, 0] / (
                    np.sum(self.params[ind_merge, 0]))
                ind_max = ind_merge[np.argmax(ess_merge_ts)]
                if self.params[ind_max, 0] == 0:
                    break
                weight = self.params[i, 0] + self.params[ind_max, 0]
                self.params[i, 1:] = (self.params[i, 0] * self.params[i, 1:] + self.params[ind_max, 0] * self.params[
                                                                                                         ind_max,
                                                                                                         1:]) / weight
                self.params[i, 0] = weight
                self.params[ind_max] = 0
            else:
                continue
            print(f'merge self at  {i} with {ind_max} ')

    def merge_one(self, component, ind, weight):
        self.params[:, 0] *= weight
        params_1 = self.params[ind, :].reshape(1, -1)
        params_2 = component.params
        params_2[:, 0] = 1 - weight
        params_merge = np.zeros_like(params_1)
        params_merge[:, 0] = np.sum(np.concatenate([params_1[:, 0], params_2[:, 0]], axis=0))
        params_merge[:, 1:] = np.average(np.concatenate([params_1[:, 1:], params_2[:, 1:]], axis=0), axis=0,
                                         weights=np.concatenate([params_1[:, 0], params_2[:, 0]], axis=0))
        self.params[ind, :] = params_merge

    # update
    def update_new(self, p, node_p, merge_ts, num):
        ess_merge = np.zeros(self.params.shape[0])
        for i in range(self.params.shape[0]):
            aux = Mixt(self.params[i, :].reshape(1, -1).copy(), dim=self.dim, df=self.df)
            aux.params[0, 0] = 1
            IS_w_i = p.IS_w(aux.pdf, node_p)
            ess_merge[i] = p.ess(IS_w_i, num)
        if any(ess_merge >= merge_ts):
            ind_merge = np.where(ess_merge >= merge_ts)[0]
            ess_merge_ts = ess_merge[ind_merge] * self.params[ind_merge, 0] / (np.sum(self.params[ind_merge, 0]))
            ind_max = ind_merge[np.argmax(ess_merge_ts)]
            weight = ess_merge[ind_max] / (ess_merge[ind_max] + 0.1)
            self.merge_one(p, ind=ind_max, weight=weight)
            # print(f"merge adding p at ind {ind_max}")
        else:
            # print('add p')
            weight = np.max(ess_merge) / (0.1 + np.max(ess_merge))
            self.add_params(p, weight=0.5)

    def add_params(self, component, weight):
        # weight = 0.5
        params_1 = self.params
        params_1[:, 0] *= weight
        params_2 = component.params
        params_2[:, 0] *= (1 - weight)
        self.params = np.concatenate([params_1, params_2], axis=0)

    def delete(self, ts):
        stay_ind = np.where(self.params[:, 0] > ts)[0]
        params_stay = self.params[stay_ind, :]
        params_stay[:, 0] = params_stay[:, 0] / np.sum(params_stay[:, 0])
        self.params = params_stay


