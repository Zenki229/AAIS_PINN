import scipy.stats as ss
import numpy as np
import torch
import logging
from libs.multi_mode import *


class Uni:
    def __init__(self, sample):
        self.sampling = sample

    def sample(self, target, node, judge, num_output, path):
        node = self.sampling(num_output, 'in')
        return node.detach().cpu().numpy(), None


class RAD:
    def __init__(self, dim, log):
        self.dim = dim
        self.log = log

    def sample(self, target, node, judge, num_output, path):
        err = target(node)
        integration = np.mean(err)
        err_normal = err/(np.sum(err))
        ind = np.random.choice(a=len(node), size=num_output, replace=False, p=err_normal)

        def proposal(node):
            return target(node)/(integration+1e-8)
        return node[ind, :], proposal


class AAISGaussian:
    def __init__(self, a_ladder, ess_dict, count_dict, dim, log, weighted_sample):
        self.dim = dim
        self.a_lad = a_ladder
        self.ess_dict = ess_dict
        self.count_dict = count_dict
        self.log = log
        self.weighted_sample = weighted_sample

    def sample(self, target, node, judge, num_output, path):
        log = self.log
        dim = self.dim
        ess_lad = self.ess_dict['ladder']
        ess_merge = self.ess_dict['merge']
        ess_add = self.ess_dict['add']
        count_lad = self.count_dict['ladder']
        count_total = self.count_dict['total']
        count_add = self.count_dict['add']
        n = int(np.ceil(node.shape[0] * 0.1))
        num = int(np.ceil(n * 0.1))
        if self.dim<=1:
          cov_init = 100/(n**2)
        else:
          cov_init = 0.1
        #cov_init = 0.01
        # validity check
        if (len(ess_lad) != len(self.a_lad)) | (len(count_lad) != len(self.a_lad)):
            raise ValueError('invalid ladder please check')
        # Initial guess
        proposal = MixGaussian(params=0, dim=dim)
        proposal.Init(target, node, num, ess_add, count_add)
        # output = MixGaussian(params = proposal.params.copy(), dim=dim)
        # start
        count_all = 0
        for enum, anneal_w in enumerate(self.a_lad):
            anneal = anneal_target(proposal, anneal_w=anneal_w)

            @anneal
            def f(node):
                return target(node)
            ess_ts = ess_lad[enum]
            if enum+1 >= len(self.a_lad):
                output = MixGaussian(params=proposal.params.copy(), dim=dim)
                node_o = output.sampling(n)
                IS_w_o = output.IS_w(f, node_o)
                ess_o = output.ess(IS_w_o, n)
            node_q = proposal.sampling(n)
            IS_w_q = proposal.IS_w(f, node_q)
            ess_q = proposal.ess(IS_w_q, n)
            count_each = 0
            while ess_q <= ess_ts:
                ind = np.argsort(IS_w_q)[::-1]
                pnt = node_q[ind][0]
                mean = pnt.copy()
                cov = np.diag(np.ones(dim) * cov_init)
                p = MixGaussian(
                    np.concatenate([np.ones(1).reshape(1, 1), mean.reshape(1, -1), cov.reshape(1, -1)], axis=1),
                    dim=dim)
                node_p = p.sampling(num)
                IS_w_p = p.IS_w(f, node_p)
                ess_p = p.ess(IS_w_p, num)
                count_p = 0
                while ess_p <= ess_add:
                    p.params = p.EM_alg(node_p, IS_w_p, del_ts=1/proposal.params.shape[0]*0.01)
                    node_p = p.sampling(num)
                    IS_w_p = p.IS_w(f, node_p)
                    ess_p = p.ess(IS_w_p, num)
                    count_p += 1
                    if count_p >= count_add:
                        break
                proposal.update_new(p, node_p, ess_merge, num)
                # proposal.add_params(p, weight=0.5)
                for i in range(np.min((int(np.ceil(proposal.params.shape[0] / 10)) + 1, 2))):
                    node_q = proposal.sampling(n)
                    IS_w_q = proposal.IS_w(f, node_q)
                    proposal.params = proposal.EM_alg(node_q, IS_w_q, del_ts=1/proposal.params.shape[0]*0.01)
                    proposal.resort()
                    proposal.delete(ts=1/proposal.params.shape[0]*0.01)
                    ess_q = proposal.ess(IS_w_q, n)
                    if ess_q >= ess_ts:
                        break
                    #np.savetxt('./proposal.txt', proposal.params, fmt='%f', delimiter=' ')
                count_all += 1
                count_each += 1
                if enum+1>=len(self.a_lad):
                    if ess_q >= ess_o:
                      output = MixGaussian(params = proposal.params.copy(), dim=dim)
                      ess_o = ess_q.copy()
                if count_all % 50 == 0:
                  log.info(f'total {count_all}:{anneal_w}-{count_each} ite, ess = {round(ess_q, 4)}, shape {proposal.params.shape[0]}')
                if (count_each >= count_lad[enum]) | (count_all >= count_total):
                    log.info(f'BREAK total {count_all}:{anneal_w}-{count_each} ite, ess = {round(ess_q, 4)}, shape {proposal.params.shape[0]}')
                    break
            if ess_q >= ess_ts:
              log.info(f'GOOD total {count_all}:{anneal_w}-{count_each} ite, ess = {round(ess_q, 4)}, shape {proposal.params.shape[0]}')

        # end here
        log.info(f'final ess = {round(ess_o,4)}')
        proposal.params = output.params.copy()
        if self.weighted_sample:
          for i in range(proposal.params.shape[0]):
            proposal.params[i, 0] *= ss.multivariate_normal.pdf(x=proposal.params[i, 1:1 + dim],
                                                          mean=proposal.params[i, 1:1 + dim].reshape((dim,)),
                                                         cov=proposal.params[i, 1 + dim:].reshape((dim, dim)))
            proposal.params[:, 0] = proposal.params[:, 0]/np.sum(proposal.params[:, 0])
        node_q = proposal.sampling(num_output)
        node_return = node_q[judge(node_q)]
        n_res = num_output-node_return.shape[0]
        while n_res > 0:
            node_add = proposal.sampling(n_res)
            node_q = np.concatenate((node_return, node_add), axis=0)
            node_return = node_q[judge(node_q)]
            n_res = num_output-node_return.shape[0]
        ess = np.load(path+'/train/'+'ess.npy')
        ess = np.concatenate((ess,np.ones((1,1))*ess_o),axis=0)
        shape = np.load(path + '/train/' + 'shape.npy')
        shape = np.concatenate((shape, np.ones((1, 1)) * proposal.params.shape[0]), axis=0)
        np.save(path+'/train/'+'ess.npy', ess)
        np.save(path + '/train/' + 'shape.npy', shape)
        return node_return, proposal.pdf

class AAISt:
    def __init__(self, a_ladder, ess_dict, count_dict, dim, df, log, weighted_sample):
        self.dim = dim
        self.a_lad = a_ladder
        self.ess_dict = ess_dict
        self.count_dict = count_dict
        self.log = log
        self.df = df
        self.weighted_sample = weighted_sample
    def sample(self, target, node, judge, num_output, path):
        log = self.log
        dim = self.dim
        ess_lad = self.ess_dict['ladder']
        ess_merge = self.ess_dict['merge']
        ess_add = self.ess_dict['add']
        count_lad = self.count_dict['ladder']
        count_total = self.count_dict['total']
        count_add = self.count_dict['add']
        n = int(np.ceil(node.shape[0] * 0.1))
        num = int(np.ceil(n * 0.1))
        if self.dim<=1:
          cov_init = 100/(n**2)
        else:
          cov_init = 0.5
        # validity check
        if (len(ess_lad) != len(self.a_lad)) | (len(count_lad) != len(self.a_lad)):
            raise ValueError('invalid ladder please check')
        # Initial guess
        proposal = Mixt(params=0, dim=dim, df=self.df)
        proposal.Init(target, node, num, ess_add, count_add)
        # output = MixGaussian(params = proposal.params.copy(), dim=dim)
        # start
        count_all = 0
        for enum, anneal_w in enumerate(self.a_lad):
            anneal = anneal_target(proposal, anneal_w=anneal_w)

            @anneal
            def f(node):
                return target(node)

            ess_ts = ess_lad[enum]
            if enum + 1 >= len(self.a_lad):
                output = Mixt(params=proposal.params.copy(), dim=dim, df=self.df)
                node_o = output.sampling(n)
                IS_w_o = output.IS_w(f, node_o)
                ess_o = output.ess(IS_w_o, n)
            node_q = proposal.sampling(n)
            IS_w_q = proposal.IS_w(f, node_q)
            ess_q = proposal.ess(IS_w_q, n)
            count_each = 0
            while ess_q <= ess_ts:
                ind = np.argsort(IS_w_q)[::-1]
                pnt = node_q[ind][0]
                mean = pnt.copy()
                cov = np.diag(np.ones(dim) * cov_init)
                p = Mixt(
                    np.concatenate([np.ones(1).reshape(1, 1), mean.reshape(1, -1), cov.reshape(1, -1)], axis=1),
                    dim=dim, df=self.df)
                node_p = p.sampling(num)
                IS_w_p = p.IS_w(f, node_p)
                ess_p = p.ess(IS_w_p, num)
                count_p = 0
                while ess_p <= ess_add:
                    p.params = p.EM_alg(node_p, IS_w_p, del_ts=1/proposal.params.shape[0]*0.01)
                    node_p = p.sampling(num)
                    IS_w_p = p.IS_w(f, node_p)
                    ess_p = p.ess(IS_w_p, num)
                    count_p += 1
                    if count_p >= count_add:
                        break
                proposal.update_new(p, node_p, ess_merge, num)
                # proposal.add_params(component=p, weight=0.5)
                for i in range(np.min((int(np.ceil(proposal.params.shape[0] / 10)) + 1, 2))):
                    node_q = proposal.sampling(n)
                    IS_w_q = proposal.IS_w(f, node_q)
                    proposal.params = proposal.EM_alg(node_q, IS_w_q, del_ts=1/proposal.params.shape[0]*0.01)
                    proposal.resort()
                    proposal.delete(ts=1/proposal.params.shape[0]*0.01)
                    ess_q = proposal.ess(IS_w_q, n)
                    if ess_q >= ess_ts:
                        break
                    # np.savetxt('./proposal.txt', proposal.params, fmt='%f', delimiter=' ')
                count_all += 1
                count_each += 1
                if enum + 1 >= len(self.a_lad):
                    if ess_q >= ess_o:
                        output = Mixt(params=proposal.params.copy(), dim=dim, df=self.df)
                        ess_o = ess_q.copy()
                if count_all % 50 == 0:
                    log.info(
                        f'total {count_all}:{anneal_w}-{count_each} ite, ess = {round(ess_q, 4)}, shape {proposal.params.shape[0]}')
                if (count_each >= count_lad[enum]) | (count_all >= count_total):
                    log.info(
                        f'BREAK total {count_all}:{anneal_w}-{count_each} ite, ess = {round(ess_q, 4)}, shape {proposal.params.shape[0]}')
                    break
            if ess_q >= ess_ts:
                log.info(
                    f'GOOD total {count_all}:{anneal_w}-{count_each} ite, ess = {round(ess_q, 4)}, shape {proposal.params.shape[0]}')

        # end here
        log.info(f'final ess = {round(ess_o, 4)}')
        proposal.params = output.params.copy()
        if self.weighted_sample:
          for i in range(proposal.params.shape[0]):
            proposal.params[i, 0] *= ss.multivariate_t.pdf(x=proposal.params[i, 1:1 + dim],
                                                                  loc=proposal.params[i, 1:1 + dim].reshape((dim,)),
                                                                  shape=proposal.params[i, 1 + dim:].reshape(
                                                                      (dim, dim)),
                                                             df=self.df)
            proposal.params[:, 0] = proposal.params[:, 0] / np.sum(proposal.params[:, 0])
        node_q = proposal.sampling(num_output)
        node_return = node_q[judge(node_q)]
        n_res = num_output - node_return.shape[0]
        while n_res > 0:
            node_add = proposal.sampling(n_res)
            node_q = np.concatenate((node_return, node_add), axis=0)
            node_return = node_q[judge(node_q)]
            n_res = num_output - node_return.shape[0]
        ess = np.load(path + '/train/' + 'ess.npy')
        ess = np.concatenate((ess, np.ones((1, 1)) * ess_o), axis=0)
        shape = np.load(path + '/train/' + 'shape.npy')
        shape = np.concatenate((shape, np.ones((1, 1)) * proposal.params.shape[0]), axis=0)
        np.save(path + '/train/' + 'ess.npy', ess)
        np.save(path + '/train/' + 'shape.npy', shape)
        return node_return, proposal.pdf


def anneal_target(q, anneal_w):
    def new_target(func):
        def new_func(node):
            return np.power(q.pdf(node), 1 - anneal_w) * np.power(func(node), anneal_w)

        return new_func

    return new_target






