import torch


def compute_loss_and_grad_mu(self, data, mu, logvar, targ, d_type, optim, opt='par', rng=None):
    optim.zero_grad()
    pi = torch.softmax(self.pi, dim=1)
    recloss, tot = self.get_loss(data, targ, mu, logvar, pi, rng)

    loss = recloss + tot

    if (d_type == 'train' or opt == 'mu'):
        loss.backward()
        optim.step()

    return recloss.item(), loss.item()


def get_pi_from_max(self, s_mu, s_var, data, targ=None, rng=None):
    n_mix = self.n_mix
    if (targ is None and self.n_class > 0):
        n_mix = self.n_mix_perclass
    pi = torch.zeros(data.shape[0], n_mix).to(self.dv)
    en = n_mix
    if (targ is not None):
        pi = pi.reshape(-1, self.n_class, self.n_mix_perclass)
        en = self.n_mix_perclass

    EE = (torch.eye(en) * 5. + torch.ones(en)).to(self.dv)
    s_mu = s_mu.reshape(-1, n_mix, self.s_dim).transpose(0, 1)
    s_var = s_var.reshape(-1, n_mix, self.s_dim).transpose(0, 1)

    x = self.decoder_and_trans(s_mu, rng)

    if targ is not None:
        x = x.transpose(0, 1)
        x = x.reshape(-1, self.n_class, self.n_mix_perclass, x.shape[-1])
        s_mu = s_mu.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim)
        s_var = s_var.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim)
        if (type(targ) == torch.Tensor):
            for c in range(self.n_class):
                ind = (targ == c)
                b = self.mixed_loss_pre(x[ind, c, :, :].transpose(0, 1), data[ind])
                KD = self.dens_apply(s_mu[ind, c, :], s_var[ind, c, :], pi[ind, c, :], pi[ind, c, :])[2]
                b = b + KD
                bb, ii = torch.min(b, dim=1)
                pi[ind, c, :] = EE[ii]
        pi = pi.reshape(-1, self.n_mix)
    else:
        # x = self.decoder_and_trans(s_mu, rng)
        b = self.mixed_loss_pre(x, data)
        KD = self.dens_apply(s_mu, s_var, pi, pi)[2]
        b = b + KD
        bb, ii = torch.min(b, dim=1)
        pi = EE[ii]
    return pi
