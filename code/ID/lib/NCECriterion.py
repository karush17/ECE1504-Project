import torch
from torch import nn
import math
import torch.nn.functional as F
eps = 1e-7

def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)  # Note JSD will be shifted
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2  # Note JSD will be shifted
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples - 1.)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq


def generator_loss(q_samples, measure, loss_type=None):
    """Computes the loss for the generator of a GAN.
    Args:
        q_samples: fake samples.
        measure: Measure to compute loss for.
        loss_type: Type of loss: basic `minimax` or `non-saturating`.
    """
    if not loss_type or loss_type == 'minimax':
        return get_negative_expectation(q_samples, measure)
    elif loss_type == 'non-saturating':
        return -get_positive_expectation(q_samples, measure)
    else:
        raise NotImplementedError(
            'Generator loss type `{}` not supported. '
            'Supported: [None, non-saturating, boundary-seek]')



class NCECriterion(nn.Module):

    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem = nLem

    def forward(self, x, targets):
        batchSize = x.size(0)
        K = x.size(1)-1
        Pnt = 1 / float(self.nLem)
        Pns = 1 / float(self.nLem)
        
        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt) 
        Pmt = x.select(1,0)
        Pmt_div = Pmt.add(K * Pnt + eps)
        lnPmt = torch.div(Pmt, Pmt_div)
        
        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        Pon_div = x.narrow(1,1,K).add(K * Pns + eps)
        Pon = Pon_div.clone().fill_(K * Pns)
        lnPon = torch.div(Pon, Pon_div)
     
        # equation 6 in ref. A
        lnPmt.log_()
        lnPon.log_()
        
        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0)
        
        loss = - (lnPmtsum + lnPonsum) / batchSize
        
        return loss


def fenchel_dual_loss(l, m, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.
    Note that vectors should be sent as 1x1.
    Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
    Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.
    Args:
        l: Local feature map.
        m: Multiple globals feature map.
        measure: f-divergence measure.
    Returns:
        torch.Tensor: Loss.
    '''
    # print(l.shape, m.shape)
    N, n_locals = l.size() # 128, 128
    N, chan, im_h, im_w = m.size() # 128, 1, 32, 32
    m = m.view(N,-1)
    u = torch.mm(l.t(), m)
    # print(u.shape)

    # Compute the positive and negative score. Average the spatial locations.
    E_pos = get_positive_expectation(u, measure, average=False).mean(1).mean(0)
    E_neg = get_negative_expectation(u, measure, average=False).mean(1).mean(0)
    # print(E_pos.shape, E_neg.shape)

    # Mask positive and negative terms for positive and negative parts of loss
    loss = E_neg - E_pos

    # N, units, n_locals = l.size()
    # n_multis = m.size(2)

    # # First we make the input tensors the right shape.
    # l = l.view(N, units, n_locals)
    # l = l.permute(0, 2, 1)
    # l = l.reshape(-1, units)

    # m = m.view(N, units, n_multis)
    # m = m.permute(0, 2, 1)
    # m = m.reshape(-1, units)

    # # Outer product, we want a N x N x n_local x n_multi tensor.
    # u = torch.mm(m, l.t())
    # u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    # # Since we have a big tensor with both positive and negative samples, we need to mask.
    # mask = torch.eye(N).to(l.device)
    # n_mask = 1 - mask

    # # Compute the positive and negative score. Average the spatial locations.
    # E_pos = get_positive_expectation(u, measure, average=False).mean(2).mean(2)
    # E_neg = get_negative_expectation(u, measure, average=False).mean(2).mean(2)

    # # Mask positive and negative terms for positive and negative parts of loss
    # E_pos = (E_pos * mask).sum() / mask.sum()
    # E_neg = (E_neg * n_mask).sum() / n_mask.sum()
    # loss = E_neg - E_pos

    return loss

