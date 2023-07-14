import torch

def sample_from_supports(supports, probs):
    '''
    supports: (n_support, dim)
    probs: (n_support), has been softmaxed
    Return: (dim)
'''
    sample_idx = torch.multinomial(probs, num_samples=1).squeeze() # scalar
    return supports[sample_idx, :]
