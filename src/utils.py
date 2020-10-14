# !/usr/bin/python3
# -*- coding:utf-8 -*-



import sys
import os
import subprocess
import torch
import torch.nn as nn
from math import ceil
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

try:
    import resource
except ImportError:
    # resource doesn't exist on Windows systems
    resource = None






def set_cuda_device(cuda_device, verbose=True):
    # cuda_device = args.cuda_device

    available_cuda_device_num = torch.cuda.device_count()
    available_cuda_device_ids = list(range(available_cuda_device_num)) + [-1]

    if available_cuda_device_num == 0 and verbose:
        print("\033[31m[WARN] No CUDA Device Found on This Machine.\n \033[0m")
        
    assert isinstance(cuda_device, int) and cuda_device in available_cuda_device_ids, \
           f"Error: Wrong CUDA Device Value Encountered! It Should in {available_cuda_device_ids}\n"

    if cuda_device >=0 and cuda_device < available_cuda_device_num:
        device = torch.device("cuda:"+str(cuda_device))
        torch.cuda.set_device(device)
        cuda_device_id = torch.cuda.current_device()
        cuda_device_name = torch.cuda.get_device_name(cuda_device_id)

        if verbose:
            print("\033[31m[INFO] Device ID: %s \033[0m" % device) 
            print("\033[31m[INFO] Device Name: %s\n\n \033[0m" % cuda_device_name)

    else:
        device = torch.device("cpu")
        if verbose:
            print("\033[31m[INFO] Device ID: %s \n\n\033[0m" % device) 

    return device




def peak_memory_mb():
    """
    Get peak memory usage for this process, as measured by
    max-resident-set size:

    https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size

    Only works on OSX and Linux, returns 0.0 otherwise.
    """
    if resource is None or sys.platform not in ('linux', 'darwin'):
        return 0.0

    # TODO(joelgrus): For whatever, our pinned version 0.521 of mypy does not like
    # next line, but later versions (e.g. 0.530) are fine with it. Once we get that
    # figured out, remove the type: ignore.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore

    if sys.platform == 'darwin':
        # On OSX the result is in bytes.
        return peak / 1_000_000

    else:
        # On Linux the result is in kilobytes.
        return peak / 1_000





def gpu_memory_mb():
    """
    Get the current GPU memory usage.
    Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Returns
    -------
    ``Dict[int, int]``
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
        Returns an empty ``dict`` if GPUs are not available.
    """
    # pylint: disable=bare-except
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                          '--format=csv,nounits,noheader'],
                                         encoding='utf-8')
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        return {gpu: memory for gpu, memory in enumerate(gpu_memory)}
    except FileNotFoundError:
        # `nvidia-smi` doesn't exist, assume that means no GPU.
        return {}
    except:
        # Catch *all* exceptions, because this memory check is a nice-to-have
        # and we'd never want a training run to fail because of it.
        logger.exception("unable to check gpu_memory_mb(), continuing")
        return {}





def color(value, pattern):
    if pattern == 1: # Red
        return "\033[31m%s\033[0m" % value
    elif pattern == 2: # Yellow
        return "\033[33m%s\033[0m" % value
    else:
        return value





def hardware_info_printer():
    peak_cpu_usage = peak_memory_mb()
    if peak_cpu_usage > 0:
        print("\033[30mPeak CPU memory usage MB: %s \n \033[0m" % peak_cpu_usage)
    for gpu, memory in gpu_memory_mb().items():
        print("\033[33mGPU:%s, Memory Usage:%s MB:  \033[0m" % (gpu, memory))
    print("\n")






def get_BLEU_score(predicted_tokens, reference_ans):
    # two references for one document
    #from nltk.translate.bleu_score import corpus_bleu
    references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
    candidates = [['this', 'is', 'a', 'test']]
    score = corpus_bleu(references, candidates)
    return score





def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.

    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()





def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.

    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.

    For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
    embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:

        - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
        - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
          query for each document)

    are valid input "vectors", producing tensors of shape:
    ``(batch_size, num_queries, embedding_dim)`` and
    ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)




def clamp_tensor(tensor, minimum, maximum):
    """
    Supports sparse and dense tensors.
    Returns a tensor with values clamped between the provided minimum and maximum,
    without modifying the original tensor.
    """
    if tensor.is_sparse:
        coalesced_tensor = tensor.coalesce()
        # pylint: disable=protected-access
        coalesced_tensor._values().clamp_(minimum, maximum)
        return coalesced_tensor
    else:
        return tensor.clamp(minimum, maximum)




def rescale_gradients(model, grad_norm = None):
    """
    Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
    """
    if grad_norm:
        parameters_to_clip = [p for p in model.parameters()
                              if p.grad is not None]
        return sparse_clip_norm(parameters_to_clip, grad_norm)
    return None


from torch.nn.utils import clip_grad_norm
def gradient_clipping(model, grad_clipping):
    if grad_clipping is not None:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda grad: nn_util.clamp_tensor(grad,
                                                                          minimum=-grad_clipping,
                                                                          maximum=grad_clipping))