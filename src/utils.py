# !/usr/bin/python3
# -*- coding:utf-8 -*-



import sys
import os
import subprocess
import torch
import torch.nn as nn
import prettytable as pt
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




# TODO 下面版本的POSitional Embedding暂时不能使用。
class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super(PositionalEmbedding, self).__init__()

        self.emb_dim = emb_dim

        inv_freq = 1 / (10000 ** (torch.arange(0.0, emb_dim, 2.0) / emb_dim))
        self.register_buffer('inv_freq', inv_freq)
        
    # pos_seq =  pos_seq = torch.arange(seq_len-1, -1, -1.0)
    def forward(self, pos_seq, batch_size=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if batch_size is not None:
            return pos_emb[:,None,:].expand(-1, batch_size, -1)
        else:
            return pos_emb[:,None,:]




class LambdaWeightGateLayer(nn.Module):
    """docstring for GateNet"""
    def __init__(self, 
                 curr_d_node, prev_y_node, cont_1_node, cont_2_node ):
        super(GateNet, self).__init__()

        self.curr_d_node = nn.Linear(in_features = curr_d_node[0], out_features = curr_d_node[1])
        self.prev_y_node = nn.Linear(in_features = prev_y_node[0], out_features = prev_y_node[1])
        self.cont_1_node = nn.Linear(in_features = cont_1_node[0], out_features = cont_1_node[1])
        self.cont_2_node = nn.Linear(in_features = cont_2_node[0], out_features = cont_2_node[1])
        
    def forward(curr_d, prev_y, content_1, content_2):
        curr_d_out = self.curr_d_node(curr_d)
        prev_y_out = self.prev_y_node(prev_y)
        content_1_out = self.cont_1_node(content_1)
        content_2_out = self.cont_2_node(content_2)
        gate_out = torch.sigmoid(curr_d_out + prev_y_out + content_1_out + content_2_out)

        return gate_out




# class InfoPrinter(object):
#     """docstring for InfoPrinter"""
#     def __init__(self, header):
#         super(InfoPrinter, self).__init__()
#         self.table = pt.PrettyTable(border=True)
#         self.table.field_names = header
#         self.screen_clear_command = "cls" if "win" in sys.platform else "clear"
#         self.col_num = len(header)
#         self.row_info_list = self.update_extra_info()
        



#     def update_extra_info(self):
#         extra_header = []
#         extra_column = []
#         row_info_list = []

#         peak_cpu_usage = peak_memory_mb()
#         gpu_memory_info = gpu_memory_mb().items()
#         extra_header.append("CPU_Info")
#         extra_column.append("\033[30m%s\033[0m" % peak_cpu_usage)
#         for gpu, memory in gpu_memory_info:
#             extra_header.append(gpu)
#             extra_column.append("\033[30m%sMb\033[0m" % memory)
#         need_row_num = ceil(len(extra_header)/(self.col_num * 1.0))

#         extra_header += [""] * (need_row_num * self.col_num - len(extra_header))
#         extra_column += [""] * (need_row_num * self.col_num - len(extra_column))

#         for row_id in range(need_row_num):
#             row_info_list.append(extra_header[row_id*self.col_num : (row_id + 1)*self.col_num])
#             row_info_list.append(extra_column[row_id*self.col_num : (row_id + 1)*self.col_num])
#         return row_info_list




#     def print_train_info(self, info, epoch, update_extroinfo_freq):
#         sys.stdout.write("\n\n\n")
#         #sys.stdout.write("------------Training Info------------\n")
#         if epoch % update_extroinfo_freq == 0:
#             self.row_info_list = self.update_extra_info()
#         self.table.clear_rows()
#         self.table.add_row(info)
#         for row in self.row_info_list:
#             self.table.add_row(row)

#         os.system(self.screen_clear_command)
#         sys.stdout.write("{0}".format(self.table))
#         sys.stdout.flush() 
#         sys.stdout.write("\n")






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