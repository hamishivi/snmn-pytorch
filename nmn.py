from controller import Controller
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import channels_last_conv, channels_last_conv_1d

MODULE_INPUT_NUM = {
    "_NoOp": 0,
    "_Find": 0,
    "_Transform": 1,
    "_Filter": 1,
    "_And": 2,
    "_Or": 2,
    "_Scene": 0,
    "_DescribeOne": 1,
    "_DescribeTwo": 2,
}

MODULE_OUTPUT_NUM = {
    "_NoOp": 0,
    "_Find": 1,
    "_Transform": 1,
    "_Filter": 1,
    "_And": 1,
    "_Or": 1,
    "_Scene": 1,
    "_DescribeOne": 1,
    "_DescribeTwo": 1,
}


class NMN(nn.Module):
    def __init__(self, cfg, module_names):
        super().__init__()
        # size stuff
        self.cfg = cfg
        self.N = cfg.TRAIN.BATCH_SIZE
        self.H = cfg.MODEL.H_FEAT
        self.W = cfg.MODEL.W_FEAT
        self.stack_len = cfg.MODEL.NMN.STACK.LENGTH
        self.mem_dim = cfg.MODEL.NMN.MEM_DIM
        # initial stack pointer, points to bottom.
        self.stack_ptr_init = F.one_hot(torch.zeros(self.N).long(), self.stack_len)
        self.mem_init = torch.zeros(self.N, self.mem_dim)
        # the set of modules and functions (e.g. "_Find" -> Find)
        self.module_names = module_names
        self.module_funcs = [getattr(self, m[1:]) for m in module_names]
        self.module_validity_mat = _build_module_validity_mat(module_names, cfg)
        ## module layers
        control_dim = cfg.MODEL.KB_DIM
        if cfg.MODEL.CTRL.USE_WORD_EMBED:
            control_dim = cfg.MODEL.EMBED_DIM
        # find
        self.find_ci = nn.Linear(control_dim, cfg.MODEL.KB_DIM)
        self.find_conv = nn.Conv2d(cfg.MODEL.KB_DIM, 1, 1)
        # transform
        self.transform_ci = nn.Linear(control_dim, cfg.MODEL.KB_DIM)
        self.transform_conv = nn.Conv2d(cfg.MODEL.KB_DIM, 1, 1)
        # scene
        self.scene_conv = nn.Conv2d(cfg.MODEL.KB_DIM, 1, 1)
        # describe one
        self.describeone_ci = nn.Linear(control_dim, cfg.MODEL.KB_DIM)
        self.describeone_mem = nn.Linear(
            cfg.MODEL.KB_DIM * 2 + control_dim, cfg.MODEL.NMN.MEM_DIM
        )
        # describe two
        self.describetwo_ci = nn.Linear(control_dim, cfg.MODEL.KB_DIM)
        self.describetwo_mem = nn.Linear(
            cfg.MODEL.KB_DIM * 2 + control_dim, cfg.MODEL.NMN.MEM_DIM
        )

    def get_att_zero(self, batch_size, device):
        return torch.zeros(batch_size, self.H, self.W, 1, device=device)

    def get_mem_zero(self, batch_size, device):
        return torch.zeros(batch_size, self.mem_dim, device=device)

    def get_stack_value(self, att_stack, stack_ptr):
        return _read_from_stack(att_stack, stack_ptr)

    def get_init_values(self, batch_size, device):
        # the versions of stuff we will use in the forward function.
        att_stack_init = torch.zeros(
            batch_size, self.H, self.W, self.stack_len, device=device
        )
        stack_ptr_init = F.one_hot(
            torch.zeros(batch_size, device=device, dtype=torch.long), self.stack_len
        )
        mem_init = torch.zeros(batch_size, self.mem_dim, device=device)
        return att_stack_init, stack_ptr_init, mem_init

    def forward(
        self,
        control_state,
        kb_batch,
        module_prob,
        mem_prev,
        att_stack_prev,
        stack_ptr_prev,
    ):
        # run all the modules, and average their results wrt module_w
        res = [
            f(kb_batch, att_stack_prev, stack_ptr_prev, mem_prev, control_state)
            for f in self.module_funcs
        ]
        att_stack_avg = (
            module_prob.view(module_prob.size(0), 1, 1, 1, module_prob.size(1))
            * torch.stack([r[0] for r in res], 4)
        ).sum(-1)
        # avg stack pointer and ongoing mem
        stack_ptr_avg = _sharpen_ptr(
            (module_prob.unsqueeze(1) * torch.stack([r[1] for r in res], 2)).sum(-1),
            self.cfg,
        )
        mem_avg = (module_prob.unsqueeze(1) * torch.stack([r[2] for r in res], 2)).sum(
            -1
        )
        return att_stack_avg, stack_ptr_avg, mem_avg

    #### MODULES #####
    # I've copied the original SNMN codebase for this, so each 'module' is simply a function within the NMN module.
    # However, I think a better way to do things for NMNs in the future would be to make each module a torch module
    # too. This way, each module's own code could define its input/outputs (rather than the hardcoded dict above),
    # and define its own layers, rather than having the NMN-specific and module-specific layers mixed as I have here.
    # I might make this change in the future, but for now I'm leaving this as-is.

    def NoOp(self, kb_batch, att_stack, stack_ptr, mem_in, control_state):
        """
        Does nothing. It leaves the stack pointer, the stack and mem vector
        as-is.
        """
        return att_stack, stack_ptr, mem_in

    def Find(self, kb_batch, att_stack, stack_ptr, mem_in, control_state):
        """
        Performs localization, and updates memory vector.
        1) linearly map the controller vectors to the KB dimension
        2) elementwise product with KB
        3) 1x1 convolution to get attention logits
        """
        c_mapped = self.find_ci(control_state)
        c_mapped = c_mapped.view(c_mapped.size(0), 1, 1, c_mapped.size(1))
        elt_prod = F.normalize(kb_batch * c_mapped, dim=-1, p=2)
        att_out = channels_last_conv(elt_prod, self.find_conv)
        # Push to stack
        stack_ptr = _move_ptr_fw(stack_ptr, self.cfg)
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)

        return (
            att_stack,
            stack_ptr,
            self.get_mem_zero(kb_batch.size(0), kb_batch.device),
        )

    def Transform(self, kb_batch, att_stack, stack_ptr, mem_in, control_state):
        """
        Transforms the previous attention, and updates memory vector.
        1) linearly map the controller vectors to the KB dimension
        2) extract attended features from the input attention
        2) elementwise product with KB
        3) 1x1 convolution to get attention logits
        """
        # Pop from stack
        att_in = _read_from_stack(att_stack, stack_ptr)
        c_mapped = self.transform_ci(control_state)
        c_mapped = c_mapped.view(c_mapped.size(0), 1, 1, c_mapped.size(1))
        kb_att_in = _extract_softmax_avg(kb_batch, att_in)
        kb_att_in = kb_att_in.view(kb_att_in.size(0), 1, 1, kb_att_in.size(1))
        elt_prod = F.normalize(kb_batch * c_mapped * kb_att_in, dim=-1, p=2)
        att_out = channels_last_conv(elt_prod, self.transform_conv)
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)

        return (
            att_stack,
            stack_ptr,
            self.get_mem_zero(kb_batch.size(0), kb_batch.device),
        )

    def Filter(self, kb_batch, att_stack, stack_ptr, mem_in, control_state):
        """
        Combo of Find + And. First run Find, and then run And.
        """
        # Run Find module
        att_stack, stack_ptr, _ = self.Find(
            kb_batch, att_stack, stack_ptr, mem_in, control_state
        )
        # Run And module
        att_stack, stack_ptr, _ = self.And(
            kb_batch, att_stack, stack_ptr, mem_in, control_state
        )

        return (
            att_stack,
            stack_ptr,
            self.get_mem_zero(kb_batch.size(0), kb_batch.device),
        )

    def And(self, kb_batch, att_stack, stack_ptr, mem_in, control_state):
        """
        Take the intersection between two attention maps
        1) Just take the elementwise minimum of the two inputs
        """
        # Pop from stack
        att_in_2 = _read_from_stack(att_stack, stack_ptr)
        stack_ptr = _move_ptr_bw(stack_ptr, self.cfg)
        att_in_1 = _read_from_stack(att_stack, stack_ptr)
        # stack_ptr = _move_ptr_bw(stack_ptr)  # cancel-out below
        att_out = torch.minimum(att_in_1, att_in_2)
        # Push to stack
        # stack_ptr = _move_ptr_fw(stack_ptr)  # cancel-out above
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)

        return (
            att_stack,
            stack_ptr,
            self.get_mem_zero(kb_batch.size(0), kb_batch.device),
        )

    def Or(self, kb_batch, att_stack, stack_ptr, mem_in, control_state):
        """
        Take the union between two attention maps
        1) Just take the elementwise maximum of the two inputs
        """
        # Pop from stack
        att_in_2 = _read_from_stack(att_stack, stack_ptr)
        stack_ptr = _move_ptr_bw(stack_ptr, self.cfg)
        att_in_1 = _read_from_stack(att_stack, stack_ptr)
        att_out = torch.maximum(att_in_1, att_in_2)
        # Push to stack
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)

        return (
            att_stack,
            stack_ptr,
            self.get_mem_zero(kb_batch.size(0), kb_batch.device),
        )

    def Scene(self, kb_batch, att_stack, stack_ptr, mem_in, control_state):
        """
        Output an attention map that looks at all the objects.
        1) 1x1 convolution on KB to get attention logits
        """
        att_out = channels_last_conv(
            F.normalize(kb_batch, dim=-1, p=2), self.scene_conv
        )
        # Push to stack
        stack_ptr = _move_ptr_fw(stack_ptr, self.cfg)
        att_stack = _write_to_stack(att_stack, stack_ptr, att_out)

        return (
            att_stack,
            stack_ptr,
            self.get_mem_zero(kb_batch.size(0), kb_batch.device),
        )

    def DescribeOne(self, kb_batch, att_stack, stack_ptr, mem_in, control_state):
        """
        Describe using one input attention. Outputs zero attention.
        1) linearly map the controller vectors to the KB dimension
        2) extract attended features from the input attention
        3) elementwise multplication
        4) linearly merge with previous memory vector, find memory
           vector and control state
        """
        att_stack_old, stack_ptr_old = att_stack, stack_ptr
        # Pop from stack
        att_in = _read_from_stack(att_stack, stack_ptr)
        c_mapped = self.describeone_ci(control_state)
        kb_att_in = _extract_softmax_avg(kb_batch, att_in)
        elt_prod = F.normalize(c_mapped * kb_att_in, dim=-1, p=2)
        mem_out = self.describeone_mem(
            torch.cat([control_state, mem_in, elt_prod], axis=1)
        )
        # Push to stack
        att_stack = _write_to_stack(
            att_stack, stack_ptr, self.get_att_zero(kb_batch.size(0), kb_batch.device)
        )
        if self.cfg.MODEL.NMN.DESCRIBE_ONE.KEEP_STACK:
            att_stack, stack_ptr = att_stack_old, stack_ptr_old

        return att_stack, stack_ptr, mem_out

    def DescribeTwo(self, kb_batch, att_stack, stack_ptr, mem_in, control_state):
        """
        Describe using two input attentions. Outputs zero attention.
        1) linearly map the controller vectors to the KB dimension
        2) extract attended features from the input attention
        3) elementwise multplication
        4) linearly merge with previous memory vector, find memory
           vector and control state
        """
        att_stack_old, stack_ptr_old = att_stack, stack_ptr
        # Pop from stack
        att_in_2 = _read_from_stack(att_stack, stack_ptr)
        stack_ptr = _move_ptr_bw(stack_ptr, self.cfg)
        att_in_1 = _read_from_stack(att_stack, stack_ptr)
        c_mapped = self.describetwo_ci(control_state)
        kb_att_in_1 = _extract_softmax_avg(kb_batch, att_in_1)
        kb_att_in_2 = _extract_softmax_avg(kb_batch, att_in_2)
        elt_prod = F.normalize(c_mapped * kb_att_in_1 * kb_att_in_2, dim=-1, p=2)
        mem_out = self.describetwo_mem(
            torch.cat([control_state, mem_in, elt_prod], dim=1)
        )
        # Push to stack
        att_stack = _write_to_stack(
            att_stack, stack_ptr, self.get_att_zero(kb_batch.size(0), kb_batch.device)
        )

        if self.cfg.MODEL.NMN.DESCRIBE_TWO.KEEP_STACK:
            att_stack, stack_ptr = att_stack_old, stack_ptr_old

        return att_stack, stack_ptr, mem_out


def _move_ptr_fw(stack_ptr, cfg):
    """
    Move the stack pointer forward (i.e. to push to stack).
    """
    filter_fw = torch.tensor([1, 0, 0], device=stack_ptr.device).view(1, 1, 3).float()
    padding_size = math.ceil(stack_ptr.size(1) / 3) * 3 - stack_ptr.size(1)
    new_stack_ptr = channels_last_conv_1d(
        stack_ptr.unsqueeze(2).float(),
        lambda x: F.conv1d(x, filter_fw, padding=padding_size).squeeze(2),
    ).squeeze(2)
    # when the stack pointer is already at the stack top, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    if cfg.MODEL.NMN.STACK.GUARD_STACK_PTR:
        stack_len = cfg.MODEL.NMN.STACK.LENGTH
        stack_top_mask = F.one_hot(
            torch.zeros(stack_ptr.size(0), dtype=torch.long, device=stack_ptr.device)
            + stack_len
            - 1,
            stack_len,
        )
        new_stack_ptr += stack_top_mask * stack_ptr
    return new_stack_ptr


def _move_ptr_bw(stack_ptr, cfg):
    """
    Move the stack pointer backward (i.e. to pop from stack).
    """
    filter_bw = torch.tensor([0, 0, 1], device=stack_ptr.device).view(1, 1, 3).float()
    padding_size = math.ceil(stack_ptr.size(1) / 3) * 3 - stack_ptr.size(1)
    new_stack_ptr = channels_last_conv_1d(
        stack_ptr.unsqueeze(2).float(),
        lambda x: F.conv1d(x, filter_bw, padding=padding_size).squeeze(2),
    ).squeeze(2)
    # when the stack pointer is already at the stack bottom, keep
    # the pointer in the same location (otherwise the pointer will be all zero)
    if cfg.MODEL.NMN.STACK.GUARD_STACK_PTR:
        stack_len = cfg.MODEL.NMN.STACK.LENGTH
        stack_bottom_mask = F.one_hot(
            torch.zeros(stack_ptr.size(0), dtype=torch.long, device=stack_ptr.device),
            stack_len,
        )
        new_stack_ptr += stack_bottom_mask * stack_ptr
    return new_stack_ptr


def _read_from_stack(att_stack, stack_ptr):
    """
    Read the value at the given stack pointer.
    """
    # The stack pointer is a one-hot vector, so just do dot product
    att = (att_stack * stack_ptr.unsqueeze(1).unsqueeze(1)).sum(-1).unsqueeze(-1)
    return att


def _write_to_stack(att_stack, stack_ptr, att):
    """
    Write value 'att' into the stack at the given stack pointer. Note that the
    result needs to be assigned back to att_stack
    """
    stack_ptr_expand = stack_ptr.unsqueeze(1).unsqueeze(1)
    att_stack = att * stack_ptr_expand + att_stack * (1 - stack_ptr_expand)
    return att_stack


def _sharpen_ptr(stack_ptr, cfg):
    """
    Sharpen the stack pointers into (nearly) one-hot vectors, using argmax
    or softmax. The stack values should always sum up to one for each instance.
    """
    hard = cfg.MODEL.NMN.STACK.USE_HARD_SHARPEN
    if hard:
        # hard (non-differentiable) sharpening with argmax
        new_stack_ptr = F.one_hot(torch.argmax(stack_ptr, dim=1), stack_ptr.size(1))
    else:
        # soft (differentiable) sharpening with softmax
        temperature = cfg.MODEL.NMN.STACK.SOFT_SHARPEN_TEMP
        new_stack_ptr = F.softmax(stack_ptr / temperature, 1)
    return new_stack_ptr


def _spatial_softmax(att_raw):
    N = att_raw.size(0)
    att_softmax = F.softmax(att_raw.view(N, -1), dim=1)
    att_softmax = att_softmax.view(att_raw.size())
    return att_softmax


def _extract_softmax_avg(kb_batch, att_raw):
    att_softmax = _spatial_softmax(att_raw)
    return (kb_batch * att_softmax).sum([1, 2])


def _build_module_validity_mat(module_names, cfg):
    """
    Build a module validity matrix, ensuring that only valid modules will have
    non-zero probabilities. A module is only valid to run if there are enough
    attentions to be popped from the stack, and have space to push into
    (e.g. _Find), so that stack will not underflow or overflow by design.
    module_validity_mat is a stack_len x num_module matrix, and is used to
    multiply with stack_ptr to get validity boolean vector for the modules.
    """
    stack_len = cfg.MODEL.NMN.STACK.LENGTH
    module_validity_mat = torch.zeros(stack_len, len(module_names))
    for n_m, m in enumerate(module_names):
        # a module can be run only when stack ptr position satisfies
        # (min_ptr_pos <= ptr <= max_ptr_pos), where max_ptr_pos is inclusive
        # 1) minimum position:
        #    stack need to have MODULE_INPUT_NUM[m] things to pop from
        min_ptr_pos = MODULE_INPUT_NUM[m]
        # the stack ptr diff=(MODULE_OUTPUT_NUM[m] - MODULE_INPUT_NUM[m])
        # ensure that ptr + diff <= stack_len - 1 (stack top)
        max_ptr_pos = stack_len - 1 + MODULE_INPUT_NUM[m] - MODULE_OUTPUT_NUM[m]
        module_validity_mat[min_ptr_pos : max_ptr_pos + 1, n_m] = 1.0

    return module_validity_mat.float()
