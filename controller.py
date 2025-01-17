import torch
import torch.nn.functional as F
import torch.nn as nn
from nn_utils import linear


class Controller(nn.Module):
    """
    Controller that decides which modules to use and maintains control state.
    Essentially same as MAC control unit.
    """

    def __init__(self, cfg, num_modules):
        super().__init__()
        self.cfg = cfg
        self.num_modules = num_modules
        control_dim = cfg.MODEL.KB_DIM
        if cfg.MODEL.CTRL.USE_WORD_EMBED:
            control_dim = cfg.MODEL.EMBED_DIM
        dim = cfg.MODEL.LSTM_DIM

        self.shared_control_proj = linear(dim, dim)
        self.position_aware = nn.ModuleList()
        for i in range(cfg.MODEL.T_CTRL):
            self.position_aware.append(linear(dim, dim))

        self.control_question = linear(dim + control_dim, dim)
        self.attn = linear(dim, 1)

        if self.cfg.MODEL.CTRL.LINEAR_MODULE_WEIGHTS:
            self.module_fc = nn.Linear(dim, num_modules, bias=False)
        else:
            self.module_fc = nn.Sequential(
                nn.Linear(dim, cfg.MODEL.LSTM_DIM),
                nn.ELU(),
                nn.Linear(cfg.MODEL.LSTM_DIM, num_modules))

    def forward(
        self, embed_context, lstm_context, question, control, question_mask, step
    ):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)

        module_logits = self.module_fc(control_question)
        module_probs = F.softmax(module_logits, 1)

        context_prod = control_question.unsqueeze(1) * lstm_context

        attn_weight = self.attn(context_prod).squeeze(-1) - 1e30 * (1 - question_mask)

        attn = F.softmax(attn_weight, 1).unsqueeze(2)

        if self.cfg.MODEL.CTRL.USE_WORD_EMBED:
            next_control = (attn * embed_context).sum(1)
        else:
            next_control = (attn * lstm_context).sum(1)

        return next_control, module_logits, module_probs, attn.squeeze(2)
