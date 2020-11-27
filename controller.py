import torch
import torch.nn.functional as F
import torch.nn as nn
from nn_utils import linear

class Controller(nn.Module):
    """
    Controller that decides which modules to use and maintains control state.
    Essentially same as MAC control unit.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_modules = 3
        dim = cfg.MODEL.LSTM_DIM

        self.shared_control_proj = linear(dim, dim)
        self.position_aware = nn.ModuleList()
        for i in range(cfg.MODEL.T_CTRL):
            self.position_aware.append(linear(dim, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.module_fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ELU(),
            nn.Linear(dim, self.num_modules)
        )

    def forward(self, context, question, control, question_mask, step):
        question = torch.tanh(self.shared_control_proj(question))
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)

        module_logits = self.module_fc(control_question)
        module_probs = F.softmax(module_logits, 1)

        context_prod = control_question.unsqueeze(1) * context

        attn_weight = self.attn(context_prod).squeeze(-1) - 1e30 * (1 - question_mask)

        attn = F.softmax(attn_weight, 1).unsqueeze(2)

        next_control = (attn * context).sum(1)

        return next_control, module_probs