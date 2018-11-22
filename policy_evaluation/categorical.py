import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils import TorchTypes
import random

class CategoricalPolicyEvaluation(object):
    def __init__(self, policy, cmdl):
        """Assumes policy returns an autograd.Variable"""
        self.name = "CP"
        self.cmdl = cmdl
        self.policy = policy

        self.dtype = dtype = TorchTypes(cmdl.cuda)
        self.support = torch.linspace(cmdl.v_min, cmdl.v_max, cmdl.atoms_no)
        self.support = self.support.type(dtype.FT)

    def get_action(self, state):
        """ Takes best action based on estimated state-action values."""
        state = state.type(self.dtype.FT)
        probs = self.policy(Variable(state, volatile=True)).data
        support = self.support.expand_as(probs)
        _, argmax_a = torch.mul(probs, support).squeeze().sum(1).max(0)
        if self.cmdl.policy_type == "eps_greedy":
            action = argmax_a[0]
        elif self.cmdl.policy_type == "sampling":
            action = torch.argmax(torch.multinomial(probs[0,:,:],1))
        elif self.cmdl.policy_type == "entropybased":
            action = argmax_a[0]
        else:
            print("Wrong argument policy_type")
            exit(0)
        return 0, action

    def explore_action(self, state):
        """ Takes best action based on estimated state-action values."""
        state = state.type(self.dtype.FT)
        probs = self.policy(Variable(state, volatile=True)).data
        support = self.support.expand_as(probs)
        _, argmax_a = torch.mul(probs, support).squeeze().sum(1).max(0)
        if self.cmdl.policy_type == "eps_greedy":
            action = torch.randint(0, probs.shape[1]-1, (1, )).int()
        elif self.cmdl.policy_type == "sampling":
            action = torch.randint(0, probs.shape[1]-1, (1, )).int()
        elif self.cmdl.policy_type == "entropybased":
            entropy = -(torch.log(probs[0,:,:])*probs[0,:,:]).sum(1)
            entropy /= entropy.sum()
            action = torch.multinomial(entropy,1)
        else:
            print("Wrong argument policy_type")
            exit(0)
        return action
