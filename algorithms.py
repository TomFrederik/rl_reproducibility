import numpy as np
import torch
import utils
from hparams import HyperParams as hp


class TargetAlgorithm:
    def __init__(self):
        pass

    def train(self, memory):
        pass

    def targets(self, memory):
        raise NotImplementedError


class ActorOnlyMC(TargetAlgorithm):
    """No critic, targets are simply sampled returns (REINFORCE targets)"""

    def targets(self, memory):
        memory = np.array(memory)
        rewards = list(memory[:, 2])
        masks = list(memory[:, 3])

        return get_returns(rewards, masks)


class BaselineCriticMC(TargetAlgorithm):
    """Baseline state value function learned by a critic is subtracted."""

    def __init__(self, critic, critic_optim):
        self.critic = critic
        self.optim = critic_optim

    def targets(self, memory):
        memory = np.array(memory)
        states = np.vstack(memory[:, 0])
        rewards = list(memory[:, 2])
        masks = list(memory[:, 3])

        return get_returns(rewards, masks) - self.critic(states)

    def train(self, memory):
        memory = np.array(memory)
        rewards = list(memory[:, 2])
        masks = list(memory[:, 3])
        states = np.vstack(memory[:, 0])

        returns = get_returns(rewards, masks)

        criterion = torch.nn.MSELoss()
        n = len(states)
        arr = np.arange(n)

        for epoch in range(5):
            np.random.shuffle(arr)

            for i in range(n // hp.batch_size):
                batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                inputs = torch.Tensor(states)[batch_index]
                target = returns.unsqueeze(1)[batch_index]

                values = self.critic(inputs)
                loss = criterion(values, target)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


class ActorAlgorithm:
    def __init__(self, target_alg):
        self.target_alg = target_alg

    def get_loss(self, returns, states, actions, *args, **kwargs):
        log_policy = self.actor.get_log_probs(states, actions)
        return returns * log_policy

    def fisher_vector_product(self, states, p):
        p.detach()
        kl = self.actor.get_kl(states)
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad = utils.flat_grad(kl_grad)  # check kl_grad == 0

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, self.actor.parameters())
        kl_hessian_p = utils.flat_hessian(kl_hessian_p)

        return kl_hessian_p + 0.1 * p

    # from openai baseline code
    # https://github.com/openai/baselines/blob/master/baselines/common/cg.py
    def conjugate_gradient(self, states, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = self.fisher_vector_product(states, p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def train(self, memory):
        memory = np.array(memory)
        states = np.vstack(memory[:, 0])
        actions = list(memory[:, 1])

        # ----------------------------
        # step 1: get targets
        returns = self.target_alg.targets(memory)

        # ----------------------------
        # step 3: get gradient of loss and hessian of kl
        loss = self.get_loss(returns, states, actions)
        loss_grad = torch.autograd.grad(loss, self.actor.parameters())
        loss_grad = utils.flat_grad(loss_grad)
        step_dir = self.conjugate_gradient(states, loss_grad.data, nsteps=10)

        # ----------------------------
        # step 4: get step direction and step size and update actor
        self.step(step_dir, states, actions, loss, loss_grad)

    def step(self, step_dir, states, *args):
        raise NotImplementedError


class NPG(ActorAlgorithm):
    def step(self, step_dir, states):
        params = utils.flat_params(self.actor)
        new_params = params + 0.5 * step_dir
        utils.update_model(self.actor, new_params)


class TRPO(ActorAlgorithm):
    def get_loss(self, targets, states, actions, old_policy=None):
        new_policy = self.actor.get_log_probs(states, actions)
        if old_policy is None:
            old_policy = new_policy.detach().clone()
        else:
            old_policy = old_policy.detach()
        return targets * torch.exp(new_policy - old_policy)

    def step(self, step_dir, states, actions, loss, loss_grad):
        params = utils.flat_params(self.actor)
        shs = 0.5 * (step_dir * self.fisher_vector_product(states, step_dir)
                     ).sum(0, keepdim=True)
        step_size = 1 / torch.sqrt(shs / hp.max_kl)[0]
        full_step = step_size * step_dir

        # ----------------------------
        # step 5: do backtracking line search for n times
        # Create a copy of the current actor
        old_actor = self.actor.__class__(self.actor.num_inputs, self.actor.num_outputs)
        utils.update_model(old_actor, params)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)

        flag = False
        fraction = 1.0
        for i in range(10):
            new_params = params + fraction * full_step
            utils.update_model(self.actor, new_params)
            new_loss = self.get_loss(states, actions)
            loss_improve = new_loss - loss
            expected_improve *= fraction
            kl = self.actor.kl_divergence(states, old_actor=old_actor)
            kl = kl.mean()

            #print('kl: {:.4f}  loss improve: {:.4f}  expected improve: {:.4f}  '
            #    'number of line search: {}'
            #    .format(kl.data.numpy(), loss_improve, expected_improve[0], i))

            # see https: // en.wikipedia.org / wiki / Backtracking_line_search
            if kl < hp.max_kl and (loss_improve / expected_improve) > 0.5:
                flag = True
                break

            fraction *= 0.5

        if not flag:
            params = utils.flat_params(old_actor)
            utils.update_model(self.actor, params)
            print('policy update does not impove the surrogate')


def get_returns(rewards, masks):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)

    running_returns = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        returns[t] = running_returns

    returns = (returns - returns.mean()) / returns.std()
    return returns
