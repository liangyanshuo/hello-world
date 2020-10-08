import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, weight_arch, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.weight_arch = weight_arch
    self.optimizer = torch.optim.Adam(self.weight_arch.parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, input_id, eta, network_optimizer):
    loss = self.model._loss(input, target)
    weight = self.weight_arch.get_weight(input, input_id)
    loss = (weight * loss).sum()
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, input_train_id, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, input_train_id, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, input_train_id, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, input_train_id, eta, network_optimizer)
    unrolled_loss = (unrolled_model._loss(input_valid, target_valid)).mean()

    unrolled_loss.backward()
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(input_train, target_train, input_train_id, vector)

    for v, g in zip(self.weight_arch.parameters(), implicit_grads):
      if v.grad is None:
        v.grad = Variable(-eta * g.data)
      else:
        v.grad.data.copy_(-eta * g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, train_input, train_target, train_id, vector):
    weight_for_weight = []
    for i in range(len(train_input)):
      input = train_input[i].view(1, train_input.size(1), train_input.size(2), train_input.size(3))
      target = train_target[i].view(-1)
      loss = self.model._loss(input, target)
      self.model.zero_grad()
      loss.backward()
      product = 0.
      for p, q in zip(vector, self.model.parameters()):
        product += (p.data * q.grad.data).sum()
      weight_for_weight.append(product)

    weight_for_weight = torch.Tensor(weight_for_weight).float()
    weight = self.weight_arch.get_weight(train_input, train_id)
    grad_p = \
      torch.autograd.grad(torch.mm(weight_for_weight, weight), self.weight_arch.parameters())
    return grad_p

    # losses = self.model._loss(train_input, train_target)
    # for loss in losses:
    #   self.model.zero_grad()
    #   loss.backward()
    #   product = 0.
    #   for p, q in zip(vector, self.model.parameters()):
    #     product += (p.data * q.grad.data).sum()
    #   weight_for_weight.append(product)
    # weight_for_weight = torch.Tensor(weight_for_weight).float()
    # weight = self.weight_arch.get_weight(train_input, train_id)
    # grad_p = \
    #   torch.autograd.grad(torch.mm(weight_for_weight, weight), self.weight_arch.parameters())
    # return grad_p


    # R = r / _concat(vector).norm()
    # for p, v in zip(self.model.parameters(), vector):
    #   p.data.add_(R, v)
    # loss = self.model._loss(input, target)
    # grads_p = torch.autograd.grad(loss, self.model.arch_parameters())
    #
    # for p, v in zip(self.model.parameters(), vector):
    #   p.data.sub_(2*R, v)
    # loss = self.model._loss(input, target)
    # grads_n = torch.autograd.grad(loss, self.model.arch_parameters())
    #
    # for p, v in zip(self.model.parameters(), vector):
    #   p.data.add_(R, v)
    #
    # return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

