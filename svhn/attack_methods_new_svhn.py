import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
from torch import autograd
import utils
import math

from utils import softCrossEntropy
from utils import one_hot_tensor_svhn, label_smoothing
import ot
import pickle


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attack_None(nn.Module):
    def __init__(self, basic_net, config, discriminator):
        super(Attack_None, self).__init__()
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.basic_net = basic_net
        self.discriminator = discriminator
        print(config)

    def forward(self, inputs, targets, attack=None, batch_idx=-1):
        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()
        outputs, _, _ = self.basic_net(inputs)
        return outputs, None


class Attack_PGD(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, discriminator, attack_net=None):
        super(Attack_PGD, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.discriminator = discriminator
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction='none') if 'loss_func' not in config.keys(
            ) else config['loss_func']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_org = x.detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            loss = self.loss_func(logits, y_tensor_adv)
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]

        return logits_pert, targets_prob.detach()


class Attack_FeaScatter(nn.Module):
    def __init__(self, basic_net, config, discriminator, D_optimizer, attack_net=None):
        super(Attack_FeaScatter, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.discriminator = discriminator
        self.D_optimizer = D_optimizer
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']

        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs, _, _ = self.basic_net(inputs)
            return outputs, None
        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        aux_net.eval()
        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size


        logits, _, test_fea_nat = aux_net(inputs)
        num_classes = logits.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()

        x_org = x.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        alpha = torch.rand(x.size(0), 10, 1, 1)

        logits_pred_nat, fea_nat, _ = aux_net(inputs)
        logits_pred_nat_D = torch.reshape(logits_pred_nat, [x.size(0), 10, 1, 1])

        num_classes = logits_pred_nat.size(1)
        y_gt = one_hot_tensor_svhn(targets, num_classes, device)
        y_sm = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)

        loss_ce = softCrossEntropy()
        adversarial_criterion = nn.BCELoss()
        mse = nn.MSELoss()

        iter_num = self.num_steps
        ones_const = Variable(torch.ones(x.size(0), 1)/2).cuda()
        for i in range(iter_num):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred, fea, test_fea_adv = aux_net(x)
            logits_pred_D = torch.reshape(logits_pred, [x.size(0), 10, 1, 1])

            #ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,logits_pred, None, None, 0.01, m, n)
            #print(logits_pred_nat.shape)


            valid = Variable(torch.Tensor(np.ones((x.size(0), 1))), requires_grad=False).cuda()
            fake = Variable(torch.Tensor(np.zeros((x.size(0), 1))), requires_grad=False).cuda()


            self.discriminator.zero_grad()
            logits_fea_nat, D_cla_nat = self.discriminator(logits_pred_nat_D)
            #print(logits_fea_nat.shape)
            logits_fea_adv, D_cla_adv = self.discriminator(logits_pred_D)

            #print(logits_fea_adv)

            loss_real = adversarial_criterion(logits_fea_nat, valid)
            loss_fake = adversarial_criterion(logits_fea_adv, fake)
            D_cla_real = loss_ce(D_cla_nat, y_sm.detach())
            D_cla_fake = loss_ce(D_cla_adv, y_sm.detach())
            discriminator_loss = loss_real + loss_fake


            aux_net.zero_grad()
            #adv_loss = ot_loss
            adv_loss = discriminator_loss
            self.D_optimizer.zero_grad()
            self.discriminator.zero_grad()
            adv_loss.backward(retain_graph=True)
            x_adv = x.data - self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv,requires_grad=True)



            logits_pred_2, fea, test_fea_adv_2 = self.basic_net(x)
            logits_pred_2_D = torch.reshape(logits_pred_2, [x.size(0), 10, 1, 1])



            for i in range(2):
                #self.D_optimizer.zero_grad()
                logits_fea_adv2, D_cla_adv2 = self.discriminator(logits_pred_2_D.detach())
                logits_fea_nat2, D_cla_nat2 = self.discriminator(logits_pred_nat_D.detach())
                #print(test_fea_nat.shape)
                #print(logits_fea_adv2)
                #print(logits_fea_nat2)
                loss_real2 = adversarial_criterion(logits_fea_nat2, valid)
                loss_fake2 = adversarial_criterion(logits_fea_adv2, fake)
                D_cla_real2 = loss_ce(D_cla_nat2, y_sm.detach())
                D_cla_fake2 = loss_ce(D_cla_adv2, y_sm.detach()) 

                discriminator_loss = loss_real2 + loss_fake2 + D_cla_real2 + D_cla_fake2

                print('\n############D loss:',discriminator_loss.item(),'############\n')
                self.D_optimizer.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                self.D_optimizer.step()


            gan_loss = mse(logits_pred_2_D,logits_pred_nat_D)
            print('\n--------gan_loss:---------*\n', gan_loss)

            #mse_loss = mse(test_fea_nat, test_fea_adv_2)
            #print('\n***********MSE:',mse_loss.item(),'***********\n')
            #print(logits_fea_nat)
            correct_num_nat = np.sum(logits_fea_nat2.detach().cpu().numpy()>0.5)
            correct_num_adv = np.sum(logits_fea_adv2.detach().cpu().numpy()<0.5)
            correct_num = correct_num_adv + correct_num_nat
            print('\n--------correct_num:',correct_num/120,'---------*\n')

            self.basic_net.zero_grad()


            adv_loss = loss_ce(logits_pred_2, y_sm.detach())

            self.discriminator.zero_grad()

            gd_gan = torch.autograd.grad(outputs=gan_loss, inputs=x,retain_graph=True)
            gd_adv = torch.autograd.grad(outputs=adv_loss, inputs=x,retain_graph=True)
            #print('adv gradient: ', gd_adv[0].mean(), gd_adv[0].max(), gd_adv[0].min())
            #print('gan gradient: ', gd[0].mean(), gd[0].max(), gd[0].min())
            scale = gd_adv[0].max() / gd_gan[0].max()
            print(scale)
            #print('adv_loss:',torch.autograd.grad(adv_loss,x,retain_graph=True)[0].max(),torch.autograd.grad(adv_loss,x,retain_graph=True)[0].min())
            #print('gan loss:',torch.max(torch.autograd.grad(gan_loss,x,retain_graph=True)[0]),torch.min(torch.autograd.grad(gan_loss,x,retain_graph=True)[0]))



        return logits_pred, adv_loss, gan_loss, scale
