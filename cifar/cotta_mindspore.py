from copy import deepcopy
import mindspore as ms
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
# import torch.jit

import PIL
import msadapter.torchvision.transforms as transforms
import my_transforms as my_transforms
from time import time
import logging


def get_tta_transforms(gaussian_std: float = 0.005, soft=False, clip_inputs=False):
    img_shape = (32, 32, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1 / 16, 1 / 16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9):
        super().__init__()
        self.model = model
        self.b_id = 0
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap

    def forward_fn(self, x):
        if self.episodic:
            self.reset()

        # x = gasuss_noise(x,mean=0, var=0.003)
        outputs = self.model(x)

        # Teacher Prediction
        anchor = self.model_anchor(x)
        anchor_prob = torch.nn.functional.softmax(anchor, dim=1).max(1)[0]

        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32
        outputs_emas = []
        lable_ema = torch.zeros(200, 10).cuda()
        for i in range(N):
            outputs_ = self.model_ema(self.transform(x)).detach()
            pre_lable = outputs_.argmax(axis=1).reshape(200, -1)

            lable_temp = torch.zeros(200, 10).cuda()
            lable_temp = lable_temp.scatter_(1, pre_lable, 1)
            lable_ema = lable_ema + lable_temp
            outputs_emas.append(outputs_)

        negLable_ema = (torch.ones(200, 10) * 32).cuda()
        negLable_ema = (negLable_ema - lable_ema) / 32

        if anchor_prob.mean(0) < self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema

        # 负学习
        pre_lable = outputs_ema.argmax(axis=1).reshape(200, -1)
        a = torch.ones(200, 10).cuda()
        pre_nlable = a.scatter_(1, pre_lable, 0)

        # t_ema = (outputs_ema_soft.scatter_(1,pre_lable,0)) #将最大置信度类别置0

        weight = softmax_entropy(outputs_ema, outputs_ema)
        weight = weight / torch.log2(torch.tensor(10))
        weight1 = torch.exp(-weight * 10)
        negWeight = torch.exp(-weight * 10)

        T = 2
        t_ce = torch.zeros(200, 10).cuda()
        # Taylor Cross Entropy
        for i in range(T):
            # temp = torch.pow((1 - (outputs_ema.softmax(1) * outputs.log_softmax(1))), i)/i
            temp = torch.pow(((outputs_ema.softmax(1) - outputs.softmax(1))), i) / i
            t_ce += temp

        t_ce = t_ce.sum(1)

        # print((-(outputs_ema.softmax(1) * outputs.log_softmax(1))).shape)

        # Student update
        # loss = (softmax_entropy(outputs, outputs_ema)).mean(0)
        # loss = (weight1 * softmax_entropy(outputs, outputs_ema)).mean(0) +  (negWeight * softmax_entropy(1-outputs, pre_nlable)).mean(0) #+ 0.1 * (softmax_entropy(outputs, pre_plable)).mean(0)
        loss = (weight1 * t_ce).mean(0) + 0.01 * (negWeight * softmax_entropy(1 - outputs, pre_nlable)).mean(0)
        # loss = (softmax_entropy(1-outputs, outputs_ema_soft)).mean(0)

        return loss, outputs_ema

    def forward(self, x):
        if self.episodic:
                self.reset()
            # steps 在配置文件里设置为 1
        grad_fn = ms.ops.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)

        # ----------------- 计算标签 ---------------
        outputs = self.model(x)

        # Teacher Prediction
        anchor = self.model_anchor(x)
        anchor_prob = torch.nn.functional.softmax(anchor, dim=1).max(1)[0]

        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32
        outputs_emas = []
        lable_ema = torch.zeros(200, 10).cuda()
        for i in range(N):
            outputs_ = self.model_ema(self.transform(x)).detach()
            pre_lable = outputs_.argmax(axis=1).reshape(200, -1)

            lable_temp = torch.zeros(200, 10).cuda()
            lable_temp = lable_temp.scatter_(1, pre_lable, 1)
            lable_ema = lable_ema + lable_temp
            outputs_emas.append(outputs_)

        negLable_ema = (torch.ones(200, 10) * 32).cuda()
        negLable_ema = (negLable_ema - lable_ema) / 32

        if anchor_prob.mean(0) < self.ap:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema

        # 负学习
        pre_lable = outputs_ema.argmax(axis=1).reshape(200, -1)
        a = torch.ones(200, 10).cuda()
        pre_nlable = a.scatter_(1, pre_lable, 0)
        # ------------------ 计算标签结束 -----------------

        def train_step(x, pre_nlable):
            (loss, _), grads = grad_fn(x, pre_nlable)
            self.optimizer(grads)
            return loss

        for _ in range(self.steps):
            res = train_step(x,pre_nlable)

        return outputs_ema
    # def forward(self, x):
    #     if self.episodic:
    #         self.reset()
    #     # steps 在配置文件里设置为 1
    #     for _ in range(self.steps):
    #         outputs = self.forward_and_adapt(x, self.model, self.optimizer)
    #
    #     return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    # @torch.enable_grad()  # ensure grads in possible no grad context for testing
    # 在整个项目中使用到这个类的地方，都是enable_grad()
    # def forward_and_adapt(self, x, model, optimizer):
    #     # x = gasuss_noise(x,mean=0, var=0.003)
    #     outputs = self.model(x)
    #
    #     # Teacher Prediction
    #     anchor = self.model_anchor(x)
    #     anchor_prob = torch.nn.functional.softmax(anchor, dim=1).max(1)[0]
    #
    #     standard_ema = self.model_ema(x)
    #     # Augmentation-averaged Prediction
    #     N = 32
    #     outputs_emas = []
    #     lable_ema = torch.zeros(200, 10).cuda()
    #     for i in range(N):
    #         outputs_ = self.model_ema(self.transform(x)).detach()
    #         pre_lable = outputs_.argmax(axis=1).reshape(200, -1)
    #
    #         lable_temp = torch.zeros(200, 10).cuda()
    #         lable_temp = lable_temp.scatter_(1, pre_lable, 1)
    #         lable_ema = lable_ema + lable_temp
    #         outputs_emas.append(outputs_)
    #
    #     negLable_ema = (torch.ones(200, 10) * 32).cuda()
    #     negLable_ema = (negLable_ema - lable_ema) / 32
    #
    #     if anchor_prob.mean(0) < self.ap:
    #         outputs_ema = torch.stack(outputs_emas).mean(0)
    #     else:
    #         outputs_ema = standard_ema
    #
    #     # 负学习
    #     pre_lable = outputs_ema.argmax(axis=1).reshape(200, -1)
    #     a = torch.ones(200, 10).cuda()
    #     pre_nlable = a.scatter_(1, pre_lable, 0)
    #
    #     # t_ema = (outputs_ema_soft.scatter_(1,pre_lable,0)) #将最大置信度类别置0
    #
    #     weight = softmax_entropy(outputs_ema, outputs_ema)
    #     weight = weight / torch.log2(torch.tensor(10))
    #     weight1 = torch.exp(-weight * 10)
    #     negWeight = torch.exp(-weight * 10)
    #
    #     T = 2
    #     t_ce = torch.zeros(200, 10).cuda()
    #     # Taylor Cross Entropy
    #     for i in range(T):
    #         # temp = torch.pow((1 - (outputs_ema.softmax(1) * outputs.log_softmax(1))), i)/i
    #         temp = torch.pow(((outputs_ema.softmax(1) - outputs.softmax(1))), i) / i
    #         t_ce += temp
    #
    #     t_ce = t_ce.sum(1)
    #
    #     # print((-(outputs_ema.softmax(1) * outputs.log_softmax(1))).shape)
    #
    #     # Student update
    #     # loss = (softmax_entropy(outputs, outputs_ema)).mean(0)
    #     # loss = (weight1 * softmax_entropy(outputs, outputs_ema)).mean(0) +  (negWeight * softmax_entropy(1-outputs, pre_nlable)).mean(0) #+ 0.1 * (softmax_entropy(outputs, pre_plable)).mean(0)
    #     loss = (weight1 * t_ce).mean(0) + 0.01 * (negWeight * softmax_entropy(1 - outputs, pre_nlable)).mean(0)
    #     # loss = (softmax_entropy(1-outputs, outputs_ema_soft)).mean(0)
    #
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     # Teacher update
    #     self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)
    #     # Stochastic restore
    #     if True:
    #         for nm, m in self.model.named_modules():
    #             for npp, p in m.named_parameters():
    #                 if npp in ['weight', 'bias'] and p.requires_grad:
    #                     mask = (torch.rand(p.shape) < self.rst).float().cuda()
    #                     with torch.no_grad():
    #                         p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
    #     return outputs_ema


# @torch.jit.script
def softmax_entropy(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect all trainable parameters.
    Walk the model's modules and collect all parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:  # isinstance(m, nn.BatchNorm2d): collect all
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            # m.running_mean = None
            # m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
