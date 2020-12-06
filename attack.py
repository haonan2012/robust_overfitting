import torch


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class Args_pgd():
    """attack args."""
    def __init__(self, attack='pgd', epsilon=8 / 255.0, alpha=2 / 255.0, lower_limit=0.0, upper_limit=1.0,
                 attack_init='rand', norm='l_inf', attack_iters=10, restarts=1, early_stop=False):
        self.attack = attack
        self.epsilon = epsilon
        self.alpha = alpha
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.attack_init = attack_init
        self.norm = norm
        self.attack_iters = attack_iters
        self.restarts = restarts
        self.early_stop = early_stop

    def check_args_value(self):
        # print(self.attack, self.epsilon, self.alpha, self.lower_limit, self.upper_limit)
        if self.attack == 'pgd':
            print(self.attack_init, self.norm, self.attack_iters, self.restarts, self.early_stop)


def set_param_grad(model, bool):
    for p in model.parameters():
        p.requires_grad_(bool)


def attack_fgsm(model, X, y, criterion, args):
    '''attack_args: epsilon, alpha, lower_limit, upper_limit
    '''
    delta = torch.zeros_like(X)
    for i in range(len(args.epsilon)):
        delta[:, i, :, :].uniform_(-args.epsilon[i][0][0], args.epsilon[i][0][0])
    delta.data = clamp(delta, 0 - X, 1 - X)
    delta.requires_grad = True
    output = model(X + delta)
    loss = criterion(output, y)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = clamp(delta.data + args.alpha * torch.sign(grad), -args.epsilon, args.epsilon)
    delta.data = clamp(delta, args.lower_limit - X, args.upper_limit - X)
    delta = delta.detach()
    return delta


def attack_pgd(model, X, y, criterion, args):
    '''attack_args: epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit,
                    attack_init='rand', norm='l_inf', early_stop=False
    '''
    set_param_grad(model, False)

    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X)

    for _ in range(args.restarts):
        delta = torch.zeros_like(X)
        if args.attack_init == 'rand':
            # print('rand init delta!')
            for i in range(len(args.epsilon)):
                delta[:, i, :, :].uniform_(-args.epsilon[i][0][0], args.epsilon[i][0][0])

        delta = clamp(delta, args.lower_limit - X, args.upper_limit - X)
        delta.requires_grad_(True)
        for _ in range(args.attack_iters):
            output = model(X + delta)
            if args.early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            loss = criterion(output, y)
            loss.backward()

            g = delta.grad[index, :, :, :]
            d = delta.data[index, :, :, :]
            x = X[index, :, :, :]

            d = clamp(d + args.alpha * torch.sign(g), -args.epsilon, args.epsilon)
            d = clamp(d, args.lower_limit - x, args.upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        with torch.no_grad():
            all_loss = criterion(model(X + delta), y, reduction='none')
            max_idx = all_loss >= max_loss
            max_delta[max_idx] = delta[max_idx]
            max_loss[max_idx] = all_loss[max_idx]
    set_param_grad(model, True)
    return max_delta


def attack_pgd_my(model, X, y, criterion, args):
    '''attack_args: epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit,
                    attack_init='rand', norm='l_inf', early_stop=False
    '''
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X)
    channel = len(args.epsilon)

    for _ in range(args.restarts):
        delta = torch.zeros_like(X)
        if args.attack_init == 'rand':
            # print('rand init delta!')
            for i in range(channel):
                delta[:, i, :, :].uniform_(-args.epsilon[i][0][0], args.epsilon[i][0][0])
        X_pert = X + delta
        for i in range(channel):
            X_pert[:, i, :, :] = torch.clamp(X_pert[:, i, :, :], args.lower_limit[i][0][0], args.upper_limit[i][0][0])
        X_pert.requires_grad_(True)
        for _ in range(args.attack_iters):
            output = model(X_pert)
            if args.early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            loss = criterion(output, y)
            loss.backward()

            g = X_pert.grad.detach()[index, :, :, :]
            x = X_pert.data[index, :, :, :]

            x = x + args.alpha * torch.sign(g)
            for i in range(channel):
                x[:, i, :, :] = torch.clamp(x[:, i, :, :], args.lower_limit[i][0][0], args.upper_limit[i][0][0])

            d = x - X[index, :, :, :]
            for i in range(channel):
                d[:, i, :, :] = torch.clamp(d[:, i, :, :], -args.epsilon[i][0][0], args.epsilon[i][0][0])

            X_pert.data[index, :, :, :] = d + X[index, :, :, :]
            X_pert.grad.zero_()
        X_pert = X_pert.detach()
        all_loss = criterion(model(X_pert), y, reduction='none')
        max_idx = all_loss >= max_loss
        max_delta[max_idx] = X_pert[max_idx]
        max_loss[max_idx] = all_loss[max_idx]
    return max_delta - X


def attack_pgd_my_1(model, X, y, criterion, args):
    '''attack_args: epsilon, alpha, attack_iters, restarts, lower_limit, upper_limit,
                    attack_init='rand', norm='l_inf', early_stop=False
    '''
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X)
    channel = len(args.epsilon)

    for _ in range(args.restarts):
        delta = torch.zeros_like(X)
        if args.attack_init == 'rand':
            # print('rand init delta!')
            for i in range(channel):
                delta[:, i, :, :].uniform_(-args.epsilon[i][0][0], args.epsilon[i][0][0])
        X_pert = X + delta
        for i in range(channel):
            X_pert[:, i, :, :] = torch.clamp(X_pert[:, i, :, :], args.lower_limit[i][0][0], args.upper_limit[i][0][0])
        for _ in range(args.attack_iters):
            X_pert.requires_grad_(True)

            output = model(X_pert)
            if args.early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            loss = criterion(output, y)
            loss.backward()

            X_grad_ = X_pert.grad.data
            X_grad = torch.zeros_like(X)
            X_grad[index] = X_grad_[index] + X_grad[index]

            X_pert = X_pert.detach()
            X_pert += args.alpha * X_grad.sign()

            d = X_pert - X

            for i in range(channel):
                d[:, i, :, :] = torch.clamp(d[:, i, :, :], -args.epsilon[i][0][0], args.epsilon[i][0][0])

            X_pert = X + d

            for i in range(channel):
                X_pert[:, i, :, :] = torch.clamp(X_pert[:, i, :, :], args.lower_limit[i][0][0], args.upper_limit[i][0][0])

        all_loss = criterion(model(X_pert), y, reduction='none')
        max_idx = all_loss >= max_loss
        max_delta[max_idx] = X_pert[max_idx]
        max_loss[max_idx] = all_loss[max_idx]
    return max_delta - X


def attack_pgd_01(model, X, y, criterion, args):
    '''from https://github.com/locuslab/fast_adversarial
    '''
    epsilon, alpha, attack_iters, restarts, early_stop = args.epsilon, args.alpha, args.attack_iters, args.restarts, args.early_stop
    attack_init, lower_limit, upper_limit = args.attack_init, args.lower_limit, args. upper_limit
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X)
        if attack_init == 'rand':
            # print('rand init delta!')
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)

            if early_stop:
                # print("with early stop")
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            loss = criterion(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = clamp(delta.detach() + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[index] = d[index]
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)

            delta.grad.zero_()
        all_loss = criterion(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def attack_pgd_02(model, X, y, criterion, args):
    '''from https://github.com/locuslab/fast_adversarial
    '''
    epsilon, alpha, attack_iters, restarts, early_stop = args.epsilon, args.alpha, args.attack_iters, args.restarts, args.early_stop
    attack_init, lower_limit, upper_limit = args.attack_init, args.lower_limit, args. upper_limit
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X)
        if attack_init == 'rand':
            # print('rand init delta!')
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)

            if early_stop:
                # print("with early stop")
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            loss = criterion(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index, :, :, :], upper_limit - X[index, :, :, :])
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = criterion(model(X + delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def attack_pgd_03(model, X, y, criterion, args):
    epsilon, alpha, attack_iters, restarts, early_stop = args.epsilon, args.alpha, args.attack_iters, args.restarts, args.early_stop
    attack_init = args.attack_init

    max_loss = torch.zeros(y.shape[0]).cuda()
    max_pert = torch.zeros_like(X)
    for _ in range(restarts):
        delta = torch.zeros_like(X)
        if attack_init == 'rand':
            # print('rand init delta!')
            delta = delta.uniform_(-epsilon, epsilon)
        X_pert = torch.clamp(X + delta, 0, 1.0)

        for _ in range(attack_iters):
            X_pert.requires_grad = True
            output = model(X_pert)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break

            loss = criterion(output, y)
            loss.backward()

            X_grad_ = X_pert.grad.data
            X_grad = torch.zeros_like(X)
            X_grad[index] = X_grad_[index] + X_grad[index]

            X_pert = X_pert.detach()
            X_pert += alpha * X_grad.sign()
            d = torch.clamp(X_pert - X, -epsilon, epsilon)
            X_pert = X + d
            X_pert = X_pert.clamp(0, 1.0)


        all_loss = criterion(model(X_pert), y, reduction='none')
        max_pert[all_loss >= max_loss] = X_pert[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_pert - X