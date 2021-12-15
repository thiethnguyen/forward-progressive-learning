import numpy as np
import time
import torch
import torch.nn.functional as f
import math
import my_functional as mf
import os

DEVICE_ = ['cuda' if torch.cuda.is_available() else 'cpu']
FOLDER = 'saved_models'


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def data_randomize(data, classes):
    idx = np.random.permutation(data.size()[0])
    x, y = data[idx], classes[idx]
    return x, y


def gain_schedule_old(loop, j):
    gain = 1
    if loop > 1:
        if j >= math.ceil(loop / 2):
            gain = 1 / 2
        if j >= math.ceil(3 * loop / 4) and loop > 4:
            gain = 1 / 4
        if j >= loop - 2 and loop > 5:
            gain = 1 / 20
        if j == loop - 1 and loop > 8:
            gain = 1 / 200
    return gain


def gain_schedule(loop, j):
    gain = 1
    if j >= math.ceil(loop / 2) and loop > 1:
        gain = 1 / 2
    if j >= math.ceil(3 * loop / 4) and loop > 3:
        gain = 1 / 4
    if j >= loop - 2 and loop > 11:
        gain = 1 / 10
    if j == loop - 1 and loop > 12:
        gain = 1 / 50
    return gain


def my_data_loader(dataset=None, batch_size=300, shuffle=False):
    if dataset is None:
        dataset = [None, None]
    # print('shapes are:', np.shape(x1), np.shape(x2))
    shape_in = np.shape(dataset[0])
    shape_out = np.shape(dataset[1])
    if shuffle:
        print('shuffle')
        rand = np.random.permutation(shape_in[0])
    else:
        print('no_shuffle')
        rand = range(shape_in[0])
    no_batch = math.ceil(shape_in[0] / batch_size)
    data_out = []
    for i in range(no_batch):
        if (i + 1) * batch_size <= shape_in[0]:
            in_images = np.zeros((batch_size, shape_in[1], shape_in[2], shape_in[3]))
            out_labels = np.zeros((batch_size, shape_out[1]))
        else:
            print(i, i * batch_size)
            in_images = np.zeros((shape_in[0] - i * batch_size, shape_in[1], shape_in[2], shape_in[3]))
            out_labels = np.zeros((shape_in[0] - i * batch_size, shape_out[1]))
        for j in range(batch_size):
            # print(i, j, batch_size*i + j )
            if batch_size * i + j < shape_in[0]:
                in_images[j] = dataset[0][rand[batch_size * i + j]]
                out_labels[j] = dataset[1][rand[batch_size * i + j]]
        in_images = torch.from_numpy(in_images)
        in_images = in_images.permute(0, 3, 1, 2)
        out_labels = torch.from_numpy(out_labels)
        data_out.append([in_images, out_labels])
    return data_out


def create_matrix_x(x, _filter, stride, pad):
    shape_filter = _filter.shape
    matrix_x = f.unfold(x, (shape_filter[2], shape_filter[3]), stride=stride, padding=pad)
    return matrix_x


def pool_backward_error(out_err, kernel=2, method='Ave'):
    in_error = 0
    if method == 'Ave':
        in_error = torch.repeat_interleave(torch.repeat_interleave(out_err, kernel, dim=2), kernel, dim=3)
    return in_error


def sum_condition_cnn(lm=0., in_matrix=None, fc_w=None, dot_value=None, fil_w=None, pool_layer='max',
                      pool_ind=None):
    sum2 = 0
    nf, _ = fil_w.shape
    phi_s, _ = dot_value.shape
    size_phij = phi_s // nf
    fc_out, fc_size = fc_w.shape
    size_fc_wj = fc_size // nf
    for j in range(nf):
        phij = dot_value[j * size_phij:(j + 1) * size_phij]
        fc_wj = fc_w[:, j * size_fc_wj:(j + 1) * size_fc_wj]
        Pj = 0
        if pool_layer:
            if pool_layer == 'avg':
                Pj = lm * phij * torch.t(fc_wj)
            elif pool_layer == 'max':
                fc_wj_pool_out = fc_wj
                temp = fc_wj_pool_out.shape
                fc_wj_pool_out = torch.reshape(fc_wj_pool_out,
                                               [1, fc_out, int(math.sqrt(temp[1])), int(math.sqrt(temp[1]))])
                pool_ind_ = pool_ind[:, j:j + 1, :, :]
                pool_ind_ = pool_ind_.repeat(1, fc_out, 1, 1)
                fc_wj_conv_out = f.max_unpool2d(fc_wj_pool_out, pool_ind_, 2)
                temp = fc_wj_conv_out.shape
                fc_w_j_ = torch.reshape(fc_wj_conv_out, [fc_out, temp[2] * temp[3]])
                Pj = lm * phij * torch.t(fc_w_j_)
        else:
            Pj = lm * phij * torch.t(fc_wj)
        sum2 = sum2 + torch.t(Pj) @ torch.t(in_matrix) @ in_matrix @ Pj
    return sum2


def inc_solve_2_layer_conv_fc(epoch_no, batch_no, in_image, out_image, pool_layer='max',
                              fil=None, fc_wei=None, fun_front=f.relu, fun_after=mf.my_identity,
                              loop=1, mix=False, stride=1, pad=0, gain=0.01, auto=True):
    out_shape = out_image.shape
    out_image = torch.reshape(out_image, [out_shape[0], out_shape[1], 1]).to(DEVICE_[0])

    shape_filter = fil.shape
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights]).to(DEVICE_[0])
    fc_w = fc_wei.to(DEVICE_[0])

    lm = gain
    pool_ind = None
    pool_out = None
    if mix:
        pass
        input()
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])
    for j in range(loop):  # Each epoch
        if batch_no == 0:
            print('= loop ', lm, ' =')
        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])

        for i in range(out_shape[0]):  # sb[0]: number of data samples (output)
            fil_w_new = fil_w
            fc_w_new = fc_w
            in_matrix = create_matrix_x(in_image[i:i + 1].to(DEVICE_[0]), fil, stride, pad)[0]
            fc_out = out_image[i]
            conv_act = fil_w_new @ in_matrix
            conv_out = fun_front(conv_act)

            conv_out_shape = conv_out.shape
            conv_flat_shape = [conv_out_shape[0] * conv_out_shape[1], 1]
            conv_act_flat = torch.reshape(conv_act, conv_flat_shape)
            conv_out = torch.reshape(conv_out, [1, conv_out_shape[0], int(math.sqrt(conv_out_shape[1])),
                                                int(math.sqrt(conv_out_shape[1]))])

            # Apply     pooling        layer
            if pool_layer:
                if pool_layer == 'avg':
                    pool_out = f.avg_pool2d(conv_out, 2, 2)
                elif pool_layer == 'max':
                    pool_out, pool_ind = f.max_pool2d(conv_out, 2, stride=2, return_indices=True)
                pool_out_shape = pool_out.shape
                pool_flatten_shape = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3] * pool_out_shape[0]
                fc_in = torch.reshape(pool_out, [pool_flatten_shape, 1])
            else:
                fc_in = torch.reshape(conv_out, conv_flat_shape)
            y_ = fun_after(fc_w_new @ fc_in)
            e_ = fc_out - y_

            # Backpropagation  to  flattening & pooling   layer
            e_fc_in = torch.t(fc_w_new) @ e_
            if pool_layer:
                e_pool_out = torch.reshape(e_fc_in, pool_out_shape)
                # Backpropagation   to   conv     layer
                if pool_layer == 'avg':
                    e_conv_out = pool_backward_error(e_pool_out, 2)
                elif pool_layer == 'max':
                    e_conv_out = f.max_unpool2d(e_pool_out, pool_ind, 2)
                e_conv_out = torch.reshape(e_conv_out, conv_flat_shape)
            else:
                e_conv_out = torch.reshape(e_fc_in, conv_flat_shape)
            dot_value = mf.derivative_fun(fun_front)(conv_act_flat)
            dot_value = dot_value.reshape(conv_flat_shape)
            e_conv_flat = dot_value * e_conv_out
            e_conv = torch.reshape(e_conv_flat, conv_out_shape)

            if auto:
                sum1 = torch.diagflat(
                    (2.0 * lm / mf.fun_max_derivative(fun_after)
                     - alpha_w * torch.sum(conv_act_flat ** 2) * lm ** 2)
                    * torch.ones(out_shape[1], 1)).to(DEVICE_[0])
                sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new, pool_layer=pool_layer,
                                                   pool_ind=pool_ind)
                lr_con = sum1 - sum2
                eig_values, _ = torch.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any(): 
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    print(alpha_v)
                    sum1 = torch.diagflat(
                        (2.0 * lm / mf.fun_max_derivative(fun_after)
                         - alpha_w * torch.sum(conv_act_flat ** 2) * lm ** 2)
                        * torch.ones(out_shape[1], 1)).to(DEVICE_[0])
                    sum2 = alpha_v * sum_condition_cnn(lm, in_matrix, fc_w_new, dot_value, fil_w_new,
                                                       pool_layer=pool_layer, pool_ind=pool_ind)
                    lr_con = sum1 - sum2
                    eig_values, _ = torch.eig(lr_con)
            fc_w = fc_w_new + alpha_w * lm * e_ @ torch.t(fc_in)
            fil_w = fil_w_new + alpha_v * lm * e_conv @ torch.t(in_matrix)

    fil_w = torch.reshape(fil_w, shape_filter)
    fc_w = fc_w
    lr = alpha_v * lm
    return fil_w, fc_w, alpha_v, lr.item()


def inc_train_2_layer(model, train_loader, test_loader, pool_layer='max', epochs=2000, gain=0.01, auto=True,
                      true_for=1, model_name='_0', avg_N = 0):
    if pool_layer:
        print(pool_layer, 'pooling')
        indx = -4
    else:
        indx = -3
    w1 = model.get_weights_index(indx)
    curr_layer_front = model.layers[indx]
    w2 = model.get_weights_index(-1)
    curr_layer_after = model.layers[-1]
    print(w1.shape, w2.shape)
    t0 = time.time()
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        alpha_vw_min = 1
        acc_last_N_epochs = 0.
        acc_curr_N_epochs = 0.
        train_acc_last_N_epochs = 0.
        train_acc_curr_N_epochs = 0.
        k = 0
        for j in range(epochs):
            t1 = time.time()
            print('============== epoch', j + 1, '/', epochs, '=============')
            gain_rate = gain_schedule(epochs, j)
            gain_ = gain * gain_rate
            if j + 1 > true_for:
                auto = False
            else:
                alpha_vw_min = 1
            gain_adj = gain * alpha_vw_min
            if gain_ > gain_adj:
                gain_ = gain_adj
            for i, (x, y) in enumerate(train_loader):
                if (i + 1) % (len(train_loader) // 10) == 0:
                    print('=========== batch', i + 1, '/', len(train_loader), '==========')
                    print('time:', time.time() - t1)
                layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), indx)
                layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
                pad = curr_layer_front.padding
                stride = curr_layer_front.stride

                w1, w2, alpha_vw, lr = inc_solve_2_layer_conv_fc(j, i, layer_in, layer_tar, pool_layer=pool_layer,
                                                                 fil=w1, fc_wei=w2,
                                                                 fun_front=curr_layer_front.activations,
                                                                 fun_after=curr_layer_after.activations, loop=1,
                                                                 stride=stride, pad=pad, gain=gain_, auto=auto)
                if alpha_vw < alpha_vw_min:
                    alpha_vw_min = alpha_vw
                    print('alpha_vm min at epoch', j + 1, ', batch', i + 1, ':', alpha_vw_min)

            print('alpha_vm min of epoch', j + 1, ':', alpha_vw_min)
            model.set_weights_index(w1, indx)
            model.set_weights_index(w2, -1)
            acc_lst = model.evaluate_both(train_loader, test_loader)
            print(acc_lst)
            # save model
            if model_name != '_0':
                if acc_lst[1] > max_acc_test:
                    max_acc_test = acc_lst[1]
                    j_max_old = j_at_max
                    j_at_max = j
                model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)
            if avg_N > 0:
                if k < avg_N - 1:
                    acc_curr_N_epochs += acc_lst[1]/avg_N
                    train_acc_curr_N_epochs += acc_lst[0]/avg_N
                    k += 1
                else:
                    acc_curr_N_epochs += acc_lst[1]/avg_N
                    train_acc_curr_N_epochs += acc_lst[0]/avg_N
                    k = 0
                    if acc_curr_N_epochs < acc_last_N_epochs - 0.1:
                        print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
                        print(acc_curr_N_epochs, ' is less than ', acc_last_N_epochs)
                        acc_last_N_epochs = acc_curr_N_epochs
                        acc_curr_N_epochs = 0
                        if train_acc_curr_N_epochs > train_acc_last_N_epochs and train_acc_last_N_epochs != 0:
                            print(train_acc_curr_N_epochs, ' is more than ', train_acc_last_N_epochs)
                            print("!!!!!!!!!!!!!!!!! STOPPING !!!!!!!!!!!!!!!!!!!!!!")
                            break
                        print('But ', train_acc_curr_N_epochs, ' is also less than ', train_acc_last_N_epochs)
                        train_acc_last_N_epochs = train_acc_curr_N_epochs
                        train_acc_curr_N_epochs = 0
                    else:
                        print("!!!!!!!!!!!!!!!!! ", avg_N, " epochs !!!!!!!!!!!!!!!!!!!!!!")
                        print('curr avg acc value: ', acc_curr_N_epochs, '; last value: ', acc_last_N_epochs)
                        acc_last_N_epochs = acc_curr_N_epochs
                        acc_curr_N_epochs = 0
                        print('curr avg train acc value: ', train_acc_curr_N_epochs, '; last train value: ', train_acc_last_N_epochs)
                        train_acc_last_N_epochs = train_acc_curr_N_epochs
                        train_acc_curr_N_epochs = 0

            print('time for epoch', j + 1, '/', epochs, ':', time.time() - t1)
    print('time:', time.time() - t0)
    return w1, w2


def inc_solve_filter(batch, lin, in_images, out_images, fil, func, loop, ran_mix, gain_rate, pad):
    if lin:
        curr_inv_f = mf.inv_fun(func)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)

        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(func)

        out_images = mf.fun_cut(out_images, func)

    in_shape = in_images.shape
    out_shape = out_images.shape

    out_images = torch.reshape(out_images, [out_shape[0], out_shape[1], out_shape[2] * out_shape[3]])

    shape_filter = fil.shape
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights])

    matrix_x = create_matrix_x(in_images, fil, pad)

    gain = 1e-6  # old: 1e-4, after resizing to (224,224)--> 1e-5
    lr = gain  # * gain_rate

    if ran_mix:
        print('inc_solve_x_random_shuffle')
        matrix_x, out_images = data_randomize(matrix_x, out_images)

    for j in range(loop):

        if loop > 1:
            if j == math.ceil(loop / 2) + 1:
                lr = lr / 2
            if j == math.ceil(3 * loop / 4) + 1 & loop > 4:
                lr = lr / 2
            if j == loop - 1 & loop > 5:
                lr = lr / 5
            if j == loop & loop > 8:
                lr = lr / 10
        if batch == 0:
            if loop <= 20:
                print(['loop ', j + 1])
                print(lr)
            elif (j + 1) % (loop / 5) == 0:
                print(['loop ', j + 1])
                print(lr)

        for k in range(in_shape[0]):
            w_new = fil_w
            in_matrix = matrix_x[k, :, :]
            y_ = w_new @ in_matrix
            if ~lin:
                # print('nonlinear coming here')
                y_ = func(y_)
            e_ = torch.squeeze(out_images[k, :, :]) - y_
            fil_w = w_new + lr * e_ @ torch.t(in_matrix)

    x = torch.reshape(fil_w, shape_filter)

    return x


def inc_solve_fc(batch, lin, in_images, out_images, weight, func, loop, ranmix, gain_, gain_rate):
    if lin:
        curr_inv_f = mf.inv_fun(func)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)
        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(func)
        out_images = mf.fun_cut(out_images, func)

    out_shape = out_images.shape
    if gain_ < 0:
        max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
        gain = 1 / max_phi2 * gain_rate
    else:
        gain = gain_ * gain_rate
    lr = gain  # .to(DEVICE_[0])

    if ranmix:
        print('inc_solve_x_random_shuffle')
        in_images, out_images = data_randomize(in_images, out_images)
    for j in range(loop):
        # print('number of loop:', loop)
        lr = lr * gain_schedule(loop, j)
        if batch == 0:
            if loop <= 20:
                print(['loop ', j + 1])
                print(lr)
                print('maxphi', max_phi2, gain_rate)
            elif (j + 1) % (loop / 5) == 0:
                print(['loop ', j + 1])
                print(lr)

        for k in range(out_shape[0]):
            w_new = weight
            in_matrix = in_images[k:k + 1, :]
            y_ = w_new @ torch.t(in_matrix)
            if ~lin:
                # print('nonlinear coming here')
                y_ = func(y_)
            e_ = torch.t(out_images[k:k + 1, :]) - y_
            e_phi = e_ @ in_matrix
            weight = w_new + lr * e_phi

    return weight, lr.item()


def inc_train_filter(model, train_loader, test_loader, epoch=2, loop=10, ran_mix=False):
    weights = model.get_weights()
    w = weights[-1]
    print(model.evaluate_both(train_loader, test_loader))

    # no_batch, batch_list, s1, s2 = batch_create_size(x_train, y_train, batch_size, ranmix=True)
    with torch.no_grad():
        for j in range(epoch):
            print('============== epoch', j + 1, '/', epoch, '=============')
            gain_rate = gain_schedule(epoch, j)
            for i, (x, y) in enumerate(train_loader):
                print('=========== batch', i + 1, '/', len(train_loader), '==========')
                w = inc_solve_filter(i, 0, x.float(), y.float(), w, f.relu, loop, ran_mix, gain_rate)
            model.set_weights([w])
            print(model.evaluate_both(train_loader, test_loader))

    return w


def inc_train_1_layer(model, at_layer, train_loader, test_loader, epoch=20, loop=1, gain_=0.1,
                      ran_mix=False, model_name='_0'):
    curr_layer = model.layers[at_layer]
    w = model.get_weights_index(at_layer)

    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        for j in range(epoch):
            print('============== epoch', j + 1, '/', epoch, '=============')
            gain_rate = gain_schedule_old(epoch, j)
            for i, (x, y) in enumerate(train_loader):
                if i % (len(train_loader) // 10) == 0:
                    print('=========== batch', i + 1, '/', len(train_loader), '==========')
                layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), at_layer)
                layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
                layer_tar = model.backward_to_layer(layer_tar, model.no_layers - at_layer - 1)
                if curr_layer.name == 'conv':
                    pad = curr_layer.padding
                    w = inc_solve_filter(i, False, layer_in, layer_tar,
                                         w, curr_layer.activations, loop, ran_mix, gain_rate, pad)
                elif curr_layer.name == 'fc':
                    w, lr = inc_solve_fc(i, False, layer_in, layer_tar,
                                         w, curr_layer.activations, loop, ran_mix, gain_, gain_rate)

            model.set_weights_index(w, at_layer)
            acc_lst = model.evaluate_both(train_loader, test_loader)
            print(acc_lst)
            # save model
            if model_name != '_0':
                if acc_lst[1] > max_acc_test:
                    max_acc_test = acc_lst[1]
                    j_max_old = j_at_max
                    j_at_max = j
                model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 0)

    return w


def inverse_layerwise_training(model, train_loader, test_loader,
                               config, no_layers=1, epoch=1, loop=1, gain_=-1, mix_data=False, model_name='_0'):
    for i in range(no_layers):
        cur_inx = model.no_layers - i - 1
        # print('cur_inx is:', cur_inx)
        cur_lay = model.layers[cur_inx]
        print('First time: ', cur_lay.name)
        if cur_lay.name in ['conv', 'fc']:
            out_weight = inc_train_1_layer(model, cur_inx, train_loader, test_loader, epoch, loop, gain_,
                                           mix_data, model_name)
            model.set_weights_index(out_weight, cur_inx)
    for i in range(no_layers - 1):
        cur_inx = i + model.no_layers - no_layers + 1
        cur_lay = model.layers[cur_inx]
        print('Second time: ', cur_lay.name)
        if cur_lay.name in ['conv', 'fc']:
            out_weight = inc_train_1_layer(model, cur_inx, train_loader, test_loader, epoch, loop, gain_,
                                           mix_data, model_name)
            model.set_weights_index(out_weight, cur_inx)
    return 0


def inc_solve_2_fc_layer(model, train_loader, test_loader, batch, lin, in_images, out_images,
                         weight_v, weight_w, fun1, fun2, loop, mix_data, gain_rate=1.0, gain_=-1.0, auto=0):
    if lin:
        curr_inv_f = mf.inv_fun(fun2)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)
        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(fun2)
        out_images = mf.fun_cut(out_images, fun2)
    #     in_images = in_images.to(DEVICE_[0])

    out_shape = out_images.shape
    t0 = time.time()
    if gain_ == None or gain_ < 0:
        max_phi2 = max(in_images[i, :] @ torch.t(in_images[i, :]) for i in range(in_images.shape[0]))
        t1 = time.time()
        print('max time', t1 - t0, max_phi2)
        gain_ = 1 / max_phi2
    alpha_v = torch.tensor(1).to(DEVICE_[0])
    alpha_w = torch.tensor(1).to(DEVICE_[0])

    if mix_data:
        print('inc_solve_x_random_shuffle')
        in_images, out_images = data_randomize(in_images, out_images)
    lr_total = gain_
    for j in range(loop):
        print('number of loop:', loop)
        lr = gain_

        alpha_v = torch.tensor(1).to(DEVICE_[0])
        alpha_w = torch.tensor(1).to(DEVICE_[0])
        for k in range(out_shape[0]):
            v_new = weight_v
            w_new = weight_w
            in_matrix = in_images[k:k + 1, :]
            vx_ = v_new @ torch.t(in_matrix)
            phi = fun1(vx_)
            if ~lin:
                # print('nonlinear coming here')
                wa = w_new @ phi
                y_ = fun2(wa)
            else:
                y_ = w_new @ phi
            e_ = torch.t(out_images[k:k + 1, :]) - y_
            # print(e_.shape, phi.shape)

            dot_f1 = mf.derivative_fun(fun1)
            dot_a_vx = dot_f1(vx_)
            dot_a_vx_ = torch.squeeze(dot_a_vx)

            if auto:
                lr_con = torch.diagflat(
                    (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2) * torch.ones(
                        out_shape[1], 1)).to(
                    DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2) * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
                    w_new) * lr
                eig_values, _ = torch.eig(lr_con)
                while (eig_values[:, 0] < -0.005).any():
                    print('%d - %d', j, k)
                    alpha_v = alpha_v / 1.1
                    alpha_w = alpha_w / 1.1
                    print(alpha_v)
                    lr_con = torch.diagflat(
                        (2.0 * lr / mf.fun_max_derivative(fun2) - alpha_w * torch.sum(phi ** 2) * lr ** 2) * torch.ones(
                            out_shape[1], 1)).to(
                        DEVICE_[0]) - alpha_v * torch.sum(in_matrix ** 2) * lr * w_new * dot_a_vx_ ** 2 @ torch.t(
                        w_new) * lr
                    eig_values, _ = torch.eig(lr_con)

            if batch == 0 and k == 0:
                if loop <= 20:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))
                elif (j + 1) % (loop / 5) == 0:
                    print(['loop ', j + 1])
                    print('gain is (not include alpha)', lr * gain_rate * gain_schedule(loop, j))

            weight_w = w_new + alpha_w * (lr * gain_rate * gain_schedule(loop, j)) * e_ @ torch.t(phi)
            weight_v = v_new + alpha_v * (lr * gain_rate * gain_schedule(loop, j)) * \
                       (dot_a_vx * (torch.t(w_new) @ e_)) @ in_matrix

        if loop > 1:
            model.set_weights_index(weight_w, -1)
            model.set_weights_index(weight_v, -2)
            print(model.evaluate_both(train_loader, test_loader))
        lr_total = alpha_w * (lr * gain_rate * gain_schedule(loop, j))
    return weight_v, weight_w, alpha_w, lr_total.item()


def conv_train_2_fc_layer_last(model, train_loader, test_loader, epoch=20, loop=1, ran_mix=False, gain_=0.001,
                               auto=False, model_name='_0'):
    curr_layer_after = model.layers[-1]
    w = model.get_weights_index(-1)
    curr_layer_front = model.layers[-2]
    v = model.get_weights_index(-2)
    alpha_w = 0
    with torch.no_grad():
        max_acc_test = 0
        j_at_max = 0
        j_max_old = 0
        for j in range(epoch):
            print('============== epoch', j + 1, '/', epoch, '=============')
            gain_rate = gain_schedule(epoch, j)
            for i, (x, y) in enumerate(train_loader):
                print('=========== batch', i + 1, '/', len(train_loader), '==========')
                layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), -2)
                layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
                v, w, alpha_w_, lr = inc_solve_2_fc_layer(model, train_loader, test_loader, i, False, layer_in, layer_tar,
                                                      v, w, curr_layer_front.activations, curr_layer_after.activations,
                                                      loop, ran_mix, gain_rate=gain_rate, gain_=gain_, auto=auto)

            model.set_weights_index(w, -1)
            model.set_weights_index(v, -2)
            acc_lst = model.evaluate_both(train_loader, test_loader)
            print(acc_lst)
            # save model
            if model_name != '_0':
                if acc_lst[1] > max_acc_test:
                    max_acc_test = acc_lst[1]
                    j_max_old = j_at_max
                    j_at_max = j
                model.save_current_state(model_name, j, lr, acc_lst, j_at_max, j_max_old, 1)

    return w


def inc_train_1_layer_error_based(model, at_layer, train_loader, test_loader, pool=True, epoch=20, loop=1,
                                  ran_mix=False):
    curr_layer = model.layers[at_layer]
    w = model.get_weights_index(at_layer)

    with torch.no_grad():
        for j in range(epoch):
            print('============== epoch', j + 1, '/', epoch, '=============')
            gain_rate = gain_schedule(epoch, j)
            for i, (x, y) in enumerate(train_loader):
                if (i + 1) % (len(train_loader) // 10) == 0:
                    print('=========== batch', i + 1, '/', len(train_loader), '==========')
                layer_in = model.forward_to_layer(x.float().to(DEVICE_[0]), at_layer)
                layer_tar = one_hot_embedding(y.long(), model.no_outputs).to(DEVICE_[0]).float()
                if curr_layer.name == 'conv':
                    pad = curr_layer.padding
                    conv_layer = curr_layer
                    if pool:
                        fc_w = model.get_weights_index(at_layer + 3)
                        pool_layer = model.layers[at_layer + 1]
                        flat_layer = model.layers[at_layer + 2]
                        w = inc_solve_filter_error_based(i, False, layer_in, layer_tar, conv_layer, pool_layer,
                                                         flat_layer,
                                                         w, fc_w, curr_layer.activations, loop, ran_mix, gain_rate, pad)
                    else:
                        fc_w = model.get_weights_index(at_layer + 2)
                        pool_layer = False
                        flat_layer = model.layers[at_layer + 1]
                        w = inc_solve_filter_error_based(i, False, layer_in, layer_tar, conv_layer, pool_layer,
                                                         flat_layer,
                                                         w, fc_w, curr_layer.activations, loop, ran_mix, gain_rate, pad)
                elif curr_layer.name == 'fc':
                    w = inc_solve_fc(i, False, layer_in, layer_tar,
                                     w, curr_layer.activations, loop, ran_mix, -1, gain_rate)

            model.set_weights_index(w, at_layer)
            print(model.evaluate_both(train_loader, test_loader))

    return w


def inc_solve_filter_error_based(batch, lin, in_images, out_images, conv_layer, pool_layer, flat_layer,
                                 fil, fc_w, func, loop, ran_mix, gain_rate, pad):
    if lin:
        curr_inv_f = mf.inv_fun(func)
        if batch <= 0:
            print('Incremental LINEAR algorithm')
            print('Calculating inverse of the target, inverse function:')
            print(curr_inv_f)

        out_images = curr_inv_f(out_images)
    else:
        if batch == 0:
            print('Incremental NON-LINEAR algorithm')
            print(func)

        out_images = mf.fun_cut(out_images, func)

    in_shape = in_images.shape

    shape_filter = fil.shape
    no_fil_weights = shape_filter[1] * shape_filter[2] * shape_filter[3]
    no_fil_channels = shape_filter[0]
    fil_w = torch.reshape(fil, [no_fil_channels, no_fil_weights])

    matrix_x = create_matrix_x(in_images, fil, pad)

    gain = 1e-5
    lr = gain * gain_rate

    if ran_mix:
        print('inc_solve_x_random_shuffle')
        matrix_x, out_images = data_randomize(matrix_x, out_images)

    for j in range(loop):

        if loop > 1:
            if j == math.ceil(loop / 2) + 1:
                lr = lr / 2
            if j == math.ceil(3 * loop / 4) + 1 & loop > 4:
                lr = lr / 2
            if j == loop - 1 & loop > 5:
                lr = lr / 5
            if j == loop & loop > 8:
                lr = lr / 10
        if batch == 0:
            if loop <= 20:
                print(['loop ', j + 1])
                print(lr)
            elif (j + 1) % (loop / 5) == 0:
                print(['loop ', j + 1])
                print(lr)

        for k in range(in_shape[0]):
            w_old = fil_w
            in_matrix = matrix_x[k, :, :]
            y_fil = w_old @ in_matrix
            y_fil_ = y_fil.reshape(conv_layer.shape)
            if pool_layer:
                y_fil__ = f.avg_pool2d(y_fil_, 2, 2)
            else:
                y_fil__ = y_fil_
            y_fil___ = y_fil__.reshape(flat_layer.shape).unsqueeze(0)
            y_ = fc_w @ torch.t(y_fil___)

            if ~lin:
                y_ = func(y_)
            e_ = torch.t(out_images[k:k + 1, :]) - y_
            # Backward error
            e_bw = torch.t(fc_w) @ e_
            if pool_layer:
                e_bw_ = e_bw.reshape(pool_layer.shape).unsqueeze(0)
                e_bw__ = pool_backward_error(e_bw_, 2)
            else:
                e_bw__ = e_bw
            e_bw___ = e_bw__.reshape(conv_layer.shape[0], -1)
            fil_w = w_old + lr * e_bw___ @ torch.t(in_matrix)

    x = torch.reshape(fil_w, shape_filter)

    return x


def inverse_layerwise_training_error_based(model, train_loader, test_loader,
                                           config, pool=True, no_layers=1, epoch=1, loop=1, mix_data=False):
    for i in range(no_layers):
        cur_inx = model.no_layers - i - 1
        # print('cur_inx is:', cur_inx)
        cur_lay = model.layers[cur_inx]
        print('First time: ', cur_lay.name)
        if cur_lay.name in ['conv', 'fc']:
            out_weight = inc_train_1_layer_error_based(model, cur_inx, train_loader, test_loader, pool, epoch, loop,
                                                       mix_data)
            model.set_weights_index(out_weight, cur_inx)
    for i in range(no_layers - 2):
        cur_inx = i + model.no_layers - no_layers + 1
        cur_lay = model.layers[cur_inx]
        print('Second time: ', cur_lay.name)
        if cur_lay.name in ['conv', 'fc']:
            out_weight = inc_train_1_layer_error_based(model, cur_inx, train_loader, test_loader, pool, epoch, loop,
                                                       mix_data)
            model.set_weights_index(out_weight, cur_inx)
    return 0
