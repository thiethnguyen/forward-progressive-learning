import torch
import torch.nn as nn
import my_functional as mf
import os


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FOLDER = 'saved_models'


class MyLayer:
    def __init__(self, name, paras, stride=1, padding=0, bias=False, activations=mf.my_identity):
        self.name = name
        self.weights = 0
        self.param = paras
        self.shape = 0
        self.pre_shape = 0
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.activations = activations


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class MyCNN(nn.Module):

    def __init__(self, n_classes):
        super(MyCNN, self).__init__()
        self.no_layers = 0
        self.layers = []
        self.no_outputs = n_classes

    def add(self, layer):
        if layer.name == 'conv':
            layer.weights = nn.Conv2d(*layer.param, stride=layer.stride, padding=layer.padding, bias=layer.bias).to(DEVICE)
            layer.activations = layer.activations
        if layer.name == 'flat':
            layer.weights = Flatten()
        if layer.name == 'pool':
            layer.weights = nn.MaxPool2d(*layer.param)
        if layer.name == 'fc':
            layer.weights = nn.Linear(*layer.param, bias=layer.bias).to(DEVICE)
            layer.activations = layer.activations
        self.layers.append(layer)
        self.no_layers += 1


    def forward(self, x):
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                x = layer.activations(layer.weights(x))
            elif layer.name in ['flat', 'pool']:
                # print(layer.weights)
                x = layer.weights(x)
        return x

    def complete_net(self, train_loader):
        x = 0
        for X, y_true in train_loader:            
            x = X.float()[0:1].to(DEVICE)
#             print(x.shape, x)
#             input()
            break
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                print(layer.name)
                print(layer.activations, layer.weights)
                x = layer.activations(layer.weights(x))
                layer.shape = x.shape[1:]
                print(x.shape)
            elif layer.name in ['flat', 'pool']:
                print(layer.name)
                print(layer.weights)
                layer.pre_shape = x.shape[1:]
                x = layer.weights(x)
                layer.shape = x.shape[1:]
                print(x.shape)

        return x

    def forward_to_layer(self, x, to_lay):
        for layer in self.layers[0:to_lay]:
            if layer.name in ['conv', 'fc']:
                x = layer.activations(layer.weights(x))
            elif layer.name in ['flat', 'pool']:
                # print(layer.weights)
                x = layer.weights(x)
        return x

    def backward_to_layer(self, y, to_lay):
        for layer in self.layers[:-to_lay-1:-1]:
            if layer.name == 'conv':
                print('conv_backward here!')
                pass
            elif layer.name == 'fc':
                y = linear_backward(y, layer.weights.weight, layer.activations, False)
            elif layer.name == 'pool':
                y = pool_backward_error(y, kernel=2)
            elif layer.name == 'flat':
                y = torch.reshape(y, torch.Size([y.shape[0]]+list(layer.pre_shape)))
        return y

    def get_weights_2(self):
        w = []
        for param in self.parameters():
            print("he he he", param)
            w.append(param.data)
        return w

    def set_weights_2(self, w):
        i = 0
        for param in self.parameters():
            # print(param)
            param.data = w[i]
            i += 1

    def get_weights(self):
        w = []
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                w.append(layer.weights.weight.data)
            elif layer.name in ['pool', 'flat']:
                w.append(0)
        return w

    def set_weights(self, w):
        i = 0
        for layer in self.layers:
            if layer.name in ['conv', 'fc']:
                layer.weights.weight.data = w[i].to(DEVICE)
                i += 1
            elif layer.name in ['pool', 'flat']:
                i += 1

    def save_weights(self, path):
        w_list = self.get_weights()
        w_dict = {str(k): v for k, v in enumerate(w_list)}
        torch.save(w_dict, path)

    def load_weights(self, path):
        w_dict = torch.load(path)
        w_list = list(w_dict.values())
        self.set_weights(w_list)

    def get_weights_index(self, index):
        return self.layers[index].weights.weight.data

    def set_weights_index(self, w, index):
        self.layers[index].weights.weight.data = w
        
    def set_bias_index(self, w, index):
        self.layers[index].weights.bias = w

    def save_current_state(self, model_name, epoch, lr, acc_lst, j_max, j_max_old, two_lay=1):
        if two_lay == 0:
            weight_name = 'layer_wise'
        else:
            weight_name = 'two_layer'
        if epoch - 2 != j_max:
            old_path = FOLDER + '/' + model_name + '_' + weight_name + '_' + str(epoch - 2)
            if os.path.exists(old_path):
                os.remove(old_path)
        if j_max_old != j_max and j_max_old != epoch - 1:
            old_path = FOLDER + '/' + model_name + '_' + weight_name + '_' + str(j_max_old)
            if os.path.exists(old_path):
                os.remove(old_path)
        path = FOLDER + '/' + model_name+'_'+ weight_name + '_' + str(epoch)
        self.save_weights(path)
        filename = FOLDER + '/' + model_name + '.txt'
        with open(filename, 'a') as out:
            if two_lay:
                out.write('R' + '\t')
            else:
                out.write('L' + '\t')
            out.write(str(epoch) + '\t')
            out.write(str(lr) + '\t')
            out.write(str(acc_lst) + '\n')

    # evaluate a fit model
    def evaluate_train(self, train_loader):
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self(images.float())
                _, predicted = torch.max(outputs, 1)
                _, true_labels = torch.max(labels, 1)
                total_train += labels.size(0)
                correct_train += (predicted == true_labels).sum()

        print('Accuracy of the network on the 50000 training images: %d %%' % (
                100 * correct_train / total_train))
        return 100 * correct_train / total_train

    # evaluate a fit model
    def evaluate_both(self, train_loader, test_loader):
        correct_train = 0
        total_train = 0
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self(images.float())
                _, predicted = torch.max(outputs, 1)
#                 _, true_labels = torch.max(labels, 1)
                true_labels = labels
                total_train += labels.size(0)
                correct_train += (predicted == true_labels).sum()

        print('Accuracy of the network on the', total_train, 'training images',
              100 * float(correct_train) / total_train)

        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = self(images.float())
                _, predicted = torch.max(outputs, 1)
#                 _, true_labels = torch.max(labels, 1)
                true_labels = labels
                total_test += labels.size(0)
                correct_test += (predicted == true_labels).sum()

        print('Accuracy of the network on the', total_test, 'test images', 100 * float(correct_test) / total_test)
        return [100 * float(correct_train) / total_train, 100 * float(correct_test) / total_test]


# def linear_backward(target, weight, func):
#     inv_f = mf.inv_fun(func)
#     return inv_f(target) @ torch.t(torch.pinverse(weight))
def linear_backward(target, weight, func, nullspace=False):
    inv_f = mf.inv_fun(func)
    if not nullspace:
        print('no nullspace --', end=' ')
        return inv_f(target) @ torch.t(torch.pinverse(weight))
    else:
        print('v random --', end=' ')
        inv_tar = inv_f(target)
        n, _ = inv_tar.size()
        _, m = weight.size()
        I = torch.eye(m).to(DEVICE)
#         print('linear backward =====================---------------')
#         print(n, m)
        v = (1 - 2*torch.rand(n,m)).to(DEVICE)
        inv_weight = torch.pinverse(weight)
        return inv_tar @ torch.t(inv_weight) + v @ torch.t(I - inv_weight @ weight)


def pool_backward_error(target, kernel=2, method='Ave'):
    if method == 'Ave':
        return torch.repeat_interleave(torch.repeat_interleave(target, kernel, dim=2), kernel, dim=3)
    return 0
