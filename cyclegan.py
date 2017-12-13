import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torchvision.utils as utils
import torch.nn.functional as F
import torch.utils as tutils
from torchvision import datasets
from torchvision import transforms
import numpy as np

IM_SIZE = 32
MODEL_PATH = 'models/'
SAMPLE_PATH = 'samples/'
MNIST_PATH = 'mnist/'
SVHN_PATH = 'svhn/'
svhn_loader = svhn_loader
mnist_loader = mnist_loader
G_CONV = 64
D_CONV = 64
N_ITER = 25000
L_RATE = 0.0002
sample_path = SAMPLE_PATH
model_path = MODEL_PATH

def get_training_generators(config):
    transform = transforms.Compose([
                    transforms.Scale(IM_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    svhn_generator = tutils.data.DataLoader(dataset=datasets.SVHN(root=SVHN_PATH, download=True, transform=transform),
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=1)

    mnist_generator = tutils.data.DataLoader(dataset=datasets.MNIST(root=MNIST_PATH, download=True, transform=transform),
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=1)
    return svhn_generator, mnist_generator


# custom layers inspired by yunjey
def deconv(c_in, c_out, k_size, stride=2, pad=1):
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class G_SVHN_MNIST(nn.Module):
    def __init__(self):
        super(G_SVHN_MNIST, self).__init__()
        cv1 = conv(1, 64, 4)
        cv2 = conv(64, 128, 4)
        conv3 = conv(128, 128, 3, 1, 1)
        conv4 = conv(128, 128, 3, 1, 1)
        decv1 = deconv(128, 64, 4)
        decv2 = deconv(64, 3, 4)

    def forward(self, x):
        # custom convs
        S = 0.05
        x = F.leaky_relu(cv1(x), S)
        x = F.leaky_relu(cv2(out), S)
        x = F.leaky_relu(conv3(out), S)
        x = F.leaky_relu(conv4(out), S)
        x = F.leaky_relu(decv1(out), S)
        x = F.tanh(decv2(x))
        return x

class G_MNIST_SVHN(nn.Module):
    def __init__(self):
        super(G_MNIST_SVHN, self).__init__()
        cv1 = conv(3, 64, 4)
        cv2 = conv(64, 128, 4)
        conv3 = conv(128, 128, 3, 1, 1)
        conv4 = conv(128, 128, 3, 1, 1)
        decv1 = deconv(128, 64, 4)
        decv2 = deconv(64, 1, 4)

    def forward(self, x):
        # custom convs
        S = 0.05
        x = F.leaky_relu(cv1(x), S)
        x = F.leaky_relu(cv2(out), S)
        x = F.leaky_relu(conv3(out), S)
        x = F.leaky_relu(conv4(out), S)
        x = F.leaky_relu(decv1(out), S)
        x = F.tanh(decv2(x))
        return x

class D_MNIST(nn.Module):
    def __init__(self, conv_dim=64, use_labels=False):
        super(D_MNIST, self).__init__()
        cv1 = conv(1, 64, 4)
        cv2 = conv(64, 128, 4)
        conv3 = conv(128, 256, 4)
        fc = conv(256 1, 4, 1, 0)

    def forward(self, x):
        S = 0.05
        x = F.leaky_relu(cv1(x), S)
        x = F.leaky_relu(cv2(out), S)
        x = F.leaky_relu(conv3(out), S)
        x = fc(x).squeeze()
        return x

class D_SVHN(nn.Module):
    def __init__(self):
        super(D_SVHN, self).__init__()
        cv1 = conv(3, 64, 4)
        cv2 = conv(64, 128, 4)
        conv3 = conv(128, 156, 4)
        fc = conv(256, 1, 4, 1, 0)

    def forward(self, x):
        S = 0.05
        x = F.leaky_relu(cv1(x), S)
        x = F.leaky_relu(cv2(x), S)
        x = F.leaky_relu(conv3(x), S)
        x = fc(x).squeeze()
        return out

# merge image code from yunjey
def merge_images(self, sources, targets, k=10):
    _, _, h, w = sources.shape
    row = int(np.sqrt(64))
    merged = np.zeros([3, row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    return merged.transpose(1, 2, 0)

def to_var(self, x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(self, x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

svhn_loader, mnist_loader = get_training_generators(config)
g_svhn_mnist = G_SVHN_MNIST()
g_svhn_mnist = G_MNIST_SVHN()
d_mnist = D_MNIST()
d_svhn = D_SVHN()
g_optim = optim.Adam(list(g_svhn_mnist.parameters()) + list(g_svhn_mnist.parameters()), L_RATE, [0.5, 0.99])
d_optim = optim.Adam(list(d_mnist.parameters()) + list(d_svhn.parameters()), L_RATE, [0.5, 0.99])

if torch.cuda.is_available():
    g_svhn_mnist.cuda()
    g_svhn_mnist.cuda()
    d_mnist.cuda()
    d_svhn.cuda()
svhn_iter = iter(svhn_loader)
mnist_iter = iter(mnist_loader)
iter_per_epoch = min(len(svhn_iter), len(mnist_iter))
svhn_static = to_var(svhn_iter.next()[0])
mnist_static = to_var(mnist_iter.next()[0])
for i in range(N_ITER+1):
    if (i+1) % iter_per_epoch == 0:
        mnist_iter = iter(mnist_loader)
        svhn_iter = iter(svhn_loader)
    svhn = svhn_iter.next()
    svhn = to_var(svhn)
    mnist, _ = mnist_iter.next()
    mnist = to_var(mnist)
    g_optim.zero_grad()
    d_optim.zero_grad()
    out = d_mnist(mnist)
    d_mnist_loss = torch.mean((out-1)**2)
    out = d_svhn(svhn)
    d_svhn_loss = torch.mean((out-1)**2)
    d_mnist_loss = d_mnist_loss
    d_svhn_loss = d_svhn_loss
    d_real_loss = d_mnist_loss + d_svhn_loss
    d_real_loss.backward()
    d_optim.step()
    g_optim.zero_grad()
    d_optim.zero_grad()
    svhn_generated = g_svhn_mnist(mnist)
    out = d_svhn(svhn_generated)
    d_svhn_loss = torch.mean(out**2)
    mnist_generated = g_svhn_mnist(svhn)
    out = d_mnist(mnist_generated)
    d_mnist_loss = torch.mean(out**2)
    d_fake_loss = d_mnist_loss + d_svhn_loss
    d_fake_loss.backward()
    d_optim.step()
    g_optim.zero_grad()
    d_optim.zero_grad()
    svhn_generated = g_svhn_mnist(mnist)
    out = d_svhn(svhn_generated)
    reconstructed_mnist = g_svhn_mnist(svhn_generated)
    g_loss = torch.mean((out-1)**2)
    g_loss += torch.mean((mnist - reconstructed_mnist)**2)
    g_loss.backward()
    g_optim.step()
    g_optim.zero_grad()
    d_optim.zero_grad()
    mnist_generated = g_svhn_mnist(svhn)
    out = d_mnist(mnist_generated)
    reconst_svhn = g_svhn_mnist(mnist_generated)
    g_loss = torch.mean((out-1)**2)
    g_loss += torch.mean((svhn - reconst_svhn)**2)
    g_loss.backward()
    g_optim.step()
    if i % 500 == 0:
        print('Saving outputs')
        svhn_generated = g_svhn_mnist(mnist_static)
        mnist_generated = g_svhn_mnist(svhn_static)
        mnist, mnist_generated = to_data(mnist_static), to_data(mnist_generated)
        svhn , svhn_generated = to_data(svhn_static), to_data(svhn_generated)
        merged = merge_images(mnist, svhn_generated)
        utils.save_image(merged, SAMPLE_PATH + 'output_m_to_s_' + str(i+1) + '.jpg')
        merged = merge_images(svhn, mnist_generated)
        utils.save_image(merged, SAMPLE_PATH + 'output_s_to_m' + str(i+1) + '.jpg')
    if i % 10 == 0:
        print("Step", i, ", Loss:", d_real_loss.data[0])
