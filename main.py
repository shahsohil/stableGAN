from __future__ import print_function
import torch.nn.init as init
import argparse
import time
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from adamPre import AdamPre
from mogdata import generate_data_SingleBatch, loglikelihood

# TODO: Needed while running on server. Change the GUI accordingly.
plt.switch_backend('agg')

parser = argparse.ArgumentParser()

# Information regarding data input
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--modes', type=int, default=8, help='total number of gaussian modes to consider')
parser.add_argument('--radius', type=int, default=1, help='radius of circle with MoG')
parser.add_argument('--sigma', type=float, default=0.01, help='variance of gaussians, default=0.01')

# Information regarding network
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--nz', type=int, default=2, help='size of the latent z vector')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")

# Training/Optimizer information
parser.add_argument('--niter', type=int, default=50000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--pdhgGLookAhead', action='store_true', help='enables generator lookahead')
parser.add_argument('--pdhgDLookAhead', action='store_true', help='enables discriminator lookahead')
parser.add_argument('--GLRatio', type=float, default=1.0, help='scaling factor for lr of generator')
parser.add_argument('--DLRatio', type=float, default=1.0, help='scaling factor for lr of discriminator')

# Miscellaneous information
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--deviceID', type=int, help='deviceID', default=0)
parser.add_argument('--verbose', action='store_true', help='displays additional information')

# Options for visualization
parser.add_argument('--viz_every', type=int, default=10000, help='plotting visualization every few iteration')
parser.add_argument('--n_batches_viz', type=int, default=10, help='number of samples used for visualization')
parser.add_argument('--markerSize', type=float, help='input batch size')
parser.add_argument('--plotRealData', action='store_true', help='saves real samples')
parser.add_argument('--plotLoss', action='store_true', help='Enables plotting of loss function')

class _netG(nn.Module):
    def __init__(self,ngpu,nz,ngf):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Linear(nz, ngf),
            nn.Tanh(),
            nn.Linear(ngf, ngf),
            nn.Tanh(),
            nn.Linear(ngf, 2),
        )

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, ndf):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(2, ndf),
            nn.Tanh(),
            nn.Linear(ndf, ndf),
            nn.Tanh(),
            nn.Linear(ndf, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

def main():
    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
        torch.backends.cudnn.enabled = False
        print("torch.backends.cudnn.enabled is: ", torch.backends.cudnn.enabled)

    cudnn.benchmark = True

    if torch.cuda.is_available():
        ngpu = int(opt.ngpu)
        if not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        if int(opt.ngpu) > 0:
            print("WARNING: CUDA not available, cannot use --ngpu =", opt.ngpu)
        ngpu = 0

    # Initializing Generator and Discriminator Network
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)

    netG = _netG(ngpu,nz,ngf)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = _netD(ngpu,ndf)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()

    input = torch.FloatTensor(opt.batchSize, 2)
    noise = torch.FloatTensor(opt.batchSize, nz)
    fixed_noise = torch.FloatTensor(opt.batchSize * opt.n_batches_viz, nz).normal_(0, 1)

    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    # Flag for disabling prediction step in the first iterate
    firstTime = True

    # setup optimizer
    optimizerD = AdamPre(netD.parameters(), lr=opt.lr/opt.DLRatio, betas=(opt.beta1, 0.999), name='optD')
    optimizerG = AdamPre(netG.parameters(), lr=opt.lr/opt.GLRatio, betas=(opt.beta1, 0.999), name='optG')

    fs = []
    np_samples = []
    np_samples_real = []

    for i in range(opt.niter):
            if opt.verbose:
                c1 = time.clock()

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # sampling input batch
            real_cpu = generate_data_SingleBatch(num_mode=opt.modes, radius=opt.radius, center=(0, 0), sigma=opt.sigma,
                                                 batchSize=opt.batchSize)

            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(batch_size).fill_(real_label)

            netD.zero_grad()

            output = netD(input)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # Update the generator weights with prediction
            # We avoid update during the first iteration
            if not firstTime and opt.pdhgGLookAhead:
                optimizerG.stepLookAhead()

            # train with fake
            noise.data.resize_(batch_size, nz)
            noise.data.normal_(0, 1)
            label.data.resize_(batch_size)
            label.data.fill_(fake_label)

            fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()

            errD = errD_real + errD_fake
            optimizerD.step()

            # restore the previous (non-predicted) weights of Generator
            if not firstTime and opt.pdhgGLookAhead:
                optimizerG.restoreStepLookAhead()

            # Set the flag to false after the first iter
            firstTime = False

            ############################
            # (2) Update G network: maximize -log(1 - D(G(z)))
            ###########################
            # Update discriminator weights with prediction; restore after the generator update.
            if opt.pdhgDLookAhead:
                optimizerD.stepLookAhead()

            # Unlike DCGAN code, we use original loss for generator. Hence we fill fake labels.
            label.data.fill_(fake_label)

            netG.zero_grad()

            fake = netG(noise)
            output = netD(fake)
            errG = -criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            # restore back discriminator weights
            if opt.pdhgDLookAhead:
                optimizerD.restoreStepLookAhead()

            if opt.plotLoss:
                f = [errD.data[0], errG.data[0]]
                fs.append(f)

            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (i, opt.niter, errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

            if opt.verbose:
                print("itr=", i, "clock time elapsed=", time.clock() - c1)

            if i % opt.viz_every == 0 or i == opt.niter - 1:

                # save checkpoints
                torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.outf, i))
                torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.outf, i))

                tmp_cpu = ((netG(fixed_noise)).data).cpu().numpy()
                np_samples.append(tmp_cpu)

                fig = plt.figure(figsize=(5, 5))
                if opt.markerSize:
                    plt.scatter(tmp_cpu[:, 0], tmp_cpu[:, 1], c='g', edgecolor='none', s=opt.markerSize)
                else:
                    plt.scatter(tmp_cpu[:, 0], tmp_cpu[:, 1], c='g', edgecolor='none')

                plt.axis('off')
                plt.savefig('%s/MoG_Fake_withP_%03d.pdf' % (opt.outf, i))
                plt.close()

    if opt.plotRealData:
        real_cpu_temp = generate_data_SingleBatch(num_mode=opt.modes, radius=opt.radius, center=(0, 0), sigma=opt.sigma,
                                                  batchSize=opt.batchSize * opt.n_batches_viz)
        tmp_cpu = real_cpu_temp.numpy()
        np_samples_real.append(tmp_cpu)

        fig = plt.figure(figsize=(5, 5))
        if opt.markerSize:
            plt.scatter(tmp_cpu[:, 0], tmp_cpu[:, 1], c='g', edgecolor='none', s=opt.markerSize)  # green is ground truth
        else:
            plt.scatter(tmp_cpu[:, 0], tmp_cpu[:, 1], c='g', edgecolor='none')  # green is ground truth

        plt.axis('off')
        plt.savefig('%s/MoG_Real.pdf' % (opt.outf))
        plt.close()


#     Final KDE plot for paper. It also plots log likelihood
    xmax = 1.3
    nLevels = 20
    np_samples_ = np_samples[::1]
    cols = len(np_samples_)
    bg_color  = sns.color_palette('Greens', n_colors=256)[0]
    plt.figure(figsize=(2*cols, 2))
    for i, samps in enumerate(np_samples_):
        if i == 0:
            ax = plt.subplot(1,cols,1)
        else:
            plt.subplot(1,cols,i+1, sharex=ax, sharey=ax)
        ax2 = sns.kdeplot(samps[:, 0], samps[:, 1], shade=True, cmap='Greens', n_levels=nLevels, clip=[[-xmax,xmax]]*2)
        ax2.set_facecolor(bg_color)
        plt.xticks([]); plt.yticks([])
        plt.title('step %d'%(i*opt.viz_every))

    plt.gcf().tight_layout()
    plt.savefig('{0}/all.png'.format(opt.outf))

    if opt.plotLoss:
        plt.figure()
        fs = np.array(fs)
        plt.plot(fs)
        plt.legend(('Discriminator loss', 'Generator loss'))
        plt.savefig('{0}/losses.pdf'.format(opt.outf))

    plt.close('all')


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.orthogonal(m.weight)
        init.constant(m.bias, 0.1)


if __name__ == '__main__':
    main()
