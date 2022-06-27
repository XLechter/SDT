from utils.utils import *
import torch
import os
import h5py
import sys
import os
proj_dir = os.path.dirname(os.path.abspath(__file__))
from models.model import Model

import subprocess

sys.path.append(os.path.join(proj_dir, "utils/ChamferDistancePytorch"))
from chamfer3D import dist_chamfer_3D
from fscore import fscore
chamLoss = dist_chamfer_3D.chamfer_3DDist()

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test(args):
    model_dir = args.model_dir
    log_test = LogString(open(os.path.join(model_dir, 'log_text.txt'), 'w'))
    dataset_test = Completion3D(args.datapath, train=False, npoints=args.num_points, use_mean_feature=args.use_mean_feature, benchmark=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    print(dataset_length)
    epochs = ['model.pth']

    odir = 'benchmark/'

    if not os.path.exists(odir):
        os.makedirs(odir)

    for epoch in epochs:
        load_path = os.path.join(args.model_dir, epoch)
        net = eval(args.model_name)(num_coarse=1024, num_fine=args.num_points, benchmark=True)
        args.load_model = load_path
        load_model(args, net, None, log_test, train=False)
        net.cuda()
        net.eval()
        log_test.log_string("Testing...")
        with torch.no_grad():
            for i, data in enumerate(dataloader_test):
                label, inputs, _ = data

                inputs = inputs.float().cuda()
                inputs = inputs.transpose(2, 1).contiguous()

                _, output = net(inputs)
                output_numpy = output.data.cpu().numpy()
                #print('output.shape', output.shape)
                for idx in range(output_numpy.shape[0]):
                    fname = label[idx].split('/')[-1]
                    #print('fname:', idx, fname)
                    outp = output_numpy[idx:idx + 1, ...].squeeze()
                    print('outp.shape', outp.shape)
                    dir = os.path.join(odir, 'all')
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    ofile = os.path.join(dir, fname)
                    print("Saving to %s ..." % (ofile))
                    # pltname = ofile.replace('h5', 'png').replace('all', 'plot')
                    # plot_pcd_three_views(pltname, [outp], ['partial'])
                    with h5py.File(ofile, "w") as f:
                        f.create_dataset("data", data=outp)

            cur_dir = os.getcwd()
            cmd = "cd %s; zip -r submission.zip * ; cd %s" % (odir, cur_dir)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, _ = process.communicate()
            print("Submission file has been saved to %s/submission.zip" % odir)

    log_test.close()

def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)
