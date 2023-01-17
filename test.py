from utils.utils import *
import torch
import os
import h5py
import sys
import os
proj_dir = os.path.dirname(os.path.abspath(__file__))
import open3d
from models.model import Model
import subprocess

sys.path.append(os.path.join(proj_dir, "utils/ChamferDistancePytorch"))
from chamfer3D import dist_chamfer_3D
from fscore import fscore
chamLoss = dist_chamfer_3D.chamfer_3DDist()


def calculate_fscore(gt_array, pr_array, th = 0.01):
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    print('gt_array.shape', gt_array.shape)
    gt = open3d.geometry.PointCloud()
    gt.points = open3d.utility.Vector3dVector(gt_array)
    pr = open3d.geometry.PointCloud()
    pr.points = open3d.utility.Vector3dVector(pr_array)

    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)

    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall


def test(args):
    model_dir = args.model_dir
    log_test = LogString(open(os.path.join(model_dir, 'log_text.txt'), 'w'))

    if args.dataset == 'SCAN':
        dataset_test = SCAN(args.datapath, npoints=args.num_points)
    elif args.dataset == 'KITTI':
        dataset_test = KITTI(args.datapath, npoints=args.num_points)
    else:
        dataset_test = PCN(args.datapath, train=False, npoints=args.num_points, test=True)

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)

    epochs = ['model.pth']
    for epoch in epochs:
        load_path = os.path.join(args.model_dir, epoch)
        net = eval(args.model_name)(num_coarse=1024, num_fine=args.num_points)
        args.load_model = load_path

        load_model(args, net, None, log_test, train=False)
        net.cuda()
        net.eval()
        log_test.log_string("Testing...")

        # pcd_file = h5py.File(os.path.join(args.model_dir, '%s_pcds.h5' % epoch.split('.')[0]), 'w')
        # pcd_file.create_dataset('output_pcds', (dataset_length, args.num_points, 3))

        test_loss_cd_p = AverageValueMeter()
        test_loss_cd_t = AverageValueMeter()
        test_f1_score = AverageValueMeter()

        with torch.no_grad():
            for i, data in enumerate(dataloader_test):
                label, inputs, gt = data

                inputs = inputs.float().cuda()
                gt = gt.float().cuda()
                inputs = inputs.transpose(2, 1).contiguous()

                coarse, output = net(inputs)

                # save pcd
                # pcd_index1 = args.batch_size * i
                # pcd_index2 = args.batch_size * (i + 1)
                # pcd_file['output_pcds'][pcd_index1:pcd_index2, :, :] = output.cpu().numpy()

                #g_input_pcd[f"{i}"] = inputs.cpu().numpy()
                #g_gt_pcd[f"{i}"] = gt.cpu().numpy()
                # g_output_pcd[f"{i}"] = output.cpu().numpy()
                # g_coarse_pcd[f"{i}"] = coarse.cpu().numpy()

                # EMD
                # dist, _ = EMD(output, gt, 0.004, 3000)
                # emd = torch.sqrt(dist).mean(1)

                # CD
                dist1, dist2, _, _ = chamLoss(gt, output)
                cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
                cd_t = dist1.mean(1) + dist2.mean(1)
                emd = cd_t

                # f1
                #f1, _, _ = fscore(dist1, dist2)

                f1, _, _ = calculate_fscore(gt.squeeze().cpu().numpy(), output.squeeze().cpu().numpy())

                f1 = torch.tensor(f1)

                test_loss_cd_p.update(cd_p.mean().item())
                test_loss_cd_t.update(cd_t.mean().item())
                test_f1_score.update(f1.mean().item())

                if i % 100 == 0:
                    log_test.log_string('test [%d/%d]' % (i, dataset_length / args.batch_size))

            log_test.log_string('Overview results:')
            log_test.log_string(
                'CD_p: %f, CD_t: %f, F1: %f' % (test_loss_cd_p.avg, test_loss_cd_t.avg,
                                                        test_f1_score.avg))
        #pcd_file.close()
        log_test.close()

