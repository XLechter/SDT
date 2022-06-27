from models.full_model import FullModel
import torch
import torch.optim as optim
from utils.utils import *
from models.model import Model

def train(args):
    # setup
    vis, log_model, log_train, log_path = log_setup(args)
    train_curve, val_curves = vis_curve_setup()
    best_val_losses, best_val_epochs = best_loss_setup()
    train_loss_meter, val_loss_meters = loss_average_meter_setup()
    dataset, dataset_test, dataloader, dataloader_test = data_setup(args, log_train)
    seed_setup(args, log_train)

    print('args.num_points', args.num_points)

    # model
    net = eval(args.model_name)(num_fine=args.num_points).cuda()
    net = torch.nn.DataParallel(FullModel(net))
    log_model.log_string(str(net.module.model) + '\n', stdout=False)

    # optim
    lrate = args.lr  # learning rate
    optimizer = optim.Adam(net.module.model.parameters(), lr=lrate)
    load_model(args, net, optimizer, log_train)

    for epoch in range(args.resume_epoch, args.nepoch):
        train_loss_meter.reset()
        net.module.model.train()

        if epoch > 0 and epoch % 20 == 0:
            lrate = max(lrate * 0.7, 1e-6)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrate

        if epoch < 5:
            alpha = 0.01
        elif epoch < 10:
            alpha = 0.1
        elif epoch < 15:
            alpha = 0.5
        else:
            alpha = 1.0

        # if epoch > 0 and epoch % 60 == 0:
        #     lrate = max(lrate * 0.7, 1e-6)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lrate
        #
        # if epoch < 5:
        #     alpha = 0.01
        # elif epoch < 15:
        #     alpha = 0.1
        # elif epoch < 30:
        #     alpha = 0.5
        # else:
        #     alpha = 1.0

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            label, inputs, gt = data
            inputs = inputs.float().cuda()
            gt = gt.float().cuda()
            inputs = inputs.transpose(2, 1).contiguous()

            # if args.loss == 'EMD':
            #     output1, output2, emd1, emd2, cd1_p, cd2_p, cd1_t, cd2_t, u1, u2 = net(inputs, gt.contiguous(), 0.005, 50, True, False)
            #     loss_net = emd1.mean() + u1.mean()*0.1 + (emd2.mean() + u2.mean()*0.1) * alpha
            #     train_loss_meter.update(emd2.mean().item())
            # else:
            output1, output2, emd1, emd2, cd1_p, cd2_p, cd1_t, cd2_t, u1, u2, origin_cd_p1, origin_cd_p2 = net(inputs, gt.contiguous(), 0.005, 50, False, True)
            loss_net = cd1_p.mean() + u1.mean()*0.1 + (cd2_p.mean() + u2.mean()*0.1) * alpha

            #loss_net = cd1_t.mean() + u1.mean() * 0.1 + (cd2_t.mean() + u2.mean() * 0.1) * alpha
            train_loss_meter.update(cd2_p.mean().item())
            loss_net.backward()
            optimizer.step()

            if i % 100 == 0:
                log_train.log_string(args.log_env + ' train [%d: %d/%d]  emd1: %f emd2: %f cd1_p: %f cd2_p: %f cd1_t: %f cd2_t: %f u1: %f u2: %f' % (
                epoch, i, len(dataset) / args.batch_size, emd1.mean().item(), emd2.mean().item(), cd1_p.mean().item(), cd2_p.mean().item(), 
                cd1_t.mean().item(), cd2_t.mean().item(), u1.mean().item(), u2.mean().item()))
                best_val_losses, best_val_epochs, save_cd_p, save_cd_t = val(args, net, epoch, dataloader_test, log_train,
                                                       best_val_losses, best_val_epochs, val_loss_meters)
                if save_cd_p:
                    save_model('%s/best_cd_p_network.pth' % log_path, net, optimizer)
                    log_train.log_string('saving best cd_p net...')

                if save_cd_t:
                    save_model('%s/best_cd_t_network.pth' % log_path, net, optimizer)
                    log_train.log_string('saving best cd_t net...')

        train_curve.append(train_loss_meter.avg)
        #if epoch % 5 == 0:
        #save_model('%s/network_%d.pth' % (log_path, epoch), net, optimizer)
        save_model('%s/network.pth' % log_path, net, optimizer)
        log_train.log_string("saving net...")

        net.module.model.eval()
        # VALIDATION
        if epoch % 1 == 0 or epoch == args.nepoch - 1:
            best_val_losses, best_val_epochs, _, _ = val(args, net, epoch, dataloader_test, log_train,
                                                   best_val_losses, best_val_epochs, val_loss_meters)
            if best_val_epochs["best_emd_epoch"] == epoch:
                save_model('%s/best_emd_network.pth' % log_path, net, optimizer)
                log_train.log_string('saving best emd net...')
            if best_val_epochs["best_cd_p_epoch"] == epoch:
                save_model('%s/best_cd_p_network.pth' % log_path, net, optimizer)
                log_train.log_string('saving best cd_p net...')
            if best_val_epochs["best_cd_t_epoch"] == epoch:
                save_model('%s/best_cd_t_network.pth' % log_path, net, optimizer)
                log_train.log_string('saving best cd_t net...')

        val_curves["val_curve_emd"].append(val_loss_meters["val_loss_emd"].avg)
        val_curves["val_curve_cd_p"].append(val_loss_meters["val_loss_cd_p"].avg)
        val_curves["val_curve_cd_t"].append(val_loss_meters["val_loss_cd_t"].avg)
        vis_curve_plot(vis, args.log_env, train_curve, val_curves)
        epoch_log(args, log_model, train_loss_meter, val_loss_meters, best_val_losses, best_val_epochs)
        net.module.model.train()

    log_model.close()
    log_train.close()
