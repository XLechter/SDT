from __future__ import print_function
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import open3d as o3d
import sys
import os
import matplotlib.pyplot as plt
from models.model_utils import edge_preserve_sampling, gen_grid_up, get_graph_feature, symmetric_sample, three_nn_upsampling

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
import pointnet2_utils as pn2

count = 0

def positional_encoding(
        tensor, num_encoding_functions=3, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)



class Coarse_Fine_SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.pos_mlp = nn.Linear(21, 1)

        #self.attn = FullAttention(mask_flag=False)
        # self.attn = ProbAttention(mask_flag=False)

    def forward(self, x, pos, pos_original):
        x_q = self.q_conv(x[:, :, :1024]).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x[:, :, 1024:])  # b, c, n
        x_v = self.v_conv(x[:, :, 1024:])
        # print('size x_q x_k x_v', x_q.size(), x_k.size(), x_v.size())
        x = x[:, :, :1024]

        pos = pos.transpose(1, 2).contiguous()
        q_pos = pos[:, :1024, :].contiguous()
        k_pos = pos[:, 1024:, :].contiguous()

        pos_original = pos_original.transpose(1, 2).contiguous()
        q_pos_original = pos_original[:, :1024, :].contiguous()
        k_pos_original = pos_original[:, 1024:, :].contiguous()


        rel_pos = q_pos[:, :, None] - k_pos[:, None, :].contiguous()
        #print('rel pos', rel_pos.shape)
        rel_pos_emb = self.pos_mlp(rel_pos).squeeze(dim=-1)
        #print('rel_pos_emb', rel_pos_emb.shape)
        # x_q = x_q[:, :2048, :]
        # x_k = x_k[:, 2048:, :]
        # x_v = x_v[:, 2048:, :]
        # print('sizes: q k v', x_q.size(), x_k.size(), x_v.size())
        energy = x_q @ x_k + rel_pos_emb  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
#         print('attention x_v', attention.shape)
#         print('attention[0][1]', attention[0][1])
#         print('max min', torch.max(attention[0][1]), torch.min(attention[0][1]))
        #plot_heatmap(attention[0].cpu().numpy(), pos=k_pos_original[0].cpu().numpy(), coarse_pos=q_pos_original[0].cpu().numpy())
        # print('x_q x_k x_v attention', x_q.shape, x_k.shape, x_v.shape, attention.shape)
        x_r = x_v @ attention.transpose(1, 2)  # b, c, n
        # print('size x x_r', x.size(), x_r.size())
        #x_r = self.act(self.after_norm(self.trans_conv(torch.cat((x, x_r), 1))))
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x.contiguous()

def dircount(DIR):
    return len(os.listdir(DIR))

def plot_pcd(ax, pcd, color):
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=color, s=15, cmap='viridis', marker='.', alpha=1)
    ax.set_axis_off()
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)

def plot_pcd_red(ax, pcd,s):
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=pcd[:, 0], s=s, cmap='Blues', vmin=-1, vmax=0.5)
    ax.set_axis_off()
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(-0.3, 0.3)

def create_plots(partial, complete, color1, color2):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')
    plot_pcd(ax, partial, color1)
    ax.set_title('Input')
    ax = fig.add_subplot(122, projection='3d')
    plot_pcd(ax, complete, color2)
    ax.set_title('Output')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)

def plot_coarse_fine(partial, complete):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121, projection='3d')
    plot_pcd_red(ax, partial, 2)
    ax.set_title('Input')
    ax = fig.add_subplot(122, projection='3d')
    plot_pcd_red(ax, complete, 5)
    ax.set_title('Output')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)

def plot_heatmap(heatmap, pos, coarse_pos):

    # heatmap = torch.squeeze(heatmap)
    # pos = torch.squeeze(pos)
    print('pos.shape', pos.shape)

    global count

    # count = dircount('/mnt/data1/zwx/ICCV_SUB/heatmaps2')
    dirpath = '/mnt/data1/zwx/ICCV_SUB/heatmaps2/' + str(count)

    #if count == 121 or count == 820 or count == 920:
    if count == 0:
        os.makedirs(dirpath)

        for i in range(1024):
            # position = pos[i]

            color1 = np.ones(1024)
            color1[i] = 0.0

            color2 = heatmap[i]

            create_plots(coarse_pos, pos, color1, color2)

            plt.savefig(dirpath + '/' + str(i) + '.png')

    count = count+1

    print('count:', count)

def plot_heatmap1(heatmap, pos, coarse_pos):

    # heatmap = torch.squeeze(heatmap)
    # pos = torch.squeeze(pos)
    print('pos.shape', pos.shape)

    og_count = dircount('/mnt/data1/zwx/ICCV_SUB/heatmaps')

    count = dircount('/mnt/data1/zwx/ICCV_SUB/heatmaps1')
    dirpath = '/mnt/data1/zwx/ICCV_SUB/heatmaps1/'+str(count)
    os.makedirs(dirpath)

    if count >= og_count:
        for i in range(1024):
            #position = pos[i]

            xyz = coarse_pos
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            translation_matrix = np.asarray(
                [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
            rotation_matrix = np.asarray(
                [[1, 0, 0, 0],
                 [0, 0, -1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])
            transform_matrix = rotation_matrix @ translation_matrix
            pcd = pcd.transform(transform_matrix)

            xyz = np.array(pcd.points)

            ax = plt.subplot(111, projection='3d')

            color = np.ones(1024)
            color[i] = 0.0

            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            ax.scatter(x, y, z, s=10, c=color, marker='.', alpha=1)

            plt.savefig(dirpath+'/coarse_'+str(i)+'.png')

            color = heatmap[i]

            xyz = pos
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            translation_matrix = np.asarray(
                [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
            rotation_matrix = np.asarray(
                [[1, 0, 0, 0],
                 [0, 0, -1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])
            transform_matrix = rotation_matrix @ translation_matrix
            pcd = pcd.transform(transform_matrix)

            xyz = np.array(pcd.points)

            ax = plt.subplot(111, projection='3d')

            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            ax.scatter(x, y, z, s=15, c=color, marker='.', alpha=1)

            plt.savefig(dirpath+'/'+str(i)+'.png')

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.pos_mlp = nn.Conv1d(channels, channels, 1)


    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention.transpose(1, 2)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class Selective_SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.pos_mlp = nn.Conv1d(channels, channels, 1)

        self.map_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.score_conv = nn.Conv1d(channels, 1, 1, bias=False)
        self.factor = 0.5
    def forward(self, x):
        B, C, N = x.shape

        #n_top = self.factor * np.ceil(np.sqrt(N)).astype('int')
        n_top = int(self.factor * N)

        q_map = self.score_conv(self.map_conv(x))
        print('q_map', q_map.shape)
        m_top = q_map.topk(n_top, sorted=False)[1].squeeze()
        print('m top', m_top.shape)

        x = x.transpose(1,2)

        select_q = x[torch.arange(B)[:, None], m_top, :]
        print('select_q.shape', select_q.shape)
        x = x.transpose(1,2)

        x_q = self.q_conv(select_q.transpose(1,2)).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ (attention.transpose(1, 2))  # b, c, n
        print('x_r', x_r.shape)

        combin_x_r = x.transpose(1, 2).clone()

        combin_x_r[torch.arange(B)[:, None], m_top, :] = x_r.transpose(1, 2)

        print('combin_x_r', combin_x_r, combin_x_r.shape)

        x_r = self.act(self.after_norm(self.trans_conv(x - combin_x_r.transpose(1, 2))))
        x = x + x_r
        return x

class SA_Layer_PCN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention.transpose(1, 2)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class PCN_encoder(nn.Module):
    def __init__(self, output_size=1024):
        super(PCN_encoder, self).__init__()
        self.conv1 = nn.Conv1d(21, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, output_size, 1)

        self.sa = SA_Layer(512)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = positional_encoding(x.transpose(1, 2).contiguous())
        x = x.transpose(1, 2).contiguous()
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(1, 1, num_points).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = self.sa(x)
        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        return global_feature.view(batch_size, -1)


class PCN_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, scale, cat_feature_num):
        super(PCN_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.scale = scale
        self.grid = gen_grid_up(2 ** (int(math.log2(scale))), 0.05).cuda().contiguous()

        self.conv1 = nn.Conv1d(cat_feature_num, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

    def forward(self, x):
        batch_size = x.size()[0]
        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)

        grid = self.grid.clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(batch_size, 1, self.num_coarse).contiguous().cuda()

        # point_feat = coarse.unsqueeze(3).repeat(1, 1, 1, self.scale).view(batch_size, 3, self.num_fine).contiguous()
        point_feat = (
            (coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                3)).transpose(1,
                                                                                                              2).contiguous()

        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)

        feat = torch.cat((grid_feat, point_feat, global_feat), 1)

        center = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(-1, self.num_fine,
                                                                                                      3)).transpose(1,
                                                                                                                    2).contiguous()

        fine = self.conv3(F.relu(self.conv2(F.relu(self.conv1(feat))))) + center
        return coarse, fine


class Stack_conv(nn.Module):
    def __init__(self, input_size, output_size, act=None):
        super(Stack_conv, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('conv', nn.Conv2d(input_size, output_size, 1))

        if act is not None:
            self.model.add_module('act', act)

    def forward(self, x):
        y = self.model(x)
        y = torch.cat((x, y), 1)
        return y


class Dense_conv(nn.Module):
    def __init__(self, input_size, growth_rate=64, dense_n=3, k=16):
        super(Dense_conv, self).__init__()
        self.growth_rate = growth_rate
        self.dense_n = dense_n
        self.k = k
        self.comp = growth_rate * 2
        self.input_size = input_size

        self.first_conv = nn.Conv2d(self.input_size * 2, growth_rate, 1)

        self.input_size += self.growth_rate

        self.model = nn.Sequential()
        for i in range(dense_n - 1):
            if i == dense_n - 2:
                self.model.add_module('stack_conv_%d' % (i + 1), Stack_conv(self.input_size, self.growth_rate, None))
            else:
                self.model.add_module('stack_conv_%d' % (i + 1),
                                      Stack_conv(self.input_size, self.growth_rate, nn.ReLU()))
                self.input_size += growth_rate

    def forward(self, x):
        y = get_graph_feature(x, k=self.k)
        y = F.relu(self.first_conv(y))
        y = torch.cat((y, x.unsqueeze(3).repeat(1, 1, 1, self.k)), 1)

        y = self.model(y)
        y, _ = torch.max(y, 3)

        return y

class EF_encoder(nn.Module):
    def __init__(self, growth_rate=24, dense_n=3, k=16, hierarchy=[1024, 256, 64], input_size=3, output_size=256):
        super(EF_encoder, self).__init__()
        self.growth_rate = growth_rate
        self.comp = growth_rate * 2
        self.dense_n = dense_n
        self.k = k
        self.hierarchy = hierarchy

        self.init_channel = 24

        self.conv1 = nn.Conv1d(21, self.init_channel, 1)
        self.dense_conv1 = Dense_conv(self.init_channel, self.growth_rate, self.dense_n, self.k)

        self.conv1_partial = nn.Conv1d(21, self.init_channel, 1)
        self.dense_conv1_partial = Dense_conv(self.init_channel, self.growth_rate, self.dense_n, self.k)

        out_channel_size_1 = (self.init_channel * 2 + self.growth_rate * self.dense_n)  # 24*2 + 24*3 = 120

        self.conv_att = nn.Conv1d(out_channel_size_1, out_channel_size_1, 1)

        self.conv2 = nn.Conv1d(out_channel_size_1 * 2, self.comp, 1)
        self.dense_conv2 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_2 = (
                out_channel_size_1 * 2 + self.comp + self.growth_rate * self.dense_n)  # 120*2 + 48 + 24*3 = 210
        self.conv3 = nn.Conv1d(out_channel_size_2 * 2, self.comp, 1)
        self.dense_conv3 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_3 = (
                out_channel_size_2 * 2 + self.comp + self.growth_rate * self.dense_n)  # 210*2 + 48 + 24*3 = 840
        self.conv4 = nn.Conv1d(out_channel_size_3 * 2, self.comp, 1)
        self.dense_conv4 = Dense_conv(self.comp, self.growth_rate, self.dense_n, self.k)

        out_channel_size_4 = out_channel_size_3 * 2 + self.comp + self.growth_rate * self.dense_n  # 840*2 + 48 + 24*3 = 1800
        self.gf_conv = nn.Conv1d(out_channel_size_4, 1024, 1)

        self.sa1 = Coarse_Fine_SA_Layer(out_channel_size_1)
        # self.sa_partial = SA_Layer(out_channel_size_1)
        # self.sa_coarse = SA_Layer(out_channel_size_1)
        self.sa1_global = SA_Layer(out_channel_size_1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1024)

        out_channel_size = out_channel_size_4 + 1024
        self.conv5 = nn.Conv1d(out_channel_size, 1024, 1)

        out_channel_size = out_channel_size_3 + 1024
        self.conv6 = nn.Conv1d(out_channel_size, 768, 1)

        out_channel_size = out_channel_size_2 + 768
        self.conv7 = nn.Conv1d(out_channel_size, 512, 1)

        out_channel_size = out_channel_size_1 + 512
        self.conv8 = nn.Conv1d(out_channel_size, output_size, 1)

    def forward(self, x):
        x = x[:, 0:3, :]
        coarse = x[:, :, :1024].contiguous()
        partial = x[:, :, 1024:].contiguous()
        # point_cloud1 = coarse.transpose(1, 2).contiguous()
        point_cloud1 = x.transpose(1, 2).contiguous()
        # print('point_cloud1:', point_cloud1.shape)
        # point_cloud1 = point_cloud1[:, :2048:, :].contiguous()

        coarse_original = coarse
        partial_original = partial

        coarse = positional_encoding(coarse.transpose(1, 2).contiguous())
        coarse = coarse.transpose(1, 2).contiguous()

        partial = positional_encoding(partial.transpose(1, 2).contiguous())
        partial = partial.transpose(1, 2).contiguous()

        x0 = F.relu(self.conv1(coarse))  # 24
        x1 = F.relu(self.dense_conv1(x0))  # 24 + 24 * 3 = 96
        x1_coarse = torch.cat((x1, x0), 1)  # 120

        x0_partial = F.relu(self.conv1_partial(partial))  # 24
        x1_partial = F.relu(self.dense_conv1_partial(x0_partial))  # 24 + 24 * 3 = 96
        x1_partial = torch.cat((x1_partial, x0_partial), 1)  # 120

        # x1_coarse = self.sa_coarse(x1_coarse)
        # x1_partial = self.sa_partial(x1_partial)

        x1 = self.sa1(torch.cat((x1_coarse, x1_partial), 2), torch.cat((coarse, partial), 2), torch.cat((coarse_original, partial_original), 2))

        # x1 = torch.cat((x1, x1_attn), 1)
        # x1 = F.relu(self.conv_att(x1))
        x1_partial = F.relu(self.conv_att(x1_partial))
        x1 = torch.cat((x1, x1_partial), 2)

        # x1_attn = self.sa1(x1)
        # x1_attn = torch.cat((x1_attn, x1[:, :, 2048:]), 2)

        # x1 = torch.cat((x1_attn, x1), 1)
        # x1 = F.relu(self.conv_att(x1))
        #
        # x1 = torch.cat((x1, x1_partial), 2)
        # print('x1 shape', x1.shape)#

        x1 = self.sa1_global(x1)
        # print('x1.shape', x1.shape)
        x1d, _, _, point_cloud2 = edge_preserve_sampling(x1, point_cloud1, self.hierarchy[0], self.k)  # 240

        x2 = F.relu(self.conv2(x1d))  # 48
        x2 = F.relu(self.dense_conv2(x2))  # 48 + 24 * 3 = 120
        x2 = torch.cat((x2, x1d), 1)  # 120 + 240 = 210
        #x2 = self.sa2(x2)
        x2d, _, _, point_cloud3 = edge_preserve_sampling(x2, point_cloud2, self.hierarchy[1], self.k)  # 720

        x3 = F.relu(self.conv3(x2d))
        x3 = F.relu(self.dense_conv3(x3))
        x3 = torch.cat((x3, x2d), 1)
        #x3 = self.sa3(x3)
        x3d, _, _, point_cloud4 = edge_preserve_sampling(x3, point_cloud3, self.hierarchy[2], self.k)

        x4 = F.relu(self.conv4(x3d))
        x4 = F.relu(self.dense_conv4(x4))
        # print('x4.shape', x4.shape)
        x4 = torch.cat((x4, x3d), 1)
        # print('x4.shape', x4.shape)
        #x4 = self.sa4(x4)
        # print('x4 after.shape', x4.shape)

        global_feat = self.gf_conv(x4)
        global_feat, _ = torch.max(global_feat, -1)
        global_feat = F.relu(self.fc1(global_feat))
        global_feat = F.relu(self.fc2(global_feat)).unsqueeze(2).repeat(1, 1, self.hierarchy[2])

        x4 = torch.cat((global_feat, x4), 1)
        x4 = F.relu(self.conv5(x4))
        idx, weight = three_nn_upsampling(point_cloud3, point_cloud4)
        x4 = pn2.three_interpolate(x4, idx, weight)

        x3 = torch.cat((x3, x4), 1)
        x3 = F.relu(self.conv6(x3))
        idx, weight = three_nn_upsampling(point_cloud2, point_cloud3)
        x3 = pn2.three_interpolate(x3, idx, weight)

        x2 = torch.cat((x2, x3), 1)
        x2 = F.relu(self.conv7(x2))
        idx, weight = three_nn_upsampling(point_cloud1, point_cloud2)
        x2 = pn2.three_interpolate(x2, idx, weight)

        x1 = torch.cat((x1, x2), 1)
        x1 = self.conv8(x1)
        return x1


class EF_expansion(nn.Module):
    def __init__(self, input_size, output_size=64, step_ratio=2, k=4):
        super(EF_expansion, self).__init__()
        self.step_ratio = step_ratio
        self.k = k
        self.input_size = input_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(input_size * 2, output_size, 1)
        self.conv2 = nn.Conv2d(input_size * 2 + output_size, output_size * step_ratio, 1)
        self.conv3 = nn.Conv2d(output_size, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()

        input_edge_feature = get_graph_feature(x, self.k, minus_center=False).permute(0, 1, 3,
                                                                                      2).contiguous()  # B C K N
        edge_feature = self.conv1(input_edge_feature)
        edge_feature = F.relu(torch.cat((edge_feature, input_edge_feature), 1))

        edge_feature = F.relu(self.conv2(edge_feature))  # B C K N
        edge_feature = edge_feature.permute(0, 2, 3, 1).contiguous().view(batch_size, self.k,
                                                                          num_points * self.step_ratio,
                                                                          self.output_size).permute(0, 3, 1,
                                                                                                    2)  # B C K N

        edge_feature = self.conv3(edge_feature)
        edge_feature, _ = torch.max(edge_feature, 2)

        return edge_feature


class Trans_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, num_input, downsample_im=False, mirror_im=False, points_label=False, benchmark=False):
        super(Trans_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine

        if not downsample_im:
            self.scale = int(np.ceil(num_fine / (num_coarse + num_input)))
        else:
            self.scale = int(np.ceil(num_fine / 2048))

        self.downsample_im = downsample_im
        self.mirror_im = mirror_im
        self.points_label = points_label

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.dense_feature_size = 256
        self.expand_feature_size = 64

        if points_label:
            self.input_size = 4
        else:
            self.input_size = 3

        self.encoder = EF_encoder(growth_rate=24, dense_n=3, k=16, hierarchy=[1024, 256, 64],
                                  input_size=self.input_size, output_size=self.dense_feature_size)

        if self.scale >= 2:
            self.expansion = EF_expansion(input_size=self.dense_feature_size, output_size=self.expand_feature_size,
                                          step_ratio=self.scale, k=4)
            self.conv1 = nn.Conv1d(self.expand_feature_size, self.expand_feature_size, 1)
        else:
            self.expansion = None
            self.conv1 = nn.Conv1d(self.dense_feature_size, self.expand_feature_size, 1)
        self.conv2 = nn.Conv1d(self.expand_feature_size, 3, 1)

        self.benchmark = benchmark
    def forward(self, global_feat, point_input):
        batch_size = global_feat.size()[0]
        coarse = F.relu(self.fc1(global_feat))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(batch_size, 3, self.num_coarse)

        if self.downsample_im:
            if self.mirror_im:
                org_points_input = symmetric_sample(point_input.transpose(1, 2).contiguous(),
                                                    int((2048 - self.num_coarse) / 2))
                org_points_input = org_points_input.transpose(1, 2).contiguous()
            else:
                org_points_input = pn2.gather_operation(point_input,
                                                        pn2.furthest_point_sample(
                                                            point_input.transpose(1, 2).contiguous(),
                                                            int(2048 - self.num_coarse)))
        else:
            org_points_input = point_input

        if self.points_label:
            id0 = torch.zeros(coarse.shape[0], 1, coarse.shape[2]).cuda().contiguous()
            coarse_input = torch.cat((coarse, id0), 1)
            id1 = torch.ones(org_points_input.shape[0], 1, org_points_input.shape[2]).cuda().contiguous()
            org_points_input = torch.cat((org_points_input, id1), 1)
            points = torch.cat((coarse_input, org_points_input), 2)
        else:
            points = torch.cat((coarse, org_points_input), 2)

        dense_feat = self.encoder(points)

        if self.scale >= 2:
            dense_feat = self.expansion(dense_feat)

        point_feat = F.relu(self.conv1(dense_feat))
        fine = self.conv2(point_feat)

        num_out = fine.size()[2]
        if self.benchmark:
            fine = pn2.gather_operation(fine,
                                        pn2.furthest_point_sample(fine.transpose(1, 2).contiguous(), 2048))
        elif num_out > self.num_fine:
            fine = pn2.gather_operation(fine,
                                        pn2.furthest_point_sample(fine.transpose(1, 2).contiguous(), self.num_fine))

        return coarse, fine


class Model(nn.Module):
    def __init__(self, num_coarse=1024, num_fine=2048, num_input=2048, downsample_im=False, mirror_im=False,
                 points_label=False, benchmark=False):
        super(Model, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        # self.scale = num_fine // num_coarse
        self.downsample_im = downsample_im
        self.mirror_im = mirror_im
        print('self.mirror_im', self.mirror_im)
        self.encoder = PCN_encoder()
        self.decoder = Trans_decoder(num_coarse, num_fine, num_input, downsample_im, mirror_im, points_label, benchmark)

    def forward(self, x):
        feat = self.encoder(x)
        coarse, fine = self.decoder(feat, x)
        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()

        print('coarse fine shape', x.transpose(1, 2).contiguous().cpu().numpy().shape, fine.cpu().numpy().shape)

#         global count
#         dirpath = '/mnt/data1/zwx/ICCV_SUB/results_sub'
#         plot_coarse_fine(x.transpose(1, 2).contiguous().cpu().numpy()[0], fine.cpu().numpy()[0])
#         plt.savefig(dirpath + '/' + str(count) + '.png')
#         count = count+1
        return coarse, fine

# if __name__ == '__main__':
#     use_cuda = False
#     Q = torch.randn(2, 1, 1024, 30)
#     K = torch.randn(2, 1, 2048, 30)
#     V = torch.ones(2, 1, 2048, 100)
#     if use_cuda:
#         Q = Q.cuda()
#         V = V.cuda()
#         K = K.cuda()
#     attn = FullAttention(100)
#     V_new, A = attn(V)
#     print(V_new. shape)
