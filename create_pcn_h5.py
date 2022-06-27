from tensorpack import dataflow
import h5py
import os

#generate train files
df = dataflow.LMDBSerializer.load('data/train.lmdb', shuffle=False)
print('df size:', df.size())
ds = dataflow.PrefetchData(df, nr_prefetch=500, nr_proc=1)
size = df.size()
output_base_folder = 'data/pcn/train'
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)
f_list = open('data/pcn/train.list', 'w')
i = 0
for id, input, gt in ds.get_data():
    ids = id.split('_')
    category_id = ids[0]
    model_id = ids[1]
    idx = len(ids) - 3

    partial_output_folder = os.path.join(output_base_folder, 'partial', category_id)
    gt_output_folder = os.path.join(output_base_folder, 'gt', category_id)
    if not os.path.exists(partial_output_folder):
        os.makedirs(partial_output_folder)
    if not os.path.exists(gt_output_folder):
        os.makedirs(gt_output_folder)

    f = h5py.File(os.path.join(partial_output_folder, '%s_%d.h5' % (model_id, idx)), 'w')
    f.create_dataset("data", data=input)
    f.close()

    f = h5py.File(os.path.join(gt_output_folder, '%s_%d.h5' % (model_id, idx)), 'w')
    f.create_dataset("data", data=gt)
    f.close()

    f_list.write(os.path.join(category_id, '%s_%d' % (model_id, idx)))
    if i != size-1:
        f_list.write('\n')
f_list.close()

#generate valid files
df = dataflow.LMDBSerializer.load('data/valid.lmdb', shuffle=False)
ds = dataflow.PrefetchData(df, nr_prefetch=500, nr_proc=1)
size = df.size()
output_base_folder = 'data/pcn/val'
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)
f_list = open('data/pcn/val.list', 'w')
i = 0
for id, input, gt in ds.get_data():
    ids = id.split('_')
    category_id = ids[0]
    model_id = ids[1]
    idx = len(ids) - 3

    partial_output_folder = os.path.join(output_base_folder, 'partial', category_id)
    gt_output_folder = os.path.join(output_base_folder, 'gt', category_id)
    if not os.path.exists(partial_output_folder):
        os.makedirs(partial_output_folder)
    if not os.path.exists(gt_output_folder):
        os.makedirs(gt_output_folder)

    f = h5py.File(os.path.join(partial_output_folder, '%s.h5' % (model_id)), 'w')
    f.create_dataset("data", data=input)
    f.close()

    f = h5py.File(os.path.join(gt_output_folder, '%s.h5' % (model_id)), 'w')
    f.create_dataset("data", data=gt)
    f.close()

    f_list.write(os.path.join(category_id, '%s' % (model_id)))
    if i != size-1:
        f_list.write('\n')
f_list.close()