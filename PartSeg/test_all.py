import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
import numpy as np
import datetime
import glob
from preprocess import get_org_data
from model import Part_Segmentation
from visualization import Vis
from miou import mIoU


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./dataset/', help='path of the dataset')
parser.add_argument('--type', dest='type', default='part_segmentation', help='part_segmentation')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='# point cloud in batch')
parser.add_argument('--load_dir', dest='load_dir', default='./checkpoints/', help='path of the checkpoints')
parser.add_argument('--vis_dir', dest='vis_dir', default='./vis/', help='path of the visualization')
parser.add_argument('--vis', dest='vis', type=bool, default=False, help='visualize 1: true, 0: false')
parser.add_argument('--max_size', dest='max_size', type=int, default=3000, help='max_size for each point cloud')
args = parser.parse_args()

TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')


def test(model, x, cls_label, seg_label, size, name=None):
    dataset_size = len(x)
    iteration = int(np.ceil(dataset_size/args.batch_size))
    miou_obj = mIoU(16)
    if args.vis:
        model_dir = "{}_{}".format(args.type, TIME)
        vis = Vis(os.path.join(args.vis_dir, model_dir))
    for i in range(iteration):
        start = i*args.batch_size
        # divisible or non divisible
        end = start+args.batch_size if i!=iteration-1 else dataset_size
        pred, matrix = model(x[start:end], cls_label[start:end], training=False)
        preds = []
        # argmax and elimiante padding
        for p, s in zip(pred.numpy(), size[start:end]):
            preds.append(np.argmax(p[:int(s[0])], 1))
        miou_obj.compute(preds, seg_label[start:end], cls_label[start:end], end-start)
        if args.vis:
            model_dir = "{}_{}".format(args.type, TIME)
            vis.visualize(x[start:end],
                          seg_label[start:end],
                          cls_label[start:end],
                          preds,
                          end-start)
    cat_iou, cat_miou, total_miou = miou_obj.get_iou()
    return cat_iou, cat_miou, total_miou

def get_saver(model):
    checkpoint_dir = sorted(glob.glob(os.path.join(args.load_dir, '*')))[-1]
    print("Checkpoint path: ", checkpoint_dir)
    checkpoint = tf.train.Checkpoint(optimizer=model.opt, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    return checkpoint, checkpoint_manager


if __name__=="__main__":
    # load test set
    test_x, test_y, test_seg, test_size = get_org_data(args)
    # load model
    model = Part_Segmentation(args.type, 0.0, args.max_size)
    # get saver
    saver, saver_manager = get_saver(model)
    # load weight to test
    checkpoint = saver_manager.latest_checkpoint
    if saver.restore(checkpoint):
        print("Load checkpoint succeeded")
    # test
    cat_iou, cat_miou, total_miou = test(model, test_x, test_y, test_seg, test_size)
    print("Non weighted category mIoU {}, Total mIoU {}".format(cat_miou, total_miou))
    print("Per category IoU:")
    print(cat_iou)
