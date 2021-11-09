import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
import numpy as np
import datetime
import glob
from preprocess import get_data, shuffle
from model import Part_Segmentation
from visualization import Vis
from miou import mIoU


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./dataset/', help='path of the dataset')
parser.add_argument('--type', dest='type', default='part_segmentation', help='part_segmentation')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--sample', dest='sampling', type=int, default=2048, help='# of sampling for each point cloud')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--alpha', dest='alpha', type=float, default=0.001, help='weight for regularization loss')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='# point cloud in batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# epoch')
parser.add_argument('--save_dir', dest='save_dir', default='./checkpoints/', help='path of the checkpoints')
parser.add_argument('--load_dir', dest='load_dir', default='./checkpoints/', help='path of the checkpoints')
parser.add_argument('--log_dir', dest='log_dir', default='./logs/', help='path of the logs')
parser.add_argument('--vis_dir', dest='vis_dir', default='./vis/', help='path of the visualization')
parser.add_argument('--vis', dest='vis', type=bool, default=False, help='visualize 1: true, 0: false')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
args = parser.parse_args()

TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

def train(model, x, cls_label, seg_label):
    # get optimizer
    optimizer = model.opt
    # sample and shuffle
    x, cls_label, seg_label = shuffle(x, cls_label, seg_label)
    dataset_size = len(x)
    iteration = int(np.ceil(dataset_size/args.batch_size))
    loss = []
    for i in range(iteration):
        start = i*args.batch_size
        # divisible or non divisible
        end = start+args.batch_size if i!=iteration-1 else dataset_size
        with tf.GradientTape() as tape:
            pred, matrix = model(x[start:end], cls_label[start:end], training=True)
            l = model.loss(pred, seg_label[start:end], matrix, args.alpha)
            loss.append(l*(end-start))
        gradients = tape.gradient(l, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return np.sum(loss)/dataset_size


def test(model, x, cls_label, seg_label, name=None):
    dataset_size = len(x)
    iteration = int(np.ceil(dataset_size/args.batch_size))
    acc = []
    miou_obj = mIoU(1)
    if args.vis:
        model_dir = "{}_{}".format(args.type, TIME)
        vis = Vis(os.path.join(args.vis_dir, model_dir))
    for i in range(iteration):
        start = i*args.batch_size
        # divisible or non divisible
        end = start+args.batch_size if i!=iteration-1 else dataset_size
        pred, matrix = model(x[start:end], cls_label[start:end], training=False)
        pred = np.argmax(pred.numpy(), 2)
        acc.append(model.accuracy(pred, seg_label[start:end]).numpy()*(end-start))
        miou_obj.compute(pred, seg_label[start:end], cls_label[start:end], end-start)
        if args.vis:
            model_dir = "{}_{}".format(args.type, TIME)
            vis.visualize(x[start:end],
                          cls_label[start:end],
                          pred,
                          end-start)
    cat_iou, cat_miou, total_miou = miou_obj.get_iou()
    return np.sum(acc)/dataset_size, cat_iou, cat_miou, total_miou


def get_saver(model):
    if args.continue_train or args.phase=='test':
        checkpoint_dir = sorted(glob.glob(os.path.join(args.load_dir, '*')))[-1]
    else:
        model_dir = "{}_{}".format(args.type, TIME)
        checkpoint_dir = os.path.join(args.save_dir, model_dir)
    print("Checkpoint path: ", checkpoint_dir)
    checkpoint = tf.train.Checkpoint(optimizer=model.opt, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    return checkpoint, checkpoint_manager


def get_writer():
    if args.continue_train or args.phase=='test':
        log_dir = sorted(glob.glob(os.path.join(args.log_dir, '*')))[-1]
    else:
        model_dir = "{}_{}".format(args.type, TIME)
        log_dir = os.path.join(args.log_dir, model_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)
    return summary_writer


if __name__=="__main__":
    if args.phase=='train':
        # load dataset
        train_x, train_y, train_seg, val_x, val_y, val_seg = get_data(args)
        # load model
        model = Part_Segmentation(args.type, args.lr, args.sampling)
        # initial epoch
        start = 0
        # get saver
        saver, saver_manager = get_saver(model)
        # load weight to continue training
        if args.continue_train:
            checkpoint = saver_manager.latest_checkpoint
            if saver.restore(checkpoint).expect_partial():
                # reset epoch
                start = int(checkpoint.split("-")[-1])+1
                print("Load checkpoint succeeded")
        # get tensorboard writer
        writer = get_writer()
        # train for # epoch
        for i in range(start, args.epoch):
            loss = train(model, train_x, train_y, train_seg)
            # test on validation set
            accuracy, cat_iou, cat_miou, total_miou = test(model, val_x, val_y, val_seg)
            # save
            saver_manager.save(i)
            # write to tensorboard 
            with writer.as_default():
                tf.summary.scalar('loss', loss, step=i)
                tf.summary.scalar('accuracy', accuracy, step=i)
                tf.summary.scalar('category mIoU', cat_miou, step=i)
                tf.summary.scalar('total mIoU', total_miou, step=i)
            print("Epoch {}, Loss {}".format(i, loss))
            print("Non weighted Category mIoU {}, Total mIoU {}".format(cat_miou, total_miou))
            print("Per category IoU:")
            print(cat_iou)
    elif args.phase=='test':
        # load test set
        test_x, test_y, test_seg = get_data(args)
        # load model
        model = Part_Segmentation(args.type, args.lr, 2048*2)
        # get saver
        saver, saver_manager = get_saver(model)
        # load weight to test
        checkpoint = saver_manager.latest_checkpoint
        if saver.restore(checkpoint):
            print("Load checkpoint succeeded")
        # test
        accuracy, cat_iou, cat_miou, total_miou = test(model, test_x, test_y, test_seg)
        print("Non weighted category mIoU {}, Total mIoU {}".format(cat_miou, total_miou))
        print("Per category IoU:")
        print(cat_iou)
