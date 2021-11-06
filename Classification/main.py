import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ""
import tensorflow as tf
import numpy as np
import datetime
import glob
from preprocess import get_data, shuffle, rotate_point_cloud, jitter_point_cloud, sample, rotate_point_cloud_angle
from model import Classification
from visualization import visualize


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./dataset/', help='path of the dataset')
parser.add_argument('--type', dest='type', default='classification', help='classification')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--sample', dest='sampling', type=int, default=1024, help='# of sampling for each point cloud')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--alpha', dest='alpha', type=float, default=0.001, help='weight for regularization loss')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='# point cloud in batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# epoch')
parser.add_argument('--save_dir', dest='save_dir', default='./checkpoints/', help='path of the checkpoints')
parser.add_argument('--load_dir', dest='load_dir', default='./checkpoints/', help='path of the checkpoints')
parser.add_argument('--log_dir', dest='log_dir', default='./logs/', help='path of the logs')
parser.add_argument('--vis_dir', dest='vis_dir', default='./vis/', help='path of the visualization')
parser.add_argument('--vis', dest='vis', type=bool, default=False, help='visualize 1: true, 0: false')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
args = parser.parse_args()

TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

def train(model, x, y, val_x=None, val_y=None):
    # get optimizer
    optimizer = model.opt
    # sample and shuffle
    x, y = shuffle(x[:, :args.sampling, :], y)
    # augment
    x = jitter_point_cloud(rotate_point_cloud(x))
    dataset_size = len(x)
    iteration = int(np.ceil(dataset_size/args.batch_size))
    loss = []
    for i in range(iteration):
        start = i*args.batch_size
        # divisible or non divisible
        end = start+args.batch_size if i!=iteration-1 else dataset_size
        with tf.GradientTape() as tape:
            pred, matrix = model(x[start:end], training=True)
            l = model.loss(pred, y[start:end], matrix, args.alpha)
            loss.append(l*(end-start))
        gradients = tape.gradient(l, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return np.sum(loss)/dataset_size


def test(model, x, y, name=None):
    dataset_size = len(x)
    iteration = int(np.ceil(dataset_size/args.batch_size))
    acc = []
    x_ = x[:, :args.sampling, :]
    for i in range(iteration):
        start = i*args.batch_size
        # divisible or non divisible
        end = start+args.batch_size if i!=iteration-1 else dataset_size
        # build committee
        committee = np.zeros((end-start, 40))
        for j in range(12):
            # rotate point cloud
            rotated_x = rotate_point_cloud_angle(x_[start:end], j/12.*np.pi*2)
            pred, matrix = model(rotated_x, training=False)
            committee[np.arange(end-start), np.argmax(pred, 1)] += 1
        acc.append(model.accuracy(committee, y[start:end]).numpy()*(end-start))
        if args.vis:
            model_dir = "{}_{}".format(args.type, TIME)
            visualize(x[start:end], 
                      y[start:end], 
                      committee, 
                      name[start:end], 
                      os.path.join(args.vis_dir, model_dir),
                      end-start)
    return np.sum(acc)/dataset_size


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
        train_x, train_y, val_x, val_y = get_data(args)
        # load model
        model = Classification(args.type, args.lr)
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
            loss = train(model, train_x, train_y)
            # test on validation set
            accuracy = test(model, val_x, val_y)
            # save
            saver_manager.save(i)
            # write to tensorboard 
            with writer.as_default():
                tf.summary.scalar('loss', loss, step=i)
                tf.summary.scalar('accuracy', accuracy, step=i)
            print("Epoch {}, Loss {}, Accuracy {}".format(i, loss, accuracy))
    elif args.phase=='test':
        # load test set
        test_x, test_y, test_name = get_data(args)
        # load model
        model = Classification(args.type, args.lr)
        # get saver
        saver, saver_manager = get_saver(model)
        # load weight to test
        checkpoint = saver_manager.latest_checkpoint
        if saver.restore(checkpoint):
            print("Load checkpoint succeeded")
        # test
        accuracy = test(model, test_x, test_y, test_name)
        print(accuracy)
