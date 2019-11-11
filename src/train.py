import os
import argparse
from mot.object_detection.dataset import register_mot
from mot.object_detection.config import config as cfg
from mot.object_detection.train import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datastore', help='data directory')

    args = parser.parse_args()
    args.load = os.path.join(args.datastore, 'COCO-MaskRCNN-R50FPN2x.npz')
    args.logdir = 'logs'

    cfg.DATA.BASEDIR = os.path.join(args.datastore, 'dataset_surfrider_cleaned')
    cfg.MODE_MASK=False
    cfg.TRAIN.LR_SCHEDULE=250,500,750
    cfg.DATA.NUM_WORKERS=0

    register_mot(cfg.DATA.BASEDIR)  # add the mot datasets to the registry

    main(args)