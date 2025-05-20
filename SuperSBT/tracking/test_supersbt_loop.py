import os
import sys
import argparse
import _init_paths

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker

import numpy as np
import os
import shutil
import argparse
import _init_paths
from lib.test.evaluation.environment import env_settings


def transform_got10k(tracker_name, cfg_name):
    env = env_settings()
    result_dir = env.results_path
    src_dir = os.path.join(result_dir, "%s/%s/got10k/" % (tracker_name, cfg_name))
    dest_dir = os.path.join(result_dir, "%s/%s/got10k_submit/" % (tracker_name, cfg_name))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    items = os.listdir(src_dir)
    for item in items:
        if "all" in item:
            continue
        src_path = os.path.join(src_dir, item)
        if "time" not in item:
            seq_name = item.replace(".txt", '')
            seq_dir = os.path.join(dest_dir, seq_name)
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            new_item = item.replace(".txt", '_001.txt')
            dest_path = os.path.join(seq_dir, new_item)
            bbox_arr = np.loadtxt(src_path, dtype=np.int64, delimiter='\t')
            np.savetxt(dest_path, bbox_arr, fmt='%d', delimiter=',')
        else:
            seq_name = item.replace("_time.txt", '')
            seq_dir = os.path.join(dest_dir, seq_name)
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            dest_path = os.path.join(seq_dir, item)
            os.system("cp %s %s" % (src_path, dest_path))
    # make zip archive
    shutil.make_archive(src_dir, "zip", src_dir)
    shutil.make_archive(dest_dir, "zip", dest_dir)

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, pth=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset.
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id, pth=pth)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default='supersbt', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='supersbt_256_light',help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, got10k_test, lasot, trackingnet, lasot_extension_subset, tnl2k).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--pth', type=str, default=None, help='pth path.')

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence
    pth = '/home/fexie/data/fei/SuperSBT/pretrained_models/supersbt/light'
    #/home/fexie/data/fei/SuperSBT/pretrained_models/supersbt/supersbt_256/supersbt_ep0400.pth.tar
    num_id = [
        'supersbt_light_fulldata.tar',

    ]
    args.dataset_name = [
        'lasot_extension_subset',
        #'lasot',
    ]
    import time
    import datetime

    for i in range(len(num_id)):
        for j in range(len(args.dataset_name)):
            start = time.time()
            pth_i = os.path.join(pth, num_id[i])
            run_tracker(args.tracker_name, args.tracker_param, i+j, args.dataset_name[j], seq_name, args.debug,
                        args.threads, num_gpus=args.num_gpus, pth=pth_i)
            end = time.time()
            test_time = end - start

            print("One dataset testing time: " + str(datetime.timedelta(seconds=test_time)))
            pth_save = args.tracker_param + '_{:03d}'.format(i)
            print(pth_save)
            try:
                if test_time > 2 and 'got' in args.dataset_name[j]:
                    transform_got10k(args.tracker_name, pth_save)
            except:
                pass




if __name__ == '__main__':
    main()
