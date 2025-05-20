class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '../AQATrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '../AQATrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '../AQATrack/pretrained_networks'
        self.lasot_dir = ''
        self.got10k_dir = '../MVTD/train'
        self.got10k_val_dir = '../MVTD/val'
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = ''
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
