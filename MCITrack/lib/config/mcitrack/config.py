from easydict import EasyDict as edict
import yaml

'''
Gohan: Dinov2 combined with One-stream framework.
'''

cfg = edict()

# MODEL
cfg.MODEL = edict()

# MODEL.ENCODER
# for more customization for encoder, please modify lib/models/mcitrack/vit.py
cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = "dinov2_vitb14" # encoder model
cfg.MODEL.ENCODER.DROP_PATH = 0
cfg.MODEL.ENCODER.PRETRAIN_TYPE = "mae" #  mae, default, or scratch. This parameter is not activated for dinov2.
cfg.MODEL.ENCODER.USE_CHECKPOINT = False # to save the memory.
cfg.MODEL.ENCODER.STRIDE = 14
cfg.MODEL.ENCODER.POS_TYPE = 'interpolate' # type of loading the positional encoding. "interpolate" or "index".
cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE = False # add a token_type_embedding to indicate the search, template_foreground, template_background
cfg.MODEL.ENCODER.INTERACTION_INDEXES = [[0, 6], [6, 12], [12, 18], [18, 24]]
cfg.MODEL.ENCODER.GRAD_CKPT = False
# MODEL.NECK
cfg.MODEL.NECK = edict()
cfg.MODEL.NECK.N_LAYERS = 4
cfg.MODEL.NECK.D_MODEL = 512
cfg.MODEL.NECK.D_STATE = 16 #MAMABA_HIDDEN_STATE
# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.TYPE = "CENTER" # MLP, CORNER, CENTER
cfg.MODEL.DECODER.NUM_CHANNELS = 256

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1  # encoder's LR = this factor * LR
cfg.TRAIN.FREEZE_ENCODER = False # for freezing the parameters of encoder
cfg.TRAIN.ENCODER_OPEN = [] # only for debug, open some layers of encoder when FREEZE_ENCODER is True
cfg.TRAIN.CE_WEIGHT = 1.0 # weight for cross-entropy loss
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.PRINT_INTERVAL = 50 # interval to print the training log
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.FIX_BN = False
cfg.TRAIN.ENCODER_W = ""
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1
cfg.TRAIN.TYPE = "normal" # normal, peft or fft
cfg.TRAIN.PRETRAINED_PATH = None

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.SAMPLER_MODE = "order"
cfg.DATA.LOADER = "tracking"
# cfg.DATA.MULTI_MODAL_VISION = True # vision multi-modal
# cfg.DATA.MULTI_MODAL_LANGUAGE = True # language multi-modalF
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1  #number of search region, only support 1 for now.
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
# DATA.TEMPLATEF
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 4.0
cfg.TEST.TEMPLATE_SIZE = 256
cfg.TEST.SEARCH_FACTOR = 2.0
cfg.TEST.SEARCH_SIZE = 128
cfg.TEST.EPOCH = 500
cfg.TEST.WINDOW = False # window penalty
cfg.TEST.NUM_TEMPLATES = 1

cfg.TEST.UPT = edict()
cfg.TEST.UPT.DEFAULT = 1
cfg.TEST.UPT.LASOT = 0
cfg.TEST.UPT.LASOT_EXTENSION_SUBSET = 0
cfg.TEST.UPT.TRACKINGNET = 0
cfg.TEST.UPT.TNL2K = 0
cfg.TEST.UPT.NFS = 0
cfg.TEST.UPT.UAV = 0
cfg.TEST.UPT.VOT20 = 0
cfg.TEST.UPT.GOT10K_TEST = 0

cfg.TEST.UPH = edict()
cfg.TEST.UPH.DEFAULT = 1
cfg.TEST.UPH.LASOT = 0
cfg.TEST.UPH.LASOT_EXTENSION_SUBSET = 0
cfg.TEST.UPH.TRACKINGNET = 0
cfg.TEST.UPH.TNL2K = 0
cfg.TEST.UPH.NFS = 0
cfg.TEST.UPH.UAV = 0
cfg.TEST.UPH.VOT20 = 0
cfg.TEST.UPH.GOT10K_TEST = 0

cfg.TEST.INTER = edict()
cfg.TEST.INTER.DEFAULT = 999999
cfg.TEST.INTER.LASOT = 0
cfg.TEST.INTER.LASOT_EXTENSION_SUBSET = 0
cfg.TEST.INTER.TRACKINGNET = 0
cfg.TEST.INTER.TNL2K = 0
cfg.TEST.INTER.NFS = 0
cfg.TEST.INTER.UAV = 0
cfg.TEST.INTER.VOT20 = 0
cfg.TEST.INTER.GOT10K_TEST = 0

cfg.TEST.MB = edict()
cfg.TEST.MB.DEFAULT = 500
cfg.TEST.MB.LASOT = 0
cfg.TEST.MB.LASOT_EXTENSION_SUBSET = 0
cfg.TEST.MB.TRACKINGNET = 0
cfg.TEST.MB.TNL2K = 0
cfg.TEST.MB.NFS = 0
cfg.TEST.MB.UAV = 0
cfg.TEST.MB.VOT20 = 0
cfg.TEST.MB.GOT10K_TEST = 0













def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)


