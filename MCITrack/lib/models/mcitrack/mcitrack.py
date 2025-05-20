"""
MCITrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from lib.models.mcitrack.encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
from .neck import build_neck
from collections import OrderedDict

class MCITrack(nn.Module):
    """ This is the base class for MCITrack """
    def __init__(self, encoder, decoder, neck,cfg,
                 num_frames=1, num_template=1, decoder_type="CENTER"):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the encoder to be used. See encoder.py
            decoder: torch module of the decoder architecture. See decoder.py
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type
        self.neck = neck

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template
        self.freeze_en = cfg.TRAIN.FREEZE_ENCODER
        self.interaction_indexes = cfg.MODEL.ENCODER.INTERACTION_INDEXES


    def forward(self, template_list=None, search_list=None, template_anno_list=None,enc_opt=None,neck_h_state=None, feature=None, mode="encoder"):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "encoder":
            return self.forward_encoder(template_list, search_list, template_anno_list)
        elif mode == "neck":
            return self.forward_neck(enc_opt,neck_h_state)
        elif mode == "decoder":
            return self.forward_decoder(feature)
        else:
            raise ValueError

    def forward_encoder(self, template_list, search_list, template_anno_list):
        # Forward the encoder
        xz = self.encoder(template_list, search_list, template_anno_list)
        return xz
    def forward_neck(self,enc_out,neck_h_state):
        x = enc_out
        xs = x[:, 0:self.num_patch_x]
        x,xs,h = self.neck(x,xs,neck_h_state,self.encoder.body.blocks,self.interaction_indexes)
        x = self.encoder.body.fc_norm(x)
        xs = xs + x[:, 0:self.num_patch_x]
        return x,xs,h

    def forward_decoder(self, feature, gt_score_map=None):
        # feature = feature[0]
        # feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)
        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_mcitrack(cfg):
    encoder = build_encoder(cfg)
    neck = build_neck(cfg,encoder)
    decoder = build_decoder(cfg, neck)
    model = MCITrack(
        encoder,
        decoder,
        neck,
        cfg,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
    )
    return model
