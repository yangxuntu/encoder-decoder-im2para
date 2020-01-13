from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .AttModel import *
from .HierModel import *
from .AttModel_acc import *
from .GraphModel import *
from .GraphModel2 import *
from .AccGraphModel2 import *

def setup(opt):
    if opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    elif opt.caption_model == 'mtopdown':
        model = MTopDownModel(opt)
    elif opt.caption_model == 'htopdown':
        model = HTopDownModel(opt)
    elif opt.caption_model == 'atopdown':
        model = AccTopDownModel(opt)
    elif opt.caption_model == 'gtopdown':
        model = GTopDownModel(opt)
    elif opt.caption_model == 'gtopdown2':
        model = GTopDownModel2(opt)
    elif opt.caption_model == 'hgtopdown2':
        model = HGTopDownModel2(opt)
    elif opt.caption_model == 'agtopdown2':
        model = AGTopDownModel2(opt)
    elif opt.caption_model == 'ahgtopdown2':
        model = AHGTopDownModel2(opt)

    # Check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.checkpoint_path)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.checkpoint_path, 'infos_'
                + opt.id + format(int(opt.start_from),'04') + '.pkl')),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(
            opt.checkpoint_path, 'model' +opt.id+ format(int(opt.start_from),'04') + '.pth')))

    return model
