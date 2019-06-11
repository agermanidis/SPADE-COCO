from PIL import Image
import torch
import numpy as np
import os
import runway
import argparse
from util.coco import label_to_id

from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from options.base_options import BaseOptions
from data.base_dataset import get_params, get_transform
import util.util as util

class Options(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')       
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(load_from_opt_file=True)
        if not torch.cuda.is_available():
            parser.set_defaults(gpu_ids="-1")
        self.isTrain = False
        return parser

opt = None

@runway.setup(options={'checkpoints_root': runway.file(is_directory=True)})
def setup(opts):
    global opt
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    options = Options()
    parser = options.initialize(parser)
    options.parser = parser
    name = opts['checkpoints_root'].split('/')[-1]
    checkpoints_dir = os.path.join(opts['checkpoints_root'], '..')
    parser.set_defaults(name=name)
    parser.set_defaults(checkpoints_dir=checkpoints_dir)
    opt = options.parse()
    opt.name = name
    opt.checkpoints_dir = checkpoints_dir
    model = Pix2PixModel(opt)
    model.eval()
    return model

label_to_color = {
    'unlabeled': (0, 0, 0),
    'grass': (29, 195, 49),
    'sky': (95, 219, 255),
    'clouds': (170, 170, 170),
    'sea': (54, 62, 167),
    'river': (0, 57, 150),
    'tree': (140, 104, 47),
    'mountain': (60, 55, 50)
}

command_inputs = {
    'semantic_map': runway.segmentation(label_to_id=label_to_id, label_to_color=label_to_color, default_label='unlabeled')
}

command_outputs = {
    'output': runway.image
}

@runway.command('convert', inputs=command_inputs, outputs=command_outputs)
def convert(model, inputs):
    img = inputs['semantic_map']
    original_size = img.size
    img = img.resize((opt.load_size, opt.load_size))
    params = get_params(opt, img.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(img) * 255.0
    label_tensor[label_tensor == 255.0] = 0
    data = {
        'label': label_tensor.unsqueeze(0),
        'instance': label_tensor.unsqueeze(0),
        'image': None
    }
    generated = model(data, mode='inference')
    output = util.tensor2im(generated[0])
    output = Image.fromarray(output).resize(original_size)
    return output

if __name__ == '__main__':
    runway.run(port=5132)