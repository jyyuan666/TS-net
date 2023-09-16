import argparse
import time

import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', default=1, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    ii = 0
    sum = 0
    sum1 = 0
    # benchmark with 2000 image and take the average
    count = 0
    sum = 0
    for i, data in enumerate(data_loader):

        # print(i)
        # print(data)
        # print(0000)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        #if i >= num_warmup:
        pure_inf_time += elapsed
        #if (i + 1) % args.log_interval == 0:
        count += 1
        print('count:',count)
        fps = (i + 1 - num_warmup) / pure_inf_time
        print('fps:', fps)
        sum += fps
        print('sum:', sum)
        # print(fps)
        print("mean fps: ", sum/count)
        #print(f'Done image [{i + 1:<3}/ 2000], fps: {fps:.1f} img / s')
        sys.stdout = Logger(sys.stdout)  # record log
        # sys.stderr = Logger(sys.stderr)  # record error
        #print('fps: ', sum/count, 'img/s')

        if (i + 1) == 3000:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            # print(fps+'00000000')

            print(f'Overall fps: {fps:.1f} img / s')

            # print(".1f")


            break

import sys, os, time
sys.setrecursionlimit(3000)

class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "./config/"  # folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #log_name = '{}.txt'.format(time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
        log_name_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        log_name = log_name_time + ".txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    main()
