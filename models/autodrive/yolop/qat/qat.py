import argparse
import os, sys
import math
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(BASE_DIR)
#sys.path.append("../..")
#sys.path.append("hmodel/thirdparty/YOLOP")
import pprint
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import sys, argparse, os, random, shutil, time, warnings, torch, torch.nn as nn, torch.nn.parallel, torch.backends.cudnn as cudnn, torch.distributed as dist, torch.optim, torch.multiprocessing as mp, torchvision.transforms as transforms, torchvision.datasets as datasets, torchvision.models as models, torch.optim.lr_scheduler as lr_scheduler, socket,torch.utils.data as data,tqdm,torch.optim as optim,cv2 as cv,torchvision,torchvision.models as models,torch.multiprocessing as mp,os.path as osp,numpy as np,torchvision.transforms as transforms,torch.utils.data.distributed as dist,torch.utils.data,torch.optim
#math,torch.optim.lr_scheduler as sche,timm,
from copy import deepcopy

from hmquant.qat_torch.apis import trace_model_for_qat,set_calib,set_fake_quant,align_hardware
from hmquant.qat_torch.apis.transforms import Rgb2Yuv2Tensor2Normalize,adap_model_to_yuv_input
from hmodel.thirdparty.YOLOP.lib.utils import DataLoaderX, torch_distributed_zero_first
from torch.utils.tensorboard import SummaryWriter
import hmodel.thirdparty.YOLOP.lib.dataset as dataset
from hmodel.thirdparty.YOLOP.lib.config import cfg
from hmodel.thirdparty.YOLOP.lib.config import update_config
from hmodel.thirdparty.YOLOP.lib.core.loss import get_loss
from hmodel.thirdparty.YOLOP.lib.models.common import Hardswish
# from lib.core.function import train
# from lib.core.function import validate
from toolchain.imodelzoo.models.autodrive.yolop.qat._train_val import train,validate
from hmodel.thirdparty.YOLOP.lib.core.general import fitness
from hmodel.thirdparty.YOLOP.lib.models import get_net
from hmodel.thirdparty.YOLOP.lib.utils import is_parallel
from hmodel.thirdparty.YOLOP.lib.utils.utils import get_optimizer
from hmodel.thirdparty.YOLOP.lib.utils.utils import save_checkpoint
from hmodel.thirdparty.YOLOP.lib.utils.utils import create_logger, select_device
from hmodel.thirdparty.YOLOP.lib.utils import run_anchor
from hmquant.qat_torch import simple_transform_model_qat
from hmquant.qat_torch.model_builder.tracer import trace_model_for_qat
from hmquant.utils.fx import wrap
from hmquant.qat_torch import align_hardware
from copy import deepcopy
from hmodel.utils.helpers import download_ptfile_from_url

from hmquant.qat_torch.apis.transforms import Rgb2Yuv2Tensor2Normalize,adap_model_to_yuv_input
mean=[0.485, 0.456, 0.406]; std=[0.229, 0.224, 0.225]
def parse_args():
    parser = argparse.ArgumentParser(description="Train Multitask network")
    # general
    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)

    # philly
    parser.add_argument("--modelDir", help="model directory", type=str, default="")
    parser.add_argument("--logDir", help="log directory", type=str, default="runs/")
    parser.add_argument("--dataDir", help="data directory", type=str, default="")
    parser.add_argument(
        "--prevModelDir", help="prev Model directory", type=str, default=""
    )

    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="use SyncBatchNorm, only available in DDP mode",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="DDP parameter, do not modify"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.001, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.6, help="IOU threshold for NMS"
    )
    parser.add_argument("--weight", type=str, default="weights/End-to-end.pth")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg.defrost()
    cfg.TRAIN.LR0 = 1e-3;cfg.TRAIN.OPTIMIZER = "adam" 
    # cfg.TRAIN.LR0 = 0.005  # initial learning rate (SGD=1E-2, Adam=1E-3)
    cfg.TRAIN.LRF = 0.01  # final OneCycleLR learning rate (lr0 * lrf)
    cfg.TRAIN.WARMUP_BIASE_LR = 0.01
    cfg.TRAIN.WD = 0.0001
    cfg.TRAIN.END_EPOCH = 10
    cfg.TRAIN.BATCH_SIZE_PER_GPU = 8

    BDD100k_PATH = os.environ.get("DATASETS_PATH") + "/bdd100k"
    cfg.DATASET.DATAROOT = BDD100k_PATH+"/bdd100k/images/100k"
    cfg.DATASET.LABELROOT = BDD100k_PATH + "/det_annotations"
    cfg.DATASET.MASKROOT = BDD100k_PATH+ "/da_seg_annotations"  # the path of da_seg_annotations folder
    cfg.DATASET.LANEROOT = BDD100k_PATH + "/ll_seg_annotations"
    args.weight = download_ptfile_from_url("http://10.10.1.53:8082/artifactory/model_zoo2/houmo/yolop/End-to-end.pth")
    cfg.freeze()
    update_config(cfg,args)
    np.random.seed(3)
    torch.manual_seed(3)
    rank = -1
    device = torch.device("cuda")
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, "train",rank=-1
    )
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    writer_dict = {
        "writer": SummaryWriter(log_dir=tb_log_dir),
        "train_global_steps": 0,
        "valid_global_steps": 0,
    }
    model = get_net(cfg)
    criterion = get_loss(cfg,device=device)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    model_dict = model.state_dict()
    checkpoint_file = args.weight
    if checkpoint_file:
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        checkpoint_dict = checkpoint["state_dict"]
        # checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
        model_dict.update(checkpoint_dict)
        model.load_state_dict(model_dict)
        logger.info("=> loaded checkpoint '{}' ".format(checkpoint_file))
    else:
        print("not load")
    model.gr = 1.0
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=(cfg.TRAIN.SHUFFLE & rank == -1),
        num_workers=cfg.WORKERS,
        # sampler=train_sampler,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    num_batch = len(train_loader)
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=False,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = DataLoaderX(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    logger.info("anchors loaded successfully")
    det = model.module.model[model.module.detector_index] if is_parallel(model) \
        else model.model[model.detector_index]
    # Trace
    from lib.models.common import Hardswish
    wrap(Hardswish)
    quant_cfg = dict(
            input_quant_params = dict(
                x = dict(
                    scale = 0.017
                ),
            ),
            op_wise_cfg=dict(
                Conv2d=dict(o_cfg = dict(calib_metric="percent-0.99999")),
                ConvBn2d=dict(o_cfg=dict(calib_metric="percent-0.99999"),freeze_bn=False),
                ConvReLU2d=dict(o_cfg=dict(calib_metric="percent-0.99999")),
                ConvBnReLU2d=dict(o_cfg=dict(calib_metric="percent-0.99999"),freeze_bn=False),
                Linear=dict(o_cfg=dict(calib_metric="percent-0.99999")),
                Add=dict(o_cfg=dict(calib_metric="percent-0.99999")),
                AdaptiveAvgPool2d=dict(o_cfg=dict(calib_metric="percent-0.99999")),
            )
            )
    traced_model = trace_model_for_qat(deepcopy(model),quant_cfg)
    align_hardware(traced_model,False)
    # from hmquant.utils.fx import FX
    # traced_model = FX(deepcopy(model.train())).op_normalize().fuse().graph_module
    raw_model = model
    model = traced_model
    model.to(device)
    raw_model.to(device)
    optimizer = get_optimizer(cfg, model)
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2)
        * (1 - cfg.TRAIN.LRF)
        + cfg.TRAIN.LRF
    )  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # Trace End
    
    logger.info(str(det.anchors))
    num_warmup = min(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 200)
    print('=> start training...')
    scaler = None
    for epoch in range(begin_epoch+1, cfg.TRAIN.END_EPOCH+1):
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, scaler,
              epoch, num_batch, num_warmup, writer_dict, logger, device,raw_model, rank)
        lr_scheduler.step()
        # evaluate on validation set
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH) and rank in [-1, 0]:
            # print('validate')
            da_segment_results,ll_segment_results,detect_results, total_loss,maps, times = validate(
                epoch,cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir, writer_dict,
                logger, device, -1,raw_model=raw_model
            )
            fi = fitness(np.array(detect_results).reshape(1, -1))  #目标检测评价指标

            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                      'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                          epoch,  loss=total_loss, da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2],
                          p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
                          t_inf=times[0], t_nms=times[1])
            logger.info(msg)

            # if perf_indicator >= best_perf:
            #     best_perf = perf_indicator
            #     best_model = True
            # else:
            #     best_model = False

        # save checkpoint model and best model
        if rank in [-1, 0]:
            savepath = os.path.join(final_output_dir, f'epoch-{epoch}.pth')
            logger.info('=> saving checkpoint to {}'.format(savepath))
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                output_dir=final_output_dir,
                filename=f'epoch-{epoch}.pth'
            )
    # save final model
    if rank in [-1, 0]:
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
        torch.save(model_state, final_model_state_file)
        writer_dict['writer'].close()
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

