import sys, argparse, os, random, shutil, time, warnings, torch, torch.nn as nn, torch.nn.parallel, torch.backends.cudnn as cudnn, torch.distributed as dist, torch.optim, torch.multiprocessing as mp, torchvision.transforms as transforms, torchvision.datasets as datasets, torchvision.models as models, torch.optim.lr_scheduler as lr_scheduler, socket,torch.utils.data as data,tqdm,torch.optim as optim,cv2 as cv,torchvision,torchvision.models as models,torch.multiprocessing as mp,math,torch.optim.lr_scheduler as sche
from copy import deepcopy
from hmquant.qat_torch.apis import trace_model_for_qat,set_calib,set_fake_quant,align_hardware
from hmquant.qat_torch.apis.transforms import Rgb2Yuv2Tensor2Normalize,adap_model_to_yuv_input

torch.manual_seed(2020)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
best_acc1 = 0

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--data", metavar="DIR", default="/data01/datasets/imagenet", help="path to dataset")
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-5,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=1,
        type=int,
        metavar="N",
        help="print frequency",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument("--pretrained", dest="pretrained", type=bool, default=True,help="use pre-trained model")
    # parser.add_argument("--")
    return parser.parse_args()

# 格式为dict(params=dict(weight=dict(weight_decay=4e-5)),lr=0.01)
def smart_optim(opt_cls: torch.optim.Optimizer, model: nn.Module, opt_cfg=dict()):
    cus_params_cfg = opt_cfg.pop("params", dict())
    local_params = dict()
    for k, v in cus_params_cfg.items():
        local_params[k] = dict(params=list())
        local_params[k].update(v)
    local_params["others"] = dict(
        params=list(), weight_decay=opt_cfg.get("weight_decay", 0)
    )
    for n, p in model.named_parameters():
        match = False
        for k in cus_params_cfg.keys():
            if k in n:
                local_params[k]["params"].append(p)
                match = True
                continue
        if not match:
            local_params["others"]["params"].append(p)
    res = list()
    for k, params in local_params.items():
        res.append(params)
    opt_cfg["params"] = res
    return opt_cls(**opt_cfg)

@torch.no_grad()
def calib(
    local_rank,
    model,
    calib_loader,
):
    set_calib(model)
    print("calibrating")
    model.eval()
    for i, (images, target) in tqdm.tqdm(enumerate(calib_loader)):
        images = images.cuda(local_rank)
        model(images)
        break
    set_fake_quant(model)
    model.train()

@torch.enable_grad()
def train(local_rank, train_loader, model: nn.Module, criterion, optimizer, epoch, args, scheduler=None):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )
    # switch to train mode
    model.train()
    parameters = list(model.parameters())
    end = time.time()
    # IMG_QUANTIZER.to(torch.device(f"cuda:{local_rank}"))
    for i, (images, target) in enumerate(train_loader):
        # warmup_scheduler
        if scheduler:
            scheduler.step()

        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(local_rank, non_blocking=True)
        # images = IMG_QUANTIZER(images)
        target = target.cuda(local_rank, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad.clip_grad_norm_(parameters, 10)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if local_rank in (-1,0):
            if i % args.print_freq == 0:
                progress.display(i)
                # breakpoint()

@torch.no_grad()
def validate(local_rank,val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    print("in val")

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            # images = IMG_QUANTIZER(images)
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if local_rank in (-1,0) and i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        if local_rank in (-1, 0):
            print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))

    return top1.avg

def main(local_rank=0, args=None):
    args.start_epoch = 0
    args.pretrained=True
    global best_acc1
    work_dir = f"output/swin_{time.strftime('%Y%m%d%H%M', time.localtime())}"
    os.makedirs(work_dir,exist_ok=True)
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    cls_net = models.__dict__[args.arch](pretrained=True)
    # 1. 将模型解析为可进行量化训练的模型
    quant_cfg = dict(
        global_wise_cfg=dict(
            o_cfg = dict(calib_metric="percent-0.99999"),
            freeze_bn = True
        ),
        input_quant_params = dict(
            x=dict(
                scale=1/128
            ),
        )
    )
    model = trace_model_for_qat(deepcopy(cls_net.train()), quant_cfg,domain="xh1")
    # 2. 将模型第一层卷积的输入调整为可以接受yuv数据的格式
    adap_model_to_yuv_input(model.conv1,mean=mean,std=std)
    # 3. 让模型的训练跟硬件进行对齐
    align_hardware(model,True) # 执行流程和硬件对齐

    model.cuda(local_rank)
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = smart_optim(  # TODO
        torch.optim.AdamW,
        model,
        dict(
            lr=args.lr,
            params=dict(weight=dict(weight_decay=1e-4), scale=dict(weight_decay=0)),
        ),
    )
    warmup_iterations = 100
    total_iterations = 10000 * args.epochs

    def lr_lambda(current_iteration):
        if current_iteration < warmup_iterations:
            # print(f"lr warmup {current_iteration}/{warmup_iterations}")
            return float(current_iteration) / float(max(1, warmup_iterations))
        progress = float(current_iteration - warmup_iterations) / float(
            max(1, total_iterations - warmup_iterations)
        )
        lr_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_decay

    optimizer = torch.optim.AdamW(
        model.parameters(),
        args.lr,
        weight_decay=0
        # weight_decay=args.weight_decay,
    )
    scheduler = sche.LambdaLR(optimizer, lr_lambda)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=f"cuda:{local_rank}")
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                Rgb2Yuv2Tensor2Normalize(), # 该变换的作用是将RGB输入的模型转化为YUV输入并进行归一化
            ]
        ),
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=None,
    )
    val_loader = data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), Rgb2Yuv2Tensor2Normalize()]),
        ),
        batch_size=256,
        shuffle=False,
        num_workers=args.workers,
    )

    calib(local_rank,model,val_loader)
    for epoch in range(args.start_epoch, args.epochs):
        train(
            local_rank, train_loader, model, criterion, optimizer, epoch, args=args, scheduler = scheduler
        )
        print("start val")
        acc1 = validate(local_rank,val_loader, model, criterion, args)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if local_rank in (-1,0):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,filename=f"{work_dir}/epoch_{epoch}.pth.tar"
            )


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
