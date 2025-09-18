from itertools import product as product
from math import ceil
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils


pretrained_urls = {
    "mobilenet": "https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth",
    "resnet50":"https://github.com/elliottzheng/face-detection/releases/download/0.0.1/Resnet50_Final.pth"
}


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(
            out_channel // 4, out_channel // 4, stride=1, leaky=leaky
        )
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(
            in_channels_list[0], out_channels, stride=1, leaky=leaky
        )
        self.output2 = conv_bn1X1(
            in_channels_list[1], out_channels, stride=1, leaky=leaky
        )
        self.output3 = conv_bn1X1(
            in_channels_list[2], out_channels, stride=1, leaky=leaky
        )

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(
            output3, size=[output2.size(2), output2.size(3)], mode="nearest"
        )
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(
            output2, size=[output1.size(2), output1.size(3)], mode="nearest"
        )
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(
            inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase="train"):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.cfg = cfg
        self.phase = phase
        backbone = None
        if cfg["name"] == "mobilenet0.25":
            backbone = MobileNetV1()
        elif cfg["name"] == "Resnet50":
            import torchvision.models as models
            pretrained = cfg["pretrain"] # remove warning in higher version torch
            assert not pretrained
            backbone = models.resnet50()

        self.body = _utils.IntermediateLayerGetter(backbone, cfg["return_layers"])
        in_channels_stage2 = cfg["in_channel"]
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg["out_channel"]
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.LandmarkHead = self._make_landmark_head(
            fpn_num=3, inchannels=cfg["out_channel"]
        )

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat(
            [self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        classifications = torch.cat(
            [self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1
        )

        if self.phase == "train":
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (
                bbox_regressions,
                F.softmax(classifications, dim=-1),
                ldm_regressions,
            )
        return output


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    landms = torch.cat(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        dim=1,
    )
    return landms


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase="train"):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [
                        x * self.steps[k] / self.image_size[1] for x in [j + 0.5]
                    ]
                    dense_cy = [
                        y * self.steps[k] / self.image_size[0] for y in [i + 0.5]
                    ]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


cfg_mnet = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 32,
    "ngpu": 1,
    "epoch": 250,
    "decay1": 190,
    "decay2": 220,
    "image_size": 640,
    "pretrain": True,
    "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
    "in_channel": 32,
    "out_channel": 64,
}

cfg_re50 = {
    "name": "Resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 24,
    "ngpu": 4,
    "epoch": 100,
    "decay1": 70,
    "decay2": 90,
    "image_size": 840,
    "pretrain": False,
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256,
}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu, network: str):
    if pretrained_path is None:
        url = pretrained_urls[network]
        if load_to_cpu:
            pretrained_dict = torch.utils.model_zoo.load_url(
                url, map_location=lambda storage, loc: storage
            )
        else:
            pretrained_dict = torch.utils.model_zoo.load_url(
                url, map_location=lambda storage, loc: storage.cuda(device)
            )
    else:
        if load_to_cpu:
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage
            )
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(
                pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
            )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_net(model_path, device, network="mobilenet"):
    if network == "mobilenet":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50
    else:
        raise NotImplementedError(network)
    # net and model
    net = RetinaFace(cfg=cfg, phase="test")
    net = load_model(net, model_path, True, network=network)
    net.eval()
    # cudnn.benchmark = True
    net = net.to(device)
    return net


def parse_det(det):
    landmarks = det[5:].reshape(5, 2)
    box = det[:4]
    score = det[4]
    return box, landmarks, score


def resize_keep_aspect_and_pad_to_corner(images, target_size=640, is_batch=True):
    """
    对图片进行等比例缩放，并填充到目标尺寸的左上角
    
    Args:
        images: 单张图片（H,W,3）或批量图片（B,H,W,3）的numpy数组
        target_size: 目标尺寸，默认640x640
        is_batch: 输入是否为批量图片
        
    Returns:
        processed_images: 处理后的图片，左上角为原图，其余部分为0
        scale_factors: 每张图片的缩放因子 (batch中每张图的缩放比例可能不同)
    """
    if not is_batch:
        images = [images]  # 将单张图片包装成列表以统一处理
        
    processed_images = []
    scale_factors = []
    
    for img in images:
        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w)  # 计算等比例缩放因子
        
        # 等比例缩放
        new_h, new_w = int(h * scale), int(w * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建目标尺寸的空白画布（全0，即黑色背景）
        padded_img = np.zeros((target_size, target_size, 3), dtype=img.dtype)
        
        # 将缩放后的图片放置在左上角
        padded_img[:new_h, :new_w] = resized_img
        
        processed_images.append(padded_img)
        scale_factors.append(scale)
    
    if is_batch:
        return np.array(processed_images), np.array(scale_factors)
    else:
        return processed_images[0], scale_factors[0]


def post_process(
    loc,
    conf,
    landms,
    prior_data,
    cfg,
    scale,
    scale1,
    resize,
    confidence_threshold,
    top_k,
    nms_threshold,
    keep_top_k,
):
    boxes = decode(loc, prior_data, cfg["variance"])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.cpu().numpy()[:, 1]
    landms_copy = decode_landm(landms, prior_data, cfg["variance"])

    landms_copy = landms_copy * scale1 / resize
    landms_copy = landms_copy.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms_copy = landms_copy[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms_copy = landms_copy[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms_copy = landms_copy[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms_copy = landms_copy[:keep_top_k, :]

    dets = np.concatenate((dets, landms_copy), axis=1)
    # show image
    dets = sorted(dets, key=lambda x: x[4], reverse=True)
    dets = [parse_det(x) for x in dets]

    return dets


def convert2dict(all_dets):
    all_dict_results = []
    for faces in all_dets:
        dict_results = []
        for face in faces:
            box, landmarks, score = face
            dict_results.append(
                {
                    "box": box,
                    "kps": landmarks,
                    "score": score,
                }
            )
        all_dict_results.append(dict_results)
    return all_dict_results


def post_process_for_corner_padding(
    loc, conf, landms, prior_data, cfg, scale, scale1, resize,
    confidence_threshold, top_k, nms_threshold, keep_top_k,
    scale_factor
):
    """
    针对左上角填充的图像进行后处理
    
    Args:
        loc, conf, landms: 模型输出
        prior_data, cfg, scale, scale1: 模型配置
        resize: 缩放因子
        confidence_threshold, top_k, nms_threshold, keep_top_k: NMS参数
        scale_factor: 等比例缩放因子（将640x640还原到原图所需的因子）
        
    Returns:
        检测结果列表，坐标已还原到原图尺度
    """
    boxes = decode(loc, prior_data, cfg["variance"])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.cpu().numpy()[:, 1]
    landms_copy = decode_landm(landms, prior_data, cfg["variance"])
    landms_copy = landms_copy * scale1 / resize
    landms_copy = landms_copy.cpu().numpy()
    
    # 筛选高置信度的检测结果
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms_copy = landms_copy[inds]
    scores = scores[inds]
    
    # NMS前保留top-k个结果
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms_copy = landms_copy[order]
    scores = scores[order]
    
    # 执行NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms_copy = landms_copy[keep]
    
    # 保留NMS后的top-k个结果
    dets = dets[:keep_top_k, :]
    landms_copy = landms_copy[:keep_top_k, :]
    
    # 特殊处理：还原到原图坐标
    # 由于是等比例缩放后左上角填充，需要除以缩放因子
    if scale_factor > 0:
        # 检测框还原
        dets[:, :4] /= scale_factor
        
        # 关键点还原
        landms_copy /= scale_factor
    
    # 合并结果
    dets = np.concatenate((dets, landms_copy), axis=1)
    dets = sorted(dets, key=lambda x: x[4], reverse=True)
    dets = [parse_det(x) for x in dets]
    
    return dets

@torch.no_grad()
def pseudo_batch_detect(net, images, device, is_tensor=False, threshold=0.5, cv=False, 
                 return_dict=False, 
                 fp16=False, 
                 max_size=-1,
                 ):
    """
    批量检测人脸
    
    Args:
        images: 图片列表或批量图片，每张可以是不同尺寸
        其他参数...
        
    Returns:
        检测结果，坐标已还原到原图尺度
    """

    if max_size == -1:
        max_size = 640

    # 记录原始尺寸
    if not is_tensor:
        orig_sizes = []
        for img in images:
            h, w = img.shape[:2]
            orig_sizes.append((w, h))
            
        # 使用等比例缩放并填充到左上角
        resized_imgs, scale_factors = resize_keep_aspect_and_pad_to_corner(
            images, target_size=max_size, is_batch=True
        )
        # 转换为tensor
        img = torch.from_numpy(np.float32(resized_imgs)).to(device)
    else:
        # 如果已经是tensor，可能需要另外的处理逻辑
        img = images.to(device)
        # 假设tensor已经是batch形式，收集原始尺寸
        orig_sizes = [(img.shape[2], img.shape[1]) for _ in range(img.shape[0])]
        scale_factors = [1.0] * img.shape[0]  # 默认缩放因子为1
    
    if fp16:
        img = img.half()
        
    img = img.permute(0, 3, 1, 2)  # BHWC -> BCHW
    
    # RGB -> BGR 如果需要
    if cv:
        img = img[:, [2, 1, 0], :, :]
        
    # 标准化
    mean = torch.as_tensor([104, 117, 123], dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    img -= mean
    
    # 模型推理
    loc, conf, landms = net(img)
    
    # 设置prior box
    cfg = net.cfg
    priorbox = PriorBox(cfg, image_size=(max_size, max_size))
    priors = priorbox.forward()
    prior_data = priors.to(device)
    
    # 设置scale和scale1
    scale = torch.as_tensor(
        [max_size, max_size, max_size, max_size],
        dtype=img.dtype, device=img.device
    )
    scale1 = torch.as_tensor(
        [max_size, max_size] * 5,
        dtype=img.dtype, device=img.device
    )
    
    # 后处理参数
    confidence_threshold = threshold
    top_k = 5000
    nms_threshold = 0.4
    keep_top_k = 750
    
    # 批量后处理，并还原到原图坐标
    all_dets = []
    for idx, (loc_i, conf_i, landms_i) in enumerate(zip(loc, conf, landms)):
        scale_factor = scale_factors[idx]
        
        # 后处理得到在640x640上的坐标
        dets = post_process_for_corner_padding(
            loc_i, conf_i, landms_i, prior_data, cfg, 
            scale, scale1, resize=1.0,
            confidence_threshold=confidence_threshold,
            top_k=top_k, nms_threshold=nms_threshold, 
            keep_top_k=keep_top_k,
            scale_factor=scale_factor,  # 传入缩放因子
        )
        all_dets.append(dets)
    
    if return_dict:
        return convert2dict(all_dets)
    else:
        return all_dets

@torch.no_grad()
def batch_detect(net, images, device, is_tensor=False, threshold=0.5, cv=False, 
                 resize = 1, 
                 max_size: int = -1, # maximum size of the image before feeding to the model
                 return_dict=False, 
                 fp16=False, 
                 resize_device='gpu'):
    """
    Perform batch face detection on a set of images using a given network.

    Args:
        net (torch.nn.Module): The face detection network.
        images (numpy.ndarray or torch.Tensor): The input images. If `is_tensor` is False, it should be a numpy array. Otherwise, it should be a torch tensor.
        device (str): The device to run the inference on (e.g., 'cpu', 'cuda').
        is_tensor (bool, optional): Whether the input images are already torch tensors. Defaults to False.
        threshold (float, optional): The confidence threshold for face detection. Defaults to 0.5.
        cv (bool, optional): Whether to convert the images from BGR to RGB. Defaults to False.
        resize (float, optional): The resize factor for the images. Defaults to 1.
        max_size (int, optional): The maximum size of the image before feeding it to the model. Defaults to -1.
        return_dict (bool, optional): Whether to return the results as a dictionary. Defaults to False.
        fp16 (bool, optional): Whether to use float16 precision for inference. Defaults to False.
        resize_device (str, optional): The device to perform image resizing on. Defaults to 'gpu'.

    Returns:
        list or list of lists: The detected faces. Each face is represented as a list containing the bounding box coordinates, landmarks, and confidence score. If `return_dict` is True, the results are returned as a list of dictionaries, where each dictionary contains the 'box', 'kps', and 'score' keys.

    Raises:
        NotImplementedError: If the input images are not of the same size.
    """
    confidence_threshold = threshold
    cfg = net.cfg
    top_k = 5000
    nms_threshold = 0.4
    keep_top_k = 750
    if fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    if not is_tensor:
        try:
            img = np.float32(images)
        except ValueError:
            raise NotImplementedError("Input images must of same size")
        img = torch.from_numpy(img)
    else:
        img = images
    
    if resize == 1 and max_size != -1:
        # compute resize factor
        resize = max_size / max(img.shape[1], img.shape[2])
    elif resize != 1 and max_size != -1:
        estimated_max_size = max(img.shape[1], img.shape[2]) * resize
        if estimated_max_size > max_size: # if the estimated size is larger than the max_size, use max_size
            resize = max_size / max(img.shape[1], img.shape[2])
    
    if resize != 1 and resize_device == 'cpu':
        initial_device = 'cpu' # resize on cpu to prevent memory error
    else:
        initial_device = device
        
    img = img.to(device=initial_device, dtype=dtype)
    img = img.permute(0, 3, 1, 2) # bhwc to bchw
    
    if resize != 1:
        img = F.interpolate(img, size=(int(img.shape[2] * resize), int(img.shape[3] * resize)), mode='bilinear', align_corners=False)

    if initial_device != device:
        img = img.to(device=device) 

    if cv:
        img = img[:, [2, 1, 0], :, :]  # rgb to bgr
    mean = torch.as_tensor([104, 117, 123], dtype=img.dtype, device=img.device).view(
        1, 3, 1, 1
    )
    img -= mean
    (
        batch_size,
        _,
        im_height,
        im_width,
    ) = img.shape
    scale = torch.as_tensor(
        [im_width, im_height, im_width, im_height],
        dtype=img.dtype,
        device=img.device,
    )
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    prior_data = priors.to(device)
    scale1 = torch.as_tensor(
        [
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
        ],
        dtype=img.dtype,
        device=img.device,
    )
    scale1 = scale1.to(device)

    all_dets = [
        post_process(
            loc_i,
            conf_i,
            landms_i,
            prior_data,
            cfg,
            scale,
            scale1,
            resize,
            confidence_threshold,
            top_k,
            nms_threshold,
            keep_top_k,
        )
        for loc_i, conf_i, landms_i in zip(loc, conf, landms)
    ]
    if return_dict:
        return convert2dict(all_dets)
    else:
        return all_dets
