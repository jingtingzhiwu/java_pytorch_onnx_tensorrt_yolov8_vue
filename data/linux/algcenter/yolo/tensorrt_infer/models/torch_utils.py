from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms, nms


def seg_postprocess(
        data: Tuple[Tensor],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = data[0][0], data[1][0]
    bboxes, scores, labels, maskconf = outputs.split([4, 1, 1, 32], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), labels.new_zeros((0, )), bboxes.new_zeros((0, 0, 0, 0))
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    idx = batched_nms(bboxes, scores, labels, iou_thres)
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx].int(), maskconf[idx]
    masks = (maskconf @ proto).sigmoid().view(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = F.interpolate(masks[None],
                          shape,
                          mode='bilinear',
                          align_corners=False)[0]
    masks = masks.gt_(0.5)[..., None]
    return bboxes, scores, labels, masks


def pose_postprocess(
        data: Union[Tuple, Tensor],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    outputs = torch.transpose(data[0], 0, 1).contiguous()
    bboxes, scores, kpts = outputs.split([4, 1, 51], 1)
    scores, kpts = scores.squeeze(), kpts.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), bboxes.new_zeros((0, 0, 0))
    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]
    xycenter, wh = bboxes.chunk(2, -1)
    bboxes = torch.cat([xycenter - 0.5 * wh, xycenter + 0.5 * wh], -1)
    idx = nms(bboxes, scores, iou_thres)
    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]
    return bboxes, scores, kpts.reshape(idx.shape[0], -1, 3)


def det_postprocess(data: Tuple[Tensor, Tensor, Tensor, Tensor]):
    # 格式1: 原始YOLOv8输出 [1, 84, 8400]
    if isinstance(data, torch.Tensor) and data.dim() == 3:
        print(f"Debug: det_postprocess input tensor shape: {data.shape}")
        # print("检测到原始YOLOv8输出格式 [batch, 84, 8400]")

        batch_size = data.shape[0]

        # 我们只处理batch_size=1的情况
        if batch_size != 1:
            raise ValueError(f"只支持batch_size=1，当前batch_size={batch_size}")

        # 当前批次预测 [84, 8400]
        pred = data[0]

        # 转置为 [8400, 84]
        pred = pred.permute(1, 0)  # [8400, 84]

        # 分离各部分
        # 前4个: bbox (cx, cy, w, h)
        # 第5个: objectness score
        # 后80个: class scores
        bbox = pred[:, :4]          # [8400, 4] - (cx, cy, w, h)
        cls_score = pred[:, 4:]     # [8400, 80] - class scores

        # 找到每个框的最大类别分数和对应的类别
        max_cls_score, cls_id = torch.max(cls_score, dim=1)  # [8400]

        # 最终分数 = objectness * 最大类别分数
        final_scores = max_cls_score  # [8400]

        # 过滤低分检测 (阈值0.25)
        score_threshold = 0.25
        keep = final_scores > score_threshold

        if keep.sum() == 0:
            # 没有检测到目标
            return (torch.zeros((0, 4), device=data.device),
                    torch.zeros((0,), device=data.device),
                    torch.zeros((0,), device=data.device, dtype=torch.long))

        bbox = bbox[keep]
        scores = final_scores[keep]
        labels = cls_id[keep]

        # 将(cx, cy, w, h)转换为(x1, y1, x2, y2)
        cx, cy, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

        # 应用NMS (IOU阈值0.45)
        if len(boxes_xyxy) > 0:
            # 使用torchvision的NMS
            keep_indices = nms(boxes_xyxy, scores, 0.45)
            boxes_xyxy = boxes_xyxy[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

        # 返回 bboxes, scores, labels
        return boxes_xyxy, scores, labels

    # 格式2: EfficientNMS输出 [num_dets, bboxes, scores, labels] - 4个输出
    elif isinstance(data, (list, tuple)) and len(data) == 4:
        # print("检测到EfficientNMS输出格式 [num_dets, bboxes, scores, labels]")
        num_dets, bboxes, scores, labels = data[0][0], data[1][0], data[2][0], data[3][0]
        nums = num_dets.item()
        if nums == 0:
            return bboxes.new_zeros((0, 4)), scores.new_zeros((0,)), labels.new_zeros((0,))
        # check score negative
        scores[scores < 0] = 1 + scores[scores < 0]
        bboxes = bboxes[:nums]
        scores = scores[:nums]
        labels = labels[:nums]
        return bboxes, scores, labels

    # 未知格式
    else:
        print(f"错误: 不支持的输出格式")
        print(f"  实际输出类型: {type(data)}")
        if isinstance(data, (list, tuple)):
            print(f"  输出数量: {len(data)}")
            for i, d in enumerate(data):
                print(f"    输出{i}: type={type(d)}")
                if hasattr(d, 'shape'):
                    print(f"      shape={d.shape}")

        # 返回空的bboxes, scores, labels
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return (torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device),
                torch.zeros((0,), device=device, dtype=torch.long))


def crop_mask(masks: Tensor, bboxes: Tensor) -> Tensor:
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device,
                     dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device,
                     dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
