import torch
import torch.nn as nn




class TaskAlignedAssigner(nn.Module):
    """A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_circles, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_circles (torch.Tensor): Predicted circles with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_circles),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_circles, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            #LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_circles, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)


    def _forward(self, pd_scores, pd_circles, anc_points, gt_labels, gt_bboxes, mask_gt):
        """Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_circles (torch.Tensor): Predicted circles with shape (bs, num_total_anchors, 3).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_circles (torch.Tensor): Ground truth circles with shape (bs, n_max_boxes, 3).
            mask_gt (torch.Tensor): Mask for valid ground truth circles with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_circles (torch.Tensor): Target circles with shape (bs, num_total_anchors, 3).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_circles, gt_labels, gt_bboxes, anc_points, mask_gt
        )  # mask_pos 与真实框匹配的topk个先验框的mask，形状为(bs, max_num_obj, h*w)；
        # align_metric 为真实框与先验框的对齐度量，形状为(bs, max_num_obj, h*w)；
        # overlaps 为预测框与真实框的iou，形状为(bs, max_num_obj, h*w)；









    def get_pos_mask(self, pd_scores, pd_circles, gt_labels, gt_circles, anc_points, mask_gt):
        """Get positive mask for each ground truth circle.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_circles (torch.Tensor): Predicted circles with shape (bs, num_total_anchors, 3).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_circles (torch.Tensor): Ground truth circles with shape (bs, n_max_boxes, 3).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth circles with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted vs ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_circles)  # 找出真实框内的锚点（掩码）（8, max_objects, 2550000）
        align_metric, overlaps, distence = self.get_box_metrics(pd_scores, pd_circles, gt_labels, gt_circles, mask_in_gts * mask_gt)




    def get_box_metrics(self, pd_scores, pd_circles, gt_labels, gt_circles, mask_gt):
        """Compute alignment metric given predicted and ground truth circles.根据预测和真实的圆计算对齐度量

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_circles (torch.Tensor): Predicted circles with shape (bs, num_total_anchors, 3).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_circles (torch.Tensor): Ground truth circles with shape (bs, n_max_boxes, 3).
            mask_gt (torch.Tensor): Mask for valid ground truth circles with shape (bs, n_max_boxes, 2550000).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_circles.shape[-2] #2550000
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_circles.dtype, device=pd_circles.device)#GT与预测框的IOU[bs,max_objects,2550000]
        distence = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_circles.dtype, device=pd_circles.device)
        circles_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)#预测框得分[bs,max_objects,2550000]

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj（表示批次索引）
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj（表示真实框的类别）
        # Get the scores of each grid for each gt cls
        circles_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_circles = pd_circles.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]#(bs,max_objects,2550000,3)*mask_gt->(bs*max_objects*2550000,3)
        gt_circles = gt_circles.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]#(bs,max_objects,2550000,3)*mask_gt->(bs*max_objects*2550000,3)
        overlaps[mask_gt] = self.circle_ious(gt_circles, pd_circles,na)#(bs,max_objects,2550000) 计算真实圆与预测圆的IOU
        distence[mask_gt] = self.center_distanceLoss(gt_circles, pd_circles,na)#(bs,max_objects,2550000) 计算真实圆与预测圆圆心的距离

        align_metric = circles_scores.pow(self.alpha) * overlaps.pow(self.beta) * distence.pow(self.beta)
        return align_metric, overlaps, distence #(bs,max_objects,2550000) 对齐度量  (bs,max_objects,2550000) IOU (bs,max_objects,2550000) 中心距离损失

    def center_distanceLoss(self, gt_circles, pd_circles,na):
        """计算真实圆与预测圆的中心距离损失

        Args:
            gt_circles (torch.Tensor): Ground truth circles with shape (bs*max_objects*2550000, 3).
            pd_circles (torch.Tensor): Predicted circles with shape (bs*max_objects*2550000, 3).

        Returns:
            center_distance_loss (torch.Tensor): Center distance loss between predicted and ground truth circles.
        """
        x1, y1, r1 = gt_circles.split(1, dim=1)  # (bs*max_objects*2550000, 1)
        x2, y2, r2 = pd_circles.split(1, dim=1)  # (bs*max_objects*2550000, 1)
        d = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # (bs*max_objects*2550000, 1)
        r = r1 + r2  # (bs*max_objects*2550000, 1)
        distance = torch.clamp((r - d) / r, min=0)  # (bs*max_objects*2550000, 1)
        return distance.squeeze(1).view(self.bs, self.n_max_boxes, na)


    def circle_ious(self, gt_circles, pd_circles,na):
        """
        🚀 闪电侠级圆交并比（IoU）计算（专为超大批次设计）
        Args:
            gt_circles: (N, 3) -> [x, y, r] for ground truth
            pd_circles: (N, 3) -> [x, y, r] for predictions
            N = bs * max_objects * 2550000 (任意大小)

        Returns:
            iou: (N,) IoU 值 (0~1)
        """
        # 1️⃣ 提取圆心和半径（0开销切片）
        c0 = gt_circles[:, :2]  # (N, 2)
        r0 = gt_circles[:, 2]  # (N,)
        c1 = pd_circles[:, :2]  # (N, 2)
        r1 = pd_circles[:, 2]  # (N,)
        # 2️⃣ 核心：高效计算交集面积（复用之前优化版）
        inter_area = self.circle_intersection_area_tensor(c0, r0, c1, r1)
        # 3️⃣ 计算并集面积 = 圆1面积 + 圆2面积 - 交集面积
        area0 = torch.pi * r0 ** 2
        area1 = torch.pi * r1 ** 2
        union_area = area0 + area1 - inter_area
        # 4️⃣ 计算 IoU (安全处理除以0)
        iou = inter_area / union_area
        iou = torch.where(union_area > 0, iou, torch.zeros_like(iou))
        return iou.view(self.bs, self.n_max_boxes, na)

    def circle_intersection_area_tensor(self,c0, r0, c1, r1):
        """（同前，已优化到效率天花板）"""
        d = torch.linalg.norm(c0 - c1, dim=-1)
        no_inter = d >= r0 + r1
        contained = d <= torch.abs(r0 - r1)
        area = torch.zeros_like(d)

        # 包含情况：小圆面积
        area = torch.where(
            contained,
            torch.pi * torch.min(r0, r1) ** 2,
            area
        )

        # 仅处理相交样本（95%+样本直接跳过！）
        mask = (~no_inter) & (~contained)
        if not mask.any():
            return area

        # 精准切片
        dm, r0m, r1m = d[mask], r0[mask], r1[mask]

        # 防浮点误差
        t0 = (dm ** 2 + r0m ** 2 - r1m ** 2) / (2 * dm * r0m)
        t1 = (dm ** 2 + r1m ** 2 - r0m ** 2) / (2 * dm * r1m)
        t0, t1 = t0.clamp(-1.0, 1.0), t1.clamp(-1.0, 1.0)

        # 海伦公式安全计算
        sq = torch.sqrt(torch.clamp(
            (-dm + r0m + r1m) * (dm + r0m - r1m) * (dm - r0m + r1m) * (dm + r0m + r1m),
            min=0.0
        ))

        # 计算相交面积
        area_m = r0m ** 2 * torch.acos(t0) + r1m ** 2 * torch.acos(t1) - 0.5 * sq
        area[mask] = area_m

        return area



    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_circles, eps=1e-9):
        """Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_circles (torch.Tensor): Ground truth circles, shape (b, n_boxes, 3).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Notes:
            - b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            - Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0] # 2550000
        bs, n_boxes, _ = gt_circles.shape # (8, max_objects, 3)
        xy, r = gt_circles.view(-1, 1, 3).split([2, 1], dim=2)  # center, radius
        diff = xy - xy_centers[None, :, :]  # (8*max_objects, 2550000, 2)
        r = r.view(-1, 1) # (8*max_objects, 1)
        distances = torch.norm(diff, dim=2)  # (8*max_objects, 2550000)
        return (distances <= r).view(bs, n_boxes, n_anchors).float()  # [8,max_objects,2550000] bool掩码

        # bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors,
        #                                                                                     -1)  # 计算锚点与真实框的偏移量
        #return bbox_deltas.amin(3).gt_(eps)  # [8,max_objects,2550000] bool掩码

class CircleLoss(nn.Module):
    """Criterion class for computing training losses for circles."""

    def __init__(self, reg_max: int = 16):
        """Initialize the CircleLoss module with regularization maximum."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None









class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max





def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)



def dist2circle(distance, anchor_points, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    l, r = distance.chunk(2, dim)
    t = l
    b = r
    lt = torch.cat((l, t), dim)
    rb = torch.cat((r, b), dim)
    x1y1 = anchor_points - lt # (8, 2550000, 2)
    x2y2 = anchor_points + rb # (8, 2550000, 2)
    xy = (x1y1 + x2y2) / 2
    r = (x2y2 - x1y1) / 2 # (8, 2550000, 2)
    r = torch.mean(r, dim=-1, keepdim=True) # (8, 2550000, 1)
    #rec = torch.cat((x1y1, x2y2), dim)

    return torch.cat((xy, r), dim)  # xyr circle
