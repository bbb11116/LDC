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
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_circles)  # 找出真实框内的锚点（掩码）


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
