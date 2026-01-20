class YOLOv8Loss(nn.Module):
    """YOLOv8 loss function - simplified but trainable version."""
    def __init__(self, model, num_classes=5, box_gain=7.5, cls_gain=0.5, dfl_gain=1.5):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.dfl_gain = dfl_gain
        self.reg_max = 16  # DFL regress bins
        self.stride = torch.tensor([8., 16., 32.])  # feature strides
        self.nc = num_classes
        self.no = num_classes + 4 * self.reg_max  # predictions per anchor

        # BCE with logits loss for classification
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # Distribution Focal Loss
        self.proj = torch.arange(self.reg_max, dtype=torch.float)

    def forward(self, predictions, targets):
        """
        predictions: list of tensors from model, each shape [B, num_anchors, H, W]
        targets: tensor of shape [B, max_objects, 6] (batch_idx, class, x, y, w, h)
        """
        device = predictions[0].device

        # Initialize losses
        loss_box = torch.tensor(0., device=device)
        loss_cls = torch.tensor(0., device=device)
        loss_dfl = torch.tensor(0., device=device)

        # Projection for DFL
        self.proj = self.proj.to(device)

        batch_size = predictions[0].shape[0]

        for pred_idx, pred in enumerate(predictions):
            stride = self.stride[pred_idx]
            batch, _, ny, nx = pred.shape

            # Reshape predictions: [B, num_anchors, H, W] -> [B, H*W, num_anchors]
            pred = pred.view(batch, self.no, -1).permute(0, 2, 1).contiguous()

            # Get predictions for this feature level
            pred_dist = pred[..., :4 * self.reg_max].view(batch, -1, 4, self.reg_max)
            pred_bbox = pred[..., 4 * self.reg_max:]

            # Decode predictions to get boxes
            pred_xywh = self.decode_bbox(pred_dist, nx, ny, stride)

            # Match targets to predictions
            for batch_idx in range(batch):
                # Get targets for this batch
                batch_targets = targets[batch_idx]
                batch_targets = batch_targets[batch_targets[:, 1] >= 0]  # filter valid

                if len(batch_targets) == 0:
                    continue

                # Convert targets to this feature level's grid
                target_boxes = batch_targets[:, 2:6]  # normalized xywh
                target_classes = batch_targets[:, 1].long()

                # Convert normalized to grid coordinates
                grid_boxes = target_boxes.clone()
                grid_boxes[:, [0, 2]] = grid_boxes[:, [0, 2]] * nx  # x, w
                grid_boxes[:, [1, 3]] = grid_boxes[:, [1, 3]] * ny  # y, h
                grid_boxes[:, 0:2] = grid_boxes[:, 0:2] - 0.5  # center to grid

                # For each target, find the best matching grid cell
                for target_idx in range(len(batch_targets)):
                    x_center, y_center, width, height = grid_boxes[target_idx]

                    # Find closest grid cell
                    grid_x = int(torch.clamp(torch.round(x_center), 0, nx-1))
                    grid_y = int(torch.clamp(torch.round(y_center), 0, ny-1))

                    # Get prediction at this grid cell
                    pred_idx_in_grid = grid_y * nx + grid_x

                    # Get predictions for this cell
                    pred_cell_dist = pred_dist[batch_idx, pred_idx_in_grid]  # [4, reg_max]
                    pred_cell_cls = pred_bbox[batch_idx, pred_idx_in_grid]    # [num_classes]
                    pred_cell_box = pred_xywh[batch_idx, pred_idx_in_grid]    # [4]

                    # Box loss (CIoU)
                    box_loss = self.box_loss(pred_cell_box, target_boxes[target_idx])
                    loss_box += box_loss

                    # Classification loss
                    cls_target = F.one_hot(target_classes[target_idx], self.num_classes).float()
                    cls_loss = self.bce(pred_cell_cls, cls_target).mean()
                    loss_cls += cls_loss

                    # DFL loss
                    dfl_loss_val = self.dfl_loss(pred_cell_dist, target_boxes[target_idx], nx, ny, stride)
                    loss_dfl += dfl_loss_val

        # Normalize by number of targets
        num_targets = max(1, (targets[..., 1] >= 0).sum().item())
        loss_box = loss_box / num_targets
        loss_cls = loss_cls / num_targets
        loss_dfl = loss_dfl / num_targets

        # Total loss
        total_loss = self.box_gain * loss_box + self.cls_gain * loss_cls + self.dfl_gain * loss_dfl

        return total_loss

    def decode_bbox(self, pred_dist, nx, ny, stride):
        """Decode bbox from distribution."""
        batch = pred_dist.shape[0]
        device = pred_dist.device

        # Create grid
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        # Correct the grid shape for broadcasting with pred_xywh[..., :2]
        grid = torch.stack((xv, yv), 2).view(1, ny * nx, 2).float().to(device)

        # Softmax over reg_max dimension
        pred_dist = pred_dist.view(batch, -1, 4, self.reg_max).softmax(dim=-1)

        # Integral over projected bins
        pred_xywh = (pred_dist * self.proj.view(1, 1, 1, -1)).sum(dim=-1)

        # Scale to grid
        pred_xywh[..., :2] = (pred_xywh[..., :2] + grid) * stride
        pred_xywh[..., 2:] = pred_xywh[..., 2:] * stride

        # Normalize by image size (640x640)
        pred_xywh = pred_xywh / 640.0

        return pred_xywh.view(batch, -1, 4)

    def box_loss(self, pred, target, eps=1e-7):
        """CIoU loss."""
        # Convert from center to corner
        pred = self.xywh2xyxy(pred.unsqueeze(0)).squeeze(0)
        target = self.xywh2xyxy(target.unsqueeze(0)).squeeze(0)

        # Intersection
        inter = (torch.min(pred[2], target[2]) - torch.max(pred[0], target[0])).clamp(0) * \
                (torch.min(pred[3], target[3]) - torch.max(pred[1], target[1])).clamp(0)

        # Union
        w1, h1 = pred[2] - pred[0], pred[3] - pred[1]
        w2, h2 = target[2] - target[0], target[3] - target[1]
        union = w1 * h1 + w2 * h2 - inter + eps

        # IoU
        iou = inter / union

        # CIoU
        cw = torch.max(pred[2], target[2]) - torch.min(pred[0], target[0])
        ch = torch.max(pred[3], target[3]) - torch.min(pred[1], target[1])
        c2 = cw**2 + ch**2 + eps
        rho2 = ((pred[0] + pred[2] - target[0] - target[2])**2 +
                (pred[1] + pred[3] - target[1] - target[3])**2) / 4
        v = (4 / (torch.pi**2)) * torch.pow(torch.atan(w2/h2) - torch.atan(w1/h1), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))

        return 1.0 - iou + rho2/c2 + alpha * v

    def dfl_loss(self, pred_dist, target_box, nx, ny, stride):
        """Distribution Focal Loss."""
        # Target box in grid coordinates
        target_box = target_box * torch.tensor([nx, ny, nx, ny], device=pred_dist.device)

        # Target distribution
        target_left = target_box.floor().long()
        target_right = target_box.ceil().long()
        weight_left = target_right.float() - target_box
        weight_right = target_box - target_left.float()

        # DFL loss
        loss = torch.tensor([0.0], device=pred_dist.device) # Initialize as a tensor with shape [1]
        for i in range(4):  # x, y, w, h
            # Left bin
            if 0 <= target_left[i] < self.reg_max:
                ce_loss = F.cross_entropy(
                    pred_dist[i:i+1],
                    target_left[i:i+1].clamp(0, self.reg_max-1), # target is already [1], ensure it stays that way
                    reduction='none'
                )
                # Explicitly ensure ce_loss is [1] for consistent broadcasting
                if ce_loss.dim() == 0:
                    ce_loss = ce_loss.unsqueeze(0)
                loss += weight_left[i] * ce_loss

            # Right bin
            if 0 <= target_right[i] < self.reg_max:
                ce_loss = F.cross_entropy(
                    pred_dist[i:i+1],
                    target_right[i:i+1].clamp(0, self.reg_max-1), # target is already [1], ensure it stays that way
                    reduction='none'
                )
                # Explicitly ensure ce_loss is [1] for consistent broadcasting
                if ce_loss.dim() == 0:
                    ce_loss = ce_loss.unsqueeze(0)
                loss += weight_right[i] * ce_loss

        return loss.mean()

    @staticmethod
    def xywh2xyxy(x):
        """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
        y = x.clone()
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y