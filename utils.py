import numpy as np
import os
import matplotlib.patches as patches

import torch
from torchvision import ops
import pandas as pd


# -------------- Data Utils -------------------


def parse_annotation(annotation_path, image_dir):
    '''
    Parse the annotations from a CSV file
    '''

    # Read the CSV file
    df = pd.read_csv(
        annotation_path,
        header=None,
        names=['img_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label'],
        dtype={
            "img_path": str,
            "xmin": float,
            "ymin": float,
            "ymax": float,
            "label": str
        }
    )

    img_paths = []
    gt_boxes_all = []
    gt_classes_all = []

    # Group by image paths to process each image and its associated annotations
    grouped = df.groupby('img_path')

    name2idx = {k: v for v, k in enumerate(sorted(df.label.unique()))}
    name2idx['pad'] = -1

    for img_path, group in grouped:
        full_img_path = os.path.join(image_dir, img_path)
        img_paths.append(full_img_path)

        # get bboxes and their labels   
        groundtruth_boxes = []
        groundtruth_classes = []

        for _, row in group.iterrows():
            bbox = [row.xmin, row.ymin, row.xmax, row.ymax]

            groundtruth_boxes.append(bbox)
            groundtruth_classes.append(row.label)

        gt_idx = torch.Tensor([name2idx[name] for name in groundtruth_classes])

        gt_boxes_all.append(torch.Tensor(groundtruth_boxes))
        gt_classes_all.append(gt_idx)

    return gt_boxes_all, gt_classes_all, img_paths, name2idx


# -------------- Prepocessing utils ----------------

def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping):
    pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:,
                                                                                                    3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:,
                                                                                                     3]

    tx_ = (gt_cx - anc_cx) / anc_w
    ty_ = (gt_cy - anc_cy) / anc_h
    tw_ = torch.log(gt_w / anc_w)
    th_ = torch.log(gt_h / anc_h)

    return torch.stack([tx_, ty_, tw_, th_], dim=-1)


def gen_anc_centers(out_size):
    out_h, out_w = out_size

    anc_pts_x = torch.arange(0, out_w) + 0.5
    anc_pts_y = torch.arange(0, out_h) + 0.5

    return anc_pts_x, anc_pts_y


def project_bboxes(bboxes, width_scale_factor, height_scale_factor, mode='a2p'):
    assert mode in ['a2p', 'p2a']

    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = (proj_bboxes == -1)  # indicating padded bboxes

    if mode == 'a2p':
        # activation map to pixel image
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        # pixel image to activation map
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor

    proj_bboxes.masked_fill_(invalid_bbox_mask, -1)  # fill padded bboxes back with -1
    proj_bboxes.resize_as_(bboxes)

    return proj_bboxes


def generate_proposals(anchors, offsets):
    # change format of the anchor boxes from 'xyxy' to 'cxcywh'
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

    # apply offsets to anchors to create proposals
    proposals_ = torch.zeros_like(anchors)
    proposals_[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
    proposals_[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
    proposals_[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
    proposals_[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])

    # change format of proposals back from 'cxcywh' to 'xyxy'
    proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

    return proposals


def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size):
    '''
    Generate anchor boxes for a single image
    '''
    n_anc_boxes = len(anc_scales) * len(anc_ratios)  # total number of anchor boxes

    Hmap, Wmap = anc_pts_x.size(dim=0), anc_pts_y.size(dim=0)
    anc_base = torch.zeros(1, Hmap, Wmap, n_anc_boxes, 4)  # shape - [1, Hmap, Wmap, n_anchor_boxes, 4]

    for ix, xc in enumerate(anc_pts_x):
        for jx, yc in enumerate(anc_pts_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4))
            c = 0
            for scale in anc_scales:
                for ratio in anc_ratios:
                    w = scale * ratio
                    h = scale

                    xmin = xc - w / 2
                    ymin = yc - h / 2
                    xmax = xc + w / 2
                    ymax = yc + h / 2

                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                    c += 1

            anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)

    return anc_base


def get_iou_matrix(batch_size, anc_boxes_all, gt_bboxes_all):
    # flatten anchor boxes
    anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)

    # create a placeholder to compute IoUs amongst the boxes
    objects_count = gt_bboxes_all.size(dim=1)

    # shape - [batch_size, n_anchor_boxes, n_objects]
    ious_mat = torch.zeros((*anc_boxes_flat.shape[:-1], objects_count))

    # compute IoU of the anc boxes with the gt boxes for all the images
    for i in range(batch_size):
        gt_bboxes = gt_bboxes_all[i]
        anc_boxes = anc_boxes_flat[i]
        ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)

    return ious_mat


def get_req_anchors(anc_boxes_all, gt_bboxes_all, gt_classes_all, pos_thresh=0.7, neg_thresh=0.2):
    '''
    Prepare necessary data required for training
    
    Input
    ------
    anc_boxes_all - torch.Tensor of shape (batch_size, Hmap, Wmap, n_anchor_boxes, 4)
        all anchor boxes for a batch of images
    gt_bboxes_all - torch.Tensor of shape (batch_size, max_objects, 4)
        padded ground truth boxes for a batch of images
    gt_classes_all - torch.Tensor of shape (batch_size, max_objects)
        padded ground truth classes for a batch of images
        
    Returns
    ---------
    positive_idxs -  torch.Tensor of shape (n_pos,)
        flattened positive indices for all the images in the batch
    negative_idxs - torch.Tensor of shape (n_pos,)
        flattened positive indices for all the images in the batch
    GT_class_positive - torch.Tensor of shape (n_pos,)
        mapped classes of positive anchors
    positive_coords - (n_pos, 4) coords of positive anchors (for visualization)
    negative_coords - (n_pos, 4) coords of nagative anchors (for visualization)
    positive_anc_ind_sep - list of indices to keep track of positive anchors
    '''

    # get the size and shape parameters
    batch_size, Hmap, Wmap, n_anchor_boxes, _ = anc_boxes_all.shape
    max_anchors = gt_bboxes_all.shape[1]  # max number of groundtruth bboxes in a batch

    # get total number of anchor boxes in a single image
    tot_anc_boxes = n_anchor_boxes * Hmap * Wmap

    # get the iou matrix which contains iou of every anchor box
    # against all the groundtruth bboxes in an image
    iou_mat = get_iou_matrix(batch_size, anc_boxes_all, gt_bboxes_all)

    # for every groundtruth bbox in an image, find the iou 
    # with the anchor box which it overlaps the most
    max_iou_per_gt_box, _ = iou_mat.max(dim=1, keepdim=True)

    # get positive anchor boxes

    # condition 1: Get this anchor box, which has the highest overlap
    # This ensures that at least one overlapping GT box will be found
    positive_mask = torch.logical_and(iou_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0)
    # condition 2: Get all anchor boxes with iou above a threshold with any of the gt bboxes
    positive_mask = torch.logical_or(positive_mask, iou_mat > pos_thresh)

    # tensor containing the indices of positive_mask where the value is True
    # Useful for later calculations and visualization
    positive_anc_ind_sep = torch.where(positive_mask)[0]

    # combine all the batches and get the idxs of the positive anchor boxes

    positive_mask = positive_mask.flatten(start_dim=0, end_dim=1)
    positive_idxs = torch.where(positive_mask)[0]

    # for every anchor box, get the iou and the idx of the
    # gt bbox it overlaps with the most
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
    max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)

    # get gt classes of the positive anchor boxes

    # expand gt classes to map against every anchor box
    gt_classes_expand = gt_classes_all.view(batch_size, 1, max_anchors).expand(batch_size, tot_anc_boxes, max_anchors)
    # for every anchor box, consider only the class of the gt bbox it overlaps with the most
    gt_classes_expand = gt_classes_expand

    max_iou_per_anc_ind = max_iou_per_anc_ind.to(device=gt_classes_expand.device)

    GT_class = torch.gather(gt_classes_expand, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1)
    # combine all the batches and get the mapped classes of the positive anchor boxes
    GT_class = GT_class.flatten(start_dim=0, end_dim=1)
    GT_class_positive = GT_class[positive_idxs]

    # expand all the gt bboxes to map against every anchor box
    gt_bboxes_expand = gt_bboxes_all.view(batch_size, 1, max_anchors, 4).expand(batch_size, tot_anc_boxes, max_anchors, 4)

    # for every anchor box, consider only the coordinates of the gt bbox it overlaps with the most
    GT_bboxes = torch.gather(gt_bboxes_expand, -2, max_iou_per_anc_ind.reshape(batch_size, tot_anc_boxes, 1, 1).repeat(1, 1, 1, 4))
    # combine all the batches and get the mapped gt bbox coordinates of the +ve anchor boxes
    GT_bboxes = GT_bboxes.flatten(start_dim=0, end_dim=2)
    GT_bboxes_pos = GT_bboxes[positive_idxs]

    # get coordinates of +ve anc boxes
    anc_boxes_flat = anc_boxes_all.flatten(start_dim=0, end_dim=-2) # flatten all the anchor boxes
    positive_coords = anc_boxes_flat[positive_idxs]

    # calculate gt offsets
    GT_offsets = calc_gt_offsets(positive_coords, GT_bboxes_pos)

    # get negative anchors

    # condition: select the anchor boxes with max iou less than the threshold
    negative_mask = (max_iou_per_anc < neg_thresh)
    negative_idxs = torch.where(negative_mask)[0]

    # sample negative samples to match the positive samples
    samples = torch.randint(0, negative_idxs.shape[0], (positive_idxs.shape[0],))
    negative_idxs = negative_idxs[samples]
    negative_coords = anc_boxes_flat[negative_idxs]

    return positive_idxs, negative_idxs, GT_class_positive, GT_offsets, positive_coords, negative_coords, positive_anc_ind_sep


# # -------------- Visualization utils ----------------

def display_img(img_data, fig, axes):
    for i, img in enumerate(img_data):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)

    return fig, axes


def display_bbox(bboxes, fig, ax, classes=None, in_format='xyxy', color='y', line_width=3):
    if isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes)
    if classes:
        assert len(bboxes) == len(classes)
    # convert boxes to xywh format
    bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt='xywh')

    c = 0
    for box in bboxes:
        x, y, w, h = box.numpy()
        # display bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # display category
        if classes:
            if classes[c] == 'pad':
                continue
            ax.text(x + 5, y + 20, classes[c], bbox=dict(facecolor='yellow', alpha=0.5))
        c += 1

    return fig, ax


def display_grid(x_points, y_points, fig, ax, special_point=None):
    # plot grid
    for x in x_points:
        for y in y_points:
            ax.scatter(x, y, color="w", marker='+')

    # plot a special point we want to emphasize on the grid
    if special_point:
        x, y = special_point
        ax.scatter(x, y, color="red", marker='+')

    return fig, ax
