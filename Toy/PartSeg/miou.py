import numpy as np
from mapping import catidToParts


# Reference: charlesq34/pointnet
class mIoU:
    def __init__(self, num_cls):
        self.total_acc_iou = 0.0
        self.total_per_cat_iou = np.zeros(num_cls)
        self.total_seen = 0.0
        self.total_per_cat_seen = np.zeros(num_cls)

    def compute(self, pred, seg_label, cls_label, batch_size):
        for shape_idx in range(batch_size):
            mask = np.int32(pred[shape_idx]==seg_label[shape_idx])
            total_iou = 0.0
            self.total_seen += 1
            shape = int(cls_label[shape_idx][0])
            self.total_per_cat_seen[shape] += 1
            for part in catidToParts[shape]:
                A = np.sum(pred[shape_idx]==part)
                B = np.sum(seg_label[shape_idx]==part)
                I = np.sum(np.int32(seg_label[shape_idx]==part)*mask)
                U = A+B-I
                if U == 0:
                    total_iou += 1
                else:
                    total_iou += float(I)/U
            avg_iou = total_iou/len(catidToParts[shape])
            self.total_acc_iou += avg_iou
            self.total_per_cat_iou[shape] += avg_iou
    
    def get_iou(self):
        cat_iou = self.total_per_cat_iou/self.total_per_cat_seen
        cat_miou = np.mean(cat_iou)
        total_miou = self.total_acc_iou/self.total_seen
        return cat_iou, cat_miou, total_miou
