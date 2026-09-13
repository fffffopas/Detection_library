import numpy as np


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    idx = np.argsort(-conf)

    tp = tp[idx]
    conf = conf[idx]
    pred_cls = pred_cls[idx]

    un_cls, num_cls = np.unique(target_cls, return_counts=True)
    nc = un_cls.shape[0]

    ap = np.zeros((nc, tp.shape[1]))

    for ci, cls in enumerate(un_cls):
        i = pred_cls == cls
        n_l = num_cls[ci]
        n_p = i.sum()

        if n_p == 0 or n_l == 0:
            continue

        tpc = tp[i].cumsum(axis=0)
        fpc = (1 - tp[i]).cumsum(axis=0)

        precision = tpc / (tpc + fpc)

        recall = tpc / (n_l + eps)

        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    return ap

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    mpre = np.flip(mpre)

    mpre = np.maximum.accumulate(mpre)
    mpre = np.flip(mpre)

    x = np.linspace(0, 1, 101)
    ap = np.trapezoid(np.interp(x, mrec, mpre), x)

    return ap