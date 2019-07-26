from model.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'aignedxception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.Xception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'drna50':
        return drn.drn_a_50(BatchNorm)
    elif backbone == 'drnc26':
        return drn.drn_c_26(BatchNorm)
    elif backbone == 'drnc42':
        return drn.drn_c_42(BatchNorm)
    elif backbone == 'drnc58':
        return drn.drn_c_58(BatchNorm)
    elif backbone == 'drnd22':
        return drn.drn_d_22(BatchNorm)
    elif backbone == 'drnd24':
        return drn.drn_d_24(BatchNorm)
    elif backbone == 'drnd38':
        return drn.drn_d_38(BatchNorm)
    elif backbone == 'drnd40':
        return drn.drn_d_40(BatchNorm)
    elif backbone == 'drnd54':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'drnd105':
        return drn.drn_d_105(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError

