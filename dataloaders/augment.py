from albumentations import (

    # Padding (image and mask)
    PadIfNeeded,

    # Non Destructive (image and mask)
    HorizontalFlip,
    VerticalFlip,
    Transpose,
    RandomRotate90,

    # Destructive (image and mask)
    Rotate,
    Resize,
    RandomSizedCrop,
    IAAAffine,
    IAAPerspective,

    # Non Rigid (image and mask)
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    IAAPiecewiseAffine,

    # Non Spatial (image only)
    CLAHE,
    RandomGamma,
    RandomContrast,
    RandomBrightness,
    RandomBrightnessContrast,
    IAAEmboss,
    IAASharpen,
    IAASuperpixels,

    # Blur (image only)
    Blur,
    MedianBlur,
    MotionBlur,
    GaussianBlur,

    # Noise (image only)
    GaussNoise,
    IAAAdditiveGaussianNoise,

    # Helper
    OneOf,
    OneOrOther,
    Lambda,
    ToFloat,
    Compose,
    Normalize,
)

from albumentations.pytorch import ToTensor


def compose_augmentations(img_height,
                          img_width,
                          flip_p=0.5,
                          translate_p=0.5,
                          distort_p=0.5,
                          color_p=0.5,
                          overlays_p=0.15,
                          blur_p=0.25,
                          noise_p=0.25):
    # Resize
    resize_p = 1 if img_height != 1024 else 0

    # Random sized crop
    if img_height == 1024:
        min_max_height = (896, 960)
    elif img_height == 512:
        min_max_height = (448, 480)
    elif img_height == 256:
        min_max_height = (224, 240)
    else:
        raise NotImplementedError

    return Compose([
        Resize(p=resize_p, height=img_height, width=img_width),
        OneOf([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Transpose(p=0.5),
            RandomRotate90(p=0.5),
        ], p=flip_p),
        OneOf([
            Rotate(p=0.25, limit=10),
            RandomSizedCrop(p=0.5,
                            min_max_height=min_max_height,
                            height=img_height,
                            width=img_width),
            OneOrOther(
                IAAAffine(p=0.1, translate_percent=0.05),
                IAAPerspective(p=0.1)
            ),
        ], p=translate_p),
        OneOf([
            ElasticTransform(p=0.5,
                             alpha=10,
                             sigma=img_height * 0.05,
                             alpha_affine=img_height * 0.03,
                             approximate=True),
            GridDistortion(p=0.5),
            OpticalDistortion(p=0.5),
            IAAPiecewiseAffine(p=0.25, scale=(0.01, 0.03)),
        ], p=distort_p),
        OneOrOther(
            OneOf([
                CLAHE(p=0.5),
                RandomGamma(p=0.5),
                RandomContrast(p=0.5),
                RandomBrightness(p=0.5),
                RandomBrightnessContrast(p=0.5),
            ], p=color_p),
            OneOf([
                IAAEmboss(p=0.1),
                IAASharpen(p=0.1),
                IAASuperpixels(p=0)
            ], p=overlays_p)
        ),
        OneOrOther(
            OneOf([
                Blur(p=0.2),
                MedianBlur(p=0.1),
                MotionBlur(p=0.1),
                GaussianBlur(p=0.1),
            ], p=blur_p),
            OneOf([
                GaussNoise(p=0.2),
                IAAAdditiveGaussianNoise(p=0.1)
            ], p=noise_p)
        ),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor(sigmoid=False),
    ])

