from .coco import CocoDataset

class HRSCL1Dataset(CocoDataset):

    CLASSES = ('ship', )

class HRSCL2Dataset(CocoDataset):

    CLASSES = (
        'aircraft carrier',
        'warship',
        'merchant ship',
        'submarine',
    )
