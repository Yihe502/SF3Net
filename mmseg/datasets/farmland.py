# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FarmlandDataset(BaseSegDataset):
    """FarmlandDataset
    """
    METAINFO = dict(
        classes=('Background', 'Other Farmland', 'Tobacco 1', 'Tobacco 2', 'Tobacco 3'),
        palette=[[0, 0, 0], [192, 128, 128], [0, 128, 0], [128, 128, 128], [128, 0, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
