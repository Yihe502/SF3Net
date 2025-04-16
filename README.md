# LSFFNet
# LSFFU-Net is implemented using the open source framework MMsegmentation, version 1.2.2, using pytorch 2.0.0.
# Train:
On Vaihingen Dataset:
python tools/train.py configs/lsffunet/mobilev2lsffunet_b4-80k_vaihingen-512.py
On Potsdam Dataset:
python tools/train.py configs/lsffunet/mobilev2lsffunet_b4-80k_potsdam-512.py
On Farmland Dataset:
python tools/train.py configs/lsffunet/mobilev2lsffunet_b4-80k_farmland-512.py

# Test:
On Vaihingen Dataset:
python tools/test.py configs/lsffunet/mobilev2lsffunet_b4-80k_vaihingen-512.py work_dirs/lsffunet/mobilev2lsffunet_vaihingen-512/best_mIoU.pth
On Potsdam Dataset:
python tools/test.py configs/lsffunet/mobilev2lsffunet_b4-80k_potsdam-512.py work_dirs/lsffunet/mobilev2lsffunet_potsdam-512/best_mIoU.pth
On Farmland Dataset:
python tools/test.py configs/lsffunet/mobilev2lsffunet_b4-80k_farmland-512.py work_dirs/lsffunet/mobilev2lsffunet_farmland-512/best_mIoU.pth
