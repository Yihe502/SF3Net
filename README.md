# SFFNet
![SFFU-Net](https://github.com/Yihe502/SFFU-Net/blob/main/SFFU-Net.png)
# SFFU-Net is implemented using the open source framework MMsegmentation, version 1.2.2, using pytorch 2.0.0.
# The Farmland dataset can be obtained from https://faculty.nuist.edu.cn/huanhai/zh_CN/zhym/62898/list/index.htm or https://pan.baidu.com/s/1Ig6f_3wKSbHNCT6kZHi25A?pwd=c308
# Train:
On Vaihingen Dataset:   
```python tools/train.py configs/sffunet/mobilev2lsffunet_b4-80k_vaihingen-512.py```    
On Potsdam Dataset:   
```python tools/train.py configs/sffunet/mobilev2lsffunet_b4-80k_potsdam-512.py```   
On Farmland Dataset:   
```python tools/train.py configs/sffunet/mobilev2lsffunet_b4-80k_farmland-512.py```   

# Test:
On Vaihingen Dataset:   
```python tools/test.py configs/sffunet/mobilev2sffunet_b4-80k_vaihingen-512.py work_dirs/sffunet/mobilev2sffunet_vaihingen-512/best_mIoU.pth```  
On Potsdam Dataset:   
```python tools/test.py configs/sffunet/mobilev2sffunet_b4-80k_potsdam-512.py work_dirs/sffunet/mobilev2sffunet_potsdam-512/best_mIoU.pth```  
On Farmland Dataset:  
```python tools/test.py configs/sffunet/mobilev2sffunet_b4-80k_farmland-512.py work_dirs/sffunet/mobilev2sffunet_farmland-512/best_mIoU.pth```  
