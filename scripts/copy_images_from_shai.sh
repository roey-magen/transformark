mkdir ~/deployment/hidden/HiDDeN-master/data
mkdir ~/deployment/hidden/HiDDeN-master/data/train
mkdir ~/deployment/hidden/HiDDeN-master/data/train/train_class
mkdir ~/deployment/hidden/HiDDeN-master/data/val
mkdir ~/deployment/hidden/HiDDeN-master/data/train/val_class

cd /home/projects/bagon/shared/coco/train2017
cp $(ls . | head -n 10000) ~/deployment/hidden/HiDDeN-master/data/train/train_class

cd /home/projects/bagon/shared/coco/val2017
cp $(ls . | head -n 1000) ~/deployment/hidden/HiDDeN-master/data/val/val_class
