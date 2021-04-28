# RD-IOD

The code is built on top of https://github.com/jwyang/faster-rcnn.pytorch.git 

### License
For Academic Research Use Only!
### Getting Started
#### the first phase
python trainval_net_ori.py --dataset pascal_voc --net res50 --bs 3 --lr 0.001 --nw 2 --epochs 20 --cuda --save_dir model_save_dir/voc07_19  

#### the second phase
python increment.py --dataset pascal_voc_07_incre --net res50 --bs 1 --lr 0.0001 --nw 2 --epochs 10 --cuda --save_dir model_save_dir/1_inc  --load_model model_baseline/voc07_19/res50/pascal_voc/faster_rcnn_1_20_3290.pth 
#### test
python test_net_increment.py --load_dir model_save_dir/1_inc/res50/pascal_voc_07_incre/faster_rcnn_1_10_557.pth --dataset pascal_voc_test --net res50 --cuda 