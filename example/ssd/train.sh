#python train.py --dataset ucar --image-set train --devkit ./data --finetune 90 --prefix ./model/vgg16_reduced --end-epoch 200
python train.py --dataset ucar --image-set train --devkit ./data --pretrained ./model/ucar_pretrained_89_epoch  --end-epoch 200 --resume 100
