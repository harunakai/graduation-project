# graduation-project
毕业设计渣作，基于Pytorch的行人重识别

使用时在命令框输入
> python train_aligned.py --save_dir 自己的路径 --root 数据集根目录 -a 模型选择，默认为ResNet50 -d 数据集选择，默认为Market1501

基本是copy罗浩博士的[AlignedReID++](https://github.com/michuanhaohao/AlignedReID）
自行修改了最后一层步长，并增加了DenseNet169骨干网络，增加了warm up，最好不要对densent使用warm up，效果很差甚至可能还不如不预热
