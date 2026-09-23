# Boosting-Crowd-Counting-via-Multifaceted-Attention
Official Implement of CVPR 2022 paper 'Boosting Crowd Counting via Multifaceted Attention'

This version replaces the VGG19 backbone with an ImageNet-pretrained
Swin-T backbone. The Swin stage-4 feature is projected from 768 to 512
channels and passed to the original multibatch MAN encoder and density
regression head. CCST/TransCrowd heads, losses, and weakly-supervised
training strategies are not used.

Training has two automatic stages. During pretrain, the complete Swin-T,
768-to-512 adapter, multibatch MAN encoder/FFN, and density head are trained
without DoRA. The best validated pretrain weights are then loaded, and DoRA is
inserted into the `qkv` and output `proj` Linear layers of all 12 Swin-T
attention blocks (24 DoRA modules). During the DoRA stage, every other model
parameter is frozen. No MoE module is used.

DoRA uses two independent training streams. Every image in `mix/train/` is an
independent counting sample with its own `.npy` annotation for Bayesian loss.
Separately, `mix/train/IMG_n.jpg` is paired with
`mix/train/night_IMG_n.jpg` for cosine feature alignment; paired images receive
identical geometric augmentation and do not use annotations in this stream. A
frozen, eval-mode copy of the best pretrain model extracts the day stage-4
feature, while the DoRA backbone extracts the night feature. The DoRA loss is
`Bayesian_mix + pair_loss_weight * paired_cosine`. Validation uses only
`mix/val/night_*.jpg`. Files named `._*.jpg` are ignored.

[arxiv](https://arxiv.org/pdf/2203.02636.pdf) | [知乎](https://zhuanlan.zhihu.com/p/478023612) | [B站](https://www.bilibili.com/video/BV13Y411u7r5?share_source=copy_web)

![image](structure.png)

## Train
1. Install dependencies with `pip install -r requirements.txt`.
2. Dowload Dataset JHU++ or UCF-QNRF.
3. Preprocess them with `preprocess_dataset_jhu.py` or `preprocess_dataset_ucf.py`.
4. Run `python train.py --data-dir <day-root> --dora-data-dir <mix-root> --batch-size 4 --pair-batch-size 2 --max-epoch 10 --dora-epochs 10 --pair-loss-weight 1.0`.
5. Wait patiently for the program to finish.
6. Then you will get a good counting model!

The ImageNet-1K Swin-T weights are downloaded automatically for a new
training run. Pass `--no-pretrained-backbone` to disable this. Old VGG19
checkpoints are not compatible with the Swin-T model.

`--max-epoch` controls pretrain, and `--dora-epochs` controls DoRA-only
training. `--data-dir` supplies day `train/` and `val/` folders;
`--dora-data-dir` supplies independent counting images and paired day/night
images from `mix/train/`, plus night-only validation from `mix/val/`. The
current `train.py` defaults point to the local SHA `day` and `mix` datasets;
pass both paths explicitly for another setup. `--batch-size` controls the mix
counting stream and `--pair-batch-size` controls the paired feature stream.
With 600 mix images, 300 pairs, and defaults 4/2, both have 150 batches. DoRA's
`--pair-loss-weight` defaults to 1.0 and should be tuned using the separate
Bayesian and pair-loss metrics. DoRA rank and alpha default to 8; change them
with `--dora-rank` and `--dora-alpha`. Use `--no-dora` for pretrain only. To
skip pretrain and begin DoRA-only training from an already trained,
**non-DoRA** Swin-MAN checkpoint, pass `--init-checkpoint <checkpoint.pth>`.
Use `--resume <phase_epoch_ckpt.tar>` to resume a stage. The best pretrain
checkpoint is `pretrain_best_model.pth`; the best DoRA checkpoint is
`dora_best_model.pth`, also available as `best_model.pth`. The final epoch of
each stage is always validated, even if it is outside the validation schedule.
DoRA `.tar` checkpoints include the frozen day teacher for reliable resuming.
Test and visualization scripts detect DoRA from the checkpoint automatically.

Training and validation progress are displayed with tqdm. TensorBoard scalar
logs are stored in `model/<run>/tensorboard/` and can be viewed with:

```
tensorboard --logdir model
```

TensorBoard metrics are grouped under `pretrain/` and `dora/`. DoRA records
the independent mix Bayesian loss, paired cosine loss, total loss, and mix
counting metrics. Epoch numbers in
each stage start at 1. `--val-start` applies to pretrain and
`--dora-val-start` to DoRA; `--val-epoch` is the shared validation interval.


## Test
1. Dowload Dataset JHU++ or UCF-QNRF.
2. Preprocess them by 'preprocess_dataset.py' or 'preprocess_dataset_ucf.py'.
3. Provide a checkpoint trained with the Swin-T model.
4. Run `python test.py --data-dir <dataset-root> --save-dir <checkpoint.pth>`.


## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{lin2022boosting,
  title={Boosting Crowd Counting via Multifaceted Attention},
  author={Lin, Hui and Ma, Zhiheng and Ji, Rongrong and Wang, Yaowei and Hong, Xiaopeng},
  booktitle={CVPR},
  year={2022}
}
```
