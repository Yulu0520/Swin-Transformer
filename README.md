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

[arxiv](https://arxiv.org/pdf/2203.02636.pdf) | [知乎](https://zhuanlan.zhihu.com/p/478023612) | [B站](https://www.bilibili.com/video/BV13Y411u7r5?share_source=copy_web)

![image](structure.png)

## Train
1. Install dependencies with `pip install -r requirements.txt`.
2. Dowload Dataset JHU++ or UCF-QNRF.
3. Preprocess them with `preprocess_dataset_jhu.py` or `preprocess_dataset_ucf.py`.
4. Run `python train.py --data-dir <pretrain-root> --dora-data-dir <dora-root> --batch-size 2 --max-epoch 10 --dora-epochs 10`.
5. Wait patiently for the program to finish.
6. Then you will get a good counting model!

The ImageNet-1K Swin-T weights are downloaded automatically for a new
training run. Pass `--no-pretrained-backbone` to disable this. Old VGG19
checkpoints are not compatible with the Swin-T model.

`--max-epoch` controls pretrain, and `--dora-epochs` controls DoRA-only
training. Both default to 10. `--data-dir` supplies pretrain `train/` and
`val/` folders; `--dora-data-dir` independently supplies the DoRA-stage
`train/` and `val/` folders. The current `train.py` defaults point to the
local SHA `day` and `mix` datasets respectively; pass both paths explicitly
for another setup. DoRA rank and alpha default to 8; change them
with `--dora-rank` and `--dora-alpha`. Use `--no-dora` for pretrain only. To
skip pretrain and begin DoRA-only training from an already trained,
**non-DoRA** Swin-MAN checkpoint, pass `--init-checkpoint <checkpoint.pth>`.
Use `--resume <phase_epoch_ckpt.tar>` to resume a stage. The best pretrain
checkpoint is `pretrain_best_model.pth`; the best DoRA checkpoint is
`dora_best_model.pth`, also available as `best_model.pth`. The final epoch of
each stage is always validated, even if it is outside the validation schedule.
Test and visualization scripts detect DoRA from the checkpoint automatically.

Training and validation progress are displayed with tqdm. TensorBoard scalar
logs are stored in `model/<run>/tensorboard/` and can be viewed with:

```
tensorboard --logdir model
```

TensorBoard metrics are grouped under `pretrain/` and `dora/`. Epoch numbers
in each stage start at 1. For example, `--max-epoch 10 --dora-epochs 10
--val-start 5 --val-epoch 5` runs 10 pretrain epochs followed by 10 DoRA-only
epochs, validating at epochs 5 and 10 in each stage. Set `--dora-val-start`
to use a different first validation epoch for DoRA; if omitted, it inherits
`--val-start`. The validation interval `--val-epoch` is shared by both stages.


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
