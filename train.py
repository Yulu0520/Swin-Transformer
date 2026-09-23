from utils.regression_trainer_cosine_multibatch import RegTrainer
import argparse
import os
import torch
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')
    parser.add_argument('--model-name', default='swin_t_trans', help='the name of the model')
    
    parser.add_argument('--save-dir', default='model',
                        help='directory to save models.')
    parser.add_argument('--save-all', type=bool, default=False,
                        help='whether to save all best model')
    parser.add_argument('--lr', type=float, default=5*1e-6,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')

    parser.add_argument('--resume', default='',
                        help='the path of resume training model')
    parser.add_argument('--init-checkpoint', default=r'',
                        help='initialize from a trained Swin-MAN model without DoRA')

    parser.add_argument('--data-dir', default=r'/home/mmslab-1080/Desktop/img2img-turbo/SHA/day',
                        help='pretrain data directory (with train/ and val/)')
    parser.add_argument('--max-model-num', type=int, default=1,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=10,
                        help='number of pretrain epochs')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='validate every N epochs (1-based)')
    parser.add_argument('--val-start', type=int, default=5,
                        help='first pretrain epoch eligible for validation (1-based)')
    
    parser.add_argument('--no-dora', dest='use_dora', action='store_false',
                        help='run pretrain only, without the DoRA stage')
    parser.set_defaults(use_dora=True)
    parser.add_argument('--dora-rank', type=int, default=8,
                        help='rank of each Swin attention DoRA layer')
    parser.add_argument('--dora-alpha', type=float, default=8.0,
                        help='scaling numerator for each DoRA layer')
    parser.add_argument('--dora-data-dir', default=r'/home/mmslab-1080/Desktop/img2img-turbo/SHA/mix',
                        help='mix directory for independent counting and paired features')
    parser.add_argument('--pair-loss-weight', type=float, default=1.0,
                        help='weight of day/night paired cosine loss in DoRA training')
    parser.add_argument('--dora-epochs', type=int, default=10,
                        help='number of DoRA-only epochs after pretrain')
    parser.add_argument('--dora-val-start', type=int, default=5,
                        help='first DoRA epoch eligible for validation (default: 5)')

    parser.add_argument('--batch-size', type=int, default=4,
                        help='counting-image train batch size')
    parser.add_argument('--pair-batch-size', type=int, default=2,
                        help='day/night pair batch size for DoRA cosine loss')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')
    parser.add_argument('--no-pretrained-backbone', dest='pretrained_backbone',
                        action='store_false',
                        help='initialize the Swin-T backbone without ImageNet weights')
    parser.set_defaults(pretrained_backbone=True)
    parser.add_argument('--tensorboard-log-interval', type=int, default=10,
                        help='steps between TensorBoard batch-loss updates')

    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=384,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=16,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = RegTrainer(args)
    trainer.setup()
    trainer.train()
