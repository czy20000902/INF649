# INF649

This is the repository for course project of INF649 at Ecole Polytechnique.

We implement a cross-modal classification model for the Memotion dataset. The model is based on MobileOne and CLIP. To use it for training or evaluation, please run `python train.py` or `python evaluate.py` with the adjusting arguments. The key arguments are listed below:
## Arguments for training
--task, type=int, default=3
--root, type=str, default='/dataset/memotion_dataset_7k/'
--save_path, type=str, default='checkpoints'
--batch_size, type=int, default=16
--img_size, nargs='+', type=int, default=[224, 224]
--epochs, type=int, default=50

--optimizer, type=str, default='Adam'
--learning_rate, type=float, default=1e-3

--min_lr, type=float, default=1e-6
--warmup_epochs, type=int, default=0
--lr_gamma, type=float, default=0.1

--num_classes, type=int, default=5)

--device, type=str, default="cuda" if torch.cuda.is_available() else "cpu"

## Arguments for evaluation
--task, type=int, default=1
--root, type=str, default='/dataset/memotion_dataset_7k/'
--img_name, type=str, default='image_1.jpg'
--save_path, type=str, default='checkpoints'
--img_size, nargs='+', type=int, default=[224, 224]
--device, type=str, default="cuda" if torch.cuda.is_available() else "cpu"
