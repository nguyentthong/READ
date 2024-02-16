import argparse
from data_preprocess.data_builder import SummaryDataModule
from models.bart import BartOrigin
from models.multi_modal_model import BartMultiModal
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import DDPPlugin
from read.model import add_read
import torch


def setup_data(summary_data):
    summary_data.train_loader.dataset.args.image_feature_path = "../data/how2/video_action_features/"
    summary_data.val_loader.dataset.args.image_feature_path = "../data/how2/video_action_features/"
    summary_data.test_loader.dataset.args.image_feature_path = "../data/how2/video_action_features/"
    return summary_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='multi_modal_bart', type=str, help='We have for models to choose, text_only_bart, multi_modal_bart,  text_only_t5 and multi_modal_t5')
    parser.add_argument('-checkpoint', default=None, type=str, help='The checkpoint path')
    parser.add_argument('-log_name', default='multi_modal_bart', type=str, help='lightning log path')
    parser.add_argument('-gpus', default='0', type=str, help='choose gpus to run the code, you can choose multipple gpus')
    parser.add_argument('-max_input_len', type=int, default=512, help='the maximun length for input dialogue')
    parser.add_argument('-max_output_len', type=int, default=64, help='the maximun length for output summary')
    parser.add_argument('-max_img_len', type=int, default=256, help='the maximun length for video features')
    parser.add_argument('-val_save_file', default='./evaluation/temp_valid_file', type=str, help='the validation results for each epoch')
    parser.add_argument('-test_save_file', default='./evaluation/results/test_summaries.txt', type=str, help='the generated summary for testing data')
    parser.add_argument('-n_beams', type=int, default=4, help='the number of beams using for generation')
    parser.add_argument('-n_shots', type=int, default=100, help='the shot for few-shot learning')
    parser.add_argument('-no_repeat_ngram_size', type=int, default=3, help='the size of no repeat ngrams during generation')
    parser.add_argument('-learning_rate', default=3e-5, type=float, help='learning rate')
    parser.add_argument('-scheduler_lambda1', default=20, type=int, help='change the learning each lambda1 epoch')
    parser.add_argument('-scheduler_lambda2', default=0.95, type=float, help='the learning rate will times lambda2 for each change')
    parser.add_argument('-num_epochs', type=int, default=100, help='maximun number of training epoches')
    parser.add_argument('-grad_accumulate', type=int, default=10, help='gradient accumulation for this number iterations')
    parser.add_argument('-random_seed', type=int, default=0, help='global random seed')
    parser.add_argument('-do_train', type=str, default='True', help='set True to training, set False to not training')
    parser.add_argument('-do_test', type=str, default='True', help='set True to testing, set False to not testing')
    parser.add_argument('-limit_val_batches', default=1.0, type=float, help='do validation for each epoch')
    parser.add_argument('-val_check_interval', type=float, default=1, help='do validation for each epoch')
    parser.add_argument('-img_lr_factor', type=float, default=1, help='the learning rate for visual guidance part will times this number')
    parser.add_argument('-use_img_trans', action='store_true', help='whether or not to use VTF')
    parser.add_argument('-use_forget_gate', action='store_true', help='whether or not to use forget gate')
    parser.add_argument('-fusion_layer', type=int, default=5, help='number of fusion layers')
    parser.add_argument('-cross_attn_type', type=int, default=0)
    parser.add_argument('-dim_common', type=int, default=256)
    parser.add_argument('-n_attn_heads', type=int, default=1) 
    parser.add_argument('-fusion_in_decoding', action='store_true')
    parser.add_argument('-vision_use_noise', action='store_true')

    args = parser.parse_args()

    seed_everything(args.random_seed)
    logger = pl_loggers.TensorBoardLogger(f'./lightning_logs/{args.log_name}')
    checkpoint_callback = ModelCheckpoint(monitor='validation_Rouge1_one_epoch',
                                          save_last=True,
                                          save_top_k=2,
                                          mode='max',)

    if args.checkpoint == 'None':
        args.checkpoint = None

    trainer = Trainer(deterministic=True,
                      num_sanity_val_steps=1,
                      resume_from_checkpoint=None,
                      logger=logger,
                      gpus=args.gpus,
                      distributed_backend='ddp',
                      plugins=DDPPlugin(find_unused_parameters=False),
                      gradient_clip_val=1.0,
                      max_epochs=args.num_epochs,
                      limit_val_batches=args.limit_val_batches,
                      val_check_interval=1.0,
                      accumulate_grad_batches=args.grad_accumulate,
                      fast_dev_run=False,
                      callbacks=[checkpoint_callback])


    summary_data = torch.load("../data/how2/summary_data.pt")
    summary_data = setup_data(summary_data)
    if args.model == 'text_only_bart':
        model = BartOrigin(args)
    elif args.model == 'multi_modal_bart':
        model = BartMultiModal(args)
    else:
        raise ValueError("Invalid model")

    add_read(model)
    for name, param in model.named_parameters():
        if 'read' not in name:
            param.requires_grad = False

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'], strict=False)
    if args.do_train == 'True':
        trainer.fit(model, train_dataloader=summary_data.train_loader, val_dataloaders=summary_data.val_loader)
    if args.do_test == 'True':
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
        trainer.test(model=model, test_dataloaders=summary_data.test_loader)