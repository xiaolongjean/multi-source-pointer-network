# !/usr/bin/python3
# -*-encoding:utf-8-*-

# ---------------------------------------------------------------------------------
# File: Multi-Scale Feature Pointer Network.
#
# Desc: Multi-Scale Feature Pointer Network Utilizing multi-scale feature and bi-source 
#       input to formulate a summary extraction task as selecting keywords from the 
#       bi-source input via a local pointer mechanism.
#           
#       Some new features are as follows:
#       1. Multi-Scale Embedding: To avoid multi OOV tokens sharing the same embedding.
#       2. Transformer Endoder: Enhancing the capability.
#       3. Local Token Indexing: Addressing the #UNK# tokens in the predicted sequence.
#       4. Masked Beam Search: Avoiding the first predicted token being 'EOS'.
# -----------------------------------------------------------------------------------


import time
import logging
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
from model import MS_Pointer
from data import DataUtils
from config import config
from torch import optim
from tensorboardX import SummaryWriter
from utils import color, hardware_info_printer, set_cuda_device
from torch.optim.lr_scheduler import ReduceLROnPlateau


logging.basicConfig(level=logging.INFO, format='\033[31m[%(levelname)s]\033[0m %(asctime)s  %(message)s')
logger = logging.getLogger(__name__)
 

def epoch_info_printer(epoch, mean_loss, epoch_time, total_time, lr, train_samples, valid_samples=0,
                       valid_loss=None, mean_blue=None, valid_time=0.0):

    valid_loss = round(valid_loss, 5) if valid_loss else None
    valid_blue = round(mean_blue, 5) if mean_blue else None

    print("\n========== Epoch Summary ==========")
    print(color("  Epoch: %s Finished. " % epoch, 1))
    print("  Train Mean Loss: %s " % color(round(mean_loss, 5), 2))
    print("  Valid Mean Loss: %s " % color(valid_loss, 2))
    print("  Valid Mean BLEU: %s " % color(valid_blue, 2))
    print("  Train Data Size: %s " % color(train_samples, 2))
    print("  Valid Data Size: %s " % color(valid_samples, 2))
    print("  Epoch Time Consumed: %ss " % color(epoch_time, 2))
    print("  Valid Time Consumed: %ss " % color(valid_time, 2))
    print("  Total Time Consumed: %ss " % color(total_time, 2))
    print("  Current Learning Rate: %s " % color(round(lr, 8), 2))
    print("===================================")
    print("\n\n\n")


def train(args):
    logger.info(color("Initializing Data Loader ... \n", 1))
    data_loader = DataUtils(args)
    device = set_cuda_device(args.cuda_device)

    logger.info(color("Processing Data ... \n", 1))
    train_data, valid_data, test_data = data_loader.obtain_formatted_data()
    train_data_num, valid_data_num = len(train_data), len(valid_data)

    model = MS_Pointer(args, word2index=data_loader.word2index, char2index=data_loader.char2index,
                       device=device).to(device)
    train_state = {"word2index": data_loader.word2index, "char2index": data_loader.char2index}

    optimizer = optim.Adam(model.parameters(), args.lr)
    train_state["optimizer"] = optimizer.state_dict()

    lr_scheduler = None
    if args.flag_lr_schedule:
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=args.lr_patience, verbose=True)
        train_state["lr_scheduler"] = lr_scheduler.state_dict()

    writer = SummaryWriter(log_dir=args.log_dir, comment="Logs For MS-Pointer Network")

    logger.info(color("Start Training \n", 1))
    train_start_time = time.time()
    
    for epoch in range(args.max_epoch):
        hardware_info_printer()

        epoch_start_time = time.time()
        train_data_batchs = data_loader.get_batch_data(train_data, with_target=True, batch_size=args.batch_size,
                                                       shuffle=True, device=device)
        valid_data_batchs = data_loader.get_batch_data(valid_data, with_target=True, batch_size=args.batch_size,
                                                       device=device)

        total_train_loss = 0.0
        tqdm_generator = tqdm(train_data_batchs, ncols=100)
        for idx, batch in enumerate(tqdm_generator):
            batch_start_time = time.time()
            # set model.training as 'True' to updating running variables for 'Dropout' and 'Normalization'.
            model.train()
            optimizer.zero_grad()

            train_loss = model.get_batch_loss(batch)
            loss = train_loss["mean_loss"]
            
            if torch.isnan(loss):
                raise ValueError("\n\n\033[31m%s\033[0m\n\n" % "【CAUTION】NAN LOSS ENCOUNTERED!")

            loss.backward()
            if args.flag_clip_grad and args.grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_norm, norm_type=2)
            optimizer.step()
            
            total_train_loss += train_loss["batch_loss"].detach().cpu().item()
            
            batch_elapsed_time = round(time.time() - batch_start_time, 2)
            
            info = color("[Train] ", 1) + "Epoch:" + color(epoch, 2) + " Batch:" + color(idx, 2) + " Loss:" + \
                   color(round(loss.detach().cpu().item(), 5), 1) + " Time:" + color(batch_elapsed_time, 2)
            tqdm_generator.set_description(desc=info, refresh=True)
            
        valid_start_time = time.time()
        total_valid_loss, mean_blue_score, valid_tokens, valid_probs = model.validation(valid_data_batchs,
                                                                                        need_pred_result=True)
        mean_valid_loss = total_valid_loss / valid_data_num if valid_data_num else 0.0
        mean_train_loss = total_train_loss / train_data_num

        writer.add_scalar(tag="scalar/train_loss", scalar_value=mean_train_loss, global_step=epoch)
        writer.add_scalar(tag="scalar/valid_loss", scalar_value=mean_valid_loss, global_step=epoch)
        writer.add_scalar(tag="scalar/valid_bleu", scalar_value=mean_blue_score, global_step=epoch)

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if args.flag_lr_schedule:
            lr_scheduler.step(mean_valid_loss)

        date = datetime.datetime.now().strftime('%Y-%m-%d')
        torch.save(model.state_dict(), f"../model/model_state_{date}_{str(epoch)}.pt")
        
        valid_elapsed_time = round(time.time() - valid_start_time, 2)
        epoch_elapsed_time = round(time.time() - epoch_start_time, 2)
        total_elapsed_time = round(time.time() - train_start_time, 2)

        epoch_info_printer(epoch=epoch, mean_loss=mean_train_loss, epoch_time=epoch_elapsed_time,
                           total_time=total_elapsed_time, lr=current_lr, train_samples=train_data_num, 
                           valid_samples=valid_data_num, valid_loss=mean_valid_loss, mean_blue=mean_blue_score, 
                           valid_time=valid_elapsed_time)

    logger.info(color("Training Task Completed! \n\n", 1))


def test(args):
    device = set_cuda_device(args.cuda_device, verbose=False)

    model_path = "../model/model_state_2020-08-03_14.st"
    vocab_path = "../model/vocab.dat"

    print("\nLoading State Dict ...\n")
    vocab_dict = torch.load(vocab_path, map_location=device)
    param_dict = torch.load(model_path, map_location=device)
    word2index, char2index = vocab_dict["word2index"], vocab_dict["char2index"]

    print("Initializing Model ...\n\n")
    model = MS_Pointer(args, word2index=word2index, char2index=char2index,
                       device=device).to(device)
    model.load_state_dict(param_dict)

    # data [ [[source1],[source2]], [[source1],[source2]] ... [[source1],[source2]] ]
    data = [
            [["雪润", "皙白", "护肤套装", "化妆品", "礼盒品", "紧致", "提拉", "补水", "保湿", "粉白", "组合", "正品", "套装"],
             ["弹力", "补水", "国产", "保湿补水"]], 
            [["面霜", "正品", "专卖", "弹力", "蛋白", "凝时", "紧致霜", "补水", "保湿", "提拉", "紧致", "正品", "紧致霜"],
             ["保湿", "紧致", "国产", "保湿补水"]], 
            [["京选", "抖音", "网红", "蜗牛", "保湿", "莹润", "嫩滑", "美肌", "补水", "保湿", "收缩", "毛孔", "护理", "套装"],
             ["保湿", "补水", "国产", "干性", "保湿补水"]], 
            [["活泉", "水动力", "补水", "保湿", "护肤品", "套装", "洗面奶", "水乳液", "化妆品", "套装", "男女", "学生", "补水套装"],
             ["保湿", "补水", "国产", "干性", "补水套装2件套"]], 
            [["乳液面霜", "白肌", "精华乳液"], 
             ["保湿", "补水", "国产", "干性", "中性", "混合性", "保湿补水"]]
           ]
    
    pred_tokens, pred_probs = model.predict([item[0: 2] for item in data])

    for idx, beam_result in enumerate(pred_tokens):
        print(beam_result[0])


if __name__ == "__main__":
    # Basic Config ...
    config = config()
    
    # Model Training ...
    train(config)
    
    # Model Testing ...
    # test(config)
