# !/usr/bin/python3
# -*- coding:utf-8 -*-


import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-random_seed", "--random_seed", type=int, default=1000, help="random seed.")
    parser.add_argument("-numpy_random_seed", "--numpy_random_seed", type=int, default=1001, help="numpy random seed.")
    parser.add_argument("-torch_random_seed", "--torch_random_seed", type=int, default=1002, help="torch random seed.")

    group = parser.add_argument_group("path")
    group.add_argument("-train_data", "--train_data", type=str, default="../data/train.dat", help="Train data path.")
    group.add_argument("-valid_data", "--valid_data", type=str, default="../data/valid.dat", help="valid data path.")
    group.add_argument("-test_data",  "--test_data", type=str, default=None,  help="Test data path.")
    group.add_argument("-vocab_file", "--vocab_file", type=str, default="../model/vocab.dat", help="Vocabulary file.")
    group.add_argument("-model_dir", "--model_dir", type=str, default="../model/", help="Model directory.")

    group = parser.add_argument_group("data")
    group.add_argument("-batch_size",  "--batch_size",  type=int, default=50, help="Batch size.")
    group.add_argument("-max_seq_len",  "--max_seq_len",  type=int, default=25, help="Maximum sentence langth.")
    group.add_argument("-max_token_len",  "--max_token_len",  type=int, default=5, help="Maximum word length.")
    group.add_argument("-max_target_len",  "--max_target_len",  type=int, default=5, help="Maximum target length.")
    group.add_argument("-PAD_idx", "--PAD_idx", type=int, default=0, help="Index of PAD token.")
    group.add_argument("-BOS_idx", "--BOS_idx", type=int, default=1, help="Index of BOS token.")
    group.add_argument("-EOS_idx", "--EOS_idx", type=int, default=2, help="Index of EOS token.")
    group.add_argument("-OOV_idx", "--OOV_idx", type=int, default=3, help="Index of OOV token.")
    group.add_argument("-PAD_token", "--PAD_token", type=str, default="#PAD#", help="symbol of PAD token.")
    group.add_argument("-BOS_token", "--BOS_token", type=str, default="#BOS#", help="symbol of BOS token.")
    group.add_argument("-EOS_token", "--EOS_token", type=str, default="#EOS#", help="symbol of EOS token.")
    group.add_argument("-OOV_token", "--OOV_token", type=str, default="#OOV#", help="symbol of OOV token.")
    
    group = parser.add_argument_group("model")
    group.add_argument("-lr", "--lr", type=float, default=5e-4, help="Learning rate.")
    group.add_argument("-emb_dim", "--emb_dim", type=int, default=128, help="Dimension of embeddings.")
    group.add_argument("-target_embedding_dim", "--target_embedding_dim", type=int, default=128)
    group.add_argument("-hidden_dim", "--hidden_dim", type=int, default=128, help="Dimension of hidden nodes.")
    group.add_argument("-encoder_output_dim_1", "--encoder_output_dim_1", type=int, default=128)
    group.add_argument("-encoder_output_dim_2", "--encoder_output_dim_2", type=int, default=128)
    group.add_argument("-decoder_output_dim", "--decoder_output_dim", type=int, default=128)
    group.add_argument("-use_position_emb", "--use_position_emb", type=bool, default=True, help="positional embedding.")
    group.add_argument("-flag_use_layernorm", "--flag_use_layernorm", type=bool, default=False, help="layernorm.")
    group.add_argument("-flag_use_dropout", "--flag_use_dropout", type=bool, default=False, help="flag use dropout.")

    group = parser.add_argument_group("training")
    group.add_argument("-max_epoch", "--max_epoch", type=int, default=50, help="Maximum number of training epochs.")
    group.add_argument("-cuda_device", "--cuda_device", type=int, default=-1, help="Id of CUDA device.")
    group.add_argument("-traing_patience", "--traing_patience", type=int, default=10, help="early stopping patience.")
    group.add_argument('-dropout_keep_prob', "--dropout_keep_prob", type=float, default=0.9, help='dropout prob')
    group.add_argument("-grad_norm", "--grad_norm", type=float, default=5.0, help="Max gradient norm,")
    group.add_argument("-lr_patience", "--lr_patience", type=int, default=5, help="Patience for reducing lr.")
    group.add_argument("-flag_clip_grad", "--flag_clip_grad", type=bool, default=True, help="clip the grad.")
    group.add_argument("-flag_lr_schedule", "--flag_lr_schedule", type=bool, default=True, help="scheduling lr.")
    
    group = parser.add_argument_group("test")
    group.add_argument("-beam_size", "--beam_size", type=int, default=3, help="Beam size.")
    group.add_argument("-max_decoding_steps", "--max_decoding_steps", type=int, default=5, help="Maxdecoding steps")
    parser.add_argument("-log_dir", "--log_dir", type=str, default="../logs", help="directory for logs")

    args = parser.parse_args()
    return args
