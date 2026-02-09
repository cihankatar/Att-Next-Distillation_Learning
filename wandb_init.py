import wandb
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

def parse_bool(s):
    return True if str(s)=="True" else False

def config_func(training_mode):
    if training_mode == "ssl":
        configs={
        "mode"              :"ssl",
        "sslmode_modelname" :"Dino",
        "imnetpr"           :False,
        "bsize"             :8, 
        "epochs"            :265,
        "imsize"            :256,
        "lrate"             :0.0001,
        "aug"               :False,
        "shuffle"           :True,
        "sratio"            :None,
        "workers"           :2,
        "cutoutpr"          :0.5,
        "cutoutbox"         :None,
        "cutmixpr"          :0.5,
        "noclasses"         :1,
        }

    elif training_mode == "supervised":

        configs={
        "mode"              :"ssl_pretrained",
        "sslmode_modelname" :"Dino",
        "imnetpr"           :True,
        "bsize"             :8,
        "epochs"            :500,
        "imsize"            :256,
        "lrate"             :0.0001,
        "aug"               :True,
        "shuffle"           :True,
        "sratio"            :0.1,
        "workers"           :2,
        "cutoutpr"          :0.5,
        "cutoutbox"         :25,
        "cutmixpr"          :0.5,
        "noclasses"         :1,
        }


    elif training_mode == "ssl_pretrained":
        ssl_config_to_load = config_func("ssl")
        configs={
        "mode"              :"ssl_pretrained",
        "sslmode_modelname" :"Dino",
        "imnetpr"           :True,
        "bsize"             :8,
        "epochs"            :500,
        "imsize"            :256,
        "lrate"             :0.0001,
        "aug"               :True,
        "shuffle"           :True,
        "sratio"            :0.1,
        "workers"           :2,
        "cutoutpr"          :0.5,
        "cutoutbox"         :25,
        "cutmixpr"          :0.5,
        "ssl_config"        :ssl_config_to_load,
        "noclasses"         :1
        }
        

    else:
        raise Exception('Invalid training type:', training_mode)

    return configs
    
def parser_init(name, op, training_mode=None):

    parser = argparse.ArgumentParser(prog=name, description=op)
    
    configs=config_func(training_mode)

    parser.add_argument("-t", "--op",               type=str,default=op)
    parser.add_argument("-m", "--mode",             type=str,default=configs["mode"])   
    parser.add_argument("-y", "--sslmode_modelname",type=str,default=configs["sslmode_modelname"])
    parser.add_argument("-p", "--imnetpr",          type=parse_bool,default=configs["imnetpr"])
    parser.add_argument("-b", "--bsize",            type=int,default=configs["bsize"])   
    parser.add_argument("-e", "--epochs",           type=int,default=configs["epochs"])
    parser.add_argument("-i", "--imsize",           type=int,default=configs["imsize"])   
    parser.add_argument("-l", "--lrate",            type=float,default=configs["lrate"])  
    parser.add_argument("-a", "--aug",              type=parse_bool,default=configs["aug"])
    parser.add_argument("-s", "--shuffle",          type=parse_bool,default=configs["shuffle"])
    parser.add_argument("-r", "--sratio",           type=float,default=configs["sratio"])
    parser.add_argument("-w", "--workers",          type=int,default=configs["workers"])
    parser.add_argument("-o", "--cutoutpr",         type=float,default=configs["cutoutpr"])
    parser.add_argument("-c", "--cutoutbox",        type=float,default=configs["cutoutbox"])
    parser.add_argument("-x", "--cutmixpr",         type=float,default=configs["cutmixpr"])
    parser.add_argument("-n", "--noclasses",        type=int,default=configs["noclasses"])

    args=parser.parse_args()
    args_key=[]
    args_value=[]
    res=[]

    for key,value in parser.parse_args()._get_kwargs():
        args_key.append(key)
        args_value.append(str(value))

    for i in range(len(args_key)):
        res.append(args_key[i]+"="+args_value[i])

    if training_mode == "ssl_pretrained":
        ssl_config = [f"op={op}"]
        for key,value in configs["ssl_config"].items():
            ssl_config.append(str(key)+"="+str(value))
        
        return args,res,ssl_config
    else:
        return args,res

def wandb_init (WANDB_API_KEY,WANDB_DIR,args,data):
    
    op                  = args.op
    training_mode       = args.mode
    ssl_mode_modelname  = args.sslmode_modelname
    imnetpr             = args.imnetpr
    batch_size          = args.bsize 
    epochs              = args.epochs
    image_size          = args.imsize
    augmentation        = args.aug
    learningrate        = args.lrate
    n_classes           = args.noclasses
    split_ratio         = args.sratio
    shuffle             = args.shuffle
    cutout_pr           = args.cutoutpr
    box_size            = args.cutoutbox
    cutmixpr            = args.cutmixpr
    workers             = args.workers

    print(f'Taining Configs:\noperation:{op}\ntraining_mode:{training_mode}\nssl_mode_modelname:{ssl_mode_modelname}\nimagenetpretrained:{imnetpr} \nbatch_size:{batch_size}, \nepochs:{epochs}, \nimagesize:{image_size}, \naugmentation:{augmentation}, \nl_r:{learningrate}, \nn_classes:{n_classes}, \nshuffle:{shuffle}, \ncutout_pr:{cutout_pr}, \ncutout_box_size:{box_size},\ncutmixpr:{cutmixpr}, \nworkers:{workers},\nsplit_ratio:{split_ratio}')

    wandb.login(key=WANDB_API_KEY)
    if op == "train": 

        if torch.cuda.is_available():
            project_name = "Att-Next-SSL"

        else:
            project_name = "Temp_Att-Next-SSL_local"
                
    else:
        project_name = data+"AAtt-Next-SSL_Test"

                
    wandb.init(project=project_name, dir=WANDB_DIR, name=f"{args.mode}_s{args.sratio}_ep{args.epochs}",
        config={
            "operation"       : op,
            "training_mode"   : training_mode,
            "ssl_mode_modelnm": ssl_mode_modelname,
            "imnetpretrained" : imnetpr,
            "epochs"          : epochs,
            "batch_size"      : batch_size,
            "learningrate"    : learningrate,
            "n_classes"       : n_classes,
            "split_ratio"     : split_ratio,
            "num_workers"     : workers,
            "image_size"      : image_size,
            "cutmix_pr"       : cutmixpr,
            "cutout_pram"     : [cutout_pr,box_size],
            "augmentation"    : augmentation,
            "shuffle"         : shuffle,
            })

    config = wandb.config

    return config
