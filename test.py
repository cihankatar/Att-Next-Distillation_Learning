import torch
import wandb
import os 
import torchvision.transforms.functional as F
from operator import add
from tqdm import tqdm, trange
from wandb_init  import *
from visualization import *
from utils.metrics import *
#from data.data_loader import batch_random_walker_pseudo_mask
from data.data_loader_ssl_pretrained import loader
from augmentation.Augmentation import Cutout
from models.Model import model_dice_bce

def using_device():
    """Set and print the device used for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    return device


def setup_paths(data):
    """Set up data paths for training and validation."""
    folder_mapping = {
        "isic_2018_1": "isic_1/",
        "kvasir_1": "kvasir_1/",
        "ham_1": "ham_1/",
        "PH2Dataset": "PH2Dataset/",
        "isic_2016_1": "isic_2016_1/"
    }
    folder = folder_mapping.get(data)
    base_path = os.environ["ML_DATA_OUTPUT"] if torch.cuda.is_available() else os.environ["ML_DATA_OUTPUT_LOCAL"]
    return os.path.join(base_path, folder)

    
if __name__ == "__main__":

    data, training_mode, op,addtopoloss = 'isic_2018_1', "supervised", "train",False
    device          = using_device()
    folder_path     = setup_paths(data)

    args, res    = parser_init("segmentation task", op, training_mode)
    res             = " ".join(res)
    res             = "["+res+"]"
    
    args.aug            = False
    args.shuffle        = False
    args.op             = "test"

    config          = wandb_init(os.environ["WANDB_API_KEY"], os.environ["WANDB_DIR"], args, data)
    def create_loader(operation):

        return loader(operation,args.mode, args.sslmode_modelname, args.bsize, args.workers,args.imsize, args.cutoutpr, args.cutoutbox, args.shuffle, args.sratio, data)

    model     = model_dice_bce().to(device)

    checkpoint_path = folder_path+str(model.__class__.__name__)+str(res)    
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    
    test_loader      = create_loader(args.op)

    print(f"model path:",res)
    print('test_loader loader transform',test_loader.dataset.tr)

    print(f"testing with {len(test_loader)*args.bsize} images")

    # try:
    #     if torch.cuda.is_available():
    #         model.load_state_dict(torch.load(checkpoint_path))
    #     else: 
    #         model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

    # except:
    #      raise Exception("******* No Checkpoint Path  *********")
    
    metrics_score = [ 0.0, 0.0, 0.0, 0.0, 0.0]
    idx=0

    image_table = wandb.Table(columns=["mode/s_ratio/epoch", "image", "pred", "target"])
    metrics_table = wandb.Table(columns=["mode/s_ratio/epoch","Jaccard", "f1", "recall", "precision", "accuracy"])

    for batch in tqdm(test_loader, desc="testing", leave=False):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            model_output = model(images)
            prediction = torch.sigmoid(model_output)    
            score = calculate_metrics(labels.detach().cpu(), prediction.detach().cpu())
            metrics_score = list(map(add, metrics_score, score))

            acc = {
                "jaccard": metrics_score[0]/len(test_loader),
                "f1": metrics_score[1]/len(test_loader),
                "recall": metrics_score[2]/len(test_loader),
                "precision": metrics_score[3]/len(test_loader),
                "acc": metrics_score[4]/len(test_loader)
            }

            print(f" F1(Dice): {acc['jaccard']:1.4f} - F1(Dice): {acc['f1']:1.4f} - Recall: {acc['recall']:1.4f} - "
                f"Precision: {acc['precision']:1.4f} - Acc: {acc['acc']:1.4f} ")

            if idx == 1 or idx==30:
                for i in range(min(1, images.size(0))):
                    img = F.to_pil_image(images[i].cpu())
                    pred_mask = F.to_pil_image((prediction[i] > 0.5).float().cpu())
                    gt_mask = F.to_pil_image(labels[i].cpu())

                    image_table.add_data(
                        f"{args.mode}, split_ratio {args.sratio},epochs {args.epochs}",
                        wandb.Image(img),
                        wandb.Image(pred_mask),
                        wandb.Image(gt_mask)
                    )
            idx += 1

    metrics_table.add_data(
    f"{args.mode}, split_ratio {args.sratio}, epochs {args.epochs}",
    acc["jaccard"],
    acc["f1"],
    acc["recall"],
    acc["precision"],
    acc["acc"]  )

    wandb.log({
        "predictions_table": image_table,
        "metrics_table": metrics_table
    })


    wandb.finish()
