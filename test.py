import torch
import wandb
import os 
import torchvision.transforms.functional as F
from operator import add
from tqdm import tqdm
from wandb_init import parser_init
from utils.metrics import calculate_metrics
#from data.data_loader import batch_random_walker_pseudo_mask
from data.data_loader_ssl_pretrained import loader
from models.Model import ATTNext

def update_res(res, **kwargs):
    # string -> dict
    res_clean = res.strip("[]")
    items = res_clean.split(" ")
    d = {}
    for item in items:
        key, val = item.split("=")
        d[key] = val

    # update values
    for k, v in kwargs.items():
        d[k] = str(v)

    # dict -> string
    new_res = "[" + " ".join([f"{k}={v}" for k, v in d.items()]) + "]"
    return new_res


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
    if folder is None:
        raise ValueError(f"Unsupported dataset: {data}")
    output_key = "ML_DATA_OUTPUT" if torch.cuda.is_available() else "ML_DATA_OUTPUT_LOCAL"
    base_path = os.environ.get(output_key) or os.environ.get("ML_DATA_OUTPUT")
    if not base_path:
        raise EnvironmentError(f"{output_key} must point to the checkpoint output directory")
    return os.path.join(base_path, folder)

    
if __name__ == "__main__":

    data = os.environ.get("TOPODISTILL_DATASET", "isic_2018_1")
    test_data = os.environ.get("TOPODISTILL_TEST_DATASET", "PH2Dataset")
    seed = int(os.environ.get("TOPODISTILL_SEED", "932"))
    training_mode, op, dinowithsegloss, addtopoloss = "supervised", "train", True, False
    device          = using_device()
    folder_path     = setup_paths(data)

    args, res    = parser_init("segmentation task", op, training_mode)
    res             = " ".join(res)
    res             = "["+res+"]"
    
    args.aug            = False
    args.shuffle        = False
    args.op             = "test"

    #config      = wandb_init(os.environ["WANDB_API_KEY"], os.environ["WANDB_DIR"], args, data, dinowithsegloss)
    def create_loader(operation,data):

        return loader(operation,args.mode, args.sslmode_modelname, args.bsize, args.workers,args.imsize, args.cutoutpr, args.cutoutbox, args.shuffle, args.sratio, data,seed)

    model     = ATTNext(args.mode).to(device)
    # model_dino     = ATTNext(args.mode).to(device)
    # model_supervised     = ATTNext(args.mode).to(device)

    res_dino = update_res(res, epochs=499)
    res_supervised = update_res(res, epochs=450, mode="supervised")

    checkpoint_path = os.environ.get("TOPODISTILL_MODEL_CHECKPOINT")
    if not checkpoint_path:
        checkpoint_path = folder_path+str(model.__class__.__name__)+str(res)+f"_seed_{seed}"
    # checkpoint_path_dino = folder_path+str(model.__class__.__name__)+str(res_dino)#+f"_seed_{seed}"
    # checkpoint_path_supervised = folder_path+str(model.__class__.__name__)+str(res_supervised)#+f"_seed_{seed}"

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. "
            "Set TOPODISTILL_MODEL_CHECKPOINT to override it."
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    # model_dino.load_state_dict(torch.load(checkpoint_path_dino, map_location=torch.device('cpu')))
    # model_supervised.load_state_dict(torch.load(checkpoint_path_supervised, map_location=torch.device('cpu')))

    test_loader      = create_loader(args.op, data=test_data)

    print(f"model:",checkpoint_path)
    print('test_loader loader transform',test_loader.dataset.tr)

    print(f"testing with {len(test_loader)*args.bsize} images")

    metrics_score = [ 0.0, 0.0, 0.0, 0.0, 0.0]
    idx=0

    image_table = wandb.Table(columns=["mode/s_ratio/epoch", "image", "pred", "target"])
    metrics_table = wandb.Table(columns=["mode/s_ratio/epoch","Jaccard", "f1", "recall", "precision", "accuracy"])

    for batch in tqdm(test_loader, desc="testing", leave=False):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            model_output = model(images)
            # model_output_dino = model_dino(images)
            # model_output_supervised = model_supervised(images)

            prediction = torch.sigmoid(model_output)
            # prediction_dino = torch.sigmoid(model_output_dino)
            # prediction_supervised = torch.sigmoid(model_output_supervised)

            score = calculate_metrics(labels.detach().cpu(), prediction.detach().cpu())
            metrics_score = list(map(add, metrics_score, score))

            acc = {
                "jaccard": metrics_score[0]/len(test_loader),
                "f1": metrics_score[1]/len(test_loader),
                "recall": metrics_score[2]/len(test_loader),
                "precision": metrics_score[3]/len(test_loader),
                "acc": metrics_score[4]/len(test_loader)
            }

            print(f" IoU: {acc['jaccard']:1.4f} - F1(Dice): {acc['f1']:1.4f} - Recall: {acc['recall']:1.4f} - "
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

    # wandb.log({
    #     "predictions_table": image_table,
    #     "metrics_table": metrics_table
    # })


    # wandb.finish()
