import os

os.environ["OPENMP_NUM_THREADS"] = "16"

import argparse
import json
import random
import imageio
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import *
from utils.imgproc import *

from datasets.real_dataset import SIDEvalDataset, ELDPairEvalDataset
from utils.model_factory import build_denoiser_from_checkpoint


def build_model(args):
    model, meta = build_denoiser_from_checkpoint(
        args.cp_dir,
        device=args.device,
        model=args.model,
        model_width=args.model_width,
        encoder_blocks=args.encoder_blocks,
        middle_blocks=args.middle_blocks,
        decoder_blocks=args.decoder_blocks,
        feature_channels=args.feature_channels,
        num_blocks=args.num_blocks,
    )
    args.resolved_model = meta["model"]
    args.resolved_graph_state = meta["graph_state"]
    args.resolved_meta = meta
    print(
        f"Loaded {meta['model']} checkpoint with {meta['graph_state']} graph "
        f"(weights: {meta['weight_source']})"
    )
    return model


def build_dataloader(args):
    DSLRValidSet = {"sid": SIDEvalDataset, "eld": ELDPairEvalDataset}
    valid_set = DSLRValidSet[args.testset_type](
        clip_low=False, clip_high=True, eval_ratio=args.eval_ratio, max_items=args.max_samples
    )

    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=args.num_workers)
    return valid_loader


@torch.inference_mode()
def valid_one_ep(model, valid_loader, args, plot_res=False):
    """Evaluate without retaining backward activations for full-resolution RAW.

    ``model.eval()`` changes layer behaviour but does not disable autograd.
    SID packed RAW is 1424x2128, so retaining the U-Net graph can exceed the
    11GB RTX 2080 Ti before the first image finishes.  Inference mode is
    numerically equivalent here and substantially lowers peak VRAM.
    """
    psnr_am = AverageMeter("valid_psnr", ":.2f")
    ssim_am = AverageMeter("valid_ssim", ":.4f")

    for data_id, data in enumerate(tqdm(valid_loader)):
        imgs_hr, imgs_lr = tensor_dim5to4(data["hr"]).to(args.device), tensor_dim5to4(data["lr"]).to(args.device)

        imgs_dn = model(imgs_lr)

        imgs_dn = ELDIlluminanceCorrect().correct(imgs_dn, imgs_hr)
        imgs_dn, imgs_hr = torch.clamp(imgs_dn, 0, 1), torch.clamp(imgs_hr, 0, 1)

        pmn_metric_dict = PMN_metric(imgs_dn, imgs_hr)
        psnr_am.update(pmn_metric_dict["psnr"].item())
        ssim_am.update(pmn_metric_dict["ssim"].item())

        ## convert raw to rgb for plotting
        if plot_res:
            wb, ccm, iso = data["wb"][0].numpy(), data["ccm"][0].numpy(), int(data["iso"].squeeze().item())
            for x, y in zip([imgs_lr[0], imgs_dn[0], imgs_hr[0]], ["inputs", "pred", "gt"]):
                x = x.clamp_(0, 1).detach().cpu().permute(1, 2, 0).numpy()
                x = rggb_to_srgb(x, wb=wb, ccm=ccm, gamma=3, format="rggb", uint8=True)
                imageio.imwrite(f"./test_res/{args.task}//{args.testset_type}/{data_id}_{y}_iso{iso}.png", x)
    return psnr_am.avg, ssim_am.avg


def main(args):
    model = build_model(args)
    valid_loader = build_dataloader(args)

    if args.plot_res:
        os.makedirs(f"./test_res/{args.task}/{args.testset_type}", exist_ok=True)

    valid_psnr, valid_ssim = valid_one_ep(model, valid_loader, args, plot_res=args.plot_res)
    print(f"valid_psnr: {valid_psnr}, valid_ssim: {valid_ssim}")
    if args.result_json:
        result = {
            "checkpoint": args.cp_dir,
            "model": args.resolved_model,
            "graph_state": args.resolved_graph_state,
            "model_config": args.resolved_meta,
            "testset_type": args.testset_type,
            "eval_ratio": args.eval_ratio,
            "max_samples": args.max_samples,
            "psnr": valid_psnr,
            "ssim": valid_ssim,
        }
        os.makedirs(os.path.dirname(args.result_json) or ".", exist_ok=True)
        with open(args.result_json, "w") as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")


##--------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## common
    parser.add_argument("--task", type=str, default="sonya7s2")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--cp-dir", "--cp_dir", dest="cp_dir", type=str, default="./checkpoints/sonya7s2.pth")
    parser.add_argument("--model", choices=["unet", "nafnet", "natnet", "mrlfn", "learning_dwt", "learning_dwt_repncb"], default=None)
    parser.add_argument("--model-width", type=int, default=None)
    parser.add_argument("--encoder-blocks", type=int, nargs="+", default=None)
    parser.add_argument("--middle-blocks", type=int, default=None)
    parser.add_argument("--decoder-blocks", type=int, nargs="+", default=None)
    parser.add_argument("--feature-channels", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    ## change below for different setups
    parser.add_argument("--plot-res", action="store_true")
    parser.add_argument("--testset-type", "--testset_type", dest="testset_type", type=str, default="sid", choices=["sid", "eld"])
    parser.add_argument("--eval-ratio", "--eval_ratio", dest="eval_ratio", type=int, default=100, help="100, 250, 300 for SID, and 100 and 200 for ELD")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None, help="Useful for a quick evaluator smoke test")
    parser.add_argument("--result-json", default=None, help="Optional machine-readable metric output")

    _args = parser.parse_args()

    # fix seed
    np.random.seed(_args.seed)
    torch.manual_seed(_args.seed)
    random.seed(_args.seed)
    torch.backends.cudnn.benchmark = True

    main(_args)
