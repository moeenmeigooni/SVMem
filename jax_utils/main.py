import argparse
import os
import pathlib

def get_args():
    parser = argparse.ArgumentParser()
    
    ROOT = pathlib.Path(__file__).parent #jax_utils directory
                        
    #Directories
    parser.add_argument('--data_dir', type=str, default=str(ROOT.parent), help="SVMem PDBs!") 
    parser.add_argument('--save_dir', type=str, default="/Scr/hyunpark-new/Protein-TDA/pickled_indiv/")  
    parser.add_argument('--load_ckpt_path', type=str, default="/Scr/hyunpark-new/Protein-TDA/saved")

    #MDAnalysis utils
    parser.add_argument('--psf', type=str, default=None)  
    parser.add_argument('--pdb', type=str, default=None)  
    parser.add_argument('--last', type=int, default=200) 
    parser.add_argument('--trajs', default=None, nargs="*") 
    parser.add_argument('--atom_selection', type=str, default="backbone")  
    parser.add_argument('--head_selection', type=str, default="backbone")  

    #PH utils
    parser.add_argument('--maxdim', type=int, default=1)  
    parser.add_argument('--multiprocessing', action="store_true")  
    parser.add_argument('--multiprocessing_backend', type=str, default="ray", choices=["multiprocessing", "dask", "joblib", "ray"])  
    parser.add_argument('--tensor', action="store_true", help="DEPRECATED!")  
    parser.add_argument('--ripspp', action="store_true", help="Rips++!")  
    parser.add_argument('--gudhi', action="store_true", help="Gudhi!")  

    #Dataloader utils
    parser.add_argument('--train_frac', type=float, default=0.8)  
    parser.add_argument('--pin_memory', type=bool, default=True)  
    parser.add_argument('--num_workers', type=int, default=0)  
    parser.add_argument('--batch_size', type=int, default=32)  
    parser.add_argument('--preprocessing_only', action="store_true", help="to get RIPSER based PH!")  
    parser.add_argument('--ignore_topologicallayer', action="store_true", help="forcefully use RIPSER for subscription!")  
    parser.add_argument('--truncated', action="store_true", help="Use only 10% of data. It MUST have original dataset, however.")  

    #Training utils
    parser.add_argument('--epoches', type=int, default=2)
    parser.add_argument('--label_smoothing', '-ls', type=float, default=0.)
    parser.add_argument('--learning_rate','-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--warm_up_split', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0)
#     parser.add_argument('--distributed',  action="store_true")
    parser.add_argument('--low_memory',  action="store_true")
    parser.add_argument('--amp', action="store_true", help="floating 16 when turned on.")
    parser.add_argument('--optimizer', type=str, default='adam', choices=["adam","lamb","sgd","torch_adam","torch_adamw","torch_sparse_adam"])
    parser.add_argument('--scheduler', type=str, default='reduce', choices=["linear","reduce"])
    parser.add_argument('--gradient_clip', type=float, default=None) 
    parser.add_argument('--accumulate_grad_batches', type=int, default=1) 
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_artifacts', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--shard', action='store_true')
    parser.add_argument('--loss', choices=["mse", "mae", "smooth", "hybrid", "distill"], default="hybrid")

    #Model utils
    parser.add_argument('--backbone', type=str, default='vit', choices=["mpnn", "vit", "swin", "swinv2", "convnext", "restv2", "clip_resnet"])
    
    #Callback utils
    parser.add_argument('--log', action="store_true", help="to log for W&B")  
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--name', type=str, default="mpnn", help="saved torch model name...")
    parser.add_argument('--teacher_name', type=str, default="mpnn", help="saved torch model name...")

    #Mode utils
    parser.add_argument('--which_mode', type=str, choices=["preprocessing", "train", "distill", "infer", "infer_custom", "xai"], default="preprocessing")  
    parser.add_argument('--which_xai', type=str, choices=["saliency", "gradcam", "lime", "attention"], default="saliency")  

    args = parser.parse_args()
    return args
