#----> pytorch imports
import torch

#----> general imports
import pandas as pd
import numpy as np
import pdb
import os
import warnings
warnings.filterwarnings('ignore')
from timeit import default_timer as timer
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment

from utils.process_args import _process_args

def main(args):

    #----> prep for 5 fold cv study
    folds = _get_start_end(args)
    
    #----> storing the val and test cindex for 5 fold cv
    all_val_cindex = []
    all_val_cindex_ipcw = []
    all_val_BS = []
    all_val_IBS = []
    all_val_iauc = []
    
    all_val_loss = []

    for i in [0,1,2,3,4]:
        
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
            fold=i
        )
        
        print("Created train and val datasets for fold {}".format(i))

        results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss) = _train_val(datasets, i, args)

        all_val_cindex.append(val_cindex)
        all_val_cindex_ipcw.append(val_cindex_ipcw)
        all_val_BS.append(val_BS)
        all_val_IBS.append(val_IBS)
        all_val_iauc.append(val_iauc)
        all_val_loss.append(total_loss)
    
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        print("Saving results...")

if __name__ == "__main__":
    start = timer()

    #----> read the args
    args = _process_args()
    
    #----> Prep
    args = _prepare_for_experiment(args)
    
    #----> create dataset factory
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        seed=args.seed, 
        print_info=True, 
        n_bins=args.n_classes, 
        label_col=args.label_col, 
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat = True if "coattn" in args.modality else False,
        is_survpath = True if args.modality == "survpath" else False,
        type_of_pathway=args.type_of_path)

    #---> perform the experiment
    results = main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))