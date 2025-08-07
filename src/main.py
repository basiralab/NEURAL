# main.py

import argparse
import stage1_train
import stage2_train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the two-stage NEURAL pipeline.")
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2], 
                        help="The stage of the pipeline to run. (1: Fine-tune, 2: NEURAL Pruning & GNN Training)")
    args = parser.parse_args()
    
    if args.stage == 1:
        stage1_train.run_stage1()
    elif args.stage == 2:
        stage2_train.run_stage2()