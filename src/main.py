"""
This script serves as the main entry point for the entire NEURAL pipeline.

It uses a command-line interface to allow the user to select and run one of
the two main stages of the project:
1.  Stage 1: Fine-tuning the vision-language model.
2.  Stage 2: Performing attention-guided pruning and training the GNN classifier.
"""

import argparse
# Import the functions that run each stage from their respective modules.
import stage1_train
import stage2_train

if __name__ == '__main__':
    # Set up the command-line argument parser to handle user inputs.
    parser = argparse.ArgumentParser(description="Run the two-stage NEURAL pipeline.")

    # Define the command-line arguments the script will accept.
    parser.add_argument(
        '--stage',
        type=int,
        required=True,
        choices=[1, 2],
        help="The stage of the pipeline to run. (1: Fine-tune, 2: NEURAL Pruning & GNN Training)"
    )

    # Parse the arguments provided by the user.
    args = parser.parse_args()

    # Conditional logic to call the correct function based on the user's input.
    if args.stage == 1:
        # If the user selects stage 1, run the fine-tuning process.
        stage1_train.run_stage1()
    elif args.stage == 2:
        # If the user selects stage 2, run the pruning and GNN training process.
        stage2_train.run_stage2()