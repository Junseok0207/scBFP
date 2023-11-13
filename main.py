import torch
from argument import parse_args
from scBFP import scBFP_Trainer

def main():
    args, _ = parse_args()
    torch.set_num_threads(3)
    
    embedder = scBFP_Trainer(args)
    embedder.train() 

if __name__ == "__main__":
    main()