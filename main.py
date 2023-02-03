import os
import argparse
from trainer import trainer




def main(config):
    if (not os.path.exists(config.model_save_path)):
        os.mkdir(config.model_save_path)
    Trainer = trainer(config)

    for i in range(config.exp):

        Trainer.train()
        Trainer.test()
    Trainer.result.to_csv(f'./{config.model}_{config.data_name}_result.csv')

    return Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')

    parser.add_argument('--model', type=str, default="PCA_GOAD")
    parser.add_argument('--data_name', type=str, default="thyroid")
    parser.add_argument('--dataset', type=str, default="tabular")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--exp', type=int, default=10)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)