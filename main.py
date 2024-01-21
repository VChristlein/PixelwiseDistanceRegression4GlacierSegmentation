from create_dataset import *
from evaluation import *
from train import *
import argparse
import time
import os



def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Glacier Front Segmentation')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='number of training epochs (integer value > 0)')
    parser.add_argument('-b', '--batch_size', default=3, type=int, help='batch size (integer value)')
    parser.add_argument('-o', '--out_path', default=f'saved_model/{time.strftime("%Y%m%d-%H%M%S")}', type=str, help='output path for results')
    parser.add_argument('-t', '--time', default=None, type=str, help='timestamp for model saving')
    parser.add_argument('--target_size', default=512, type=int, help='input size of images')
    parser.add_argument('--gamma', default=7, type=int, help='Gamma value for distance map creation')
    parser.add_argument('--second_model', default=0, type=int, help='create second model')
    parser.add_argument('--loss', default=None, type=str)

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    gamma = args.gamma
    out_path = f'{args.out_path}_{gamma}'
    target_size = args.target_size
    second_model = False
    crf_model = False
    enze_model = False

    if args.second_model:
        second_model = True
        out_path = out_path + '_second'

    dir_list = os.listdir('saved_model/')
    model_path = [s for s in dir_list if (s.endswith(f'_{gamma}'))][0]
    model_path = f'saved_model/{model_path}'

    if not os.path.isfile(f'data_{target_size}_{gamma}/gamma{gamma}'):
        # create new dataset with new gamma value
        create_dataset(gamma=gamma, model_path=model_path, second_model=second_model)

    train(epochs=epochs, batch_size=batch_size, out_path=out_path, gamma=gamma, loss_func=args.loss, second_model=second_model)

    evaluate(gamma=gamma, model_path=out_path, test_path=f'data_{target_size}_{gamma}/test', enze_model=enze_model, crf=crf_model)

    # delete_data(f'data_{TARGET_SIZE}_{gamma}')

    '''
    dir_list = os.listdir('saved_model/')
    for g in range(1, 11):
        gamma = g
        model_path = [s for s in dir_list if (s.endswith(f'_{g}'))][0]
        model_path = f'saved_model/{model_path}'

        if not os.path.isfile(f'data_{TARGET_SIZE}_{gamma}/gamma{gamma}'):
            # create new dataset with new gamma value
            create_dataset(gamma=gamma, model_path=model_path, second_model=True)

        train(epochs=epochs, batch_size=batch_size, out_path=out_path, gamma=gamma, second_model=True)

        evaluate(gamma=gamma, model_path=out_path, test_path=f'data_{TARGET_SIZE}_{gamma}/test')
    '''


if __name__ == "__main__":
    main()
