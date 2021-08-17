import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_gan', type=bool, help='enable gan training')
parser.add_argument('--enable_stable_loss', type=bool, help='enable stable loss')
parser.add_argument('--input_dim', type=int,default=4096, help='number of input dimensions')
parser.add_argument('--task', type=str,default="scene_depth", help='task to perform')
parser.add_argument('--pretrained_gen', type=str,default=None, help='file to pretrained generator')
parser.add_argument('--learn_save_file', type=str,default=None, help='learner save file')
parser.add_argument('--file_name', type=str,default=None, help='evaluation file name')



args = parser.parse_args()
args = vars(args)
