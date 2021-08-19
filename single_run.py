from core import run_all
import click

@click.command()
@click.option('--skip_training', type=bool,default=False, help='create learner', metavar='BOOL')
@click.option('--train_gan', type=bool,default=False, help='enable gan training',  metavar='BOOL')
@click.option('--stable_loss', type=bool,default=False, help='enable stable loss', metavar='BOOL')
@click.option('--input_dim', type=int,default=4096, help='number of input dimensions', metavar='INT')
@click.option('--task', type=str,default="scene_depth", help='task to perform')
@click.option('--pretrained_gen', type=str,default=None, help='file to pretrained generator')
@click.option('--learn_save_file', type=str,default=None, help='learner save file')
@click.option('--file_name', type=str,default=None, help='evaluation file name')
@click.option('--results_folder', type=str,default="results", help='results folder')
@click.option('--calc_importance', type=bool,default=False, help='calculate importance')
@click.option('--sub_folder', type=str,default=None, help='sub folder name')
def main(**args):
    run_all(args)

main()