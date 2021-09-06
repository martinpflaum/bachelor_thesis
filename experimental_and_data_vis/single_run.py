"""
MIT License

Copyright (c) 2021 martinpflaum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
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