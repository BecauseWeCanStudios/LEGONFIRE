#!/usr/bin/python
import os
import glob
import utils
import random
import argparse
import platform
import threading
import numpy as np
from PIL import Image
from tqdm import tqdm
from math import ceil
from itertools import chain

def apply_background(back, img, path):
	back.paste(img, (0, 0), img)
	utils.make_dirs(path)
	back.save(path)

def apply_noise_func(min, max):
	def f(img):
		arr = np.array(img).astype(int)
		arr += np.random.randint(min, max, (*img.size, 3))
		return Image.fromarray(arr.clip(0, 255).astype('uint8'))
	return f

def choose_backgrounds(backgrounds, count):
	return [(Image.open(file).convert('RGB'), os.path.splitext(os.path.basename(file))[0]) for file in random.sample(backgrounds, count)]

def function(n, images, backgrounds, args):
	pbar = tqdm(images, position=n)
	for file in pbar:
		pbar.set_description(utils.cut_string(file))
		img = Image.open(file)
		p = os.path.join(args.output, os.path.dirname(file))
		n = os.path.splitext(os.path.basename(file))[0]
		for back, name in backgrounds:
			apply_background(noise(back.resize(img.size, Image.ANTIALIAS)), img, os.path.join(p, './{}_{}.png'.format(n, name)))
		if args.mask:
			Image.frombytes('1', img.size, np.packbits(np.array(img)[::,::,3].astype(bool), axis=1)).save(os.path.join(p, './{}.mask'.format(n)), 'png')
		for i in range(args.random_backgrounds):
			apply_background(Image.fromarray(np.random.randint(0, 256, (*img.size, 3), 'uint8')), img, os.path.join(p, './{}_{}.png'.format(n, i)))
		if args.rebackground:
			backgrounds = choose_backgrounds(args.backgrounds, args.backgrounds_number)

parser = argparse.ArgumentParser(description='Enlarge your dataset with new backgrounds and/or generate masks')
parser.add_argument('filenames', help='Image filenames', nargs='+', metavar='IMG')
parser.add_argument('-b', '--backgrounds', default=glob.glob('./backgrounds/**', recursive=True), help='Background filenames', nargs='+', metavar='BG')
parser.add_argument('-n', '--noise', nargs=2, help='Apply noise [MIN, MAX)', metavar=('MIN', 'MAX'), type=int)
parser.add_argument('-m', '--mask', action='store_true', help='Generate mask')
parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads')
parser.add_argument('-bn', '--backgrounds_number', type=int, default=1, help='Apply N backgrounds', metavar='N')
parser.add_argument('-rb', '--random_backgrounds', type=int, default=0, help='Generate K images with random noise backgrounds', metavar='K')
parser.add_argument('--output', help='Output dir', type=str, default='./result', metavar='OUT')
parser.add_argument('--rebackground', action='store_true', help='Choose new backgrounds for every image')
args = parser.parse_args()

if platform.system() == 'Windows':
	args.filenames = list(chain(*(glob.glob(i, recursive=True) for i in args.filenames)))
	args.backgrounds = list(chain(*(glob.glob(i, recursive=True) for i in args.backgrounds)))

args.filenames = list(filter(lambda x: os.path.isfile(x), args.filenames))
args.backgrounds = list(filter(lambda x: os.path.isfile(x), args.backgrounds))

if args.backgrounds_number < 0:
	args.backgrounds_number = len(args.backgrounds)

backgrounds = choose_backgrounds(args.backgrounds, args.backgrounds_number)
noise = apply_noise_func(args.noise[0], args.noise[1]) if args.noise else lambda x: x

threads = []
tn = ceil(len(args.filenames) / args.threads)

for i in range(args.threads):
	threads.append(threading.Thread(target=function, args=(i, args.filenames[i * tn:(i + 1) * tn], backgrounds, args)))
	threads[-1].start()

for i in threads:
	i.join()

for i in threads:
	print()
