#!/usr/bin/python
import os
import io
import sys
import cv2
import ctypes
import tempfile
import argparse
from tqdm import tqdm
from contextlib import contextmanager
from utils import make_dirs, cut_string

libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
tfile = tempfile.TemporaryFile(mode='w+b')

@contextmanager
def stdout_redirector(stream):
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        libc.fflush(c_stdout)
        sys.stdout.close()
        os.dup2(to_fd, original_stdout_fd)
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        _redirect_stdout(tfile.fileno())
        yield
        _redirect_stdout(saved_stdout_fd)
    finally:
        os.close(saved_stdout_fd)


parser = argparse.ArgumentParser(description='Extract frames from videos')
parser.add_argument('files', metavar='VID', nargs='+', help='Videos')
parser.add_argument('-s', '--skip_frames', type=int, metavar='N', help='Extract every N\'s frame', default=1)
parser.add_argument('--roi', action='store_true', help='Extract ROI')
parser.add_argument('--height', type=int, default=None, help='ROI height', metavar='H')
parser.add_argument('--width', type=int, default=None, help='ROI width', metavar='W')
parser.add_argument('--output', type=str, default='./result', help='Output directory')
args = parser.parse_args()

def save_frame(frame, path, name, i, r):
	if r:
		frame = frame[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
	path = os.path.join(path, str(c) + '.png')
	make_dirs(path)
	cv2.imwrite(path, frame)

f = io.StringIO()
pbar = tqdm(args.files, position=0)

for v in pbar:
	pbar.set_description(cut_string(v))
	vidcap = cv2.VideoCapture(v)
	name = os.path.splitext(os.path.basename(v))[0]
	path = os.path.join(args.output, name)
	s, image = vidcap.read()
	r, l = None, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	if args.roi:
		with stdout_redirector(f):
			r = list(cv2.selectROI(image))
		if args.width:
			diff = (args.width - r[2]) // 2
			r[0], r[2] = max(0, int(r[0] - diff)), args.width
		if args.height:
			diff = (args.height - r[3]) // 2
			r[1], r[3] = max(int(r[1] - diff), 0), args.height
	bar = tqdm(list(range(0, l, args.skip_frames)), position=1)
	for c in bar:
		bar.set_description(cut_string(str(c)))
		vidcap.set(1, c)
		s, image = vidcap.read()
		if s:
			save_frame(image, path, name, c, r)
	vidcap.release()

print()
tfile.close()
