import os

def make_dirs(path):
	path = os.path.dirname(path)
	try:
		if not os.path.exists(path):
			os.makedirs(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

def cut_string(s, n=20):
	if len(s) > n:
		return '...' + s[3 - n::]
	return ' ' * (n - len(s)) + s
	