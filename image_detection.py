#!/usr/bin/env python
# hieuza@gmail.com
# 27.Dec.2012

# Given an image, detect a matching image in a image data base.
# feature detector and descriptor: SURF
# feature classification: nearest neighbors (using flann library)

# test on OpenCV 2.4.3

import cv2
import scipy as sp
import pyflann as pf
from PIL import Image
import os
import time

maxsize = 640

def compute_size(h, w, ms=maxsize):
  return (ms, int(1.0*ms*h/w)) if w > h else (int(1.0*ms*w/h), ms)


def extract_feature(detector, img_path, img=None):
	if img is None:
		img = cv2.imread(img_path, cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)

	if img is None:
		raise Exception('cannot read image %s' % img_path)

	h, w = img.shape[:2]
	small_size = compute_size(h, w)
	small_img = cv2.resize(img, small_size)

	# detector = cv2.SURF(800)
	keypoints, desc = detector.detectAndCompute(small_img, None)

	# print keypoints, desc
	# print len(keypoints), len(desc), len(desc[0])

	return desc


def show_image(img_path, img=None):
	if img is None:
		img = cv2.imread(self.img_path)

	h, w = img.shape[:2]
	small_size = compute_size(h, w)
	# small_size = (640, 480) if w > h else (480, 640)
	small_img = cv2.resize(img, small_size)
	cv2.imshow("image", small_img)
	cv2.waitKey()
	cv2.destroyWindow("image")


def show_result(img_path, img=None, res=[]):
	if img is None:
		img = cv2.imread(img_path)

	imgs = [img]
	imgs.extend([cv2.imread(ipath) for ipath in res])

	maxsize = 160
	view = sp.zeros((200, maxsize * (len(res) + 1), 3), sp.uint8)

	sh, sw = 0, 0
	res2 = [None] + res
	for image_path, image in zip(res2, imgs):
		h, w = image.shape[:2]
		# small_size = (160, 120) if w > h else (120, 160)
		small_size = compute_size(h, w, maxsize)
		small = cv2.resize(image, small_size)
		h, w = small.shape[:2]
		view[sh:sh+h, sw:sw+w, :] = small

		# print the date
		if image_path is not None:
			# EXTRACT THE DATE
			# get the file's EXIF
			info = Image.open(image_path)._getexif()

			if info is not None:

				img_date = info.get(36867)

				if img_date is not None:
					date = '/'.join(reversed(img_date.split()[0].split(':')))

					print date
					cv2.putText(view, date, (sw, 180),\
						cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

		sw += 160

	cv2.imshow("result", view)
	# cv2.waitKey()
	# cv2.destroyAllWindows()


def test_show_result():
	show_result('/tmp/test//IMG_9010.JPG', None,\
		['/tmp/test//IMG_9010.JPG', '/tmp/test//IMG_9431.JPG',\
			'/tmp/test//IMG_9010.JPG'])


class Detective(object):
	def __init__(self, detector, img_list, id_list, neighbors):
		"""
		Parameters
		----------
			detector: FeatureDetector
			img_list: list
				list of image paths.
			id_list: list
				list of image id.
			neighbors: FLANN object
				nearest neighbor classifier.
		"""
		self.detector = detector
		self.img_list = img_list
		self.id_list = id_list
		self.neighbors = neighbors
		self.num_image = len(self.img_list)


	def detect(self, image_path, image=None):
		"""
		detect the given image, by path or by image object.
		"""
		desc = extract_feature(self.detector, image_path, image)

		hist = [0] * self.num_image
		idx, _ = self.neighbors.nn_index(sp.array(desc), 1)

		for i in idx:
			hist[self.id_list[i]] += 1

		print 'hist:', hist

		top = sorted([(hist[i], i) for i in xrange(len(hist))])[-5:]
		top.reverse()
		print top
		res_img = [self.img_list[i] for _, i in top]

		show_result(image_path, image, res_img)

		
def form_detective(root_folder):
	"""make a detective trained on the images in the given folder"""

	t0 = time.time()
	files = os.listdir(root_folder)
	detector = cv2.SURF(800)

	img_list = []
	id_list = []
	feature_list = []
	img_id = 0
	for f in files:
		img_path = '%s/%s' % (root_folder, f)

		if not os.path.isfile(img_path) or img_path[-3:].lower() != 'jpg':
			continue

		# print img_path

		desc = extract_feature(detector, img_path)

		print '%.5f' % (time.time() - t0), img_id, img_path, len(desc)

		num_feat = len(desc)
		id_list.extend([img_id] * num_feat)
		img_list.append(img_path)
		feature_list.extend(desc)

		img_id += 1

	# print '%.5f' % (time.time() - t0)
	# print len(feature_list), len(img_list), len(id_list)

	# build flann
	neighbors = pf.FLANN()
	neighbors.build_index(sp.array(feature_list), algorithm='kdtree', trees=20)

	return Detective(detector, img_list, id_list, neighbors)


def test_recognizer(root_folder, test_img):
	detective = form_detective(root_folder)

	print detective.detect(test_img)


def recognizer(root_folder):
	detective = form_detective(root_folder)

	cap = cv2.VideoCapture(0)

	if cap is None:
		raise Exception("Cannot open video stream")

	while True:
		_, frame = cap.read()

		cv2.imshow("video", frame)

		c = '%c' % (cv2.waitKey(20) & 255)

		if c in ['d', 'D', ' ']:
			detective.detect(None, frame)
		elif c in ['q', 'Q', '\x1b']:
			break

	cv2.destroyAllWindows()


if __name__ == '__main__':
	import sys

	if len(sys.argv) < 2:
		print 'given a folder'
		sys.exit(1)

	# test_recognizer(sys.argv[1], sys.argv[2])
	root_folder = sys.argv[1]
	recognizer(root_folder)

	# test_show_result()
