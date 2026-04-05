import os

from collections import defaultdict
import numpy as np
import cv2

from . import functions as f

class Data():
	"""
	A class to hold ground truth or predictions data in an easy to work with format.
	Note that any time they appear, bounding boxes are [x, y, width, height] and masks
	are either a list of polygons or pycocotools RLEs.

	Also, don't mix ground truth with predictions. Keep them in separate data objects.
	
	'max_dets' specifies the maximum number of detections the model is allowed to output for a given image.
	"""

	def __init__(self, name:str, max_dets:int=100, task_type:str='2d_det', coordinate_frame:str=None):
		valid_frames = {None, 'camera', 'lidar', 'ego', 'global', 'vehicle'}
		if coordinate_frame not in valid_frames:
			raise ValueError('coordinate_frame must be one of {}'.format(sorted([x for x in valid_frames if x is not None])))

		self.name     = name
		self.max_dets = max_dets
		self.task_type = task_type
		self.coordinate_frame = coordinate_frame
		self.is_3d = self.task_type == '3d_det'

		self.classes     = {}  # Maps class ID to class name 
		self.annotations = []  # Maps annotation ids to the corresponding annotation / prediction
		
		# Maps an image id to an image name and a list of annotation ids
		self.images      = defaultdict(lambda: {'name': None, 'anns': []})


	def _get_ignored_classes(self, image_id:int) -> set:
		anns = self.get(image_id)

		classes_in_image = set()
		ignored_classes  = set()

		for ann in anns:
			if ann['ignore']:
				if ann['class'] is not None and ann['bbox'] is None and ann['mask'] is None:
					ignored_classes.add(ann['class'])
			else:
				classes_in_image.add(ann['class'])
		
		return ignored_classes.difference(classes_in_image)


	def _make_default_class(self, id:int):
		""" (For internal use) Initializes a class id with a generated name. """

		if id not in self.classes:
			self.classes[id] = 'Class ' + str(id)

	def _make_default_image(self, id:int):
		if self.images[id]['name'] is None:
			self.images[id]['name'] = 'Image ' + str(id)

	def _prepare_box(self, box:object):
		return box

	def _prepare_mask(self, mask:object):
		return mask

	def _prepare_box_3d(self, box_3d:object):
		if box_3d is None:
			return None

		if isinstance(box_3d, dict):
			center = box_3d.get('center')
			size   = box_3d.get('size')
			yaw    = box_3d.get('yaw')

			if center is None or size is None or yaw is None:
				raise ValueError('3D box dict must include center, size, and yaw')

			box_3d = list(center) + list(size) + [yaw]

		if isinstance(box_3d, (list, tuple)) and len(box_3d) == 7:
			return list(box_3d)

		raise ValueError('3D box must be a 7-element list/tuple or dict with center/size/yaw')

	def _add(self, image_id:int, class_id:int, box:object=None, mask:object=None, score:float=1, ignore:bool=False, velocity:object=None):
		""" Add a data object to this collection. You should use one of the below functions instead. """
		self._make_default_class(class_id)
		self._make_default_image(image_id)
		new_id = len(self.annotations)

		ann = {
			'_id'   : new_id,
			'score' : score,
			'image' : image_id,
			'class' : class_id,
			'bbox'  : self._prepare_box(box),
			'mask'  : self._prepare_mask(mask),
			'ignore': ignore,
		}
		if velocity is not None:
			ann['velocity'] = velocity
		self.annotations.append(ann)

		self.images[image_id]['anns'].append(new_id)

	def add_ground_truth(self, image_id:int, class_id:int, box:object=None, mask:object=None):
		""" Add a ground truth. If box or mask is None, this GT will be ignored for that mode. """
		self._add(image_id, class_id, box, mask)

	def add_detection(self, image_id:int, class_id:int, score:int, box:object=None, mask:object=None):
		""" Add a predicted detection. If box or mask is None, this prediction will be ignored for that mode. """
		self._add(image_id, class_id, box, mask, score=score)

	def add_ground_truth_3d(self, image_id:int, class_id:int, box_3d:object, velocity:object=None):
		""" Add a 3D ground truth. box_3d should be [x, y, z, l, w, h, yaw] or a dict with center/size/yaw. """
		if not self.is_3d:
			raise ValueError('add_ground_truth_3d requires Data(task_type="3d_det")')
		self._add(image_id, class_id, box=self._prepare_box_3d(box_3d), mask=None, velocity=velocity)

	def add_detection_3d(self, image_id:int, class_id:int, score:float, box_3d:object, velocity:object=None):
		""" Add a 3D detection. box_3d should be [x, y, z, l, w, h, yaw] or a dict with center/size/yaw. """
		if not self.is_3d:
			raise ValueError('add_detection_3d requires Data(task_type="3d_det")')
		self._add(image_id, class_id, box=self._prepare_box_3d(box_3d), mask=None, score=score, velocity=velocity)

	def add_ignore_region(self, image_id:int, class_id:int=None, box:object=None, mask:object=None):
		"""
		Add a region inside of which background detections should be ignored.
		You can use these to mark a region that has deliberately been left unannotated
		(e.g., if is a huge crowd of people and you don't want to annotate every single person in the crowd).

		If class_id is -1, this region will match any class. If the box / mask is None, the region will be the entire image.
		"""
		self._add(image_id, class_id, box, mask, ignore=True)

	def add_class(self, id:int, name:str):
		""" Register a class name to that class ID. """
		self.classes[id] = name
	
	def add_image(self, id:int, name:str):
		""" Register an image name/path with an image ID. """
		self.images[id]['name'] = name


	def get(self, image_id:int):
		""" Collects all the annotations / detections for that particular image. """
		return [self.annotations[x] for x in self.images[image_id]['anns']]
