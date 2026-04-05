from tidecv import TIDE, datasets

gt = datasets.KITTI3D('/tmp/kitti')
preds = datasets.KITTI3DResult('/tmp/preds.json')

gt.add_ground_truth_3d(
	image_id=1,
	class_id=1,
	box_3d=[0.0, 0.0, 0.0, 4.0, 1.8, 1.6, 0.0]
)

preds.add_detection_3d(
	image_id=1,
	class_id=1,
	score=0.9,
	box_3d=[0.0, 0.0, 0.0, 4.0, 1.8, 1.6, 0.0]
)

print('gt is_3d:', gt.is_3d, 'frame:', gt.coordinate_frame)
print('pred is_3d:', preds.is_3d, 'frame:', preds.coordinate_frame)

try:
	TIDE().evaluate_3d(gt, preds, protocol='kitti_3d')
except NotImplementedError as e:
	print('evaluate_3d scaffold reached:', e)
