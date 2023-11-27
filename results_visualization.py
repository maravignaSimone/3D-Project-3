# Authors: Simone Maravigna, Francesco Marotta
# Date: 2023-11
# Project: 3D Object Detection on nuScenes Dataset
# File: annotations_fusion.py
# Description: This file contains the code to fuse the annotations from the 3D object detection taken by Lidar and the 2D object detection taken by Camera.

#-------------------------------------------
# imports
#-------------------------------------------
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from PIL import Image
import matplotlib.pyplot as plt
import json
from nuscenes import NuScenes
from pyquaternion import Quaternion
import os
#-------------------------------------------
# helper functions
#-------------------------------------------

def create_box(sample_data_token, ann_rec):
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None
    
    box = Box(ann_rec['translation'], ann_rec['size'], Quaternion(ann_rec['rotation']), name=ann_rec['detection_name']) 
    # Move box to ego vehicle coord system.
    box.translate(-np.array(pose_record['translation']))
    box.rotate(Quaternion(pose_record['rotation']).inverse)

    #  Move box to sensor coord system.
    box.translate(-np.array(cs_record['translation']))
    box.rotate(Quaternion(cs_record['rotation']).inverse)
    
    if sensor_record['modality'] == 'camera' and not \
            box_in_image(box, cam_intrinsic, imsize, vis_level=BoxVisibility.ALL):
        return data_path, None, None
 
    return data_path, box, cam_intrinsic

#-------------------------------------------
# main
#-------------------------------------------

# open dataset and json files
nusc = NuScenes(version='v1.0-test', dataroot='D:/nuscenes', verbose=False)
pred_file = open('transfusion_results.json', 'r')
pred_data = json.load(pred_file)
margin = 50
view_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
# iterate over all samples
for i, d in enumerate(pred_data['results']):
    scene = nusc.get('scene', nusc.get('sample', str(d))['scene_token'])['name']
    # Create subplot with 2 columns and 1 row
    fig = plt.figure(figsize=(18, 9), tight_layout=True)
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    subfigs = fig.subfigures(1, 2, width_ratios=[2,1], wspace=0, hspace=0) # 1 row, 2 columns for camera and lidar views
    
    left = subfigs[0].subplots(2,3) # 2 rows, 3 columns for camera views
    right = subfigs[1].subplots(1,1) # 1 row, 1 column for lidar view
    subfigs[0].subplots_adjust(wspace=0, hspace=0)
    subfigs[1].subplots_adjust(wspace=0, hspace=0)
    # plot lidar predictions
    # iterate over all predictions for the current sample
    for ann_rec in pred_data['results'][str(d)]:
        if(ann_rec['detection_score'] < 0.25):
            continue
        sample_record = nusc.get('sample', str(d))

        # Plot LIDAR view.
        lidar = sample_record['data']['LIDAR_TOP']
        data_path, box, camera_intrinsic = create_box(lidar, ann_rec)
        LidarPointCloud.from_file(data_path).render_height(right, view=np.eye(4), y_lim=(-50,50))
        if box is not None:
            box.render(right, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=2)
            corners = view_points(box.corners(), np.eye(4), False)[:2, :]
            right.text(corners[0, 0], corners[1, 0], ann_rec['detection_name'], color='b', fontsize=8)
        #right.set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        #right.set_ylim([np.min(corners[1, :]) - 2*margin, np.max(corners[1, :]) + 2*margin])
        right.axis('off')
        right.set_aspect('equal')

        # Plot CAMERA view.
        for j, view in enumerate(view_list):
            if(ann_rec['detection_score'] < 0.25):
                continue           
            cam = sample_record['data'][view]
            data_path, box, camera_intrinsic = create_box(cam, ann_rec)
            im = Image.open(data_path)
            
            left[j//3][j%3].imshow(im)
            left[j//3][j%3].axis('off')
            left[j//3][j%3].set_aspect('equal')
            if box is not None:
                box.render(left[j//3][j%3], view=camera_intrinsic, normalize=True, colors=('b', 'b', 'b'), linewidth= 1)
                corners = view_points(box.corners(), camera_intrinsic, True)[:2, :]
                left[j//3][j%3].text(corners[0, 0], corners[1, 0], ann_rec['detection_name'], color='b', fontsize=8)
        
    # save figure
    #plt.show()
    output_dir = 'output/'+ str(scene)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir +'/eval_' + str(i)+ '_' + str(d) +'.png')


# close json file
pred_file.close()