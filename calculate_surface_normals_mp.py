from PIL import Image
import math
#import matplotlib
import numpy as np
import os
#import pathlib
#import random
import scenenet_pb2 as sn
#import sys
#import scipy.misc
import argparse
import time
import multiprocessing as mp


def normalize(v):
    return v/np.linalg.norm(v)

def load_depth_map_in_m(file_name):
    image = Image.open(file_name)
    pixel = np.array(image)
    return (pixel * 0.001)

def pixel_to_ray(pixel,vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    x, y = pixel
    x_vect = math.tan(math.radians(hfov/2.0)) * ((2.0 * ((x+0.5)/pixel_width)) - 1.0)
    y_vect = math.tan(math.radians(vfov/2.0)) * ((2.0 * ((y+0.5)/pixel_height)) - 1.0)
    return (x_vect,y_vect,1.0)

def normalised_pixel_to_ray_array(width=320,height=240):
    pixel_to_ray_array = np.zeros((height,width,3))
    for y in range(height):
        for x in range(width):
            pixel_to_ray_array[y,x] = normalize(np.array(pixel_to_ray((x,y),pixel_height=height,pixel_width=width)))
    return pixel_to_ray_array

def points_in_camera_coords(depth_map,pixel_to_ray_array):
    assert depth_map.shape[0] == pixel_to_ray_array.shape[0]
    assert depth_map.shape[1] == pixel_to_ray_array.shape[1]
    assert len(depth_map.shape) == 2
    assert pixel_to_ray_array.shape[2] == 3
    camera_relative_xyz = np.ones((depth_map.shape[0],depth_map.shape[1],4))
    for i in range(3):
        camera_relative_xyz[:,:,i] = depth_map * pixel_to_ray_array[:,:,i]
    return camera_relative_xyz

# A very simple and slow function to calculate the surface normals from 3D points from
# a reprojected depth map. A better method would be to fit a local plane to a set of 
# surrounding points with outlier rejection such as RANSAC.  Such as done here:
# http://cs.nyu.edu/~silberman/projects/indoor_scene_seg_sup.html
def surface_normal(points):
    # These lookups denote y,x offsets from the anchor point for 8 surrounding
    # directions from the anchor A depicted below.
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 | A | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    d = 2
    lookups = {0:(-d,0),1:(-d,d),2:(0,d),3:(d,d),4:(d,0),5:(d,-d),6:(0,-d),7:(-d,-d)}
    surface_normals = np.zeros((240,320,3))
    for i in range(240):
        for j in range(320):
            min_diff = None
            point1 = points[i,j,:3]
             # We choose the normal calculated from the two points that are
             # closest to the anchor points.  This helps to prevent using large
             # depth disparities at surface borders in the normal calculation.
            for k in range(8):
                try:
                    point2 = points[i+lookups[k][0],j+lookups[k][1],:3]
                    point3 = points[i+lookups[(k+2)%8][0],j+lookups[(k+2)%8][1],:3]
                    diff = np.linalg.norm(point2 - point1) + np.linalg.norm(point3 - point1)
                    if min_diff is None or diff < min_diff:
                        normal = normalize(np.cross(point2-point1,point3-point1))
                        min_diff = diff
                except IndexError:
                    continue
            surface_normals[i,j,:3] = normal
    return surface_normals

def depth_path_from_view(data_root_path, render_path, view):
    photo_path = os.path.join(render_path, 'depth')
    depth_path = os.path.join(photo_path, '{0}.png'.format(view.frame_num))
    return os.path.join(data_root_path, depth_path)

def normals_path_from_view(data_root_path, render_path, view):
    photo_path = os.path.join(render_path, 'normals')
    normals_dir = os.path.join(data_root_path, photo_path)
    if not os.path.exists(normals_dir):
        os.makedirs(normals_dir)
    return os.path.join(normals_dir, '{0}.png'.format(view.frame_num))


def calculate_normals(paths):
    depth_path, surface_normal_path = paths

    print('Converting depth image:{0} to surface_normal image:{1}'.format(
        depth_path, surface_normal_path))

    #start_time = time.time()

    # This stores for each image pixel, the cameras 3D ray vector 
    cached_pixel_to_ray_array = normalised_pixel_to_ray_array()

    # load depth map data
    depth_map = load_depth_map_in_m(str(depth_path))

    # When no depth information is available (because a ray went to
    # infinity outside of a window) depth is set to zero.  For the purpose
    # of surface normal, here we set it simply to be very far away
    depth_map[depth_map == 0.0] = 50.0

    # This is a 320x240x3 array, with each 'pixel' containing the 3D point in camera coords
    points_in_camera = points_in_camera_coords(
        depth_map, cached_pixel_to_ray_array)
    surface_normals = surface_normal(points_in_camera)

    # Write out surface normal image.
    img = Image.fromarray(np.uint8((surface_normals + 1.0) * 128.0))
    img.save(surface_normal_path)

    #elapsed_time = time.time() - start_time
    #print('Elapsed time: {}'.format(elapsed_time))


def CalculateSurfaceNormalsApp():
    parser = argparse.ArgumentParser(description='Normals generator application.')
    parser.add_argument('--data-root-path', type=str, required=True,
                        help='the path to the dataset root.')
    parser.add_argument('--protobuf-path', type=str, required=True,
                        help='the path to the protobuf to extract the data.')
    parser.add_argument('--num-steps', type=int, default=30,
                        help='the number samples to skip.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='the number of workers to use to compute the normals.')
    args = parser.parse_args()

    trajectories = sn.Trajectories()
    try:
        with open(args.protobuf_path,'rb') as f:
            trajectories.ParseFromString(f.read())
    except IOError:
        print('Scenenet protobuf data not found at location:{0}'.format(data_root_path))
        print('Please ensure you have copied the pb file to the data directory')

    global_idx = 0

    paths = []

    for traj_idx, traj in enumerate(trajectories.trajectories):
        for view_idx, view in enumerate(traj.views):
            if global_idx % args.num_steps != 0:
                global_idx += 1
                continue
            global_idx += 1

            # prepare data paths
            depth_path = depth_path_from_view(
                args.data_root_path, traj.render_path, view)
            surface_normal_path = normals_path_from_view(
                args.data_root_path, traj.render_path, view)

            paths.append((depth_path, surface_normal_path))

    p = mp.Pool(processes=args.num_workers)
    p.map(calculate_normals, paths)


if __name__ == '__main__':
    CalculateSurfaceNormalsApp()
