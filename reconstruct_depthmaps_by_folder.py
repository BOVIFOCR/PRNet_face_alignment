from __future__ import print_function

import numpy as np
import os, sys
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import random

from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture


def find_images(folder_path, extensions):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for ext in extensions:
            pattern = os.path.join(root, '*' + ext)
            matching_files = glob(pattern)
            image_paths.extend(matching_files)
    return sorted(image_paths)


def find_subfolders_videos(dataset_path, img_types=('.jpg', '.png')):
    subfolders = []
    for root, dirs, files in os.walk(dataset_path):
        if any(file.lower().endswith(img_types) for file in files):
            subfolders.append(root)
    subfolders = list(set(subfolders))   # remove repeated subfolders
    return sorted(subfolders)


def main(args):
    if args.isShow or args.isTexture:
        import cv2
        from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box

    # ---- init PRN
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # GPU number, -1 for CPU
    prn = PRN(is_dlib = args.isDlib)

    # ------------- load data
    args.inputDir = args.inputDir.rstrip('/')
    args.outputDir = args.outputDir.rstrip('/')
    image_folder = args.inputDir
    if args.inputDir.split('/')[-1] == args.outputDir.split('/')[-1]:
        output_folder = args.outputDir
    else:
        output_folder = os.path.join(args.outputDir, args.inputDir.split('/')[-1])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # types = ('*.jpg', '*.png')
    # image_path_list= []
    # for files in types:
    #     image_path_list.extend(glob(os.path.join(image_folder, files)))
    # total_num = len(image_path_list)
    # if total_num == 0:
    #     raise Exception('No input images found in \''+ image_folder +'\'')
    types = ('.jpg', '.png')
    print('Searching images of type', types, '...')
    subfolders_videos = find_subfolders_videos(image_folder, types)
    total_subfolders = len(subfolders_videos)
    print('Found', total_subfolders, 'video folders')
    # for i, subfolder in enumerate(subfolders_videos):
    #     print('i:', i, 'subfolder:', subfolder)
    # sys.exit(0)

    # frames-per-video
    for s, subfolder_video in enumerate(subfolders_videos):
        # print('subfolder', str(s)+'/'+str(total_subfolders), '    subfolder_video:', subfolder_video)

        image_path_list = find_images(subfolder_video, types)
        total_num = len(image_path_list)

        image_path_list_sampled = image_path_list
        if args.frames_per_video > 0:
            image_path_list_sampled = random.sample(image_path_list, args.frames_per_video)
        total_num_sampled = len(image_path_list_sampled)

        if total_num == 0:
            raise Exception('No input images found in \''+ image_folder +'\'')

        for i, image_path in enumerate(sorted(image_path_list_sampled)):
            start_time = time()
            print('subfolder_video', str(s)+'/'+str(total_subfolders)+':', subfolder_video)
            print('    frames_per_video:', str(total_num_sampled)+'/'+str(total_num))
            print('    image_path', str(i)+'/'+str(total_num_sampled)+':', image_path)

            name = image_path.strip().split('/')[-1][:-4]

            # read image
            image = imread(image_path)
            [h, w, c] = image.shape
            if c>3:
                image = image[:,:,:3]

            # the core: regress position map
            if args.isDlib:
                max_size = max(image.shape[0], image.shape[1])
                if max_size> 1000:
                    image = rescale(image, 1000./max_size)
                    image = (image*255).astype(np.uint8)
                pos = prn.process(image) # use dlib to detect face
            else:
                if image.shape[0] == image.shape[1]:
                    image = resize(image, (256,256))
                    pos = prn.net_forward(image/255.) # input image has been cropped to 256x256
                else:
                    box = np.array([0, image.shape[1]-1, 0, image.shape[0]-1]) # cropped with bounding box
                    pos = prn.process(image, box)
            
            image = image/255.
            if pos is None:
                continue

            save_folder = '/'.join(image_path.split('/')[:-1])
            save_folder = save_folder.replace(image_folder, output_folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            if args.is3d or args.isMat or args.isPose or args.isShow:
                # 3D vertices
                vertices = prn.get_vertices(pos)
                if args.isFront:
                    save_vertices = frontalize(vertices)
                else:
                    save_vertices = vertices.copy()
                save_vertices[:,1] = h - 1 - save_vertices[:,1]

            if args.isImage:
                # imsave(os.path.join(save_folder, name + '.jpg'), image)   # original
                path_output_image = os.path.join(save_folder, name + '.png')
                print('    output_image:', path_output_image)
                imsave(path_output_image, image)     # Bernardo

            if args.is3d:
                # corresponding colors
                colors = prn.get_colors(image, vertices)

                if args.isTexture:
                    if args.texture_size != 256:
                        pos_interpolated = resize(pos, (args.texture_size, args.texture_size), preserve_range = True)
                    else:
                        pos_interpolated = pos.copy()
                    texture = cv2.remap(image, pos_interpolated[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
                    if args.isMask:
                        vertices_vis = get_visibility(vertices, prn.triangles, h, w)
                        uv_mask = get_uv_mask(vertices_vis, prn.triangles, prn.uv_coords, h, w, prn.resolution_op)
                        uv_mask = resize(uv_mask, (args.texture_size, args.texture_size), preserve_range = True)
                        texture = texture*uv_mask[:,:,np.newaxis]
                    write_obj_with_texture(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, texture, prn.uv_coords/prn.resolution_op)#save 3d face with texture(can open with meshlab)
                else:
                    write_obj_with_colors(os.path.join(save_folder, name + '.obj'), save_vertices, prn.triangles, colors) #save 3d face(can open with meshlab)

            if args.isDepth:
                depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
                # depth = get_depth_image(vertices, prn.triangles, h, w)
                # imsave(os.path.join(save_folder, name + '_depth.jpg'), depth_image)   # original
                path_output_depth = os.path.join(save_folder, name + '_depth.png')
                print('    output_depth:', path_output_depth)
                imsave(path_output_depth, depth_image)     # Bernardo
                # sio.savemat(os.path.join(save_folder, name + '_depth.mat'), {'depth':depth})

            if args.isMat:
                sio.savemat(os.path.join(save_folder, name + '_mesh.mat'), {'vertices': vertices, 'colors': colors, 'triangles': prn.triangles})

            if args.isKpt or args.isShow:
                # get landmarks
                kpt = prn.get_landmarks(pos)
                np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)

            # if args.isPose or args.isShow:
            #     # estimate pose
            #     camera_matrix, pose = estimate_pose(vertices)
            #     np.savetxt(os.path.join(save_folder, name + '_pose.txt'), pose) 
            #     np.savetxt(os.path.join(save_folder, name + '_camera_matrix.txt'), camera_matrix) 
                
            # if args.isShow:
            #     # ---------- Plot
            #     image_pose = plot_pose_box(image, camera_matrix, kpt)
            #     cv2.imshow('sparse alignment', plot_kpt(image, kpt))
            #     cv2.imshow('dense alignment', plot_vertices(image, vertices))
            #     cv2.imshow('pose', plot_pose_box(image, camera_matrix, kpt))
            #     cv2.waitKey(0)

            spent_time = time() - start_time
            est_time_folder = spent_time*(total_num_sampled-i)
            est_time_total = est_time_folder*(total_subfolders-s)
            print('    Spent time (frame): %.2f seconds' % (spent_time))
            print('    Estimated time (folder): %.2fs / %.2fm / %.2fh' % (est_time_folder, est_time_folder/60., est_time_folder/3600))
            print('    Estimated time (total): %.2fs / %.2fm / %.2fh' % (est_time_total, est_time_total/60., est_time_total/3600))
            print('    ----------------')

        print('--------------------------------')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='TestImages/', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='TestImages/results', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('--gpu', default='0', type=str,
                        help='set gpu id, -1 for CPU')
    parser.add_argument('--isDlib', default=True, type=ast.literal_eval,
                        help='whether to use dlib for detecting face, default is True, if False, the input image should be cropped in advance')
    parser.add_argument('--is3d', default=False, type=ast.literal_eval,
                        help='whether to output 3D face(.obj). default save colors.')
    parser.add_argument('--isMat', default=False, type=ast.literal_eval,
                        help='whether to save vertices,color,triangles as mat for matlab showing')
    parser.add_argument('--isKpt', default=False, type=ast.literal_eval,
                        help='whether to output key points(.txt)')
    parser.add_argument('--isPose', default=True, type=ast.literal_eval,
                        help='whether to output estimated pose(.txt)')
    parser.add_argument('--isShow', default=False, type=ast.literal_eval,
                        help='whether to show the results with opencv(need opencv)')
    parser.add_argument('--isImage', default=True, type=ast.literal_eval,
                        help='whether to save input image')
    # update in 2017/4/10
    parser.add_argument('--isFront', default=False, type=ast.literal_eval,
                        help='whether to frontalize vertices(mesh)')
    # update in 2017/4/25
    parser.add_argument('--isDepth', default=True, type=ast.literal_eval,
                        help='whether to output depth image')
    # update in 2017/4/27
    parser.add_argument('--isTexture', default=False, type=ast.literal_eval,
                        help='whether to save texture in obj file')
    parser.add_argument('--isMask', default=False, type=ast.literal_eval,
                        help='whether to set invisible pixels(due to self-occlusion) in texture as 0')
    # update in 2017/7/19
    parser.add_argument('--texture_size', default=256, type=int,
                        help='size of texture map, default is 256. need isTexture is True')
    parser.add_argument('--frames-per-video', default=-1, type=int,
                        help='Number of frames per video (folder) to process')
    main(parser.parse_args())
