import os
import cv2
import shutil
from tqdm import tqdm

video_folder = <path_to_store_processed_data> # e.g. './data/droid_processed'
cnt = 0
# TODO plz check before uncommenting the delete operations
for traj in tqdm(os.listdir(video_folder)): 
    traj_full_path = os.path.join(video_folder, traj)
    if sorted(os.listdir(traj_full_path)) != ['exterior_image_1_left', 'exterior_image_2_left', 'language_instruction', 'language_instruction_2', 'language_instruction_3', 'other_data.pkl']:
        print(f"1 Removing {traj_full_path}")
        # shutil.rmtree(traj_full_path)
        cnt+=1
        continue
    if os.listdir(os.path.join(traj_full_path, 'language_instruction')) != ['0.txt']:
        print(f"2 Removing {traj_full_path}")
        # shutil.rmtree(traj_full_path)
        cnt+=1
        continue
    if os.listdir(os.path.join(traj_full_path, 'language_instruction_2')) != ['0.txt']:
        print(f"3 Removing {traj_full_path}")
        # shutil.rmtree(traj_full_path)
        continue
    if os.listdir(os.path.join(traj_full_path, 'language_instruction_3')) != ['0.txt']:
        print(f"4 Removing {traj_full_path}")
        # shutil.rmtree(traj_full_path)
        cnt+=1
        continue

    if len(os.listdir(os.path.join(traj_full_path, 'exterior_image_1_left'))) < 40:
        print(f"5 Removing {traj_full_path} {len(os.listdir(os.path.join(traj_full_path, 'exterior_image_1_left')))}")
        # shutil.rmtree(traj_full_path)
        cnt+=1
        continue

    # if len(os.listdir(os.path.join(traj_full_path, 'wrist_image_left'))) != len(os.listdir(os.path.join(traj_full_path, 'exterior_image_2_left'))):
    #     print(f"6 Removing {traj_full_path}")
    #     shutil.rmtree(traj_full_path)
    #     continue

    # if len(os.listdir(os.path.join(traj_full_path, 'wrist_image_left'))) != len(os.listdir(os.path.join(traj_full_path, 'exterior_image_1_left'))):
    #     print(f"7 Removing {traj_full_path}")
    #     shutil.rmtree(traj_full_path)
    #     continue

    with open(os.path.join(traj_full_path, 'language_instruction', '0.txt'), 'r') as file:
        label = file.read()
    if len(label.split()) <= 1:
        print(f"8 Removing {traj_full_path}")
        # shutil.rmtree(traj_full_path)
        cnt+=1
        continue

    with open(os.path.join(traj_full_path, 'language_instruction_2', '0.txt'), 'r') as file:
        label = file.read()
    if len(label.split()) <= 1:
        print(f"9 Removing {traj_full_path}")
        # shutil.rmtree(traj_full_path)
        cnt+=1
        continue

    with open(os.path.join(traj_full_path, 'language_instruction_3', '0.txt'), 'r') as file:
        label = file.read()
    if len(label.split()) <= 1:
        print(f"10 Removing {traj_full_path}")
        # shutil.rmtree(traj_full_path)
        cnt+=1
        continue

print(f"Done! {cnt}")