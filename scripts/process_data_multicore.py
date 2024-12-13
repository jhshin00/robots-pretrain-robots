import os
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json
import numpy as np
import pickle

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def process_step(base_name, step, i, save_root_dir, other_data_dict):
    observation = step['observation']
    # Process images
    for image_type in ['exterior_image_1_left', 'exterior_image_2_left']: # remove wrist_image_left to save space
        if image_type in observation:
            image_data = observation[image_type]  # Receive NumPy array directly
            save_dir = os.path.join(save_root_dir, base_name, image_type)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{i}.png")
            plt.imsave(save_path, image_data)
            #print("Saved to " + save_path)

    # process state vectors
    for state_type in ['cartesian_position', 'gripper_position', 'joint_position']:
        if state_type in observation:
            state_vector = observation[state_type]
            if state_type not in other_data_dict:
                other_data_dict.update({state_type: [state_vector]})
            else:
                other_data_dict[state_type].append(state_vector)

    # process other type of data
    for data_type in ['action', 'action_dict', 'discount', 'is_first', 'is_last', 'is_terminal']:
        if data_type in step:
            data = step[data_type]
            if data_type not in other_data_dict:
                other_data_dict.update({data_type: [data]})
            else:
                other_data_dict[data_type].append(data)

    # Process language instructions if first step, because a whole traj shares same instructions
    if i == 0:
        for lang_field in ['language_instruction', 'language_instruction_2', 'language_instruction_3']:
            if lang_field in step:
                instruction = step[lang_field]  # Receive decoded string directly
                assert instruction != '', f"language instruction empty: '{instruction}', plz check"
                save_dir = os.path.join(save_root_dir, base_name, lang_field)
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, f"{i}.txt")
                with open(file_path, 'w') as f:
                    f.write(instruction)


def process_episode(data):
    e, save_root_dir = data
    try:
        episode_steps = e['steps']
        episode_metadata = e['episode_metadata']
        episode_reward = e['reward']
        file_path_tensor = episode_metadata['file_path']
        base_name = file_path_tensor.split('/')[-3:-1]  # Assume conversion to string has been done beforehand
        base_name = '_'.join(base_name)
        check_name = os.path.join(save_root_dir, base_name)
        other_data_dict = dict()
        if os.path.exists(check_name):
            print(f"{check_name} already exists, return!")
            return 

        for i, step in enumerate(episode_steps):
            process_step(base_name, step, i, save_root_dir, other_data_dict)
        
        # save other_data_dict as pickle file
        # with open(os.path.join(save_root_dir, base_name, "other_data.json"), 'w') as f:
        #     json.dump(other_data_dict, f, cls=NumpyEncoder)
        with open(os.path.join(save_root_dir, base_name, "other_data.pkl"), 'wb') as f:
            pickle.dump(other_data_dict, f)

    except Exception as ex:
        print(f"Error processing episode: {ex}")
    

def main(load_dataset_path, save_root_dir):
    try:
        loaded_dataset = tfds.builder_from_directory(load_dataset_path).as_dataset(split='all')
        i=1
        part_e=[]
        for e in tqdm(loaded_dataset):
            
            #print(i)
            episode_steps = [{
                            'observation': {k: v.numpy() for k, v in step['observation'].items()},
                            'language_instruction': step['language_instruction'].numpy().decode('utf-8') if 'language_instruction' in step else None,
                            'language_instruction_2': step['language_instruction_2'].numpy().decode('utf-8') if 'language_instruction_2' in step else None,
                            'language_instruction_3': step['language_instruction_3'].numpy().decode('utf-8') if 'language_instruction_3' in step else None,
                            'action': step['action'].numpy() if 'action' in step else None,
                            'action_dict': {k: v.numpy() for k, v in step['action_dict'].items()},
                            'discount': step['discount'].numpy() if 'discount' in step else None,
                            'is_first': step['is_first'].numpy() if 'is_first' in step else None,
                            'is_last': step['is_last'].numpy() if 'is_last' in step else None,
                            'is_terminal': step['is_terminal'].numpy() if 'is_terminal' in step else None,
                            }
                            for step in e['steps']]
            # remove traj w/o language
            if episode_steps[0]['language_instruction'] == '' or episode_steps[0]['language_instruction_2'] == '' or episode_steps[0]['language_instruction_3'] == '':
                episode_steps = []
                continue
            episode_metadata = {'file_path': e['episode_metadata']['file_path'].numpy().decode('utf-8'),
                                'recording_folderpath': e['episode_metadata']['recording_folderpath'].numpy().decode('utf-8')}
            episode_reward = {'reward': e['reward'].numpy() if 'reward' in e else None }
            part_e.append({'steps': episode_steps, 'episode_metadata': episode_metadata, 'reward': episode_reward})
            if i % (96 * 2)==0:
                print(i)
                st=time.time()
                with Pool(96) as pool:  # TODO change speficied number of cores     
                    result=list(pool.imap_unordered(process_episode,[(ee,save_root_dir) for ee in part_e]))
                et=time.time()
                print(et-st)
                part_e=[]
            i=i+1
        with Pool(64) as pool:      
            pool.imap(process_episode,[(ee,save_root_dir) for ee in part_e])
        
    finally:  
        print("finally")
        print(i) # 37616
   

if __name__ == "__main__":
    # TODO replace the path and mind space
    load_dataset_path = <path_to_raw_droid> # e.g. './data/dataset/droid/1.0.0'
    save_root_dir = <path_to_store_processed_data> # "./data/droid_processed"
    main(load_dataset_path, save_root_dir)
