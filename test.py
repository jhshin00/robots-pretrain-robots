import os
import h5py

def print_hdf5_structure(group, indent=0):
    """
    HDF5 Group 또는 File 객체를 넘기면
    하위 그룹/데이터셋 이름과 shape, dtype을 들여쓰기와 함께 출력합니다.
    """
    for key, item in group.items():
        prefix = "  " * indent
        if isinstance(item, h5py.Group):
            print(f"{prefix}{key}/")
            print_hdf5_structure(item, indent+1)
        else:  # Dataset
            print(f"{prefix}{key}  — dataset, shape={item.shape}, dtype={item.dtype}")

if __name__ == "__main__":
    h5_dir = "./data/libero/libero_spatial/"   # HDF5 파일들이 들어있는 폴더
    for filename in os.listdir(h5_dir):
        if not filename.endswith(".hdf5"):
            continue
        h5_path = os.path.join(h5_dir, filename)
        print(f"\n--- Structure of {filename} ---")
        with h5py.File(h5_path, "r") as f:
            print_hdf5_structure(f)