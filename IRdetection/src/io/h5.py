import h5py
import numpy as np
import pandas as pd
import os

class H5Manager:
    """
    A utility class to more easily manage h5 files for experiments.
    Enforces a specific structure of the files that is compatible with all experiment automation modules.

    The components of the structures are **Groups** and **Datasets**. Groups should be understand as *folders*,
    while Datasets contain (ND) data arrays (all elements in a same dataset must be of the same type).

    We then have **Metadata** used to save parameters of the experiment, the run or the specific acqusition.
    Metadata can be found either as:
    - global (file) metadata, when it refers to something valide for the whole file (usually run paramenters)
    - group metadata, which refers only to the what's inside that group
    - dataset metadata, refered only to the dataset (for example parameters for a sweep step)

    Note that groups can have subgroups in a nested way (they could also have datasets and subgroups)

    An example of a file structure:

    File\\
    ├─ global metadata\\
    ├─ Group <group name>\\
        ├─ group metadata\\
        ├─ Dataset <dataset name>\\
            ├─ dataset metadata\\
            └─ data (ND array)


    The class manages this files by keeping for each element (being it a group a dataset or the file itself) a
    dictionary of the form:
    \<file name\>:{
        type: file - gorup - dataset
        metadata: dictionary of metadata, elements are {meta_key : value}
        content: list of content, i.e. other elements
    }
    
    Parameters
    ----------
    file_path : str
        Path to the h5 file to manage. You don't have to include the '.h5' extension.
        If the file is not present it will be created.

    Methods
    -------
    """

    def __init__(self, file_path):
        self.file = file_path if file_path.endswith('.h5') else file_path + '.h5'
        # Check that the file exists, othewise create it
        if not os.path.exists(self.file):
            with h5py.File(self.file, 'w') as f:
                pass
        
        file_name = os.path.basename(self.file)
        # Get file metadata
        with h5py.File(self.file, 'r') as f:
            file_meta = dict(f.attrs)
        self.structure = {file_name: {'type': 'file', 'metadata': file_meta, 'content': []}}

    def _crawl_file(self):
        """
        Crawls through the h5 file and builds the structure dictionary
        that represents the hierarchical organization of groups and datasets.
        """
        file_name = os.path.basename(self.file)
        
        # Improved implementation that actually works
        # Reset structure with correct file metadata
        with h5py.File(self.file, 'r') as f:
            self.structure = {
                file_name: {
                    'type': 'file',
                    'metadata': dict(f.attrs),
                    'content': []
                }
            }
            
            # Simple recursive function to build the structure
            def build_structure(name, obj, parent_dict):
                base_name = name.split('/')[-1] if '/' in name else name
                
                if isinstance(obj, h5py.Group) and name != '':  # Skip root group
                    group_dict = {
                        'type': 'group',
                        'metadata': dict(obj.attrs),
                        'content': []
                    }
                    parent_dict['content'].append({base_name: group_dict})
                    
                    # Process all children of this group
                    for child_name, child_obj in obj.items():
                        full_path = f"{name}/{child_name}" if name else child_name
                        build_structure(full_path, child_obj, group_dict)
                        
                elif isinstance(obj, h5py.Dataset):
                    dataset_dict = {
                        'type': 'dataset',
                        'metadata': dict(obj.attrs),
                        'content': obj[()]
                    }
                    parent_dict['content'].append({base_name: dataset_dict})
            
            # Start building from the root
            for name, obj in f.items():
                build_structure(name, obj, self.structure[file_name])


# TESTING
if __name__ == '__main__':
    h5_manager = H5Manager("IRdetection/Experiments/PhotodiodeArea/run-1/raw_data.h5")
    h5_manager._crawl_file()
    print(h5_manager.structure)
               