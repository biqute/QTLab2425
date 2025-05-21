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
    ├─ Group \\<group name\\>\\
        ├─ group metadata\\
        ├─ Dataset \\<dataset name\\>\\
            ├─ dataset metadata\\
            └─ data (ND array)


    The class manages this files by keeping for each element (being it a group a dataset or the file itself) a
    dictionary of the form:
    \\<file name\\>:{
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
        self._crawl_file() # Initialize the structure with the file metadata

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

    def get_datasets(self):
        """
        Returns a list of all datasets in the file.

        Recursively traverses the structure to find all datasets.
        
        Returns
        -------
        list
            A list of tuples, each containing the path to the dataset and its metadata.
        """
        datasets = []
        
        def traverse(items, current_path):
            for item in items:
                # Each item is a dictionary with a single key (name)
                for name, details in item.items():
                    path = f"{current_path}/{name}" if current_path else name
                    
                    if details['type'] == 'dataset':
                        datasets.append(path)
                    elif details['type'] in ['group', 'file'] and 'content' in details:
                        traverse(details['content'], path)
        
        # Start traversal from file's content
        file_name = os.path.basename(self.file)
        traverse(self.structure[file_name]['content'], '')
        
        return datasets
    
    def get_dataset(self, dataset_name):
        """
        Returns the data of a specific dataset.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to retrieve.

        Returns
        -------
        np.ndarray
            The data contained in the specified dataset.
        """
        with h5py.File(self.file, 'r') as f:
            dataset = f[dataset_name]
            data = dataset[()]
            return data
        
    def get_metadata(self, element_name):
        """
        Returns the metadata of a specific element (file, group, or dataset).

        Parameters
        ----------
        element_name : str
            The name of the element whose metadata is to be retrieved.

        Returns
        -------
        dict
            The metadata dictionary of the specified element.
        """
        with h5py.File(self.file, 'r') as f:
            return dict(f[element_name].attrs)
        
    def add_metadata(self, element_name, metadata):
        """
        Adds metadata to a specific element (file, group, or dataset).

        Parameters
        ----------
        element_name : str
            The name of the element to which metadata is to be added.
        metadata : dict
            The metadata dictionary to be added.

        Returns
        -------
        None
        """
        with h5py.File(self.file, 'a') as f:
            for key, value in metadata.items():
                f[element_name].attrs[key] = value

        # Update the structure dictionary
        self._crawl_file()

    def add_dataset(self, dataset_name, data, group_name='/', metadata=None):
        """
        Adds a new dataset to the HDF5 file.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to be added.
        data : np.ndarray
            The data to be stored in the dataset.
        group_name : str, optional
            The name of the group under which the dataset will be created.
        metadata : dict, optional
            Metadata to be associated with the dataset.

        Returns
        -------
        None
        """
        with h5py.File(self.file, 'a') as f:
            group = f.require_group(group_name)
            dataset = group.create_dataset(dataset_name, data=data)
            if metadata:
                for key, value in metadata.items():
                    dataset.attrs[key] = value

        # Update the structure dictionary
        self._crawl_file()

# TESTING
if __name__ == '__main__':
    h5_manager = H5Manager("IRdetection/Experiments/PhotodiodeArea/run-1/raw_data.h5")
    datasets = h5_manager.get_datasets()
    print("Datasets in the file:")
    for dataset in datasets:
        print(dataset)

    # # Example of adding a dataset
    # data = np.random.rand(10, 10)
    # h5_manager.add_dataset("new_dataset", data, group_name="Group1", metadata={"description": "Random data"})
    # print("Added new dataset 'new_dataset' to 'Group1'.")
    # # Example of getting a dataset
    # dataset_data = h5_manager.get_dataset("Group1/new_dataset")
    # print("Data from 'Group1/new_dataset':")
    # print(dataset_data)
