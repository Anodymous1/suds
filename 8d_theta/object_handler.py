from __future__ import annotations
from typing import Any
import os
import pickle
import pandas as pd
from numpy import ndarray, array
from torch import Tensor, tensor


def _is_overwrite(file_path:str):
    """
    Returns true if the action is going to overwrite files
    """
    
    if os.path.exists(file_path):
        print(f"Warning: {file_path} already exists.")
        return True
    else: 
        return False
    
    
def save_csv(obj:ndarray|Tensor, file_path:str, override:bool = False) -> None:
    """
    Saves a csv file using pandas
    
    params:
    - obj: the object
    - file_path: save to file path
    - override: True if overide the overwrite protection
    """
    if not _is_overwrite(file_path) or override:
        pd.DataFrame(obj).to_csv(file_path, index=None, header=None)
        
def save_pickle(obj:Any, file_path:str, override:bool = False) -> None:
    """
    Saves a .pkl file using pickle
    
    params:
    - obj: the object
    - file_path: save to file path
    - override: True if overide the overwrite protection
    """
    
    if not _is_overwrite(file_path) or override:
        with open(file_path, "wb") as handle:
            pickle.dump(obj, handle)

def load_pickle(file_path:str) -> Any:
    """
    Load a .pkl file
    
    params:
    - file_path: file directory
    """
    
    with open(file_path, "rb") as file:
        return pickle.load(file)
    
def load_csv(file_path:str, type:str) -> ndarray|Tensor:
    """
    Loads a csv file as a numpy array or torch tensor
    
    params:
    - file_path: file directory
    - type: output type
    
    Preconditions:
    - type in ["ndarray", "Tensor"]
    """
    df = pd.read_csv(file_path, header=None)
    if type == "ndarray":
        df = array(df)
    elif type == "Tensor":
        df = tensor(df.values).float()
        
    return df
        
def load_galaxies(file_path:str, type:str) -> ndarray|Tensor:
    """
    Loads a csv file as a numpy array or torch tensor for galaxy
    Returns the parameters and the number of stars in the galaxy
    
    params:
    - file_path: file directory
    - type: output type
    
    Preconditions:
    - type in ["ndarray", "Tensor"]
    """
    df = load_csv(file_path, "Tensor")
    theta, k = df[:,:-1], df[:,-1].long()
    
    if type == "ndarray":
        theta = array(theta)
        k = array(k)
    
    return theta, k 
