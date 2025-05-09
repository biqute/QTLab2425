
import sys; sys.path.append("../utils")
from dictionary_to_csv import dictionary_to_csv

metadata = {
    "info": "empty cavity, three port configuration for S21 = H_i + H_o", 
    "info2": "empty cavity, three port configuration for S21 = H_i + H_o", 
    "info3": "empty cavity, three port configuration for S21 = H_i + H_o", 
}
print(dictionary_to_csv(metadata))