import csv

def read_metadata(filename):
    with open(filename, mode="r", newline='') as file:
        reader = csv.reader(file)
        return {row[0]: row[1] for row in reader if len(row) == 2}  # TODO: check length