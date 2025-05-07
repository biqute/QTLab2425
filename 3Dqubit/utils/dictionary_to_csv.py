import csv
import io

def dictionary_to_csv(dictionary):
    # Create an in-memory text stream
    output = io.StringIO()
    writer = csv.writer(output)

    # Write key-value pairs
    for key, value in dictionary.items():
        writer.writerow([key, value.strip()])  # Strip to remove newline characters

    # Get CSV string
    csv_string = output.getvalue()
    output.close()

    return csv_string
