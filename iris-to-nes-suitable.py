import os

file_path = 'data/iris/iris.data'
to_save = 'data/iris/iris-nes.data'
label_to_int = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

processed_lines = []

print(f"Reading data from: {file_path}")

try:
    with open(file_path, 'r') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')

            if len(parts) == 5:
                try:
                    features = []
                    for i in range(4):
                        feature_val = float(parts[i])
                        processed_feature = int(feature_val * 10)
                        features.append(str(processed_feature))

                    label_str = parts[4]
                    if label_str in label_to_int:
                        processed_label = str(label_to_int[label_str])
                    else:
                        print(f"Warning: Unknown label '{label_str}' found in line: {line}. Skipping.")
                        continue

                    processed_parts = features + [processed_label]
                    processed_line = ",".join(processed_parts)
                    processed_lines.append(processed_line)

                except ValueError as e:
                    print(f"Warning: Could not process line due to value error: {line}. Error: {e}. Skipping.")
                except IndexError:
                     print(f"Warning: Line format incorrect: {line}. Skipping.")
            else:
                print(f"Warning: Line does not have 5 parts: {line}. Skipping.")

    print(f"Writing processed data to: {to_save}")
    with open(to_save, 'w') as outfile:
        for line in processed_lines:
            outfile.write(line + '\n')

    print(f"Successfully processed {file_path}")
    print(f"Total lines processed and written: {len(processed_lines)}")

except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")