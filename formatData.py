# Define paths to input and output files
input_file_path = 'labeled_name_data.txt'
output_file_path = 'output.txt'

# Open the input and output files
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        word, label = line.strip().split('\t')
        if label == '0':
            label = '__label__non_name'
        elif label == '1':
            label = '__label__name'
        else:
            continue
        # Write formatted line to output file
        outfile.write(f'{label} {word}\n')

print(f'Data reformatted and saved to {output_file_path}')
