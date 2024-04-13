import os
import shutil
import subprocess
import pysam
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Delete folder.
def delete_folder(folder_path):
    shutil.rmtree(folder_path, ignore_errors=True)

# NGS data preprocessing.
def data_preprocessing(output_folder, bam_file):
    # Remove duplicate reads.
    input_bam = bam_file
    output_bam = os.path.join(output_folder, os.path.basename(bam_file).replace('.bam', '.remove_dup.bam'))
    marked_dup_metrics = os.path.join(output_folder, os.path.basename(bam_file).replace('.bam', '.marked_dup_metrics.txt'))
    # PICARD = '/home/tinhnh/aidNIPT/picard/picard.jar'
    # PICARD = '/home/tori/Tori/picard/picard.jar'
    command = ['java', '-jar', PICARD, 'MarkDuplicates', f'I={input_bam}', f'O={output_bam}', f'M={marked_dup_metrics}', 'REMOVE_DUPLICATES=true']
    subprocess.run(command, check=True)
    os.remove(marked_dup_metrics)

    # Remove mapping quality below 30.
    input_bam = output_bam
    output_bam = os.path.join(output_folder, os.path.basename(input_bam).replace('.bam', '.above_30.bam'))
    command = ['samtools', 'view', '-b', '-h', '-q', '30', f'{input_bam}', '-o', f'{output_bam}']
    subprocess.run(command, check=True)
    os.remove(input_bam)

    # TODO: Adjust the GC content and mappability.

    # Index bam file.
    input_bam = output_bam
    output_bai = os.path.join(output_folder, os.path.basename(input_bam).replace('.bam', '.bam.bai'))
    command = ['samtools', 'index', f'{input_bam}']
    subprocess.run(command, check=True)
    return output_bam, output_bai

# Calculation of FD.
def fd_calculatation(output_file, bam, tc, icc):
    # Calculate FD of chromosome.
    def calculate_fd_of_chr(bam, chr):
        # Removal of lower 10% and upper 10% bins of the normalized FD representative values.
        def remove_outliers(values):
            sorted_values = sorted(values)
            lower_index = int(len(values) * 0.1)
            upper_index = int(len(values) * 0.9)
            trimmed_values = sorted_values[lower_index:upper_index]
            result = [value for value in values if value in trimmed_values]
            return result 

        # Non-overlapping binning at 1 Mbp.
        chr_bam = bam.fetch(chr)
        positions = []
        for read in chr_bam:
            forward = not(read.flag & (1 << 4))
            if forward: 
                positions.append(read.reference_start + 80)
            else:
                positions.append(read.reference_end - 80)
        positions = sorted(positions)
        start_position = positions[0]
        positions = [position - start_position for position in positions]
        end_position = positions[-1]
        bin_size = 1e6
        fragment_distance = [[] for _ in range(int(end_position // bin_size) + 1)]
        for i in range(len(positions) - 1):
            position = positions[i]
            position_next = positions[i+1]
            if position == position_next:
                continue
            bin_position = int(position // bin_size)
            bin_position_next = int(position_next // bin_size)
            position_distance = position_next - position
            bin_seperate = bin_position_next * bin_size
            if bin_position == bin_position_next:
                fragment_distance[bin_position].append(position_distance)
            elif bin_seperate - position < position_next - bin_seperate:
                fragment_distance[bin_position].append(position_distance)
            else:
                fragment_distance[bin_position_next].append(position_distance)
        fd_mean = []; fd_median = []; fd_iqr = []
        for i in range(len(fragment_distance)):
            bin = fragment_distance[i]
            if len(bin) == 0:
                continue
            fd_mean.append(np.mean(bin))
            fd_median.append(np.median(bin))
            q1 = np.percentile(bin, 25)
            q3 = np.percentile(bin, 75)
            iqr = q3 - q1
            fd_iqr.append(iqr)
        
        fd_mean_median = np.median(fd_mean)
        fd_median_median = np.median(fd_median)
        fd_iqr_median = np.median(fd_iqr)
        fd_mean = [fd_mean_value / fd_mean_median for fd_mean_value in fd_mean]
        fd_mean = remove_outliers(fd_mean)
        fd_median = [fd_median_value / fd_median_median for fd_median_value in fd_median]
        fd_median = remove_outliers(fd_median)
        fd_iqr = [fd_iqr_value / fd_iqr_median for fd_iqr_value in fd_iqr]
        fd_iqr = remove_outliers(fd_iqr)
        return fd_mean, fd_median, fd_iqr

    # Save 1D array to text file.
    def save_1d_array_to_file(output_file, arr, header):
        with open(output_file, 'a') as file:
            file.write(header + '\n')
            for item in arr:
                file.write(str(item) + '\n')
    
    # Save 2D array to text file.
    def save_2d_array_to_file(output_file, arr, header):
        with open(output_file, 'a') as file:
            file.write(header + '\n')
            for row in arr:
                file.write(' '.join(map(str, row)) + '\n')

    # Calculate values.
    fd_mean_tc, fd_median_tc, fd_iqr_tc = calculate_fd_of_chr(bam, tc)
    fd_mean_icc = [0] * 3
    fd_median_icc = [0] * 3
    fd_iqr_icc = [0] * 3
    fd_mean_icc[0], fd_median_icc[0], fd_iqr_icc[0] = calculate_fd_of_chr(bam, icc[0])
    fd_mean_icc[1], fd_median_icc[1], fd_iqr_icc[1] = calculate_fd_of_chr(bam, icc[1])
    fd_mean_icc[2], fd_median_icc[2], fd_iqr_icc[2] = calculate_fd_of_chr(bam, icc[2])

    # Save data to txt file.
    save_1d_array_to_file(output_file, fd_mean_tc, '#fd_mean_tc')
    save_1d_array_to_file(output_file, fd_median_tc, '#fd_median_tc')
    save_1d_array_to_file(output_file, fd_iqr_tc, '#fd_iqr_tc')
    save_2d_array_to_file(output_file, fd_mean_icc, '#fd_mean_icc')
    save_2d_array_to_file(output_file, fd_median_icc, '#fd_median_icc')
    save_2d_array_to_file(output_file, fd_iqr_icc, '#fd_iqr_icc')

# Read array from text file.
def read_array_from_file(input_file):
    with open(input_file, 'r') as file:
        data = {}
        current_list = None
        arr_type = None 
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                current_list = line[1:]
                data[current_list] = []
            else:
                line = line.split(' ')
                if len(line) > 1:
                    line = [float(x) for x in line]
                else:
                    line = ''.join(line)
                    line = float(line)
                data[current_list].append(line)
        return data

# Get list files from folder.
def list_files(data_folder, value_folder, bam_files, value_files):
    # Create value folder and image folder if not exist.
    if not os.path.exists(value_folder):
        os.makedirs(value_folder)

    # List data files with absolute paths.
    bam_listdir = os.listdir(data_folder)
    for file in bam_listdir:
        if file.endswith('.bam'):
            bam_files.append(os.path.join(data_folder, file))
            bai_file = os.path.basename(file).replace('.bam', '.bam.bai')
            if bai_file not in bam_listdir:
                logging.info(f'Indexing {os.path.basename(value_file)}')
                input_bam = os.path.join(data_folder, file)
                command = ['samtools', 'index', f'{input_bam}']
                subprocess.run(command, check=True)

    # List value files with absolute paths.
    for file in os.listdir(value_folder):
        if file.endswith('.txt'):
            value_files.append(os.path.join(value_folder, file))

# Calculate value files.
def calculate_value_files(value_folder, bam_files, value_files, tc, icc):
    for bam_file in bam_files:
        sample_name = os.path.basename(bam_file).split('.bam')[0]
        value_file = os.path.join(value_folder, os.path.basename(bam_file).replace('.bam', '.txt'))
        if value_file not in value_files:
            # preprocessed_bam, preprocessed_bai = data_preprocessing(value_folder, bam_file)
            logging.info(f'Calculating value file: {sample_name}')
            bam = pysam.AlignmentFile(str(bam_file), 'rb')
            fd_calculatation(value_file, bam, tc, icc)
            value_files.append(value_file)
            # os.remove(preprocessed_bam)
            # os.remove(preprocessed_bai)

# Get max list from 2 lists.
def find_max_list(list_1, list_2):
    max_list = [
        max(list_1[0], list_2[0]), 
        max(list_1[1], list_2[1]), 
        max(list_1[2], list_2[2]), 
        max(list_1[3], list_2[3])
    ]
    return max_list

# Find y lim of sequential line.
def find_y_lim(value_files):
    fd_mean_max = [0] * 4
    fd_median_max = [0] * 4
    fd_iqr_max = [0] * 4
    for value_file in value_files:
        sample_name = os.path.basename(value_file).split('.txt')[0]
        # Get data from file.
        data = read_array_from_file(value_file)
        fd_mean_tc = data['fd_mean_tc']
        fd_median_tc = data['fd_median_tc']
        fd_iqr_tc = data['fd_iqr_tc']
        fd_mean_icc = data['fd_mean_icc']
        fd_median_icc = data['fd_median_icc']
        fd_iqr_icc = data['fd_iqr_icc']
        
        fd_mean_max = find_max_list(
            fd_mean_max, 
            [max(fd_mean_tc), max(fd_mean_icc[0]), max(fd_mean_icc[1]), max(fd_mean_icc[2])]
        )
        fd_median_max = find_max_list(
            fd_median_max,
            [max(fd_median_tc), max(fd_median_icc[1]), max(fd_median_icc[1]), max(fd_median_icc[2])]
        )
        fd_iqr_max = find_max_list(
            fd_iqr_max,
            [max(fd_iqr_tc), max(fd_iqr_icc[0]), max(fd_iqr_icc[1]), max(fd_iqr_icc[2])]
        )
    return fd_mean_max, fd_median_max, fd_iqr_max

# Generate image files.
def generate_image_files(image_folder, value_files, y_lim):
    # TRS image generation.
    def trs_image_generation(output_image, fd_tc, fd_icc, y_lim):
        fig, axs = plt.subplots(6, 1, figsize=(4, 2), dpi=200)

        for i in range(3):
            axs[i*2].plot(fd_icc[i], color='black')
            axs[i*2].axis('off')
            axs[i*2].set_xlim(0, len(fd_icc[i]))
            axs[i*2].set_ylim(0, y_lim[i + 1])
            axs[i*2].fill_between(range(len(fd_icc[i])), fd_icc[i], color = 'black')

            axs[i*2+1].plot(fd_tc, color='black')
            axs[i*2+1].axis('off')
            axs[i*2+1].set_xlim(0, len(fd_tc))
            axs[i*2+1].set_ylim(0, y_lim[0])
            axs[i*2+1].fill_between(range(len(fd_tc)), fd_tc, color = 'black')
            
        plt.axis('off')
        plt.savefig(output_image, bbox_inches='tight', pad_inches=0)

        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close(fig)  

    # Reset image folder.
    delete_folder(image_folder)
    os.makedirs(image_folder)
    
    for value_file in value_files:
        sample_name = os.path.basename(value_file).split('.txt')[0]
        # Get data from file.
        logging.info(f'Generating image file: {sample_name}')
        data = read_array_from_file(value_file)
        fd_mean_tc = data['fd_mean_tc']
        fd_median_tc = data['fd_median_tc']
        fd_iqr_tc = data['fd_iqr_tc']
        fd_mean_icc = data['fd_mean_icc']
        fd_median_icc = data['fd_median_icc']
        fd_iqr_icc = data['fd_iqr_icc']
        # Generate image.
        image_file_mean = os.path.join(image_folder, os.path.basename(value_file).replace('.txt', '.mean.png'))
        image_file_median = os.path.join(image_folder, os.path.basename(value_file).replace('.txt', '.median.png'))
        image_file_iqr = os.path.join(image_folder, os.path.basename(value_file).replace('.txt', '.iqr.png'))
        trs_image_generation(image_file_mean, fd_mean_tc, fd_mean_icc, y_lim[0])
        trs_image_generation(image_file_median, fd_median_tc, fd_median_icc, y_lim[1])
        trs_image_generation(image_file_iqr, fd_iqr_tc, fd_iqr_icc, y_lim[2])    

def main():
    # NEGATIVE_DATA_FOLDER_1 = '/data/tinhnh/NIPT/data/negatives'
    # NEGATIVE_DATA_FOLDER_2 = '/home/tinhnh/negatives2'
    # POSITIVE_DATA_FOLDER_1 = '/data/tinhnh/NIPT/data/positives'
    # POSITIVE_DATA_FOLDER_2 = '/home/tinhnh/positives2'
    NEGATIVE_DATA_FOLDER_1 = 'data/negatives'
    NEGATIVE_DATA_FOLDER_2 = 'data/negatives'
    POSITIVE_DATA_FOLDER_1 = 'data/positives'
    POSITIVE_DATA_FOLDER_2 = 'data/positives'

    negative_data_folder_1 = NEGATIVE_DATA_FOLDER_1
    negative_data_folder_2 = NEGATIVE_DATA_FOLDER_2
    negative_value_folder = 'data/negative_values'
    negative_image_folder = 'data/negative_images'

    positive_data_folder_1 = POSITIVE_DATA_FOLDER_1
    positive_data_folder_2 = POSITIVE_DATA_FOLDER_2
    positive_value_folder = 'data/positive_values'
    positive_image_folder = 'data/positive_images'

    # Calculate value files.
    negative_bam_files = []
    negative_value_files = []
    list_files(negative_data_folder_1, negative_value_folder, negative_bam_files, negative_value_files)
    calculate_value_files(
        negative_value_folder, 
        negative_bam_files, 
        negative_value_files, 
        'chr13', 
        ['chr4', 'chr5', 'chr6']
    )
    list_files(negative_data_folder_2, negative_value_folder, negative_bam_files, negative_value_files)
    calculate_value_files(
        negative_value_folder, 
        negative_bam_files,
        negative_value_files, 
        'chr13', 
        ['chr4', 'chr5', 'chr6']
    )

    positive_bam_files = []
    positive_value_files = []
    list_files(positive_data_folder_1, positive_value_folder, positive_bam_files, positive_value_files)
    calculate_value_files(
        positive_value_folder,
        positive_bam_files, 
        positive_value_files,
        'chr13', 
        ['chr4', 'chr5', 'chr6']
    )
    list_files(positive_data_folder_2, positive_value_folder, positive_bam_files, positive_value_files)
    calculate_value_files(
        positive_value_folder, 
        positive_bam_files, 
        positive_value_files,
        'chr13', 
        ['chr4', 'chr5', 'chr6']
    )

    # Find y_lim.
    negative_fd_mean_max, negative_fd_median_max, negative_fd_iqr_max = find_y_lim(negative_value_files)
    positive_fd_mean_max, positive_fd_median_max, positive_fd_iqr_max = find_y_lim(positive_value_files)
    fd_mean_max = find_max_list(negative_fd_mean_max, positive_fd_mean_max)
    fd_median_max = find_max_list(negative_fd_median_max, positive_fd_median_max)
    fd_iqr_max = find_max_list(negative_fd_iqr_max, positive_fd_iqr_max)

    # Generate image files.
    generate_image_files(negative_image_folder, negative_value_files, [fd_mean_max, fd_median_max, fd_iqr_max])
    generate_image_files(positive_image_folder, positive_value_files, [fd_mean_max, fd_median_max, fd_iqr_max])

if __name__ == '__main__':
    main()