#include <stdio.h>
#include <stdlib.h>

#define SAMPLE_RATE 125 // Sample rate (samples per second)
#define SEGMENT_LENGTH (SAMPLE_RATE * 10) // Segment length (10 seconds)
#define STRIDE_LENGTH (SAMPLE_RATE * 6) // Stride length (6 seconds)

char input_file[20] = "filtered_data.csv";

int main() { 
    double data[60002]; // Adjust the size as per your maximum data size
    int data_length = 0;
    int num_segments = 0;

    // Open the CSV file for reading
    FILE* file = fopen(input_file, "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    // Read data from CSV file
    while (fscanf(file, "%lf,", &data[data_length]) != EOF) {
        data_length++;
    }

    fclose(file);

    // Calculate the number of segments
    num_segments = (data_length - SEGMENT_LENGTH) / STRIDE_LENGTH + 1;
    printf("%d", num_segments);

    // Open a new CSV file for writing segmented data
    FILE* file2 = fopen("segmented_data.csv", "w");
    if (file2 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    // Write headers for each segment
    for (int i = 0; i < num_segments; i++) {
        fprintf(file2, "Segment_%d,", i + 1);
    }
    fprintf(file2, "\n");

    // Write segmented data to the file with overlap
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        for (int j = 0; j < num_segments; j++) {
            int start_index = j * STRIDE_LENGTH + i;
            if (start_index < data_length) {
                fprintf(file2, "%lf,", data[start_index]);
            } else {
                fprintf(file2, ",");
            }
        }
        fprintf(file2, "\n");
    }

    fclose(file2);

    return 0;
}
