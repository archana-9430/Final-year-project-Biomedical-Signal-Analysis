#include <stdio.h>
#include <stdlib.h>

#define SAMPLE_RATE 125 // Sample rate (samples per second)
#define SEGMENT_LENGTH (SAMPLE_RATE * 10) // Segment length (10 seconds)

char input_file[20] = "filtered_data.csv";

int main() { 
    double data[10000]; // Adjust the size as per your maximum data size
    int data_length = 0;
    int num_segments;

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
    num_segments = data_length / SEGMENT_LENGTH;

    // Open a new CSV file for writing segmented data
    File* file2 = fopen("segmented_data.csv", "w");
    if (file2 == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    // Write headers for each segment
    for (int i = 0; i < num_segments; i++) {
        fprintf(file2, "Segment_%d,", i + 1);
    }
    fprintf(file2, "\n");

    // Write segmented data to the file
    for (int i = 0; i < SEGMENT_LENGTH; i++) {
        for (int j = 0; j < num_segments; j++) {
            fprintf(file2, "%lf,", data[i + j * SEGMENT_LENGTH]);
        }
        fprintf(file2, "\n");
    }

    fclose(file2);

    return 0;
}
