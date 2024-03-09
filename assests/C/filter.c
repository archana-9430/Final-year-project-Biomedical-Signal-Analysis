#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define PI 3.141592653589793

char input_file[20] = "bidmc03m.csv";
char output_file[40] = "filtered_bidmc03m.csv";


void filtfilt(double *data, int data_length, double *b, double *a, int order) {
    // Allocate memory for temporary array
    double *tmp = (double *)malloc(data_length * sizeof(double));
    int order_half = order / 2;
    // forward filter
    for (int i = 0; i < data_length; i++) {
        tmp[i] = 0;
        for (int j = 0; j <= order; j++) {
            if (i - j >= 0) {
                tmp[i] += b[j] * data[i - j];
            }
            if (j > 0 && i - j >= 0) {
                tmp[i] -= a[j] * tmp[i - j];
            }
        }
    }

    // backward filter
    for (int i = data_length - 1; i >= 0; i--) {
        data[i] = 0;
        for (int j = 0; j <= order; j++) {
            if (i + j < data_length) {
                data[i] += b[j] * tmp[i + j];
            }
            if (j > 0 && i + j < data_length) {
                data[i] -= a[j] * data[i + j];
            }
        }
    }

    free(tmp);
}

int main() {
    // Filter specifications
    int order = 4;
    double low_cutoff = 0.2;  // Lower cutoff frequency in Hz
    double high_cutoff = 4.0;  // Upper cutoff frequency in Hz
    double fs = 125.0;  // Sampling frequency in Hz

    // Coefficient arrays
    double b[order + 1];
    b[0] = 0.00801714;
    b[1] = 0.00000000;
    b[2] = -0.01603429;
    b[3] =  0.00000000;
    b[4] =  0.00801714;
    double a[order + 1];
    a[0] = 1.00000000;
    a[1] = -3.72743886;
    a[2] = 5.21865042;
    a[3] = -3.25449722;
    a[4] = 0.76328926;

/*b: [ 0.00801714  0.         -0.01603429  0.          0.00801714]  
a:  [ 1.         -3.72743886  5.21865042 -3.25449722  0.76328926]*/

    FILE *file = fopen(input_file, "r");
    if (!file) {
        printf("Error opening file.\n");
        return 1;
    }
    int data_length = 0;
    char c;
    while((c = fgetc(file)) != EOF){
        if(c == '\n') data_length++;
    }
    // printf("data length %d", data_length); // 60002
    rewind(file);

    // Allocate memory for data
    double *data = malloc(data_length * sizeof(double));
    if(data == NULL) {
        printf("Data allocation error");
        return 1;
    }
    // Read data from file
    for (int i = 0; i < data_length; ++i) {
        fscanf(file, "%lf", &data[i]);
    }
    // data has total 60002 elements
    // Close file
    fclose(file);
 
    filtfilt(data, data_length, b, a, order);

    // Output the filtered data
    for (int i = 0; i < data_length; i++) {
        printf("%d: %f\n", i, data[i]);
    }
    // Save filtered data to a new CSV file
    // FILE *file2 = fopen("filtered_data.csv", "w");
    // if (file2 == NULL) {
    //     printf("Error opening file!\n");
    //     return 1;
    // }

    // // Write filtered data to the file
    // for (int i = 0; i < data_length; i++) {
    //     fprintf(file, "%f\n", data[i]);
    // }

    // fclose(file2);
    
    return 0;
}
