# Final-year-project-Biomedical-Signal-Analysis
-----------------------------------------------

Machine Learning based PPG signal quality analysis and scavenging maximum information from corrupted segments with related hardware implementation.
## Workflow:
### Dataset preparation: 
* PPG signal from a total of 71 patients was data taken from different publicly avaialble datasets are used like, BIDMC dataset(15 subjects), MIMIC III dataset(17 subjects), MIMIC PERForm AF dataset(19 subjects), CSL Benchmark dataset(2 subjects) and DaLiA dataset (15 subjects).
* The frequency of all the dataset is made equal to 125Hz.
* The duration of the data taken from each subjects was kept 10min long except for BIDMC, which was 8min long.

### Segmentation:
* Smaller segments of 10 secs window are created. Overlapping segmentation is used with an overlap of about 4sec(10sec/6sec).

### Annotation:
* Each 10s segments were classified as Clean, Partly Corrupted and Corrupted signals depending upon the noise levels present.
*  A segment having noise level:
      *    noise level < 10% were labelled as Clean(0)
      *  10% < noise level < 30% were labelled as Partly Corrupted(1)
      *   noise level > 30% were labelled as Corrupted(2)
*  Annotation result:
    *  Class 0: 4506 segments: 67.5% of entire dataset
    *  Class 1: 1402 segments: 21% of entire dataset
    *  Class 2: 761 segments: 11.5% of entire segments

### Filteration:
* Butterworth 4th order bandpass filter with cut-off frequency of 0.2 - 4Hz was used.
* Filter evaluation was done on filters like Chebyshev-I, Chebyshev-II and Elliptic filter with varying specifications(order, cutoff frequency, etc).
*  Filter evaluation was done using different features like SNR improvement, Pearson coefficient correlation(PCC or NCC), Smoothness factor(SF), Mean square error, Root mean square error(RMSE), Peak Signal-to-Noise (PSNR).
* DaliA Noise dataset was used for adding noise for analysis.

### Feature Extracted: 
* A total of 145 features from time domain, Freqency domain, DWT domain (mother wavelet), Entropy measures were calculated.

### Feature Selection:
* Different methods like Recursive feature elimination(RFE),  Maximum Relevance — Minimum Redundancy(mRMR), Grey Wolf optimisation were used.

### Training:
* Different classification models like Random Forest model(RF), Support Vector Machine(SVM), Decision Trees(DT), Naive Bayes(NB) _k_-NN(_k_-Nearest Neighbours)  were trained and tested.
* Total number of segments obtained are 6669, out of which 3335 are used for train and remaining 3334 are used for test (50:50 split).
* The highest accuracy recorded was 86.746% using RF model with 200 trees and max_depth of 10

# Exploring Phase:
## Initial work done on BIDMC datatset(exploring phase):
#### 1. Started working on a 125Hz PPG signal dataset with 8min long segments.
#### 2. The flow of work is: 
    a) Text to csv (make it columnwise, ie., having a single row) 
    b) Perform filering (using 4th order butterworth bandpass filter(0.2-5Hz)) 
    c) Non-beat Segmentation (10sec) and Annotations (2 classes: Clean / Corrupted) 
    d) Feature extraction (Entropy, Skewness, Kurtosis, Morphological features) 
    e) Build a RF and a SVM model based on this.

#### 3. Procedure for Annotations: 
    a) Divide the Signals into fixed length segments(say, 10sec, for segmentation approach). 
    b) Check the beats present in that window. Please note the annotation values we used are Class-1(clean) and Class-2(corrupted). 
    c) Fixed a threshold level for the number of perfectly clean beats in the segment for the entire segment to be identified as a clean segment. For the BIDMC dataset, we set the level value to be 0.9, i.e., if 9 out of 10 beats are clean then the segment will be annotated as clean (1). However, thuis threshold value may vary for different dataset and go upto 0.75. 
    d) For both partly clean and corrupted, we annotated it as a corrupted signal (Class-2). 

# Foundation work:
1. Went through the NPTEL lectures on Machine Learning.
2. Read 31 papers in the doamin of Signal Qulaity assessment of PPG signal.
3. Explored the disease dataset(available on Kaggle) and perform ML techniques on that(for learning phase).
4. Designed the basics model(Linear regression, <em>k</em>-NN, Decision Tree, Random Forest, SVM, Naive Bayes) on those freely available datasets.
5. Learnt about different feature extraction and minimization tools like PCA, DWT, EMD, AE.



