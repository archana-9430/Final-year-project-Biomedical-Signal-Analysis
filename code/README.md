# Final-year-project-Biomedical-Signal-Analysis

ML/DL based PPG signal analysis and processing for disease identification and related hardware implementation.

Current work:
1. Started working on a 125Hz PPG signal dataset with 8min long
2. The flow of work is:
    a) Text to csv (make it columnwise, ie., having a single row)
    b) Perform filering (using 4th order butterworth bandpass filter(0.2-5Hz))
    c) Non-beat Segmentation (10sec) and Annotations (2 classes: Clean / Corrupted)
    d) Feature extraction (Entropy, Skewness, Kurtosis, Morphological features)
    e) Build a RF and a SVM model based on this.

3. Procedure fo Annotations:
    a) Divide the Signals into fixed length segments(say, 10sec, for segmentation approach).
    b) Check the beats present in that window. Please note the annotation values we used are Class-1(clean) and Class-2(corrupted).
    c) Fixed a threshold level for the number of perfectly clean beats in the segment for the entire segment to be identified as a clean segment. For the BIDMC dataset, we set the level value to be 0.9, i.e., if 9 out of 10 beats are clean then the segment will be annotated as clean (1). However, thuis threshold value may vary for different dataset and go upto 0.75.
    d) For both partly clean and corrupted, we annotated it as a corrupted signal (Class-2).

Foundation work:
1. Went through the NPTEL lectures in Machine Learning.
2. Explored the disease dataset(available on Kaggle) and perform ML techniques on that(for learning phase).
3. Designed the basics model(Linear regression, k-NN, Decision Tree, Random Forest, SVM, Naive Bayes) on those freely available datasets.
4. Leart about different feature extraction and minimisation tools like PCA, DWT, EMD, AE.



