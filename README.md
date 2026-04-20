Personalized Healthcare Recommendations
Using RFMT-Based Blood Donation Data and Logistic Regression
A Machine Learning Approach to Predictive Donor Classification

2024 • Healthcare Analytics & Machine Learning
Abstract
This paper presents a machine learning-based framework for predicting blood donation behaviour using donor history encoded in the RFMT (Recency, Frequency, Monetary, Time) model. The dataset comprises 748 records from a blood transfusion service, with each donor labelled as either a future donor (Class 1) or non-donor (Class 0). A Logistic Regression classifier was developed, trained, and evaluated using standard performance metrics. The model achieved an overall accuracy of 76.00%, a ROC-AUC score of 0.748, and a cross-validated accuracy of 77.28% across five folds. Findings indicate that recency of donation is the strongest negative predictor, while monetary volume exhibits a modest positive influence. The study highlights the feasibility of deploying binary classification models in healthcare settings to prioritise donor outreach, optimise hospital blood inventory, and personalise engagement strategies. Limitations such as class imbalance and feature scope are discussed, along with directions for future research including ensemble methods and deep learning architectures.
1. Introduction
Blood donation is a critical component of global healthcare infrastructure. The World Health Organization estimates that a stable and sufficient blood supply requires approximately 1% of the population to donate regularly. Despite this need, blood banks frequently face shortages due to unpredictable donor turnout, seasonal fluctuations, and attrition among repeat donors. Identifying which donors are likely to donate in the near future remains a practical and computationally tractable challenge.
Machine learning offers a compelling pathway to address this challenge. By learning patterns from donor history, predictive models can flag high-probability donors for targeted communication, reduce operational waste, and help healthcare institutions allocate recruitment resources more efficiently. This work proposes and evaluates a binary classification system built on logistic regression, applied to a publicly available blood donation dataset originally compiled by the Blood Transfusion Service Center in Hsin-Chu City, Taiwan.
The dataset is structured around the RFMT model, a behavioural analytics framework adapted from customer relationship management. In this context, Recency refers to the number of months since the donor's last donation, Frequency to the total number of donations made, Monetary to the total volume of blood donated in cubic centimetres, and Time to the number of months since the first donation. The binary target variable (Class) indicates whether the donor donated blood in March 2007.
This paper is organised as follows: Section 2 provides a review of relevant literature. Section 3 describes the dataset and exploratory analysis. Section 4 details the methodology including preprocessing and model design. Section 5 presents experimental results. Section 6 offers a discussion of findings, and Section 7 concludes with future directions.
2. Literature Review
2.1 Predictive Analytics in Healthcare
The application of predictive analytics in healthcare has expanded considerably over the past decade. Researchers have applied machine learning to domains including disease prognosis, readmission risk stratification, treatment outcome prediction, and patient triage. Logistic regression, as one of the foundational supervised learning algorithms, has been widely employed in clinical settings due to its interpretability, probabilistic output, and robustness on tabular data. Studies have demonstrated that logistic regression can achieve competitive performance relative to more complex models when datasets are well-structured and sample sizes are moderate.
2.2 Blood Donation Behaviour Models
The RFMT framework was introduced to blood donation contexts by Yeh et al. (2009), adapting the RFM model from marketing analytics. Their study on the Taiwanese donor dataset demonstrated that donor behaviour follows predictable patterns that can be captured with relatively few variables. Subsequent studies have explored gradient boosted trees, support vector machines, and neural networks on this dataset, achieving accuracy values between 74% and 82%. A recurring finding across this literature is that recency is the most predictive individual feature, as a recent donor is significantly more likely to donate again in the short term.
2.3 Class Imbalance in Medical Datasets
A consistent challenge in blood donation prediction, as in many medical classification tasks, is class imbalance. In most publicly available donor datasets, the proportion of non-donors substantially outweighs donors, which can bias classifiers towards the majority class and reduce sensitivity. Techniques such as SMOTE (Synthetic Minority Oversampling Technique), class weight adjustment, and threshold tuning have been proposed and evaluated to address this issue. The present study acknowledges this limitation and discusses its effect on the model's recall for the minority class.
3. Dataset Description and Exploratory Analysis
3.1 Dataset Overview
The dataset used in this study contains 748 donor records, each described by four numerical features and one binary class label. The data was sourced from the Blood Transfusion Service Center in Hsin-Chu City, Taiwan, and is widely used as a benchmark in classification research. No missing values were detected in any feature column, making preprocessing straightforward. Table 1 summarises the feature descriptions.
Table 1: Dataset Feature Descriptions
Feature	Description	Unit / Scale	Example Value
Recency	Months since last donation	Months (0 – 74)	2
Frequency	Total number of donations	Count (1 – 50)	5
Monetary	Total blood donated	Cubic cm (250 – 12,500)	1,250
Time	Months since first donation	Months (2 – 99)	34
Class (Target)	Donated in March 2007?	Binary: 0 = No, 1 = Yes	1
3.2 Statistical Summary
Descriptive statistics reveal several noteworthy characteristics of the dataset. The mean recency is approximately 9.5 months, with a standard deviation of 8.1 months, suggesting substantial variation in how recently donors last gave blood. Frequency ranges from 1 to 50 donations, with a mean of 5.5 and a right-skewed distribution, indicating that most donors have donated fewer than 10 times while a small cohort are highly frequent donors. Monetary values are exactly 250 times the frequency values, reflecting a fixed 250 cc per donation, which means monetary is a perfectly linear function of frequency. Table 2 presents the full statistical summary.
Table 2: Descriptive Statistics
Statistic	Recency	Frequency	Monetary	Time
Count	748	748	748	748
Mean	9.51	5.52	1,378.68	34.28
Std Dev	8.10	5.84	1,459.83	24.38
Minimum	0	1	250	2
25th Percentile	2.75	2	500	16
Median (50th)	7.00	4	1,000	28
75th Percentile	14.00	7	1,750	50
Maximum	74	50	12,500	99
3.3 Class Distribution
The target variable exhibits a notable class imbalance: 570 donors (76.2%) belong to Class 0 (did not donate in March 2007) while 178 donors (23.8%) belong to Class 1 (donated). This approximately 3:1 imbalance is a structural characteristic of blood donation data and must be considered when interpreting precision-recall trade-offs in model evaluation.
4. Methodology
4.1 Data Preprocessing
The dataset required minimal preprocessing. A null-value check confirmed that no features contained missing entries. Label encoding was applied to any categorical columns, though the dataset as structured contains only numerical features. The Monetary feature is linearly redundant with Frequency (Monetary = Frequency x 250), which was noted as a potential source of multicollinearity. Despite this, both features were retained to align with standard reporting on this benchmark dataset and to assess whether the model independently assigns different weights to the two variables.
No normalisation or standardisation was applied in the primary experiment, as logistic regression in scikit-learn supports both scaled and unscaled input. The dataset was split into training and test subsets using a 80:20 ratio with a fixed random seed of 42, yielding 598 training samples and 150 test samples.
4.2 Model Architecture
Logistic Regression was selected as the primary classifier for this study. It models the probability that a given input belongs to the positive class (Class 1) using the logistic sigmoid function applied to a linear combination of the input features. The model is well-suited to binary classification problems with numerical features and provides interpretable coefficients that allow direct assessment of each feature's influence on the prediction.
The scikit-learn implementation was used with a maximum iteration ceiling of 1,000 to ensure convergence, using the default L2 regularisation penalty (C=1.0) and the lbfgs solver. No threshold tuning was applied; predictions were generated using the default 0.5 probability cutoff.
4.3 Evaluation Metrics
Model performance was evaluated using the following metrics:
•	Accuracy: The proportion of correct predictions out of all predictions made.
•	Precision: Of all records predicted as Class 1, the fraction that truly belong to Class 1.
•	Recall (Sensitivity): Of all true Class 1 records, the fraction correctly identified.
•	F1-Score: The harmonic mean of precision and recall, balancing both concerns.
•	ROC-AUC: The area under the Receiver Operating Characteristic curve, measuring the model's ability to discriminate between classes across all thresholds.
•	5-Fold Cross-Validation Accuracy: Average accuracy across five stratified folds to assess generalisation stability.
5. Experimental Results
5.1 Model Performance
The trained logistic regression model achieved an accuracy of 76.00% on the held-out test set of 150 samples, marginally consistent with the 5-fold cross-validated accuracy of 77.28% (standard deviation: 2.66%), indicating stable generalisation with low variance across data splits. The ROC-AUC score of 0.748 suggests the classifier possesses meaningful discriminatory ability beyond a random baseline (AUC = 0.5). Table 3 presents the primary performance metrics.
Table 3: Model Performance Summary
Metric	Value
Test Accuracy	76.00%
ROC-AUC Score	0.748
5-Fold CV Accuracy	77.28% (±2.66%)
Macro-Average F1-Score	0.52
Weighted-Average F1-Score	0.69
5.2 Classification Report
A per-class breakdown reveals the asymmetric performance of the model. For Class 0 (non-donors), the model achieves a precision of 0.77, recall of 0.97, and F1-score of 0.86, indicating strong identification of the majority class. For Class 1 (donors), precision is 0.57 but recall drops sharply to 0.11, yielding an F1-score of only 0.18. This reflects the model's tendency to default towards the majority class due to the inherent class imbalance in the dataset. Table 4 presents the full classification report.
Table 4: Per-Class Classification Report
Class	Precision	Recall	F1-Score	Support
Class 0 (Non-Donor)	0.77	0.97	0.86	113
Class 1 (Donor)	0.57	0.11	0.18	37
Macro Average	0.67	0.54	0.52	150
Weighted Average	0.72	0.76	0.69	150
5.3 Confusion Matrix
The confusion matrix (Table 5) quantifies prediction outcomes across both classes. Of 113 true non-donors, 110 were correctly classified and 3 were incorrectly predicted as donors (false positives). Of 37 true donors, only 4 were correctly identified, while 33 were misclassified as non-donors (false negatives). The high false negative rate underscores the cost of class imbalance on minority-class recall.
Table 5: Confusion Matrix
	Predicted: Class 0	Predicted: Class 1
Actual: Class 0	110 (True Negative)	3 (False Positive)
Actual: Class 1	33 (False Negative)	4 (True Positive)
5.4 Feature Importance (Logistic Coefficients)
The logistic regression coefficients reflect each feature's contribution to the log-odds of donation. Table 6 presents the coefficient values. Recency carries the largest absolute coefficient (-0.0987), confirming that more time since last donation reduces the probability of donating again. Time since first donation also has a negative coefficient (-0.0232), indicating that longer-tenured donors are modestly less likely to donate in a given window. Monetary exhibits a small positive coefficient (0.0005), and Frequency yields a near-zero coefficient (0.0000), likely because its information is almost entirely captured by Monetary given their linear dependency.
Table 6: Logistic Regression Feature Coefficients
Feature	Coefficient	Direction	Interpretation
Recency	-0.0987	Negative	Higher recency → lower donation probability
Frequency	0.0000	Neutral	Absorbed by Monetary (linear redundancy)
Monetary	+0.0005	Positive	Higher total volume → marginally higher probability
Time	-0.0232	Negative	Longer tenure → slightly lower recent activity
6. Discussion
6.1 Interpretation of Findings
The results demonstrate that logistic regression is a viable baseline model for blood donation prediction, attaining accuracy and AUC scores comparable to published benchmarks on the same dataset. The model's ability to correctly flag 97% of non-donors is operationally useful in contexts where conserving outreach resources is a priority, as it correctly avoids contacting donors who are unlikely to respond. However, the low recall for actual donors (11%) limits its utility as a standalone tool for blood bank replenishment planning.
The dominance of Recency as the most influential predictor aligns with well-established donor behavioural theory: a person who donated recently is far more likely to donate again in the near future. This has a direct operational implication: blood banks should prioritise outreach to donors who gave blood within the past two to three months, rather than distributing communication uniformly.
6.2 Limitations
Several limitations should be acknowledged. First, the class imbalance (76% vs. 24%) substantially suppresses recall for the minority class. Without resampling or class weight adjustment, the classifier gravitates toward predicting the majority class. Second, the linear dependency between Monetary and Frequency reduces the effective feature dimensionality to three independent signals, limiting the expressiveness of the model. Third, the dataset lacks demographic variables (age, gender, blood type, geographic region) that are known to influence donation behaviour, constraining the potential predictive ceiling of any model trained on this data alone. Fourth, the study is limited to a single country and service centre, reducing external generalisability.
6.3 Implications for Personalised Healthcare
Beyond blood donation, the principles demonstrated here are directly transferable to personalised healthcare applications. The RFMT model is analogous to engagement tracking frameworks used in preventive care, medication adherence monitoring, and patient re-engagement. A hospital system could apply similar classification logic to identify patients at risk of disengaging from chronic disease management programmes, or to prioritise follow-up outreach after discharge. The ability to predict individual-level health-related behaviour from structured tabular records, using interpretable and computationally lightweight models, makes this approach particularly suitable for deployment in resource-constrained healthcare settings.
7. Conclusion and Future Work
This study developed and evaluated a logistic regression model for predicting blood donation behaviour based on RFMT features extracted from a blood transfusion service dataset. The model achieved an accuracy of 76%, a ROC-AUC of 0.748, and a stable cross-validated accuracy of 77.28%, establishing a competent baseline for this binary classification task. The analysis confirmed Recency as the primary predictor of future donation, offering a clear and actionable insight for healthcare practitioners.
Future work should explore several avenues. Ensemble methods such as Random Forest and Gradient Boosting have shown improved performance on this dataset and should be evaluated with explicit handling of class imbalance through SMOTE or class weighting. Feature engineering, including derived ratios such as Frequency-to-Time and Recency-to-Time, may improve predictive power. Threshold tuning at inference time can also shift the precision-recall trade-off to better serve specific operational objectives, such as maximising donor identification at the cost of increased false positives. Finally, extending the feature set with demographic, geographic, and health record data would be a natural and valuable enhancement for real-world deployment.
In summary, this work demonstrates that interpretable machine learning models, even at their simplest form, can provide meaningful and actionable predictions that support the personalisation of healthcare outreach and resource allocation strategies.
References
Yeh, I.-C., Yang, K.-J., and Ting, T.-M. (2009). Knowledge discovery on RFM model using Bernoulli sequence. Expert Systems with Applications, 36(3), 5866–5871.
Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
Chawla, N. V., Bowyer, K. W., Hall, L. O., and Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321–357.
Hosmer, D. W., Lemeshow, S., and Sturdivant, R. X. (2013). Applied Logistic Regression (3rd ed.). Wiley.
Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
World Health Organization (2023). Blood Safety and Availability. WHO Fact Sheet. Geneva: WHO Press.
Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.
