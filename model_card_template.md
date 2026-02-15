# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Gradient Boosting Classifier implemented using scikit-learn's `GradientBoostingClassifier` with default hyperparameters. It was trained to predict whether an individual's annual income exceeds $50K based on census demographic attributes. The model uses one-hot encoding for categorical features and a label binarizer for the binary target variable. The model is serialized and stored as a pickle file.

## Intended Use

This model is intended to predict whether an individual earns more than $50K per year based on publicly available census data. It is designed for educational and research purposes as part of a scalable ML pipeline deployed with FastAPI. It should not be used for making consequential decisions about individuals, such as credit approval, hiring, or eligibility determinations.

## Training Data

The training data is derived from the UCI Census Income dataset (also known as the "Adult" dataset). The full dataset contains 32,562 rows and 15 columns, including 8 categorical features (workclass, education, marital-status, occupation, relationship, race, sex, native-country) and 6 continuous features (age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week). The target label is "salary," a binary classification of <=50K or >50K. An 80/20 train-test split was applied using `train_test_split` with `random_state=42`, resulting in approximately 26,049 training samples.

## Evaluation Data

The evaluation data consists of the 20% holdout split from the original census dataset, resulting in approximately 6,513 test samples. The same one-hot encoder and label binarizer fitted on the training data were applied to the test set to ensure consistent feature representation.

## Metrics

The model was evaluated using three metrics: **Precision**, **Recall**, and **F1 score** (fbeta with beta=1).

**Overall performance on the test set:**
- Precision: ~0.8030
- Recall: ~0.6170
- F1: ~0.6980

**Slice performance highlights (by categorical feature):**

| Slice | Precision | Recall | F1 |
|---|---|---|---|
| sex: Male | 0.8037 | 0.6428 | 0.7143 |
| sex: Female | 0.7973 | 0.5064 | 0.6194 |
| race: White | 0.8056 | 0.6211 | 0.7015 |
| race: Black | 0.8000 | 0.5538 | 0.6545 |
| education: Bachelors | 0.7580 | 0.7867 | 0.7721 |
| education: HS-grad | 0.9118 | 0.2696 | 0.4161 |
| education: Masters | 0.8381 | 0.8502 | 0.8441 |
| education: Doctorate | 0.8361 | 0.8947 | 0.8644 |

Full per-slice metrics across all 8 categorical features are available in `slice_output.txt`.

## Ethical Considerations

The model is trained on census data that reflects historical societal biases. Features such as race, sex, and native-country are included as inputs, and the model may encode and perpetuate existing disparities in income distribution. Slice analysis shows performance gaps across demographic groups — for example, recall for females (0.5064) is notably lower than for males (0.6428), meaning the model is less effective at identifying high earners among women. Similarly, performance varies across racial groups and education levels. This model should not be used in any decision-making context where fairness and equity are required without additional bias mitigation steps.

## Caveats and Recommendations

- The model uses default hyperparameters for `GradientBoostingClassifier` and has not been tuned. Hyperparameter optimization (e.g., grid search or random search) could improve performance.
- The dataset may contain missing values encoded as "?" in categorical features (e.g., workclass and occupation), which are treated as a distinct category rather than being imputed.
- Performance on small population slices (e.g., native-country with fewer than 10 samples) may be unreliable due to limited sample sizes.
- The model's recall is notably lower than precision, suggesting it is conservative in predicting the >50K class. Depending on the use case, threshold tuning could help balance precision and recall.
- It is recommended to periodically retrain the model as demographic and economic patterns evolve over time.
