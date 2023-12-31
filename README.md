# Priority Attention Score Prediction

This project aims to predict an "Priority Score" for data based on a variety of factors such as the days since creation, their status, and the statuses of various child data. The higher the attention score, the more attention the record might require.

## Overview

Using a synthetic dataset, we predict the attention score by transforming categorical statuses into weighted values, combining them with other features, and training a regression model on TensorFlow. This helps in ranking records based on the predicted urgency of their requirements.

## Key Results
|Creation Date|Days Since Creation|Parent Status|Child1 Status|Child2 Status|Child3 Status|Child4 Status|Child5 Status|Weighted Days Since Creation|Weighted Parent Status|Weighted Child1 Status|Weighted Child2 Status|Weighted Child3 Status|Weighted Child4 Status|Weighted Child5 Status|Priority Score (Attention Score)|Predicted Priority Score|
|-------------|-------------------|-------------|-------------|-------------|-------------|-------------|-------------|----------------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|--------------------------------|------------------------|
|2023-09-11   |17                 |1            |0            |1            |3            |2            |1            |0.009315                    |0.120                 |0.015                 |0.040                 |0.010                 |0.025                 |0.040                 |0.708333                        |0.656048                |
|2023-08-26   |33                 |1            |3            |0            |0            |1            |1            |0.018082                    |0.120                 |0.010                 |0.015                 |0.015                 |0.040                 |0.040                 |0.666667                        |0.637899                |
|2022-11-24   |308                |1            |3            |1            |0            |2            |2            |0.168767                    |0.120                 |0.010                 |0.040                 |0.015                 |0.025                 |0.025                 |0.645833                        |0.621230                |
|2022-12-13   |289                |2            |1            |2            |0            |2            |1            |0.158356                    |0.075                 |0.040                 |0.025                 |0.015                 |0.025                 |0.040                 |0.583333                        |0.495786                |
|2023-04-02   |179                |2            |1            |0            |3            |1            |1            |0.098082                    |0.075                 |0.040                 |0.015                 |0.010                 |0.040                 |0.040                 |0.583333                        |0.497784                |
|2023-04-23   |158                |1            |1            |0            |0            |3            |0            |0.086575                    |0.120                 |0.040                 |0.015                 |0.015                 |0.010                 |0.015                 |0.562500                        |0.567729                |
|2023-05-09   |142                |2            |3            |3            |2            |1            |1            |0.077808                    |0.075                 |0.010                 |0.010                 |0.025                 |0.040                 |0.040                 |0.500000                        |0.478082                |
|2023-01-10   |261                |2            |0            |3            |1            |3            |1            |0.143014                    |0.075                 |0.015                 |0.010                 |0.040                 |0.010                 |0.040                 |0.458333                        |0.453264                |
|2023-06-10   |110                |0            |1            |3            |1            |1            |3            |0.060274                    |0.045                 |0.040                 |0.010                 |0.040                 |0.040                 |0.010                 |0.437500                        |0.378995                |
|2022-10-29   |334                |0            |2            |3            |1            |2            |1            |0.183014                    |0.045                 |0.025                 |0.010                 |0.040                 |0.025                 |0.040                 |0.437500                        |0.382849                |
|2023-06-26   |94                 |1            |0            |0            |3            |3            |3            |0.051507                    |0.120                 |0.015                 |0.015                 |0.010                 |0.010                 |0.010                 |0.416667                        |0.514782                |
|2023-06-24   |96                 |0            |0            |1            |1            |0            |3            |0.052603                    |0.045                 |0.015                 |0.040                 |0.040                 |0.015                 |0.010                 |0.354167                        |0.359646                |
|2023-01-20   |251                |2            |3            |2            |2            |3            |0            |0.137534                    |0.075                 |0.010                 |0.025                 |0.025                 |0.010                 |0.015                 |0.333333                        |0.396326                |
|2023-09-03   |25                 |0            |1            |2            |0            |3            |0            |0.013699                    |0.045                 |0.040                 |0.025                 |0.015                 |0.010                 |0.015                 |0.291667                        |0.318809                |
|2022-10-15   |348                |3            |1            |2            |2            |0            |0            |0.190685                    |0.030                 |0.025                 |0.025                 |0.015                 |0.015                 |0.015                 |0.291667                        |0.287373                |
|2022-10-10   |353                |0            |0            |3            |2            |0            |1            |0.193425                    |0.045                 |0.015                 |0.010                 |0.025                 |0.015                 |0.040                 |0.291667                        |0.323153                |
|2023-08-25   |34                 |2            |0            |0            |0            |3            |0            |0.018630                    |0.075                 |0.015                 |0.015                 |0.015                 |0.010                 |0.015                 |0.270833                        |0.370571                |
|2023-05-04   |147                |0            |0            |2            |3            |2            |0            |0.080548                    |0.045                 |0.015                 |0.025                 |0.010                 |0.025                 |0.015                 |0.229167                        |0.301750                |
|2023-04-02   |179                |3            |2            |2            |0            |0            |0            |0.098082                    |0.030                 |0.025                 |0.025                 |0.015                 |0.015                 |0.015                 |0.187500                        |0.258929                |
|2023-05-04   |147                |3            |3            |0            |1            |0            |0            |0.080548                    |0.030                 |0.010                 |0.015                 |0.040                 |0.015                 |0.015                 |0.187500                        |0.258929                |

Key metrics after training the model:
- **R-squared (R2):** 0.8582038773298546
  - This value suggests that approximately 87.05% of the variance in the dependent variable (Attention Score) can be explained by the independent variables (features). 
- **Mean Absolute Error (MAE):** 0.05185547471046448
  - On average, the model's predictions are 0.0496 units away from the actual values.
- **Mean Squared Error (MSE):** 0.06473353466935941
  - This metric represents the average of the squares of the errors between predicted and actual values.
- **Root Mean Squared Error (RMSE):** 0.07526811877386448
  - This is the square root of MSE and indicates the absolute fit of the model to the data.
- **Variance of True Attention Scores:** 0.029552506510416653
- **Variance of Predicted Attention Scores:** 0.01600724086165428

## How It Works

1. **Data Generation:** Synthetic data is generated to simulate record statuses and other record child statuses over a period of a year.
2. **Weight Assignment:** Weights are assigned to each status based on their importance.
3. **Feature Engineering:** The data is then processed to calculate weighted features which are used to predict the attention score.
4. **Model Training:** A TensorFlow model is trained on the processed data.
5. **Prediction:** The trained model is used to predict attention scores on the test set.
6. **Evaluation:** Various metrics like R2, MAE, MSE, etc., are calculated to evaluate the performance of the model.
