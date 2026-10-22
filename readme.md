# One-zone direct writing! CEEMDAN decomposition + Informer-LSTM + XGBoost ensemble prediction model

Preface:
In this issue, we introduce a groundbreaking predictive model: the combination of CEEMDAN decomposition, Informer-LSTM, and XGBoost for forecasting. By employing the adaptive signal decomposition of CEEMDAN, the original sequence is decoupled into multiple frequency components, establishing a dual-level prediction channel structure: high-frequency and low-frequency. 

The high-frequency component, due to its complexity, utilizes an Informer-LSTM parallel model with rich parameters. This model, which integrates attention mechanisms and long short-term memory networks, better captures long-range dependencies and complex dynamic changes. 

On the other hand, the low-frequency component employs XGBoost, an efficient gradient boosting decision tree model capable of processing simple and low-frequency features quickly, thus avoiding overfitting. 

Each model independently predicts its corresponding component, generating the predicted values for each IMF; then, all predicted IMFs are recombined to construct the complete prediction signal. By reasonably assigning tasks to complex and simple models, we achieve optimal prediction performance on signals with different frequency characteristics. 

This model is suitable for single-variable dataset predictions (e.g., wind speed, load, lithium battery life prediction tasks).

1. Innovation Model Overview
1.1 Model Evaluation:
1.2 Forecast Visualization:
We also provide detailed materials, instructional documents, and video tutorials, including how to replace your own dataset, parameter tuning tutorials, and task replacements, along with code comments that explain each line of code and provide detailed parameter introductions:
● Data Set: Single feature (variable) dataset
● Environment Framework: Python 3.9 or higher, PyTorch 2.1 or higher versions can be run
● Stepwise Prediction Model Score: Test set 0.98
● User: Paper requirements, graduation design requirements
● Code Guarantee: Code comments are detailed, ready to run without modification.

2. Innovation Points of the Model
2.1 Decomposition-Combination Forecasting Strategy
1. Prediction of Component Features with Complex Models
Due to the large parameter size of complex models, they are suitable for predicting high-frequency complex component features. However, low-frequency component features are relatively simple. If complex models are still used, it may lead to overfitting and negatively impact the prediction results. Therefore, for low-frequency component features, we use simpler models (or machine learning models) to predict, followed by reconstruction of the predicted components to achieve high-precision predictions.
2.2 Data Preprocessing and Decomposition
The raw time series data is input into the CEEMDAN algorithm for decomposition. CEEMDAN is an improved ensemble empirical mode decomposition method that can effectively decompose signals into several intrinsic mode functions (IMFs), each representing a signal component at different frequencies.
Sample entropy is a method used to measure the complexity of a sequence, which can be evaluated by calculating the uncertainty in the sequence to assess its complexity. The higher the sample entropy, the greater the complexity of the sequence. Based on the frequency characteristics and complexity of each component, it is classified into high-frequency complex components and low-frequency simple components.
2.3 Prediction of High-Frequency Complex Components
Informer: Proficient in handling long sequences and capable of parallel computation, enhancing computational efficiency and prediction performance. Informer has been improved based on the Transformer, making it more suitable for time series data, especially those with long-term dependencies.
LSTM: Outperforms in capturing both short-term and long-term dependencies in sequence data, providing good handling of temporal relationships within the sequence data.
By parallelly using these two models, better patterns at different time scales can be captured, improving prediction accuracy and robustness.
2.4 Prediction of Low-Frequency Complex Components
Utilize XGBoost for modeling. XGBoost, with its fast training speed and strong generalization capabilities, provides stable prediction performance on low-frequency features.
2.5 Prediction Results Reconstruction and Evaluation: Recombine all the predicted IMFs to construct a complete prediction signal. This process ensures that the prediction results of each component are integrated into one whole, maintaining the original signal's structure and characteristics. Validate the combined prediction results by comparing them with real data, assessing the model's prediction accuracy, and making optimization adjustments based on feedback.
3 Data CEEMDAN Decomposition and Visualization:
3.1 Import Data:
3.2 CEEMDAN Decomposition:
According to the decomposition result, CEEMDAN has decomposed into 11 components, and then calculate the sample entropy value of each component for analysis. Sample entropy is a method used to measure the complexity of a sequence, which can be evaluated by calculating the uncertainty in the sequence to assess its complexity. The higher the sample entropy, the greater the complexity of the sequence. We use the first 6 high-sample entropy complex components as input for Informer-LSTM parallel model predictions, and the last 5 low-sample entropy simple components as input for XGBoost model predictions.
3.4 Data Set Preparation and Preprocessing:
Detailed descriptions are provided in the document!
4 Combination Prediction Model Based on CEEMDAN Decomposition + Informer-LSTM + XGBoost:
4.1 Define the Informer-LSTM Parallel Prediction Network Model:
4.2 Set parameters and train the model:
50 epochs, MSE is 0.000879, the Informer-LSTM parallel prediction effect is significant, the model can fully utilize the long-term dependency modeling ability of Informer and the short-term dependency capturing ability of LSTM, converges quickly, has excellent performance, predicts with high accuracy, and further improves the model prediction performance by appropriately adjusting model parameters.
4.3 Predictions Based on XGBoost Model:
Data loading, training data and test data grouping, 5 components, divided into 5 datasets
Save the predicted data, other components' predictions are consistent with the above process, just keep the final model results.
5 Results Visualization and Model Evaluation
5.1 Visualization of component prediction results
5.2 Visualization of combined prediction results
5.3 Model evaluation
From the component prediction results, it can be seen that the first 6 complex components fit well under the Informer-LSTM parallel prediction model, while the last 5 simple components fit very well under the XGBoost model's prediction, showing significant combined prediction effects!
Click to download: Full original data, Python code
Previously interesting content:
Time series forecasting: Analysis and comparison of LSTM, ARIMA, Holt-Winters, SARIMA models
Wind speed forecast (I) Data introduction and preprocessing
Wind speed forecast (II) EMD-LSTM model based on Pytorch
Wind speed forecast (III) EMD-LSTM-Attention model
Wind speed forecast (IV) EMD-Transformer model based on Pytorch
Wind speed forecast (V) EMD-CNN-LSTM model based on Pytorch
Wind speed forecast (VI) EMD-CNN-GRU parallel model based on Pytorch
CEEMDAN + combined prediction model (BiLSTM-Attention + ARIMA)
CEEMDAN + combined prediction model (CNN-LSTM + ARIMA)
CEEMDAN + combined prediction model (Transformer - BiLSTM + ARIMA)
CEEMDAN + combined prediction model (CNN-Transformer + ARIMA)
Multi-feature variable sequence forecasting (I) - CNN-LSTM wind speed forecasting model
Multi-feature variable sequence prediction (II) - CNN-LSTM-Attention wind speed prediction model
Multi-feature variable sequence prediction (III) - CNN-Transformer wind speed prediction model
Multi-feature variable sequence prediction (IV) Transformer-BiLSTM wind speed prediction model
Multi-feature variable sequence prediction (V) CEEMDAN+CNN-LSTM wind speed prediction model
Multi-feature variable sequence prediction (VI) CEEMDAN+CNN-Transformer wind speed prediction model
Prediction model based on the SSA optimization algorithm for CEEMDAN-BiLSTM-Attention:
Prediction model based on the SSA optimization algorithm for CEEMDAN-Transformer-BiGRU:
Superior predictive model: Secondary decomposition-combined prediction
Multi-feature variable sequence prediction (VII) CEEMDAN+Transformer-BiLSTM prediction model
Multi-feature variable sequence prediction (VIII) Prediction model based on the SSA optimization algorithm for CEEMDAN-SSA-BiLSTM:
Multi-feature variable sequence prediction (11) TCN-GRU prediction model based on Pytorch
VMD + CEEMDAN secondary decomposition, CNN-LSTM prediction model
VMD + CEEMDAN secondary decomposition, CNN-Transformer prediction model
VMD + CEEMDAN secondary decomposition, Transformer-BiGRU prediction model
Prediction model based on the SSA optimization algorithm for - code family kit
 | Superior combined prediction model!
High innovation | CEEMDAN + SSA-TCN-BiLSTM-Attention prediction model
 | Superior combined prediction model!
Lowest price in the network | Family kit continuously updated!
| Based on TCN-SENet + BiGRU-GlobalAttention Parallel Prediction Model
Secondary Decomposition - Innovative Model for Predicting the Entire Family Bundle
 | BiTCN-BiGRU-CrossAttention Fusion of Spatiotemporal Features for High Innovation Prediction Models
Combination Prediction Update | Superior Combination Prediction Model!
Spatiotemporal Feature Fusion BiTCN-Transformer Parallel Prediction Model
 | Based on Multi-Level Attention Mechanism for Parallel Prediction Models
 | CEEMDAN-CNN-GRU-GlobalAttention + XGBoost Combination Prediction
Multi-Step Prediction Series | Collection of LSTM, CNN, Transformer, TCN, Serial, and Parallel Models
END
\t2. Friends who like this can click to follow, give the article a "like", share it in WeChat Moments or discuss groups.

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
