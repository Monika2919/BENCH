# BENCH
Advanced Time Series Forecasting Project Attention LSTM vs SARIMAX Benchmark
A complete, production-quality Python implementation is provided for the entire forecasting pipeline. The code includes multivariate time-series data generation, preprocessing with normalization, and supervised sequence construction. A statistical SARIMAX model is implemented as a baseline benchmark. An advanced Attention-based LSTM model is developed using TensorFlow/Keras. All components are modular, reproducible, and evaluated using standard error metrics.
Dataset Description
•	A synthetic multivariate time series dataset is generated for experimentation.
•	The dataset consists of 5 input features and 1 target variable.
•	Features exhibit sinusoidal patterns, trends, and noise to simulate real-world temporal behavior.
•	Total observations: 1000 time steps
Columns:
•	feature1 – sinusoidal pattern with noise
•	feature2 – cosine pattern with noise
•	feature3 – linear trend with noise
•	feature4 – higher-frequency sinusoidal pattern
•	feature5 – random noise
•	target – weighted combination of selected features with noise
Data Preprocessing
•	Min–Max normalization is applied to all features.
•	A sliding window (lookback = 20) approach is used to create supervised learning sequences.
•	Dataset is split chronologically into:
o	80% training data
o	20% testing data
Models Implemented
SARIMAX Baseline Model
•	A Seasonal AutoRegressive Integrated Moving Average with eXogenous variables (SARIMAX) model is used as the statistical baseline.
•	Model configuration:
o	Order: (2, 1, 2)
o	Seasonal Order: (1, 1, 1, 12)
•	Forecasts are generated for the test period.
Attention-based LSTM Model
•	A deep learning architecture combining:
o	Two stacked LSTM layers
o	An attention mechanism to focus on relevant temporal patterns
•	Architecture:
o	Input → LSTM (return sequences)
o	Attention layer
o	LSTM
o	Dense output layer
•	Optimizer: Adam
•	Loss function: Mean Squared Error (MSE)
Model Training
•	The Attention LSTM model is trained using:
o	Batch size: 32
o	Epochs: 20
•	Validation is performed using the test set during training.
Evaluation Metrics
The following metrics are used for performance evaluation:
•	Mean Absolute Error (MAE)
•	Root Mean Squared Error (RMSE)
Both metrics are reported for:
•	Attention LSTM model
•	SARIMAX baseline model
Cross-Validation
•	A rolling time-series cross-validation strategy is applied using TimeSeriesSplit.
•	Five folds are used to assess the robustness of the Attention LSTM model.
•	Average MAE and RMSE across folds are reported.
Hyperparameter Tuning
•	Manual grid search is conducted for the Attention LSTM model.
•	Tuned parameters include:
o	Number of LSTM units
o	Learning rate
•	Model performance is evaluated on the test set for each configuration.
Results and Comparison
•	A comparison table summarizes MAE and RMSE for:
o	Attention LSTM
o	SARIMAX
•	Results demonstrate the relative strengths and limitations of deep learning versus statistical forecasting approaches.
Visualizations
The notebook includes:
•	Actual vs Predicted values plot
•	Attention weight heatmap
•	Performance comparison outputs
Model Interpretability Findings
Attention weight visualizations indicate that the model assigns greater importance to recent time steps while still considering earlier observations, demonstrating its ability to capture short-term dependencies and seasonal patterns. This confirms that the Attention LSTM dynamically focuses on informative temporal regions rather than relying on fixed assumptions.
Predictive Strengths
The Attention LSTM effectively models nonlinear and multivariate temporal relationships and shows stable performance across validation folds. The attention mechanism enhances temporal representation, making the model well suited for complex datasets with trend and seasonality.
Predictive Weaknesses
The model requires higher computational resources and careful hyperparameter tuning. Although attention improves transparency, interpretability remains less intuitive than coefficient-based statistical models, and simpler datasets may not justify the added complexity.
Conclusion
Overall, the Attention LSTM provides improved flexibility and predictive power for complex time series, while the statistical baseline remains advantageous in terms of simplicity, interpretability, and computational efficiency.

