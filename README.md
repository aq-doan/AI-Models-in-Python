# RasPi Temperature Sensor Data Analysis and Classification

This project aims to analyze and classify temperature data collected from a RasPi (Raspberry Pi) temperature sensor. The goal is to develop various machine learning models capable of classifying patterns accurately for determining the driver behaviour.

## Project Structure

- **Data:** The dataset used for this project is located at `/content/drive/My Drive/datasets/sensor_raw.csv`. It contains readings from the temperature sensor, including gyroscope (GyroX, GyroY, GyroZ), accelerometer (AccX, AccY, AccZ), and the corresponding temperature class (`Target(Class)`).
- **Models:** The following models are implemented and evaluated:
    - RNN (Recurrent Neural Network)
    - CNN (Convolutional Neural Network)
    - TCN (Temporal Convolutional Network)
- **Libraries:** The project utilizes libraries such as numpy, pandas, scikit-learn, XGBoost, Keras, and TensorFlow for various tasks.

## Results and Evaluation

The project evaluates each model's accuracy and F1 score on the testing set. The training time for each model is also recorded. You can find the performance metrics for each model (LSTM, GRU, RNN, CNN) printed in the output section of the code.


