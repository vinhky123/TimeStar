# TimeStar: Time Series Forecasting with Time-based Attention and Refinement

TimeStar is an innovative time series forecasting model that I introduce for my graduation thesis. This project represents a significant advancement in time series prediction, offering both high accuracy and interpretability.

## 🌟 Key Features

- **Time-based Attention**: Specialized attention mechanism designed specifically for temporal patterns
- **STar module**: A MLP - Pooling based layer for capturing variate dependencies

## 🏗️ Architecture

The model architecture's overview is shown in this figure:

![Time Star Model Architecture](img/Model%20Architecture.svg)

The models' encoder is a combination of multihead self attention and STAR module. Which helps capturing more information from dataset while keep the model lightweight:

![Time Star Encoder](img/Model%20encoder.svg)

## 📈 Performance

TimeStar demonstrates exceptional performance in:

- Long-term forecasting accuracy
- Handling non-stationary time series
- Capturing complex temporal patterns
- Computational efficiency

The experiment is leverage with look back length S = 96 for all models, and predict length T = {96, 192, 336, 720} to ensure the fairness.

![Full result](img/Final%20result.svg)

Not only the performance, Time Star is also surpass the SOTA model for its computaional efficiency with 26% less training time and 10% less GPU RAM.

![Time and space complexity](img/Time%20%20performance%20chart.svg)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

Thanks ![Time Series Library][https://github.com/thuml/time-series-library] for the baseline benchmarks and dataset. Your works inspired me a lot.
