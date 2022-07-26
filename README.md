# Stock_Price_Prediction

I implemented this project based on the ideas presented in this paper: [Stock Price Prediction Using Hidden Markov Models](https://users.cs.duke.edu/~bdhingra/papers/stock_hmm.pdf)


## Methodology
1. Preprocessing the dataset.
2. Feature engineering.
3. Experimenting with HMMGMM model and fine-tuning the hyperparameters.
4. Experimenting with an LSTM model and fine-tuning the hyperparameters.

**I have also implemented a discrete HMM model from scratch, which is available in [HMM_from_scratch.py](https://github.com/taravatp/Stock_Price_Prediction/blob/main/HMM_from_scratch.py)**

## Results

Model  |     MSE loss                  
------ | ----------------
HMMGMM |    239.2     
LSTM   |    273.70      

Suprisingly, The HMMGMM model performed better than the LSTM model. this must be due to the fact that we have a small dataset that satisfies the markov property.
