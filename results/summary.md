# Results

FI-2010, no-auction, decimal-precision normalised. Trained on days 1-7 (last 20% held out for early stopping), tested on days 8-10. Macro F1 over up / stationary / down.

Share of 'stationary' labels in the test set by horizon: k=10: 71%, k=20: 62%, k=30: 56%, k=50: 47%, k=100: 34%

|   horizon_k | model             |   accuracy |   f1_macro |   precision |   recall |   latency_p50_us |      params |
|------------:|:------------------|-----------:|-----------:|------------:|---------:|-----------------:|------------:|
|          10 | transformer       |     0.8679 |     0.8093 |      0.8031 |   0.8159 |         724.7535 | 102915.0000 |
|          10 | logistic          |     0.8669 |     0.7848 |      0.8798 |   0.7288 |         nan      |    nan      |
|          10 | xgboost           |     0.8489 |     0.7708 |      0.7945 |   0.7511 |         nan      |    nan      |
|          10 | deeplob           |     0.8200 |     0.6935 |      0.8071 |   0.6506 |        2681.4635 | 143907.0000 |
|          10 | xgboost_book_only |     0.4130 |     0.3577 |      0.3895 |   0.4186 |         nan      |    nan      |
|          50 | transformer       |     0.8823 |     0.8771 |      0.8936 |   0.8652 |         776.3215 | 102915.0000 |
|          50 | xgboost           |     0.8321 |     0.8199 |      0.8349 |   0.8096 |         nan      |    nan      |
|          50 | logistic          |     0.7055 |     0.6968 |      0.7033 |   0.6933 |         nan      |    nan      |
|          50 | deeplob           |     0.6911 |     0.6269 |      0.8245 |   0.6092 |        4094.4595 | 143907.0000 |
|          50 | xgboost_book_only |     0.3850 |     0.3839 |      0.4008 |   0.4061 |         nan      |    nan      |
|         100 | transformer       |     0.9196 |     0.9201 |      0.9225 |   0.9193 |         763.5635 | 102915.0000 |
|         100 | xgboost           |     0.8634 |     0.8629 |      0.8659 |   0.8622 |         nan      |    nan      |
|         100 | deeplob           |     0.7314 |     0.7192 |      0.8284 |   0.7220 |        3278.7285 | 143907.0000 |
|         100 | logistic          |     0.5118 |     0.5137 |      0.5178 |   0.5172 |         nan      |    nan      |
|         100 | xgboost_book_only |     0.4057 |     0.4002 |      0.4080 |   0.4092 |         nan      |    nan      |
