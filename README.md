<h1 align="center">CoreTab</h1>

  <p align="center">
    SOTA coreset creation algorithm for tabular data

<!-- GETTING STARTED -->
## Installation

Clone the repo
   ```sh
   git clone https://github.com/avivhadar33/coretab.git 
   ```

<!-- USAGE EXAMPLES -->
## Usage

This is how you create a coreset using coretab.

```py
from coretab.coreset_algorithms import CoreTabXGB

coretab_xgb = CoreTabXGB()
X_filter, y_filter = coretab_xgb.create_coreset(X_train, y_train)
```
Use the "Training Enhancement" inference method to maximize you algorithm performance.

```py
import xgboost as xgb  # or any model you want

model = xgb.XGBClassifier()
model.fit(X_filter, y_filter)
te_predictions = coretab_xgb.te_predict(X_test, prediction_model=model)
```

<!-- CONTACT -->
## Contact

Aviv Hadar - aviv0040@gamil.com

Project Link: https://github.com/avivhadar33/coretab
