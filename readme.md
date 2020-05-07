## Offense detection

Dataset: [OffensEval 2020](https://sites.google.com/site/offensevalsharedtask/home) - [zip](https://sites.google.com/site/offensevalsharedtask/olid/OLIDv1.0.zip)
<br/> *(unpack it as it is in ```data/``` folder)*

### Metrics comparison

| Method      | Accuracy | Precision| Recall    |
| --------    | -------- | -------- | ------    |
| Naive Bayes | 79.77% & | 83.89%   |**89.03%** |
| W2V + LSTM  | 73.14%   | 82.69%   |79.35%     | 
| BERT + LSTM |**80.70%**|**85.47%**|88.23%     |

