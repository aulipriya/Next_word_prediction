# Next word prediction

This project was started with the aim to clear the concepts of rnn and learning data preprocessing for sequence data. This project follows [this](https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470 ) tutorial for concept and codes.

### Dataset 

From [this](https://www.patentsview.org/query/) link all information on patents related to neural network are downloaded in a csv format and that is used as our data. 

### Model 

This is a simple model with an lstm layer with 64 cells followed by a dense layer and an output layer 


### Output
 It yields 16.24% training accuracy and 17.29 validation accuracy with pretrained embeddings. 