Synopsis
=====================================

This repo is a fork of: https://github.com/deepsound-project/genre-recognition

Two models are provided for training and validation:

* The first model is a modification on the LSTM model developed by Piotr Kozakowski, Jakub Królak, Łukasz Margas and Bartosz Michalak. The model used the weighted average of probability distribution of genres at each time step. 

* The second model was implemented by Nikhil George Titus based on http://benanne.github.io/2014/08/05/spotify-cnns.html This model acheived an accuracy of 83% in the current data split of GTZAN data set. 



Usage
-----

In a fresh virtualenv type:  

```shell
pip install -r requirements.txt
```

You can train your own model by modifying and running train\_model.py. If you wish to train a model by yourself, download the [GTZAN dataset](http://opihi.cs.uvic.ca/sound/genres.tar.gz) (or provide analogous) to the data/ directory, extract it, run create\_data\_pickle.py to preprocess the data and then run train\_model.py to train the model:

```shell
cd data
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar zxvf genres.tar.gz
cd ..
python create_data_pickle.py
 python train_model.py
```

By default the model 2 is chosen. The model can be chosen using an optional model_choice parameter. eg: 

```shell
python train_model.py -c 1

```


Acknowledgments
----------

* This repo is a fork of: https://github.com/deepsound-project/genre-recognition. Many thanks to Piotr Kozakowski & Bartosz Michalak. 
* Models were trained and tested as part of the neural networks project of COMPSCI 682 at UMASS Amherst. https://compsci682.github.io/. Please read the project report here: [REPORT](https://github.com/nikhiltitus/genre-recognition/blob/master/report/finalreport.pdf)
* The second model was trained based on http://benanne.github.io/2014/08/05/spotify-cnns.html
