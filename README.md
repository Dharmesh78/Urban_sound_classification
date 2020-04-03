# Urban_sound_classification

PLEASE READ THE BELOW INSTRUCTIONS FOR RUNNING THE PROJECT:

the project is divided in 6 files:
1. UrbanSound8K.csv
2. audioData.csv
3. datamaking.py
4. model_train.py
5. sound_classifier_model
6. gui_applicn.py

**For running the application only the files named 'gui_applicn.py' and 'sound_classifier_model' are needed as latter is the trained model which is already saved and gives accuracy of 87% with CNN model.**

** A video named 'proj_demonstration.mkv' is also given. **

**For self demonstration purpose 6 audio files in .wav format are also present in the repository.**

**NOVELTY OF THE PROJECT LIES IN THE TYPE OF INPUT DATA i.e. AUDIO DATA, WITH METHODOLOGY USED AS MOST OF THE WORK RELATED TO CNN ARE DONE IN GRAPHICAL/IMAGE FIELD.**

*(I have done the data preparation and GUI development part, whereas for feature extraction and model description i have taken some help from the internet.)*

In 'datamaking.py' , feature extraction step is performed. Every audio file from UrbanSound8k audio dataset is retrieved and its corresponding MFCC feature is extracted, in the form of numpy array and then with its corresponding class label (from 'UrbanSound8k.csv' file) is retrieved.
Both the extracted feature and class label are structred in dataframe and stored as audioData.csv file.

In 'model_train.py', a CNN model is trained on the dataset after reshaping them in the required format. The CNN model has 6 layers including 1 input layer, 2 hidden convolutional layer, 1 maxpooling layer, 2 dense layers including output layer of 10 neurons. The model is also given a dropout of 0.5.
The above model after compilation and training is saved with the name 'sound_classifier_model'.

In 'gui_applicn.py', tkinter library of python is used, to develop a simple and user friendly interface for our task of urban sound classification.

The interface has 4 buttons with detail as follows:
a. 'Load': to load the file from the local machine
b. 'Play to Listen':  This enables the user to first listen the sound so that he/she can verify the class label later predicted.
c. 'Predict': to predict the class label of the loaded sound file with the help of 'sound_classifier_model'
d. 'Close': to close the tkinter window normally. 


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*******************************
   DHARMESH SINGH
   M.Tech Data Analytics
   NIT Tiruchirappalli
*******************************
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
