# DL-for-MI-BCI
Please find here the code accompanying The Promise of Deep Learning for BCIs: Classification of Motor Imagery EEG using Convolutional Neural Network by Navneet Tibrewal, Nikki Leeuwis and Maryam Alimardani (2021). 

https://www.biorxiv.org/content/biorxiv/early/2021/06/18/2021.06.18.448960.full.pdf

The data accompanying thiscode can be downloaded from Leeuwis, N., Paas, A., & Alimardani, M. (2021). Psychological and Cognitive Factors in Motor Imagery Brain Computer Interfaces. 

https://dataverse.nl/dataset.xhtml?persistentId=doi%3A10.34894%2FZ7ZVOD&version=&q=&fileTypeGroupFacet=&fileAccess=Public&fileSortField=date

Guidelines to use the code:

The dataset has EEG recordings from 67 participants with 3 trial each.

Subject No. 1, 2, 3, 4, 5, 6, 20, 35, 39, 40, 53, 62 and 64 are removed because the readings for these subjects were not proper.


Instruction 

For running Python code for the CNN model and CSP+LDA Model

Please follow the steps below to run the code:

1.	Create an empty folder
2.	In the empty folder create folders named ‘dataset’ and ‘graphs’
3.	In the ‘dataset’ folder transfer EEG recordings of all the subjects excluding the 12 subjects mentioned above from the MAIN DATSET
4.	First Run Preprocessing_Data_EEG_MI_Dataset in Python
5.	Then Run codes for CSP, LDA and CNN 
