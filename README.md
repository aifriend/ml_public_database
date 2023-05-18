## Top 23 Best Public Datasets for Practicing Machine Learning

1. **Palmer Penguin Dataset**
1. **Bike Sharing Demand Dataset**
1. **Wine Classification Dataset**
1. **Boston Housing Dataset**
1. **Ionosphere Dataset**
1. **Fashion MNIST Dataset**
1. **Cats vs Dogs Dataset**
1. **Breast Cancer Wisconsin (Diagnostic) Dataset**
1. **Twitter Sentiment Analysis and Sentiment140 Datasets**
1. **BBC News Datasets**
1. **Spam SMS Classifier Dataset**
1. **CelebA Dataset**
1. **YouTube-8M Dataset**
1. **Amazon Reviews Dataset**
1. **Banknote Authentication Dataset**
1. **LabelMe Dataset**
1. **Sonar Dataset**
1. **Pima Indians Diabetics Dataset**
1. **Wheat Seeds Dataset**
1. **Jeopardy! Dataset**
1. **Abalone Dataset**
1. **Fake News Detection Dataset**
1. **ImageNet Dataset**

### 1. Palmer Penguin Dataset

It is created by Dr.Kristen Gorman and the Palmer Station, Antarctica LTER. This dataset is essentially composed of two datasets, each containing the data of 344 penguins. 

Just like in the Iris dataset, there are 3 different species of penguins coming from 3 islands in the Palmer Archipelago. These three classes are Adelie, Chinstrap, and Gentoo. If ‘Gentoo’ sounds familiar that is because Gentoo Linux is named after it! Also, these datasets contain culmen dimensions for each species. The culmen is the upper ridge of a bird’s bill. In the simplified penguin’s data, culmen length and depth are renamed as variables culmen\_length\_mm and culmen\_depth\_mm.

#### 1.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\penguins\_size.csv")

data.head()

Note that we use [Pandas library](https://rubikscode.net/2020/12/07/python-for-data-science-numpy-pandas-cheatsheet/) for data visualization. Also, we are loading a simpler dataset.

#### 1.3 This Public Dataset is Best used for

It is a good dataset to practice solving classification and clustering problems. Here you can try out a wide range of classification algorithms like [Decision Tree, Random Forest, SVM](https://rubikscode.net/ultimate-guide-to-machine-learning-with-python/), or adapt it to clustering problems and practice using unsupervised learning

You can find more information about PalmerPenguins dataset and download it from:

- [**Info**](https://allisonhorst.github.io/palmerpenguins/articles/intro.html)
- [**Github**](https://github.com/allisonhorst/palmerpenguins)
- [**Kaggle**](https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data)

### 2. Bike Sharing Demand Dataset

This dataset is really interesting. It is a bit complicated for beginners, however, that is why it is good for practicing. It contains data of bike rental demand in the Capital Bikeshare program in Washington, D.C. Bike sharing and rental systems are in general good sources of information. This one, in particular, contains information about the duration of travel, departure location, arrival location, and time elapsed is explicitly recorded, but it also contains information about the weather of each particular hour and day.

#### 2.1 Sample of the Dataset

Let’s load the data and see what it looks like. First we do so with hourly part of the dataset:

data = pd.read\_csv(f".\\Datasets\\hour.csv")

data.head()

And here is what the daily data looks like:

data = pd.read\_csv(f".\\Datasets\\day.csv")

data.head()

#### 2.3 This Public Dataset is Best used for

Because of the variety of information that this dataset contains it is good for practicing solving regression problems. You can try using Multiple Linear Regression on it, or using neural networks.

You can find more information about dataset and download it from:

- [**UCI**](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)
- [**Kaggle**](https://www.kaggle.com/c/bike-sharing-demand)

### 3. Wine Classification Dataset

This is one is one of the classics. Expecially if you like vine and or planing to become somalier. This dataset is composed of two datasets. Both are containg chemical measures of wine from the Vinho Verde region of Portugal, one for red wine and the other one for white. Due to privacy constraints, there is no data about grape types, wine brand, wine selling price, however, there is information about wine quality.

#### 3.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\winequality-white.csv")

data.head()

#### 3.3 This Public Dataset is Best used for

It is a multi-class classification problem, but could also be framed as a regression problem. The classes are not balanced (e.g. there are many more normal wines than excellent or poor ones), which is great for practicing classification in an imbalanced dataset. Apart from that, not all features are relevant, so feature engineering and feature selection can be practiced as well.

You can find more information about dataset and download it from:

- [**Info**](https://www.vinhoverde.pt/en/about-vinho-verde)
- [**UCI**](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

### 4. Boston Housing Dataset

I know that I said that I will try not to suggest datasets that everyone suggests, but this one is old and gold. The famous Boston Housing Dataset is used in many tutorials, examples, and [books](https://rubikscode.net/ultimate-guide-to-machine-learning-with-python/), and for a good reason. This dataset is composed of 14 features and contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts. It is a small dataset with only 506 samples. 

#### 4.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\boston\_housing.csv")

data.head()

#### 4.3 This Public Dataset is Best used for

This dataset is great for practicing regression tasks. Be aware that because this is a small dataset, you might get optimistic results.

You can find more information about dataset and download it from:

- [**Info**](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
- [**Kaggle**](https://www.kaggle.com/c/boston-housing)

### 5. Ionosphere Dataset
This one is another old dataset. It actually originates from 1989. However, it is really interesting. This dataset contains data collected by a radar system in Goose Bay, Labrador. This system consists of a phased array of 16 high-frequency antennas designed to detect free electrons in the ionosphere. In general, there are two types of structures in the ionosphere: “Good” and “Bad”. These radars detected these structures and passed the signal. There are 34 independent variables and one dependant, and 351 observations in total.

#### 5.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\ionsphere.csv")

data.head()

#### 5.3 This Public Dataset is Best used for

This is obviously a binary (2-class) classification problem. The interesting thing is that this is an imbalanced dataset, so you can practice that as well. It is also not easy to achieve high accuracy on this dataset and the baseline performance is around 64%, while the top accuracy is around 94%.

You can find more information about the dataset and download it from:

- [**UCI**](https://archive.ics.uci.edu/ml/datasets/Ionosphere)

### 6. Fashion MNIST Dataset
MNIST dataset is a famous dataset for practicing image classification and image recognition. However, it is a bit overused. If you want a simple dataset for practicing image classification you can try out FashionMNIST. It is used for image classification examples in Ultimate Guide to machine learning. 

In essence, this dataset is a variation of the MNIST dataset, and it has the same structure as the MNIST dataset, i.e., it has a training set of 60,000 samples and a testing set of 10,000 clothes images. All images have been size-normalized and centered. The size of the images is also fixed to 28×28, so the preprocessing image data is minimized. It is also available as a part of some frameworks like TensorFlow or PyTorch.

#### 6.2 Sample of the Dataset

Let’s load the data and see what it looks like:

#### 6.3 This Public Dataset is Best used for

Image classification and image generating tasks. You can try it out with simple [Convolutional Neural Networks (CNN)](https://rubikscode.net/ultimate-guide-to-machine-learning-with-python/), or use it to generate images using [Generative Adversarial Networks (GANs)](https://rubikscode.net/deep-learning-for-programmers/).

You can find more information about PalmerPenguins dataset and download it from:

- [**Github**](https://github.com/zalandoresearch/fashion-mnist)
- [**Kaggle**](https://www.kaggle.com/zalando-research/fashionmnist)

### 7. Cats vs Dogs Dataset
It is a dataset with images of cats and dogs, of course, it will be included in this list 🙂 This dataset contains 23,262 images of cats and dogs, and it is used for binary image classification. In the main folder, you will find two folders train1 and test. 

The train1 folder contains training images while the test contains test images (duh!). Notice that image names start with cat or dog. These are essentially our labels, which means that target will be defined using these names.

#### 7.2 Sample of the Dataset

Let’s load the data and see what it looks like:

#### 7.3 This Public Dataset is Best used for

The purpose of this dataset is twofold. First, it can be used for practicing image classification, as well as to object detection. Second, it is an endless source of ‘awwwww’s 🙂

You can find more information about dataset and download it from:

- [**Info**](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- [**Kaggle**](https://www.kaggle.com/c/dogs-vs-cats)

### 8. Breast Cancer Wisconsin (Diagnostic) Dataset

There is a steady increase in using Machine Learning and Deep Learning techniques in healthcare. If you would like to practice and see how it looks like working with such data, this dataset is a good choice. In this dataset, data is extracted by processing digitized images of a fine needle aspirate (FNA) of a breast mass. Each feature in this dataset describes characteristics of the cell nuclei that are found in mentioned digitalized images. 

Dataset is composed of 569 examples which include 357 benign and 212 malignant instances. There are three types of features in this dataset, of which real-valued features are most interesting. They are calculated from digitalized images and contain information about the area, the radius of the cell, texture, etc.

#### 8.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\breast-cancer-wisconsin.csv")

data.head()

#### 8.3 This Public Dataset is Best used for

This is one of those healthcare datasets that are good for practicing classification and playing around with algorithms such as [Random Forest, SVM](https://rubikscode.net/ultimate-guide-to-machine-learning-with-python/), etc.

You can find more information about PalmerPenguins dataset and download it from:

- [**Kaggle**](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- [**UCI**](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+\(Diagnostic\))

### 9. Twitter Sentiment Analysis and Sentiment140 Datasets

In the past couple of years, sentiment analysis became one of the essential tools to monitor and understand customer feedback. This way detection of underlying emotional tone that messages and responses carry is fully automated, which means that businesses can better and faster understand what the customer needs and provide better products and services.

This is done by applying various NLP (Natural Language Processing) techniques. These datasets can help you practice such techniques and it is in fact perfect for beginners in this area. Sentiment140 contains 1,600,000 tweets extracted using the Twitter API. Their structures differ a little.

#### 9.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\training.1600000.processed.noemoticon.csv")

data.head()

#### 9.3 This Public Dataset is Best used for

As already mentioned, this is a dataset for sentiment analysis. Sentiment Analysis is the most common text classification tool. It’s the process of analyzing pieces of text to determine the sentiment, whether they’re positive, negative, or neutral. Understand the social sentiment of brand and product is one of the essential tools of the modern business.

You can find more information about dataset and download it from:

- [**Kaggle**](https://www.kaggle.com/c/twitter-sentiment-analysis2)
- [**Kaggle**](https://www.kaggle.com/kazanova/sentiment140)

### 10. BBC News Datasets
Let’s stay in a similar category and explore another interesting textual dataset. This dataset comes from BBC news. It is comprised of 2225 articles and every article is labeled. There are 5 categories: tech, business, politics, entertainment and sport. The dataset is not disbalanced and there is a similar number of articles in each category.

#### 10.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\BBC News Train.csv")

data.head()

#### 10.3 This Public Dataset is Best used for

Naturally this dataset is best used for text classification. You can go one step further and analyze the sentiment of each article too. Overall it is good for various NLP tasks and practieces.

You can find more information about dataset and download it from:

- [**Kaggle**](https://www.kaggle.com/c/learn-ai-bbc)

### 11. Spam SMS Classifier Dataset

Spam detection was one of the first Machine Learning tasks that was used in the Internet. This task falls under NLP and text classification jobs, as well. So, if you want to practice solving this kind of problem, Spam SMS Dataset is a good choice. It is heavily used in literature and it is great for beginners. 

The really cool thing about this dataset is that is was built from multiple sources from the internet. For example, 425 SMS spam messages were scrapped from the Grumbletext Web site, 3,375 SMS randomly chosen ham messages of the NUS SMS Corpus (NSC) from the National University of Singapore, 450 SMS ham messages is taken from Caroline Tag’s Ph.D. Thesis, etc. The dataset itself is composed by two columns: the label (ham or spam) and the raw text.

#### 11.2 Sample of the Dataset

Let’s load the data and see what it looks like:

ham What you doing?how are you?

ham Ok lar... Joking wif u oni...

ham dun say so early hor... U c already then say...

ham MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H\*

ham Siva is in hostel aha:-.

ham Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he started guessing who i was wif n he finally guessed darren lor.

spam FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop

spam Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B

spam URGENT! Your Mobile No 07808726822 was awarded a L2,000 Bonus Caller Prize on 02/09/03! This is our 2nd attempt to contact YOU! Call 0871-872-9758 BOX95QU

#### 11.3 This Public Dataset is Best used for

As the name suggests, this dataset is best used for spam detection and text classification. It is often used in job interviews as well, so it is good to practice o

You can find more information about dataset and download it from:

- [**UCI**](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [**Kaggle**](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

### 12. CelebA Dataset
If you want to work on a face detection solution, build your own face generator, or create your deep fake model, this dataset is the way to go. With more than 200K celebrity images and 40 attribute annotations for each image, this dataset provides a good starting point for your research project. Also, it covers large pose and background variations

#### 12.2 Sample of the Dataset

Let’s load the data and see what it looks like:

#### 12.3 This Public Dataset is Best used for

There are multiple problems that we can solve with this dataset. For starters we can work on various face recognition and computer vision problems. It can be used for generating images with different generative algorithms. Finally, you can use it to develop your novel [deep fake model](https://rubikscode.net/2021/05/31/create-deepfakes-in-5-minutes-with-first-order-model-method/) or a model for deep-fake detection.

You can find more information about dataset and download it from:

- [**Info**](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

### 13. YouTube-8M Dataset
This is the largest multi-label video classification dataset. It coming from Google with 8M classified YouTube Videos with its’ annotations and IDs. Annotations are created by YouTube video annotation system using the vocabulary of 48000 visual entities. This vocabulary is also available for download. 

Note that this dataset is available as TensorFlow Record files. Apart from this, you can check the extension of this dataset – The YouTube-8M Segments dataset. It contains human-verified segment annotations. 

` `Download them using commands:

mkdir -p ~/yt8m/2/frame/train

cd ~/yt8m/2/frame/train

curl data.yt8m.org/download.py | partition=2/frame/train mirror=us python

#### 13.2 This Public Dataset is Best used for

There are multiple things that you can do with this dataset. You can follow Googles competitions with this dataset and develop classification algorithms which accurately assign video-level labels. 

Another thing you can do is to create video classification model uned budget. Finally, you can find and share specific video moments known as temporal concept localization.

You can find more information about dataset and download it from:

- [**Info**](https://arxiv.org/abs/1609.08675)
- [**Download**](http://research.google.com/youtube8m/)

### 14. Amazon Reviews Dataset

Sentiment Analysis is, in a nutshell, the most common text classification tool. It’s the process of analyzing pieces of text to determine the sentiment, whether they’re positive, negative, or neutral. Understand the social sentiment of your brand, product, or service while monitoring online conversations is one of the essential tools of the modern business and sentiment analysis is the first step towards that. This dataset contains product reviews and metadata from Amazon, including 233.1 million reviews spanning May 1996 – Oct 2018.

#### 14.2 This Public Dataset is Best used for

This is the dataset for creating a starter model for sentiment analysis for any product. You can use it to quickly have a model which can be used in production.

You can find more information about dataset and download it from:

- [**Info and Download**](https://jmcauley.ucsd.edu/data/amazon/)

### 15. Banknote Authentication Dataset

This is a fun dataset. You can use it for creating the solution that can detect genuine and forged banknotes. This dataset contains a number of measures taken from digitalized images. Images are created using an industrial camera that is usually used for print inspection. Images are 400x 400 pixels. It is a clean dataset with 1372 examples and no missing values.

#### 15.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\data\_banknote\_authentication.csv")

data.head()

#### 15.3 This Public Dataset is Best used for

It is a great dataset to practice binary classification and apply various algorithms. Also, you could modify it and use it for clustering and come up with the algorithm that will cluster this data with unsupervised learning.

You can find more information about dataset and download it from:

- [**UCI**](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
- [**Kaggle**](https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data)

### 16. LabelMe Dataset
LabelMe is another computer vision dataset. LabelMe is a large database of images with ground truth labels. It is used for object detection and recognition. The annotations come from two different sources, including the LabelMe online annotation tool. 

In a nutshell, there are two ways to utilize this dataset. You can either downloading all the images via the LabelMe Matlab toolbox, either use the images online via the LabelMe Matlab toolbox. 

#### 16.2 Sample of the Dataset

Labeled data looks like this:

#### 16.3 This Public Dataset is Best used for

It is a great dataset for working on object detection and object recognition solutions.

You can find more information about dataset and download it from:

- [**Info and Download**](http://labelme.csail.mit.edu/Release3.0/index.php)

### 17. Sonar Datasets

If you are into geology, you will find this dataset quite interesting. It is made by using a sonar signal and it is composed of two parts. The first part, named “sonar.mines” contains 111 patterns made by bouncing sonar signals off a metal cylinder at various angles and under various conditions. 

The second part, named “sonar.rocks” is composed of 97 patterns, again obtained by bouncing sonar signals, but this is done on the rocks. It is an imbalanced dataset with 208 examples, 60 input features, and one output feature.

#### 17.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\sonar.csv")

data.head()

#### 17.3 This Public Dataset is Best used for

This dataset is great for practicing binary classification. The goal is to detect whether the input is a mine or a rock. It is an interesting problem since the top results achieved an accuracy of 88%.

You can find more information about dataset and download it from:

- [**Info**](https://www.is.umk.pl/projects/datasets.html#Sonar)
- [**UCI**](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+\(Sonar,+Mines+vs.+Rocks\))

### 18. Pima Indians Diabetic Dataset

This is another healthcare dataset for practicing classification. It originates from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict whether or not a patient has diabetes, based on certain diagnostic measurements.

This dataset contains 768 observations, with 8 input features and 1 output feature. It is not a balanced dataset and it is assumed that missing values are replaced with 0.

#### 18.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\pima-indians-dataset.csv")

data.head()

#### 18.3 This Public Dataset is Best used for

It is another dataset suitable for practicing binary classification.

You can find more information about dataset and download it from:

- [**Info**](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)
- [**Kaggle**](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

### 19. Wheat Seeds Dataset

This dataset is very interesting and simple. It is very good for beginners and it can be used instead of Iris Dataset. This dataset contains information about seeds belonging to three different varieties of wheat: Kama, Rosa and Canadian. It is a balanced dataset and each class has 70 instances. Measurements of the internal kernel structure was detected using a soft X-ray technique.

#### 19.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\seeds\_dataset.csv")

data.head()

#### 19.3 This Public Dataset is Best used for

Good for sharpening classification skills.

You can find more information about dataset and download it from:

- [**UCI**](https://archive.ics.uci.edu/ml/datasets/seeds)
- [**Kaggle**](https://www.kaggle.com/jmcaro/wheat-seedsuci)

### 20. Jeopardy! Questions Dataset

This is one beautiful dataset that contains 216,930 Jeopardy questions, answers, and other data. It is a brilliant dataset for your NLP project. Apart from questions and answers this dataset also contains information about the category and value of the question.

#### 20.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\joepardy.csv")

data.head()

#### 20.3 This Public Dataset is Best used for

This is a rich dataset that can be used for multiple purposes. You can run classification algorithms and predict the category of the question, or the value of the question. However, probably the coolest thing you can do with it is to train the [**BERT** ](https://rubikscode.net/2021/04/19/machine-learning-with-ml-net-nlp-with-bert/)model with it.

You can find more information about dataset and download it from:

- [**Kaggle**](https://www.kaggle.com/tunguz/200000-jeopardy-questions)

### 21. Abalone Dataset
In its essence, this is a multi-classification problem, however, this dataset can be framed as a regression problem too. The goal is to predict the age of abalone using provided measures. The dataset is not balanced and 4,177 instances have 8 input variables and 1 output variable.

#### 21.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\abalone.csv")

data.head()

#### 21.3 This Public Dataset is Best used for

This dataset can be framed as both, a regression and a classification task. It is a good chance to use algorithms like multiple linear regression, SVM, Random Forest, or building a neural network that can solve this problem.

You can find more information about dataset and download it from:

- [**UCI**](https://archive.ics.uci.edu/ml/datasets/abalone)
- [**Kaggle**](https://www.kaggle.com/rodolfomendes/abalone-dataset)

### 22. Fake News Dataset

We live in a wird era. Fake news, deep fakes, and other types of deception are part of our everyday lives, whether we like that or not. This dataset provides another NLP task that is really good for practicing. It contains labeled real and fake news, with their text and author.

#### 22.2 Sample of the Dataset

Let’s load the data and see what it looks like:

data = pd.read\_csv(f".\\Datasets\\fake\_news\\train.csv")

data.head()

#### 22.3 This Public Dataset is Best used for

It is another NLP text-classification task. 

You can find more information about dataset and download it from:

- [**Kaggle**](https://www.kaggle.com/c/fake-news/overview)

### 23. ImageNet Dataset
Last but not the least, the king of all computer vision datasets – ImageNet. This dataset is a benchmark for any new deep learning and computer vision brake through. Without it world of **deep learning** would’t be shaped in a way it is shaped today. **ImageNet** is an large image database organized according to the [WordNet](http://wordnet.princeton.edu/) hierarchy. This means that each entity is described with a set of words and phrases called – synset. For each synset around 1000 images is assigned. Basically, each node of the hierarchy is described by hundreds and thousands of images.


