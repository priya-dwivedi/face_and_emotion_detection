# Facial Detection, Recognition and Emotion Detection



![](https://mirlab.org:444/demo/emotionclassification/images/20180901150822_new.jpg)



## Introduction

Humans have always had the innate ability to recognize and distinguish between faces. The same has been achieved by computers using opencv and deep learning. This blog briefly throws some light on this ability of computers to excel in facial detection, facial recognition and emotion detection by using the results of experimentation and analysis done on these topics. The blog  has been divided into  three parts:

1. ### Facial Detection

2. ### Facial Recognition

3. ### Emotion Detection

We will walk around through these topics one by one briefly.

### Facial Detection

Detecting all the faces from an image. The facial detection is an first and important part in bringing out the results of facial recognition. It can be achieved by using the amazing python library "face_recognition" which performs very well in detecting location of faces from an image. The following image shows there are two faces detected from the given image.




![](https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png)

The below snippet shows how to use the face_recognition library for detecting faces.

```
face_locations = face_recognition.face_locations(image)
top, right, bottom, left = face_locations[0]
face_image = image[top:bottom, left:right]
```

The full code can be taken from the github.

### Facial Recognition

Facial Recognition verifies if  two faces are same. The use of facial recognition is huge in security, bio-metrics, entertainment, personal safety, etc. The python library "face_recognition" offers a very good performance in recognizing if two faces match with each other giving the result as True or False. The steps involved in facial recognition are

- Find face in an image
- Analyze facial feature
- Compare against both the faces
- Returns True if matched or else False.

Let us understand by seeing the images below

###### Image1

![](data:image/jpeg;base64,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)

###### Image2

![](data:image/jpeg;base64,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)

The  face_recognition library compares the above two faces and returns the result as "True"  stating that two images having different pose are recognized as same

The below snippet shows how to use the face_recognition library for recognizing faces.

```
image1 = face_recognition.load_image_file("../test_images/index1.jpg")
image2 = face_recognition.load_image_file("../test_images/index2.jpeg")
encoding_1 = face_recognition.face_encodings(image1)[0]
encoding_2 = face_recognition.face_encodings(image1)[0]
results = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.50)
print results
[True]
```



Similarly,  let us consider an image of a different person(Image3) and compare it with the Image1

###### Image3

![](test_images/rajeev.jpg)

The  same is done for Image1 and Image3 which are the images of two persons  and the result returned after comparison is "False" denoting the two  images are not recognized as same.

###### So, we can see clearly, the two images of same person though in different poses are recognized as same and those of different persons are not recognized as same.



### Emotion Detection

Humans are used to non verbal communication. The emotions expressed increases the clarity of any thoughts and ideas. It becoms quite interesting when a computer can capture this complex feature of humans, ie emotions. The above topic talks about building a model which can detect an emotion from an image. There key points to be followed are:

1. Data gathering and  augmentation

   The dataset taken was **"fer2013"**. It can be downloaded through the link "https://github.com/npinto/fer2013". Image augmentation was performed on this data.

2. Model building

   The model architecture consists of CNN Layer, Max Pooling, Batch Normalization and fully convolutional layer. 

3. Training

   The model was trained  by  using variants of above layers mentioned in model building and by varying hyperparameters. The best model was able to achieve 61.3% of validation accuracy

4. Testing

   The model was tested with sample images. It can be seen below:

   ![](test_images/39.jpg)

   The image shows the emotion of "Surprise"

   Let us see the prediction of the model trained:

   

   ```
   emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
   face_image  = cv2.imread("..test_images/39.jpg")
   face_image = cv2.resize(face_image, (48,48))
   face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
   face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
   model = load_model("./emotion_detector_models/model.hdf5")
   predicted_class = np.argmax(model.predict(face_image))
   label_map = dict((v,k) for k,v in emotion_dict.items()) 
   predicted_label = label_map[predicted_class]
   Surprise
   ```

   

   We see the predicted label by the model is same as actual label. 

### Future Work:

​	More image augmentation and hyperparameter tuning to further increase the accuracy of model. 

​	Facial Recognition of two faces having different profile(left and front profile)

​	Emotion Detection of different profiles(other than front profile)

### Conclusion:

​	We can clearly see the wonders of AI in facial recognition. The amazing python library of 			  face_recognition, pretrained  deep learning models and open-cv  have already gained so much performance and have made our life easier. There are lots of other materials that are helpful and bring into picture different approaches used in achieving the same goal.

You can see the full code on the following github link:

https://github.com/priya-dwivedi/face_and_emotion_detection

