import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import locale

def create_mlp(dim, regress=False):
	# define our MLP network
	model = Sequential()
	model.add(Dense(8, input_dim=dim, activation="relu"))
	model.add(Dense(4, activation="relu"))

	# check to see if the regression node should be added
	if regress:
		model.add(Dense(1, activation="linear"))

	# return our model
	return model
def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(4)(x)
	x = Activation("relu")(x)

	# check to see if the regression node should be added
	if regress:
		x = Dense(1, activation="linear")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model

def process_house_attributes(df, train, test):
	continuous = ["ACB Wiring", "Control", "Busbar Termination", "CT & VT Tapping", "Metering", "Component",
				  "MCCB Wiring/Termination"]

	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])

	zipBinarizer = LabelBinarizer().fit(df["Risk"])
	trainCategorical = zipBinarizer.transform(train["Risk"])
	testCategorical = zipBinarizer.transform(test["Risk"])

	# construct our training and testing data points by concatenating
	# the categorical features with the continuous features
	trainX = np.hstack([trainCategorical, trainContinuous])
	testX = np.hstack([testCategorical, testContinuous])

	# return the concatenated training and testing data
	return (trainX, testX)

dataset_path = "C:/Users/mi/PycharmProjects/tensorEnv/linear_regression/dataset/dataset.csv"
path = "C:/Users/mi/PycharmProjects/tensorEnv/linear_regression/dataset/image"
column_names= ['Risk','Component']
raw_dataset = pd.read_csv(dataset_path,names=column_names,na_values= "?",comment='\t',sep=",",skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

component = dataset.pop('Component')

dataset['Metering'] = (component == 'Metering')*1.0
dataset["ACB Wiring"] = (component == "ACB Wiring")*1.0
dataset["Busbar Termination"] = (component == "Busbar Termination")*1.0
dataset["CT & VT Tapping"] = (component == "CT & VT Tapping")*1.0
dataset["Component"] = (component == "Component")*1.0
dataset["MCCB Wiring/Termination"] = (component == "MCCB Wiring/Termination")*1.0
dataset["Control"] = (component == "Control")*1.0

images = []
for img in os.listdir(path):
	img_array = cv2.imread(os.path.join(path, img))
	outputImage = np.zeros((64, 64, 3), dtype="uint8")
	image = cv2.resize(img_array, (64, 64))
	outputImage[0:64, 0:64] = image
	images.append(outputImage)
imgg= np.array(images)


split = train_test_split(dataset, imgg, test_size=0.25)

(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

maxRisk = trainAttrX["Risk"].max()
trainY = trainAttrX["Risk"] / maxRisk
testY = testAttrX["Risk"] / maxRisk

(trainAttrX, testAttrX) = process_house_attributes(dataset,
	trainAttrX, testAttrX)

mlp = create_mlp(trainAttrX.shape[1], regress=False)
cnn = create_cnn(64, 64, 3, regress=False)

combinedInput = concatenate([mlp.output, cnn.output])

x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_squared_error", optimizer=opt)
print('meh2')

model.fit(
	x=[trainAttrX, trainImagesX], y=trainY,
	validation_data=([testAttrX, testImagesX], testY),
	epochs=200, batch_size=8)

