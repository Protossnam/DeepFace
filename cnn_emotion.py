# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:59:16 2020

@author: pbourdon
"""

import os, glob, re, sys, argparse, itertools
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import regularizers
import sklearn, sklearn.preprocessing, sklearn.model_selection, sklearn.metrics
import keras, keras.backend, keras.utils.np_utils, keras.preprocessing.image

class KerasFaceAnalyzerBase():
    _g_ck_emotion_dict = {None:'???', 0:'Neutral', 1:'Anger', 2:'???', 3:'Disgust', 4:'Fear', 5:'Happiness', 6:'Sadness', 7:'Surprise'}
    _g_ck_gender_dict = {0:'Male', 1:'Female'}
    _g_ck_genders = {5:1, 10:1, 11:1, 14:1, 22:1, 26:1, 28:1, 29:1, 32:1, 34:1, 35:1, 37:0, 42:1, 44:0, 45:0,
                     46:1, 50:0, 51:1, 52:1, 53:0, 54:0, 55:1, 56:0, 57:1, 58:0, 59:1, 60:1, 61:1, 62:0,
                     63:0, 64:0, 65:1, 66:1, 67:1, 68:1, 69:1, 70:1, 71:1, 72:1, 73:0, 74:1, 75:0, 76:1,
                     77:1, 78:1, 79:1, 80:1, 81:0, 82:0, 83:1, 84:1, 85:1, 86:1, 87:1, 88:0, 89:0, 90:1,
                     91:1, 92:0, 93:0, 94:1, 95:1, 96:0, 97:0, 98:1, 99:0, 100:1, 101:0, 102:1, 103:1,
                     104:1, 105:0, 106:1, 107:0, 108:1, 109:1, 110:1, 111:1, 112:1, 113:0, 114:1, 115:1,
                     116:0, 117:1, 118:0, 119:0, 120:0, 121:1, 122:1, 124:1, 125:1, 126:0, 127:1, 128:1,
                     129:1, 130:1, 131:0, 132:0, 133:1, 134:0, 135:0, 136:1, 137:1, 138:1, 139:1, 147:1,
                     148:1, 149:1, 151:1, 154:1, 155:1, 156:1, 157:1, 158:1, 160:0, 501:1, 502:1, 503:1,
                     504:0, 505:0, 506:0, 895:1, 999:1}

    _g_emotion_labels = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise']
    _g_gender_labels = ['Male','Female']
    _g_class_targets = ['emotion', 'gender', 'identity']

    _g_fname_ext_model_weights = '.weights.h5'
    _g_fname_ext_model_history = '.history.csv'
    _g_fname_ext_model_per_epoch = '.epoch{epoch:02d}-loss{val_loss:.2f}.h5'
    _g_fname_ext_model = '.json'

    def __init__(self, target_class):
        self._img_dim = (64,64)
        self._data_generator_preproc = None
        self._X_train_fname = 'X_train.npy'
        self._Y_train_fname = 'Y_train.npy'
        self._X_test_fname = 'X_test.npy'
        self._Y_test_fname = 'Y_test.npy'
        
        assert target_class in self._g_class_targets, 'Wrong target class. Should be one of {}'.format(self._g_class_targets)
        self._target_class = target_class
        if self._target_class=='emotion':
            self._class_labels = self._g_emotion_labels
        elif self._target_class=='gender':
            self._class_labels = self._g_gender_labels
        else:
            raise NotImplementedError

    @classmethod
    def _load_ck_sample(cls, path):
        basename = os.path.basename(path)
        r = re.compile('S(\d+)_(\d+)_(\d+)_landmarks.txt')
        res = r.findall(basename)
        assert len(res)==1, 'File does not match Cohn-Kanade pattern'
        subject_id, session_id, _ = res[0]
        subject_id = int(subject_id)
        session_id = int(session_id)
        name = basename.replace('_landmarks.txt', '')
    
        assert os.path.exists(path), 'Landmarks file {} does not exist'.format(path)
        img_path = path.replace('_landmarks.txt', '.png')
        assert os.path.exists(img_path), 'Image file {} does not exist'.format(img_path)
        img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE).astype(np.float)/255.0

        emotion_path = path.replace('_landmarks.txt', '_emotion.txt')
        assert os.path.exists(emotion_path), 'Emotion label file {} does not exist'.format(emotion_path)
    
        with open(path, 'r') as f:
            landmarks = np.array([[float(x) for x in line.split()] for line in f])
        assert landmarks.shape[0]!=0, 'No landmark found in file {}'.format(path)
    
        with open(emotion_path, 'r') as f:
            emotion_label = cls._g_ck_emotion_dict.get(float(f.readline()))

        assert cls._g_ck_genders.get(subject_id) is not None, "Error: unknown gender for subject {}".format(subject_id)
        gender_label = cls._g_ck_gender_dict.get(cls._g_ck_genders.get(subject_id))

        return subject_id, session_id, img, landmarks, emotion_label, gender_label

    def _load_ck_data(self, input_dir):
        search_pattern = os.path.join(input_dir, 'S[0-9][0-9][0-9]_[0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_landmarks.txt')
        files = glob.glob(search_pattern)
        assert len(files)>0, 'No file found in input directory'
        # print('Found {} files'.format(len(files)))

        ck_data_dict = {'subject_id':[], 'img':[], 'class_index':[], 'class_label':[]}
        class_map = {e:idx for idx, e in enumerate(self._class_labels)}
    
        n_landmarks = None
        for path in files:
            subject_id, session_id, img, landmarks, emotion_label, gender_label = self._load_ck_sample(path)
            if emotion_label not in self._g_emotion_labels: continue

            if self._target_class=='emotion':
                class_label = emotion_label
            elif self._target_class=='gender':
                class_label = gender_label
            class_index = class_map.get(class_label)
            
            x_min, x_max = int(np.min(landmarks[:,0])), int(np.max(landmarks[:,0]))
            y_min, y_max = int(np.min(landmarks[:,1])), int(np.max(landmarks[:,1]))
            img_crop = img[y_min:y_max, x_min:x_max]
            img_crop = cv2.resize(img_crop, self._img_dim)

            if n_landmarks is None: n_landmarks = landmarks.shape[0]
            assert landmarks.shape[0]==n_landmarks, 'Mismatch in number of landmarks'

            if keras.backend.image_data_format() == 'channels_last':
                img_crop = img_crop.reshape(img_crop.shape[0], img_crop.shape[1], 1)    # tensorflow
            elif keras.backend.image_data_format() == 'channels_first':
                img_crop = img_crop.reshape(1, img_crop.shape[0], img_crop.shape[1])    # theano
            ck_data_dict['subject_id'].append(subject_id)
            ck_data_dict['img'].append(img_crop)
            ck_data_dict['class_index'].append(class_index)
            ck_data_dict['class_label'].append(class_label)
            
        return ck_data_dict
    
    def prepare_data(self, input_dir, output_dir):
        print('Loading data...')
        ck_data_df = pd.DataFrame.from_dict(self._load_ck_data(input_dir))
        print('Found {} samples'.format(len(ck_data_df)))
        print(ck_data_df['class_label'].value_counts().apply(lambda x: '{:.2f}%'.format(x/len(ck_data_df)*100)))

        print('Normalizing...')
        # create min max normalized column
        X = np.asarray(ck_data_df['img'].values.tolist())
        X = X.reshape(len(X), -1)
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        X_min_max = min_max_scaler.fit_transform(X)
        if keras.backend.image_data_format() == 'channels_last':
            ck_data_df['normalized_img'] = [arr.reshape(self._img_dim[0], self._img_dim[1], 1) for arr in X_min_max]    # tensorflow
        elif keras.backend.image_data_format() == 'channels_first':
            ck_data_df['normalized_img'] = [arr.reshape(1, self._img_dim[0], self._img_dim[1]) for arr in X_min_max]    # theano

        print('Splitting for cross-validation...')
        split_ = lambda X, y: sklearn.model_selection.train_test_split(X, y, test_size=.3, random_state=42)
        if self._target_class=='gender':
            subjects, selection_indexes = np.unique(ck_data_df['subject_id'].values.tolist(), return_index=True)
            y_dummy = np.array(ck_data_df['class_index'].values.tolist())[selection_indexes] # used for stratification
            subjects_train, subjects_test, _, _ = split_(subjects, y_dummy)
            df_train = ck_data_df.loc[ck_data_df['subject_id'].isin(subjects_train)]
            df_test = ck_data_df.loc[ck_data_df['subject_id'].isin(subjects_test)]
            X_train = np.array(df_train['normalized_img'].values.tolist())
            y_train = np.array(df_train['class_index'].values.tolist())
            X_test = np.array(df_test['normalized_img'].values.tolist())
            y_test = np.array(df_test['class_index'].values.tolist())
        else:
            X = np.array(ck_data_df['normalized_img'].values.tolist())
            y = np.array(ck_data_df['class_index'].values.tolist())
            assert len(X)==len(y), 'Groundtruth error'
            X_train, X_test, y_train, y_test = split_(X, y)
        
        print('Training Set: {} samples'.format(len(X_train)))
        print('Test Set: {} samples'.format(len(X_test)))

        print('Saving Keras-compliant data (one-hot encoded)...')
        Y_train = keras.utils.np_utils.to_categorical(y_train)
        Y_test = keras.utils.np_utils.to_categorical(y_test)
        print('X:', X_train.shape, X_test.shape)
        print('Y:', Y_train.shape, Y_test.shape)

        for arr, fn in ((X_train, self._X_train_fname), (Y_train, self._Y_train_fname), (X_test, self._X_test_fname), (Y_test, self._Y_test_fname)):
            path = os.path.join(output_dir, fn)
            print('{}'.format(path))
            np.save(path, arr)

    def _load_training_data(self, output_dir):
        print('Loading training data...')
        X_train = np.load(os.path.join(output_dir, self._X_train_fname))
        Y_train = np.load(os.path.join(output_dir, self._Y_train_fname))
        return X_train, Y_train

    def _load_test_data(self, output_dir):
        print('Loading test data...')
        X_test = np.load(os.path.join(output_dir, self._X_test_fname))
        Y_test = np.load(os.path.join(output_dir, self._Y_test_fname))
        return X_test, Y_test

    def _build_train_model(self):
        raise NotImplementedError('Keras model architecture needs to be implemented') 

    def _compile_train_model(self, model):
        raise NotImplementedError('Keras model compilation needs to be implemented') 
    
    def _build_data_generator(self, output_dir):
        X_train, Y_train = self._load_training_data(output_dir)

        print('Building image data generator...')
        # keras.backend.set_image_data_format('channels_last')
        data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=10,
                                                                      shear_range=0.2, width_shift_range=0.2,
                                                                      height_shift_range=0.2, horizontal_flip=True,
                                                                      preprocessing_function=self._data_generator_preproc)
        data_generator.fit(X_train)
        print(data_generator.flow(X_train, Y_train, batch_size=50000).next()[0].shape)
        
        return data_generator

    def train(self, output_dir, model_fname, epochs=5):
        X_train, Y_train = self._load_training_data(output_dir)
        balance = [np.sum(Y_train[:,i])/Y_train.shape[0] for i in range(len(self._class_labels))]
        for e, b in zip(self._class_labels, balance):
            print('{}: {:.2f}%'.format(e, 100*b))

        data_generator = self._build_data_generator(output_dir)

        print('Building model...')
        model = self._build_train_model()
        self._compile_train_model(model)
        print(model.summary())

        print('Training...')
        model_path = os.path.join(output_dir, model_fname)
        history_path = model_path+self._g_fname_ext_model_history
        model_cb = keras.callbacks.ModelCheckpoint(filepath=model_path+self._g_fname_ext_model_per_epoch)
        history_cb = keras.callbacks.CSVLogger(history_path, separator=",", append=False)
        '''
        weights = [1/b for b in balance]
        weights[:] = [w/np.sum(weights) for w in weights]
        class_weight = {i:w for i,w in enumerate(weights)}
        
        history = model.fit(X_train, Y_train, class_weight=class_weight, verbose=1, batch_size=32, shuffle=True, epochs=epochs, validation_split=.2, callbacks=[model_cb, history_cb])
        '''
        history = model.fit(X_train, Y_train, verbose=1, batch_size=32, shuffle=True, epochs=epochs, validation_split=.2, callbacks=[model_cb, history_cb])

        print('Saving model...')
        with open(model_path+self._g_fname_ext_model, 'w') as json_file:
            json_file.write(model.to_json())
        model.save_weights(model_path+self._g_fname_ext_model_weights)
        model.save(model_path)

    def _plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title if not normalize else title+' (normalized)')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            value = cm[i, j] if not normalize else '{:.2f}%'.format(100*cm[i, j])
            plt.text(j, i, value, horizontalalignment='center', color='red' if cm[i, j] > thresh else 'black')
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
    def display_results(self, output_dir, model_fname):
        model_path = os.path.join(output_dir, model_fname)
        history_path = model_path+self._g_fname_ext_model_history

        print('Loading model...')
        # with open(model_path+self._g_fname_ext_model, 'r') as json_file:
        #     model = keras.models.model_from_json(json_file.read())
        # model.load_weights(model_path+self._g_fname_ext_model_weights)
        model = keras.models.load_model(model_path)
        self._compile_train_model(model)

        print('Loading training history...')
        history = pd.read_csv(history_path, sep=',', engine='python')

        plt.figure()
        plt.plot(history['epoch'], history['accuracy'])
        plt.plot(history['epoch'], history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        plt.figure()
        plt.plot(history['epoch'], history['loss'])
        plt.plot(history['epoch'], history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        X_train, Y_train = self._load_training_data(output_dir)
        print('Evaluating model on train data...')
        score = model.evaluate(X_train, Y_train, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    def test(self, output_dir, model_fname):
        model_path = os.path.join(output_dir, model_fname)
        history_path = model_path+self._g_fname_ext_model_history

        print('Loading model...')
        model = keras.models.load_model(model_path)
        self._compile_train_model(model)

        X_test, Y_test = self._load_test_data(output_dir)
        print('Evaluating model on test data...')
        score = model.evaluate(X_test, Y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

        print('Computing confusion matrix on test data...')
        Y_test_pred = model.predict(X_test)
        Y_test_pred = np.argmax(Y_test_pred, axis=1)
        Y_test = np.argmax(Y_test, axis=1)
        
        Y_test_pred = [self._class_labels[y] for y in Y_test_pred]
        Y_test = [self._class_labels[y] for y in Y_test]
        confusion_matrix = sklearn.metrics.confusion_matrix(Y_test, Y_test_pred, labels=self._class_labels)
        print('Confusion matrix:\n', confusion_matrix)
        # accuracy = sklearn.metrics.accuracy_score(Y_test, Y_test_pred)
        # print('Accuracy: {:.2f}%'.format(100*accuracy))
        self._plot_confusion_matrix(confusion_matrix, self._class_labels, normalize=False)
        self._plot_confusion_matrix(confusion_matrix, self._class_labels, normalize=True)

class MyKerasFaceAnalyzer(KerasFaceAnalyzerBase):
    def _build_train_model(self):
        if keras.backend.image_data_format() == 'channels_last':
            input_shape=(self._img_dim[0], self._img_dim[1], 1)
        elif keras.backend.image_data_format() == 'channels_first':
            input_shape=(1, self._img_dim[0], self._img_dim[1])
        
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape)) 
        model.add(keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')) 
        model.add(keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')) 
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) 
        model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')) 
        model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')) 
        model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')) 
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) 
        model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')) 
        model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')) 
        model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')) 
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) 
        model.add(keras.layers.Flatten()) 
        model.add(keras.layers.Dense(128, activation='relu')) 
        model.add(keras.layers.Dropout(0.5)) 

        # Output layer
        if self._target_class=='gender':
            model.add(keras.layers.Dense(len(self._g_gender_labels), activation="softmax"))
        else:
            model.add(keras.layers.Dense(len(self._g_emotion_labels), activation="softmax"))
        
        return model

    def _compile_train_model(self, model):
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.0001), metrics=['accuracy'])
        
class AlexNetFaceAnalyzer(KerasFaceAnalyzerBase):
     def _build_train_model(self):
        if keras.backend.image_data_format() == 'channels_last':
            input_shape=(self._img_dim[0], self._img_dim[1], 1)
        elif keras.backend.image_data_format() == 'channels_first':
            input_shape=(1, self._img_dim[0], self._img_dim[1])
            
        model = keras.models.Sequential()
        
        # First layer
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', name="conv1", activation="relu", input_shape=input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same'))

        # Second layer
        model.add(keras.layers.Conv2D(filters=96, kernel_size=(5, 5), strides=(1, 1), padding='same', name="conv2", activation="relu"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same'))
        
        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(384, activation='relu'))
        model.add(keras.layers.Dropout(0.6))
        model.add(keras.layers.Dense(192, activation='relu'))
        model.add(keras.layers.Dropout(0.6))
        
        # Output layer
        if self._target_class=='gender':
            model.add(keras.layers.Dense(len(self._g_gender_labels), activation="softmax"))
        else:
            model.add(keras.layers.Dense(len(self._g_emotion_labels), activation="softmax"))

        return model
            
     def _compile_train_model(self, model):
        #model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.0001), metrics=['accuracy'])

class LeNetFaceAnalyzer(KerasFaceAnalyzerBase):
     def _build_train_model(self):
        if keras.backend.image_data_format() == 'channels_last':
            input_shape=(self._img_dim[0], self._img_dim[1], 1)
        elif keras.backend.image_data_format() == 'channels_first':
            input_shape=(1, self._img_dim[0], self._img_dim[1])
            
        model = keras.models.Sequential()

        # C1 layer
        model.add(keras.layers.Conv2D(32, (5,5), strides=(1,1), padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(1e-2)))
        model.add(keras.layers.Activation('relu'))

        # S2 layer
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        # C3 layer
        model.add(keras.layers.Conv2D(64, (5,5), kernel_regularizer=regularizers.l2(1e-2)))
        model.add(keras.layers.Activation('relu'))

        # S4 layer
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        # F5 layer
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(120))
        model.add(keras.layers.Dropout(0.5))

        # F6 layer
        model.add(keras.layers.Dense(84))
        model.add(keras.layers.Dropout(0.5))

        # Output layer
        if self._target_class=='gender':
            model.add(keras.layers.Dense(len(self._g_gender_labels), activation="softmax"))
        else:
            model.add(keras.layers.Dense(len(self._g_emotion_labels), activation="softmax"))
  
        return model
            
     def _compile_train_model(self, model):
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

def main(argv):
    plt.close('all')

    parser = argparse.ArgumentParser(description='CNN facial analysis')
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('-O', '--output_dir', type=str, default=None, help='output directory')
    parser.add_argument('-M', '--model_fname', type=str, default=None, help='model filename')
    parser.add_argument('-E', '--epochs', type=int, default=1, help='number of epochs (default 1)')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir is not None else os.path.curdir
    model_fname = os.path.abspath(args.model_fname) if args.model_fname is not None else 'cnn_face_emotion.model'
    # Set epoch here
    #epochs = args.epochs
    epochs = 100

    #analyzer = KerasFaceAnalyzerBase()
    #analyzer = MyKerasFaceAnalyzer('emotion')
    analyzer = AlexNetFaceAnalyzer('emotion')
    #analyzer = LeNetFaceAnalyzer('emotion')
    
    analyzer.prepare_data(input_dir, output_dir)
    analyzer.train(output_dir, model_fname, epochs=epochs)
    analyzer.display_results(output_dir, model_fname)
    analyzer.test(output_dir, model_fname)

if __name__ == '__main__':
    main(sys.argv)



    
