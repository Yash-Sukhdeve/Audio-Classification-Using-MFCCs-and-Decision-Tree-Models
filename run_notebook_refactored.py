import os
import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fftpack
from scipy.stats import kurtosis,skew,mode
import sklearn.preprocessing,sklearn.decomposition
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold,train_test_split
from sklearn.feature_selection import f_classif
from keras import utils
import keras
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv1D, Conv2D, Flatten,Reshape, BatchNormalization, ZeroPadding2D,MaxPooling1D,AveragePooling1D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras import regularizers,optimizers
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.utils import to_categorical
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

def get_training(original_path):
    df = pd.read_csv(os.path.join(original_path,'train.csv'))
    if not os.path.exists(os.path.join(original_path,'train_extracted')):
        os.makedirs(os.path.join(original_path,'train_extracted'))
    audio_files = np.array(df['new_id'])
    for i in range(len(audio_files)):    
        file_name = str(audio_files[i])
        d,r = librosa.load(os.path.join("Train",file_name + ".wav"),mono=True)
        if (np.isnan(d).any()):
            print(file_name)
        np.save(os.path.join(original_path, 'train_extracted',file_name+'.npy'),d)

def get_testing(original_path):
    df = pd.read_csv(os.path.join(original_path,'test_idx.csv'))
    if not os.path.exists(os.path.join(original_path,'test_extracted')):
        os.makedirs(os.path.join(original_path,'test_extracted'))
    audio_files = np.array(df['new_id'])
    for i in range(len(audio_files)):   
        file_name = str(audio_files[i])
        d,r = librosa.load(os.path.join("Test",file_name + ".wav"),mono=True)
        if (np.isnan(d).any()):
            print(file_name)
        np.save(os.path.join(original_path, 'test_extracted',file_name+'.npy'),d)

def get_mfcc_features(original_path, csv_file, extracted_folder,num_coeffs):
  df = pd.read_csv(os.path.join(original_path, csv_file ))
  audio_extracted = np.array(df['new_id'])
  mfcc_features=list()
  for i in range(len(audio_extracted)):
    audio_file_data = np.load(os.path.join(original_path, extracted_folder, str(audio_extracted[i])+'.npy'))
    audio_file_data = librosa.effects.preemphasis(y=audio_file_data, coef=0.90)
    mfcc_data = librosa.feature.mfcc(y=audio_file_data,sr=8000, n_mfcc = num_coeffs)
    rms_data = librosa.feature.rms(y=audio_file_data, frame_length=128, hop_length=128)
    rms_db = librosa.power_to_db(rms_data**2, ref=np.max)
    mean_mfcc = np.mean(mfcc_data, axis=1)
    addList = np.concatenate((np.array([np.mean(rms_db[0], axis=0)]), mean_mfcc))
    mfcc_features.append(addList) 
    D = np.abs(librosa.stft(audio_file_data))
  return mfcc_features

def svm_classifier(X_train,Y_train,X_test, num_pcas):
  svm_model = svm.SVC(decision_function_shape='ovr')
  svm_model.fit(X_train, Y_train)
  return svm_model.predict(X_test)

def get_all_features(original_path, csv_file, extracted_folder):
  df = pd.read_csv(os.path.join(original_path, csv_file ))
  audio_extracted = np.array(df['new_id'])
  all_features=list()
  for i in range(len(audio_extracted)):
    audio_file_data= np.load(os.path.join(original_path, extracted_folder, str(audio_extracted[i])+'.npy'))
    mfcc_data = librosa.feature.mfcc(y=audio_file_data,sr=22050)
    rmse= librosa.feature.rms(y=audio_file_data)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio_file_data, sr=22050),axis=1)
    spec_cent = librosa.feature.spectral_centroid(y=audio_file_data, sr=22050)
    spec_bw = librosa.feature.spectral_bandwidth(y=audio_file_data, sr=22050)
    rolloff = librosa.feature.spectral_rolloff(y=audio_file_data, sr=22050)
    zcr =librosa.feature.zero_crossing_rate(audio_file_data)
    addList = np.concatenate((np.mean(mfcc_data, axis=1),np.median(mfcc_data,axis=1),np.std(mfcc_data, axis=1),skew(mfcc_data, axis=1),kurtosis(mfcc_data, axis=1),np.atleast_1d(np.mean(rmse)),np.atleast_1d(np.median(rmse)),np.atleast_1d(np.std(rmse)),np.atleast_1d(skew(rmse,axis=1)),np.atleast_1d(kurtosis(rmse,axis=1)),np.atleast_1d(np.mean(chroma_stft)),np.atleast_1d(np.median(chroma_stft)),np.atleast_1d(np.std(chroma_stft)),np.atleast_1d(skew(chroma_stft)),np.atleast_1d(kurtosis(chroma_stft)),np.atleast_1d(np.mean(spec_cent)),np.atleast_1d(np.median(spec_cent)),np.atleast_1d(np.std(spec_cent)),np.atleast_1d(skew(spec_cent,axis=1)),np.atleast_1d(kurtosis(spec_cent,axis=1)),np.atleast_1d(np.mean(spec_bw)),np.atleast_1d(np.median(spec_bw)),np.atleast_1d(np.std(spec_bw)),np.atleast_1d(skew(spec_bw,axis=1)),np.atleast_1d(kurtosis(spec_bw,axis=1)),np.atleast_1d(np.mean(rolloff)),np.atleast_1d(np.median(rolloff)),np.atleast_1d(np.std(rolloff)),np.atleast_1d(skew(rolloff,axis=1)),np.atleast_1d(kurtosis(rolloff,axis=1)),np.atleast_1d(np.mean(zcr)),np.atleast_1d(np.median(zcr)),np.atleast_1d(np.std(zcr)),np.atleast_1d(skew(zcr,axis=1)),np.atleast_1d(kurtosis(zcr,axis=1)),np.amax(mfcc_data, axis=1),np.amin(mfcc_data, axis=1)))
    all_features.append(addList) 
    if (np.isnan(addList).any()):
      print(audio_extracted[i])
  return all_features

def get_pca_features(original_path, train_csv, train_extracted,test_csv, test_extracted,no_of_components):
  train_features=get_all_features(original_path, train_csv, train_extracted)
  test_features =get_all_features(original_path, test_csv, test_extracted)
  sc = StandardScaler(with_mean=False)
  train_features = sc.fit_transform(train_features)
  test_features = sc.transform(test_features)
  pca = sklearn.decomposition.PCA()
  pca.fit(train_features)
  cumsum = np.cumsum(pca.explained_variance_ratio_)
  pca = sklearn.decomposition.PCA(n_components=no_of_components)
  train_features = pca.fit_transform(train_features)
  test_features = pca.transform(test_features)
  return train_features,test_features

def random_forest(X_train,Y_train,X_test, num_trees):
  model = RandomForestClassifier(n_estimators= num_trees)
  model.fit(X_train, Y_train)
  return model.predict(X_test)

def get_labels(original_path,csv_file):
  df = pd.read_csv(os.path.join(original_path, csv_file ))
  labels = np.array(df['genre'])
  speakers = np.array(df['speaker'])
  return labels, speakers

def standardize_features(X_train,X_test):
  sc = StandardScaler(with_mean=True)
  X_train= sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  return X_train,X_test

def plot_confusion_matrix(y_true,y_pred,label_names,classifier, num_ceoffs, num_components):
    confusion_mat = confusion_matrix(y_true, y_pred, labels=label_names)
    fig = plt.figure(figsize=(10,10))
    plt.imshow(confusion_mat, cmap=plt.cm.Blues, interpolation='nearest')
    plt.xlabel("Predicted Outputs", fontsize=10)
    plt.ylabel("True Outputs", fontsize=10)
    plt.title(f"Confusion Matrix of {classifier} classifier",fontsize=12)
    plt.xticks(np.arange(len(label_names)), label_names, rotation='vertical')
    plt.yticks(np.arange(len(label_names)), label_names)
    plt.tick_params(axis='both', labelsize='10')
    plt.tight_layout()
    for (y, x), label in np.ndenumerate(confusion_mat):
        if label != 0:
            plt.text(x,y,label,ha='center',va='center', size='12')
    if not os.path.exists("Images"):
        os.makedirs("Images")
    filename = f'Images/confusion_matrix_{classifier}_coeffs{num_ceoffs}_comps{num_components}.png'
    plt.savefig(filename)
    plt.close()

def cross_validate(X_train, Y_train, clf, classifier, num_ceoffs, num_components):
    if classifier=="Random Forest":
        sc= StandardScaler()
        X_train = sc.fit_transform(X_train)

    label_names=['0','1','2','3','4','5','6','7','8','9']
    k_fold = StratifiedKFold(n_splits=5,shuffle=True)
    accuracies_clf = list()
    predictions_clf = list()
    actual_predictions = list()
    for train_index, test_index in k_fold.split(X_train,Y_train):
        x_train, x_test = X_train[train_index], X_train[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]    
        pred = clf(x_train,y_train,x_test, num_components)
        predictions_clf.append(pred)
        actual_predictions.append(y_test)
    
    predictions_clf = np.concatenate(predictions_clf)
    actual_predictions = np.concatenate(actual_predictions)

    plot_confusion_matrix(actual_predictions, predictions_clf, label_names, classifier, num_ceoffs, num_components)
    
    f =  f1_score(actual_predictions, predictions_clf, average='macro')
    accuracy_clf = accuracy_score(actual_predictions, predictions_clf)
    
    radius_clf_acc = 2.58 * np.sqrt(accuracy_clf*(1-accuracy_clf)/len(predictions_clf))
    accuracy_clf_min =  (accuracy_clf - radius_clf_acc)
    accuracy_clf_max =  (accuracy_clf + radius_clf_acc)
    
    radius_clf_f1 = 2.58 * np.sqrt(f*(1-f)/len(predictions_clf))
    f_clf_min =  (f - radius_clf_f1)
    f_clf_max =  (f + radius_clf_f1)
    
    print(f'\n\n Accuracy of {classifier} on Validation Dataset is {accuracy_clf}')
    print('\n At 99% Confidence Interval:')
    print(f'\n The Accuracy of {classifier} is likely between {accuracy_clf_min} and {accuracy_clf_max}')
    
    print(f'\n\n F1 Score of {classifier} on Validation Dataset is {f}')
    print('\n At 99% Confidence Interval:')
    print(f'\n The F1 Score of {classifier} is likely between {f_clf_min} and {f_clf_max}')
    
    return(accuracy_clf, f)

original_path = os.path.realpath(".")
get_training(original_path)
get_testing(original_path)

# --- MFCC and Random Forest ---
accuracy = []
f1_total = []
trees = 20
for i in range(8,17):
    X_train = get_mfcc_features(original_path,'train.csv','train_extracted', i)
    Y_train, _ = get_labels(original_path,'train.csv')
    X_test = get_mfcc_features(original_path,'test_idx.csv','test_extracted', i)
    X_train,X_test = standardize_features(X_train,X_test)
    
    acc, f1 = cross_validate(np.array(X_train),np.array(Y_train).flatten(), random_forest, "Random Forest", i, trees)
    accuracy.append(acc)
    f1_total.append(f1)
    
    y_test_rf = random_forest(X_train,Y_train,X_test, trees)
    Y_test = pd.read_csv(os.path.join(original_path,'test_idx.csv'))
    Y_test['genre'] = y_test_rf.tolist()
    Y_test = Y_test.rename(columns={"new_id":"id"})
    Y_test.to_csv(os.path.join(original_path,'predict_rf.csv'),index=False)

plt.plot(range(8,17),accuracy)
plt.ylabel("Accuracy")
plt.xlabel("# MFCC Coeff")
if not os.path.exists("Images"):
    os.makedirs("Images")
plt.savefig('Images/Accuracy.png')
plt.show()

plt.plot(range(8,17),f1_total)
plt.ylabel("F1 Score")
plt.xlabel("# MFCC Coeff")
plt.savefig('Images/F1Total.png')
plt.show()

# --- PCA and SVM ---
accuracy_svm = []
f1_svm = []
for i in range(2,14):
    X_train,X_test = get_pca_features(original_path,'train.csv','train_extracted','test_idx.csv','test_extracted',i)
    Y_train, _ = get_labels(original_path,'train.csv')
    
    acc_svm, f1_svm_val = cross_validate(np.array(X_train),np.array(Y_train), svm_classifier, "SVM", i, i)
    accuracy_svm.append(acc_svm)
    f1_svm.append(f1_svm_val)
    
    y_test_svm = svm_classifier(X_train,Y_train,X_test, i)
    Y_test = pd.read_csv(os.path.join(original_path,'test_idx.csv'))
    Y_test['genre'] = y_test_svm.tolist()
    Y_test = Y_test.rename(columns={"new_id":"id"})
    Y_test.to_csv(os.path.join(original_path,'predict_svm.csv'),index=False)

plt.plot(range(2,14),accuracy_svm)
plt.ylabel("Accuracy")
plt.xlabel("# PCA Coeff")
plt.savefig('Images/SVMAccuracy.png')
plt.show()

plt.plot(range(2,14),f1_svm)
plt.ylabel("F1 Score")
plt.xlabel("# PCA Coeff")
plt.savefig('Images/SVMF1Total.png')
plt.show()
