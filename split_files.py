import os
import random
import shutil
import pandas as pd



if not os.path.exists("Train"):
    #shutil.rmtree(str(os.path.realpath("Train")))
    os.makedirs("Train")

if not os.path.exists("Test"):
    #shutil.rmtree(str(os.path.realpath(".")) + "\\Test")
    os.makedirs("Test")


files = os.listdir("free-spoken-digit-dataset-master/recordings")

i = 0

ran_index = random.sample(range(50),50)

train_list = []
test_list = []
train_genre = []
test_genre = []
train_speaker = []
test_speaker = []


for filename in files:
    # if (filename[0] == '0' or filename[0] == '1'):
    # if ('george' in filename):
    start_name = filename.find('_')
    speaker = filename[start_name+1:]
    end_name = speaker.find('_')
    speaker = speaker[:end_name]
    print(speaker)


    print(ran_index[i])
    if (ran_index[i] >= 40):
        shutil.copyfile(os.path.join("free-spoken-digit-dataset-master/recordings/", filename),os.path.join("Test/", filename))
        test_list.append(filename[0:filename.find('.')])
        # test_list.append('Test/'+filename)
        test_genre.append(filename[0])
        test_speaker.append(speaker)
    else:
        shutil.copyfile(os.path.join("free-spoken-digit-dataset-master/recordings/", filename),os.path.join("Train/", filename))
        train_list.append(os.path.join(filename[0:filename.find('.')]))
        # train_list.append(os.path.join('Train/'+filename))
        train_genre.append(filename[0])
        train_speaker.append(speaker)

        print(filename)

    i = i + 1
    if i > 49:
        ran_index = random.sample(range(50),50)
        i = 0


#print(train_list)
#print(test_list)
train_df = pd.DataFrame(train_list, columns = ['new_id'])
train_df['genre'] = train_genre
train_df['speaker'] = train_speaker
train_df.to_csv("train.csv")

test_df = pd.DataFrame(test_list, columns = ['new_id'])
test_df['genre'] = test_genre
test_df['speaker'] = test_speaker
test_df.to_csv("test_idx.csv")