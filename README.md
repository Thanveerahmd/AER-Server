# AER-Server 

 Rest API to infer the emotion result of an audio by using audio-emotion-classifier that designed using tensorflow , python and flask. 

# Setting up Guidance 

### Step-1 : Clone GitHub repository

Clone GitHub repository and cd into it.

```
$ git clone https://github.com/Thanveerahmd/AER-Server.git
$ cd AER-Server
```

### Step-2 : Setting up the environment.

in the terminal type following commands in the given order.

```

  $ python3 -m venv venv
  $ venv\Scripts\activate
  $ pip install -r requirements.txt

```

### Step-3 : Setting up the files 

 Create a folder called model_weight_file inside the root directory and add audio_emotion.hdf5 model weight file inside 
that folder.

    model_weight_file/audio_emotion.hdf5

 download this file from this link : https://drive.google.com/file/d/1Ku6l3bjyWJVFgSnbk4nH6fgaNjsLZgLx/view?usp=sharing

### Step-4 :  running the project (using PowerShell)

```
$ flask run
```

now server will up and run in localHost : http://127.0.0.1:5000/