from re import T
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import librosa
import tensorflow as tf
import matplotlib.pyplot as plot
from scipy.io import wavfile
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('finalModel.h5')

app = Flask(__name__)

def freq_thresh_index(fs,n_fft,thresh=4000):
  x=librosa.fft_frequencies(fs, n_fft=512)
  print(x.shape)
  for i in range(len(x)):
    if(x[i]>thresh):
      N=i-1
      break
  return N

n=freq_thresh_index(fs=16000,n_fft=512,thresh=4000)

## Short Time Fourier Transform function
def stft(y,sr):
  S = abs(librosa.stft(y,n_fft=512,hop_length=256))
  R1= np.array(S)
  r = R1[0:n+1, :]
  return r


def prediction(test_filepath,offset=0,duration=1):
  signal, sr = librosa.load(test_filepath,sr=16000,duration=duration,offset=offset) ## offset 
  feature = stft(signal,sr)
  size = feature.shape
  feature = feature.reshape(1,size[0],size[1])
  output = model.predict(feature)
  output = np.argmax(output)
  return output

def plotCounts(test_filepath,output):
  signalData, samplingFrequency = librosa.load(test_filepath,sr=160000)
  plot.figure(figsize=(15,12))
  plot.subplot(211)
  plot.title('Audio Signal')
  duration = len(signalData)/samplingFrequency
  time = np.arange(0,duration,1/samplingFrequency)
  for i in range(len(signalData)//samplingFrequency):
    if output[i]==0:
      plot.axvspan(i, i+1, color='plum', alpha=0.5, lw=0,label='count=0')
      plot.plot(time[i*samplingFrequency:(i+1)*samplingFrequency],signalData[i*samplingFrequency:(i+1)*samplingFrequency],color='violet')
    elif output[i]==1:
      plot.axvspan(i, i+1, color='turquoise', alpha=0.5, lw=0,label='count=1')
      plot.plot(time[i*samplingFrequency:(i+1)*samplingFrequency],signalData[i*samplingFrequency:(i+1)*samplingFrequency],color='darkturquoise')
    elif output[i]==2:
      plot.axvspan(i, i+1, color='palegreen', alpha=0.5, lw=0,label='count=2')
      plot.plot(time[i*samplingFrequency:(i+1)*samplingFrequency],signalData[i*samplingFrequency:(i+1)*samplingFrequency],color='green')
    else:
      plot.axvspan(i, i+1, color='indianred', alpha=0.5, lw=0,label='count=>2')
      plot.plot(time[i*samplingFrequency:(i+1)*samplingFrequency],signalData[i*samplingFrequency:(i+1)*samplingFrequency],color='red')
        
    Count_0 = mpatches.Patch(color='plum', label='Count=0')
    Count_1 = mpatches.Patch(color='turquoise', label='Count=1')
    Count_2 = mpatches.Patch(color='palegreen', label='Count=2')
    Count_greater_2 = mpatches.Patch(color='indianred', label='Count=>2')
    plot.legend(handles=[Count_0, Count_1,Count_2,Count_greater_2],loc=1,prop={'size': 16})
    plot.xlabel('Time(s)')
    plot.ylabel('Amplitude')
    plot.savefig('templates/image/plotCounts')
    plot.close()
    
    plot.figure(figsize=(19,13))
    plot.subplot(212)
    plot.specgram(signalData,Fs=samplingFrequency)
    plot.title('Spectrogram')
    plot.colorbar()
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    plot.savefig('templates/image/Spectogram')
    plot.close()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    file=request.files['file']
    output = []
    if file.filename != '':
        file.save(file.filename)
        test_filepath = file.filename
        signal, sr = librosa.load(test_filepath,sr=16000)
        for i in range(len(signal)//sr):
            output.append(prediction(test_filepath,i,1))
        #plotCounts(test_filepath,output)
    return render_template('index.html', prediction_text='Source Counts $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)