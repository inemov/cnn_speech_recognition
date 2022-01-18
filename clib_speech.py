# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 13:51:38 2020

@author: Ivan Nemov
"""
from threading import Thread
import wave
import struct
from PyQt5 import QtCore
import os
import numpy as np
import random
import time
import csv
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pyaudio
import matplotlib.pyplot as plt
from scipy.signal import stft
from datetime import datetime

# message class
class signal_message(QtCore.QObject):
    progress_update_bit = QtCore.pyqtSignal(int)                       #starts taking silence sample
    volume_update_bit = QtCore.pyqtSignal(int)                         #updates voice volume bar
    silence_threshold_update_bit = QtCore.pyqtSignal(int)              #updates silence threshold based on sound amplitude analysis
    audio_record_is_ready_bit = QtCore.pyqtSignal(str, int)            #updates GUI with filename of newly created audio record
    record_next_audio_bit = QtCore.pyqtSignal(int)                     #update thread with user input to proceed with record
    audio_recording_completed_bit = QtCore.pyqtSignal(bool)            #overall task of audio recording is completed
    wav_to_csv_conversion_completed_bit = QtCore.pyqtSignal(bool)      #updates GUI about completion of the conversion
    wav_to_csv_spectrogram_is_ready_bit = QtCore.pyqtSignal(str)       #updates GUI with filename of newly created spectrogram
    cnn_training_over_bit = QtCore.pyqtSignal(bool)                    #update GUI about completion of CNN training
    cnn_save_action_bit = QtCore.pyqtSignal(int)                       #dialog reply from GUI to save network
    cnn_saved_name_bit = QtCore.pyqtSignal(str,list, str)                    #updates GUI with name of saved CNN
    spectrogram_filter_abnormal_found_bit = QtCore.pyqtSignal(str)     #call dialog in GUI to remove spectrogramm
    spectrogram_filter_action_bit = QtCore.pyqtSignal(int)             #perfrom action of file
    spectrogram_filter_completed_bit = QtCore.pyqtSignal(int)          #update GUI on completion of filtering
    cnn_loaded_bit = QtCore.pyqtSignal(bool)                           #update GUI about CNN being loaded
    cnn_unloaded_bit = QtCore.pyqtSignal(bool)                         #update GUI about CNN being unloaded
    cnn_waiting_for_speech_bit = QtCore.pyqtSignal(bool)               #update GUI about CNN waiting for sound
    cnn_idle_bit = QtCore.pyqtSignal(bool)                             #update GUI about CNN being idle
    cnn_recognized_bit = QtCore.pyqtSignal(str)                        #return recognized word
    cnn_spectrogram_is_ready_bit = QtCore.pyqtSignal(str)              #return ready spectrogram for plotting in GUI
    cnn_update_mic_and_silence_threshold_bit = QtCore.pyqtSignal(int,int) #update CNN with new microphone and silence threshold
    cnn_change_switch_bit = QtCore.pyqtSignal(int)                     #update CNN to start listen for speech
    
class data_augmentation():
    def __init__(self):
        ''' Constructor. '''
    
    def read_from_csv(self, file):
        dataarray = []
        freqseriesarray = []
        timeseriesarray = []
        readcsv=csv.reader(open(file))
        i=1
        for row in readcsv:
            if i==1:
                freqseriesarray.append(row)
            if i==2:
                timeseriesarray.append(row)
            if i>2:
                dataarray.append(row)
            i=i+1
        data=np.array(dataarray,dtype=np.float32)
        freqseries=np.array(freqseriesarray,dtype=np.float16)
        timeseries=np.array(timeseriesarray,dtype=np.float16)
        return [data, freqseries, timeseries] 
    
class indentify_silence_threshold(Thread):
    def __init__(self, 
                 _signal_message, 
                 _mic, 
                 _rate, 
                 _freq, 
                 _chunk, 
                 _format, 
                 _channels, 
                 _sampleduration):
        ''' Constructor. '''
        Thread.__init__(self)
        self._signal_message = _signal_message
        self._mic = _mic
        self._rate = _rate
        self._freq = _freq
        self._chunk = _chunk
        self._format = _format
        self._channels = _channels
        self._sampleduration = _sampleduration
        
    def run(self):
        # create audio stream
        try:
            pa = pyaudio.PyAudio()
            pa_stream = pa.open(format=self._format, 
                                channels=self._channels, 
                                rate=self._rate, input=True, 
                                frames_per_buffer=self._chunk, 
                                input_device_index=self._mic)
        except:
            pa = None
            pa_stream = None
            return None
        
        # identify silence threshold
        silence_amplitude_array = []
        for i in range(1,int(self._sampleduration*self._freq+1)):
            data = np.frombuffer(pa_stream.read(self._chunk), dtype=np.int16)
            amplitude = np.linalg.norm(data, np.inf)
            silence_amplitude_array.append(amplitude)
            self._signal_message.progress_update_bit.emit(int(i*100/int(self._sampleduration*self._freq)))
            self._signal_message.volume_update_bit.emit(amplitude*100/32767)
        self._signal_message.progress_update_bit.emit(int(100))
        self._signal_message.volume_update_bit.emit(0)
        time.sleep(0.1)
        self._signal_message.progress_update_bit.emit(0)
        silence_amplitude_npArray = np.array(silence_amplitude_array, dtype=np.int16)
        silence_amplitude_avg = np.average(silence_amplitude_npArray)
        silence_amplitude_stdv = np.std(silence_amplitude_npArray)
        silince_threshold = silence_amplitude_avg + 3.890592*silence_amplitude_stdv
        if silince_threshold < 32767*0.1:
            silince_threshold = 32767*0.1
        self._signal_message.silence_threshold_update_bit.emit(silince_threshold*100/32767)
        pa_stream.stop_stream()
        pa_stream.close()
        pa.terminate()
        
class record_audio_csv(Thread):
    def __init__(self, 
                 _signal_message, 
                 _mic, 
                 _rate,
                 _chunk, 
                 _commandchunk, 
                 _format, 
                 _channels, 
                 _directory, 
                 _numberoffiles,
                 _silencethreshold,
                 _stftsegmentsize,
                 _overlappercent,
                 _stftfreqres,
                 _freqhighlim,
                 _stft_lin_log_norm):
        ''' Constructor. '''
        Thread.__init__(self)
        self._signal_message = _signal_message
        self._mic = _mic
        self._rate = _rate
        self._chunk = _chunk
        self._commandchunk = _commandchunk
        self._format = _format
        self._channels = _channels
        self._directory = _directory
        self._numberoffiles = _numberoffiles
        self._silencethreshold = _silencethreshold
        self._stftsegmentsize = _stftsegmentsize
        self._overlappercent = _overlappercent
        self._stftfreqres = _stftfreqres
        self._freqhighlim = _freqhighlim
        self._stft_lin_log_norm = _stft_lin_log_norm
        
        self.switch = 0
        
    def run(self):
        # create audio stream
        try:
            pa = pyaudio.PyAudio()
            pa_stream = pa.open(format=self._format, 
                                channels=self._channels, 
                                rate=self._rate, input=True, 
                                frames_per_buffer=self._chunk, 
                                input_device_index=self._mic)
        except:
            pa = None
            pa_stream = None
            return None
        
        # record audio
        self._signal_message.audio_record_is_ready_bit.emit("",0)
        for j in range(1,self._numberoffiles+1):
            while self.switch == 0:
                time.sleep(0.025)
            if self.switch == -1:
                break
            elif self.switch == 1:
                while self.switch == 1:
                    data = np.frombuffer(pa_stream.read(self._chunk), dtype=np.int16)
                    amplitude = np.linalg.norm(data, np.inf)
                    if amplitude > self._silencethreshold*32767/100:
                        voice_sample_array=np.array([])
                        voice_sample_array=np.concatenate((voice_sample_array, data))
                        data = np.frombuffer(pa_stream.read(self._commandchunk), dtype=np.int16)
                        
                        pa_stream.stop_stream()
                        
                        voice_sample_array=np.concatenate((voice_sample_array, data))
                        voice_sample_array=np.ravel(voice_sample_array)
                        freq_series, time_series, data_stft = stft(voice_sample_array, 
                                                                       fs = self._rate, 
                                                                       nperseg = self._stftsegmentsize, 
                                                                       noverlap = int(self._stftsegmentsize*self._overlappercent/100), 
                                                                       nfft = int(self._rate/self._stftfreqres), 
                                                                       padded = False, 
                                                                       boundary = None)
                        data_stft_np = np.array(data_stft, dtype = np.float16)
                        data_stft_abs = np.abs(data_stft_np)
                        if self._stft_lin_log_norm == 0:
                            data_stft_abs_norm = data_stft_abs / np.linalg.norm(data_stft_abs,np.inf)
                        else:
                            base = 10
                            data_stft_abs_norm = (np.log(data_stft_abs+1) / np.log(base)) / np.linalg.norm((np.log(data_stft_abs+1) / np.log(base)),np.inf)
                        freq_series = freq_series[:int(self._freqhighlim/self._stftfreqres+1)]
                        data_stft_abs_norm_limited = data_stft_abs_norm[:int(self._freqhighlim/self._stftfreqres+1),:]
                        zero_array = np.zeros((data_stft_abs_norm_limited.shape[0]-int(8000/self._stftfreqres+1),data_stft_abs_norm_limited.shape[1]), dtype = np.float16)
                        data_stft_abs_norm_limited[int(8000/self._stftfreqres+1):,:] = zero_array
                    
                        date_ = datetime.now().date()
                        time_ = datetime.now().time()
                        date_time = "__" + str(date_.year) + "_" + str(date_.month) + "_" + str(date_.day) + "__" + str(time_.hour) + "_" + str(time_.minute) + "_" + str(time_.second)
                        filename = str(self._directory + "/" + date_time + ".csv")
                        with open(filename, 'w', newline='') as csvfile:
                            data_stft_csv = csv.writer(csvfile, dialect='excel')
                            data_stft_csv.writerow(freq_series)
                            data_stft_csv.writerow(time_series)
                            for i in range(0,data_stft_abs_norm_limited.shape[0]):
                                data_stft_csv.writerow(data_stft_abs_norm_limited[i,:])

                        self._signal_message.progress_update_bit.emit(int(j*100/self._numberoffiles))
                        self.switch = 0
                        self._signal_message.audio_record_is_ready_bit.emit(filename, j)
                        
                        pa_stream.start_stream()
                        
                    else:
                        self._signal_message.volume_update_bit.emit(amplitude*100/32767)
        
        self._signal_message.volume_update_bit.emit(0)              
        self._signal_message.progress_update_bit.emit(int(100))
        time.sleep(0.1)
        self._signal_message.progress_update_bit.emit(int(0))
        
        pa_stream.stop_stream()
        pa_stream.close()
        pa.terminate()
        self._signal_message.audio_recording_completed_bit.emit(True)
    
    def change_switch(self, value):
        self.switch = value

class record_audio_wav(Thread):
    def __init__(self, 
                 _signal_message, 
                 _mic, 
                 _rate,
                 _chunk, 
                 _commandchunk, 
                 _format, 
                 _channels, 
                 _directory, 
                 _numberoffiles,
                 _silencethreshold):
        ''' Constructor. '''
        Thread.__init__(self)
        self._signal_message = _signal_message
        self._mic = _mic
        self._rate = _rate
        self._chunk = _chunk
        self._commandchunk = _commandchunk
        self._format = _format
        self._channels = _channels
        self._directory = _directory
        self._numberoffiles = _numberoffiles
        self._silencethreshold = _silencethreshold
        
        self.switch = 0
        
    def run(self):
        # create audio stream
        try:
            pa = pyaudio.PyAudio()
            pa_stream = pa.open(format=self._format, 
                                channels=self._channels, 
                                rate=self._rate, input=True, 
                                frames_per_buffer=self._chunk, 
                                input_device_index=self._mic)
        except:
            pa = None
            pa_stream = None
            return None
        
        # record audio
        self._signal_message.audio_record_is_ready_bit.emit("",0)
        for j in range(1,self._numberoffiles+1):
            while self.switch == 0:
                time.sleep(0.025)
            if self.switch == -1:
                break
            elif self.switch == 1:
                while self.switch == 1:
                    data = np.frombuffer(pa_stream.read(self._chunk), dtype=np.int16)
                    amplitude = np.linalg.norm(data, np.inf)
                    if amplitude > self._silencethreshold*32767/100:
                        voice_sample_array=np.array([])
                        voice_sample_array=np.concatenate((voice_sample_array, data))
                        data = np.frombuffer(pa_stream.read(self._commandchunk), dtype=np.int16)
                        
                        pa_stream.stop_stream()
                        
                        voice_sample_array=np.concatenate((voice_sample_array, data))
                        voice_sample_array=np.ravel(voice_sample_array)
                        
                        date_ = datetime.now().date()
                        time_ = datetime.now().time()
                        date_time = "__" + str(date_.year) + "_" + str(date_.month) + "_" + str(date_.day) + "__" + str(time_.hour) + "_" + str(time_.minute) + "_" + str(time_.second)
                        filename = str(self._directory + "/" + date_time + ".wav")
                        
                        wavefile = wave.open(filename, 'wb')
                        wavefile.setnchannels(self._channels)
                        wavefile.setsampwidth(pa.get_sample_size(self._format))
                        wavefile.setframerate(self._rate)
                        packed_voice_sample_array = voice_sample_array.astype('h').tostring()
                        wavefile.writeframes(packed_voice_sample_array)
                        wavefile.close()
                        
                        self._signal_message.progress_update_bit.emit(int(j*100/self._numberoffiles))
                        self.switch = 0
                        self._signal_message.audio_record_is_ready_bit.emit(filename, j)
                        
                        pa_stream.start_stream()
                    
                    else:
                        self._signal_message.volume_update_bit.emit(amplitude*100/32767)
        
        self._signal_message.volume_update_bit.emit(0)               
        self._signal_message.progress_update_bit.emit(int(100))
        time.sleep(0.1)
        self._signal_message.progress_update_bit.emit(int(0))
        
        pa_stream.stop_stream()
        pa_stream.close()
        pa.terminate()
        self._signal_message.audio_recording_completed_bit.emit(True)
    
    def change_switch(self, value):
        self.switch = value

class convert_wav_to_csv(Thread):
    def __init__(self, 
                 _signal_message,
                 _source_directory, 
                 _target_directory,
                 _stftsegmentsize_ref,
                 _overlappercent,
                 _stftfreqres,
                 _freqhighlim,
                 _stft_lin_log_norm,
                 _rate_ref,
                 _number_of_time_steps):
        ''' Constructor. '''
        Thread.__init__(self)
        self._signal_message = _signal_message
        self._source_directory = _source_directory
        self._target_directory = _target_directory
        self._stftsegmentsize_ref = _stftsegmentsize_ref
        self._overlappercent = _overlappercent
        self._stftfreqres = _stftfreqres
        self._freqhighlim = _freqhighlim
        self._stft_lin_log_norm = _stft_lin_log_norm
        self._rate_ref = _rate_ref
        self._number_of_time_steps = _number_of_time_steps
        
    def run(self):
        #get array of files in source directory
        file_names_array = []
        wav_file_names_array = []
        file_names_array = os.listdir(self._source_directory)
        if file_names_array != []:
            for file_item in file_names_array:
                if file_item[-3:] == "wav":
                    wav_file_names_array.append(file_item)
        
        wav_file_count = 0
        wav_file_total = len(wav_file_names_array)
        
        for wav_file in wav_file_names_array:
            #read wav file to array
            wav_file_path = self._source_directory + "/" + wav_file
            wav_read = wave.open(wav_file_path, 'rb')
            rate = wav_read.getframerate()
            nframes = wav_read.getnframes()
            wav_data_byte_like = wav_read.readframes(nframes)
            data_conversion_format = "<" + str(nframes) + "h"
            wav_data_array = struct.unpack(data_conversion_format, wav_data_byte_like)
            wav_data_array_np=np.array(wav_data_array, dtype=np.int16)
            wav_data_array_np=np.ravel(wav_data_array_np)
            wav_read.close()
            
            #cut silence in the beginning
            silence_threshold = 0.2 * np.linalg.norm(np.abs(wav_data_array_np),np.inf)
            try:
                voice_start_position = int(np.max([np.argwhere(np.abs(wav_data_array_np)>silence_threshold)[0]-rate*0.075,0]))
            except:
                voice_start_position = int(0)
            wav_data_array_np=wav_data_array_np[voice_start_position:]
            
            #define size of data array and extend with noise if required
            stftsegmentsize = int(self._stftsegmentsize_ref * rate / self._rate_ref)
            n_samples_in_stft_time_step = stftsegmentsize - int(stftsegmentsize * self._overlappercent / 100)
            voice_end_position = int(self._number_of_time_steps * n_samples_in_stft_time_step)
            if wav_data_array_np.shape[0] < voice_end_position:
                increase_size = voice_end_position - wav_data_array_np.shape[0]
                rand_limit = np.max([0.005*silence_threshold,1])
                increase_array = np.random.randint(-rand_limit, rand_limit, size = increase_size)
                wav_data_array_np=np.concatenate((wav_data_array_np, increase_array))
                wav_data_array_np=np.ravel(wav_data_array_np)
            else:
                wav_data_array_np = wav_data_array_np[:voice_end_position]

            #peform stft on data
            freq_series, time_series, data_stft = stft(wav_data_array_np, 
                                                       fs = rate, 
                                                       nperseg = stftsegmentsize, 
                                                       noverlap = int(stftsegmentsize * self._overlappercent / 100), 
                                                       nfft = int(rate/self._stftfreqres), 
                                                       padded = False, 
                                                       boundary = None)
            data_stft_np = np.array(data_stft, dtype = np.float16)
            data_stft_abs = np.abs(data_stft_np)
            if self._stft_lin_log_norm == 0:
                data_stft_abs_norm = data_stft_abs / np.linalg.norm(data_stft_abs,np.inf)
            else:
                base = 10
                data_stft_abs_norm = (np.log(data_stft_abs+1) / np.log(base)) / np.linalg.norm((np.log(data_stft_abs+1) / np.log(base)),np.inf)
        
            freq_series_np = np.array(freq_series, dtype = np.int16)
            if freq_series_np.shape[0] >= int(self._freqhighlim/self._stftfreqres+1):
                freq_series_np_limited = freq_series_np[:int(self._freqhighlim/self._stftfreqres+1)]
                data_stft_abs_norm_limited = data_stft_abs_norm[:int(self._freqhighlim/self._stftfreqres+1),:]
            else:
                freq_series_np_limited = np.linspace(0, self._freqhighlim, num=int(self._freqhighlim/self._stftfreqres+1), dtype = np.int16)
                data_stft_abs_norm_limited = np.zeros((int(self._freqhighlim/self._stftfreqres+1),data_stft_abs_norm.shape[1]), dtype = np.float16)
                data_stft_abs_norm_limited[:data_stft_abs_norm.shape[0],:data_stft_abs_norm.shape[1]] = data_stft_abs_norm
            
            time_series_np = np.array(time_series, dtype = np.float16)

            #save file with original name in the target directory
            filename = str(self._target_directory + "/" + wav_file[:len(wav_file)-4] + ".csv")
            with open(filename, 'w', newline='') as csvfile:
                data_stft_csv = csv.writer(csvfile, dialect='excel')
                data_stft_csv.writerow(freq_series_np_limited)
                data_stft_csv.writerow(time_series_np)
                for i in range(0,data_stft_abs_norm_limited.shape[0]):
                    data_stft_csv.writerow(data_stft_abs_norm_limited[i,:])
            
            wav_file_count = wav_file_count + 1
            self._signal_message.progress_update_bit.emit(int(wav_file_count*100/wav_file_total))
            #every 10 files - show spectrogram
            if wav_file_count == int(wav_file_count/10)*10:
                self._signal_message.wav_to_csv_spectrogram_is_ready_bit.emit(filename)
                
        self._signal_message.progress_update_bit.emit(int(100))
        time.sleep(0.1)
        self._signal_message.wav_to_csv_conversion_completed_bit.emit(True)
        self._signal_message.progress_update_bit.emit(int(0))
        
class spectrogram_filter(Thread):
    def __init__(self, 
                 _signal_message,
                 _source_directory):
        ''' Constructor. '''
        Thread.__init__(self)
        self._signal_message = _signal_message
        self._source_directory = _source_directory
        
        self.switch = 0
        
    def run(self):
        #get array of files in source directory
        file_names_array = []
        csv_file_names_array = []
        file_names_array = os.listdir(self._source_directory)
        if file_names_array != []:
            for file_item in file_names_array:
                if file_item[-3:] == "csv":
                    csv_file_names_array.append(file_item)
        
        csv_file_count = 0
        csv_file_total = len(csv_file_names_array)
        
        #read wav file to array
        _DA = data_augmentation()
        for csv_file in csv_file_names_array:
            csv_file_path = self._source_directory + "/" + csv_file
            
            data, freqseries, timeseries = _DA.read_from_csv(csv_file_path)
                
            #test for emptiness
            emptiness_TH_03 = 0.0006
            if np.std(data)<emptiness_TH_03:
                emptiness_03 = True
            else:
                emptiness_03 = False
                
            #test for empty first half
            fisrst_half_percent = 35
            emptiness_TH_04 = 0.0006
            emptiness_TH_05 = 0.22
            index_fisrst_half_percent = int(timeseries.shape[1]*fisrst_half_percent/100)
            if np.std(data[:,0:index_fisrst_half_percent]) < emptiness_TH_04 or (np.std(data[:,0:index_fisrst_half_percent]) - np.std(data))/np.std(data)<-emptiness_TH_05:
                emptiness_04 = True
            else:
                emptiness_04 = False
            
            #apply tests
            if emptiness_03 == True or emptiness_04 == True:
                self.switch = 0
                self._signal_message.spectrogram_filter_abnormal_found_bit.emit(csv_file_path)
            else:
                self.switch = 1
            
            while self.switch == 0:
                time.sleep(0.025)
            if self.switch == -1:
                break
            elif self.switch == 2:
                os.remove(csv_file_path)
            
            self.switch = 0
            
            csv_file_count = csv_file_count + 1
            self._signal_message.progress_update_bit.emit(int(csv_file_count*100/csv_file_total))
        
        self._signal_message.progress_update_bit.emit(int(100))
        time.sleep(0.1)
        self._signal_message.spectrogram_filter_completed_bit.emit(True)
        self._signal_message.progress_update_bit.emit(int(0))
            
    def change_switch(self, value):
        self.switch = value

class data_generator():
    def __init__(self, _CNN_batches, _command_list):
        ''' Constructor. '''
        self._CNN_batches = _CNN_batches
        self.command_list = _command_list
        
    def train_generator(self, data_set):
        _DA = data_augmentation() 
        while True:
            index_array = range(0, len(data_set))
            # Create data batches
            for start in range(0, len(index_array), self._CNN_batches):
                x_batch = []
                y_batch = []
                
                end = min(start + self._CNN_batches, len(index_array))
                i_batch = index_array[start:end]
                i_batch = random.sample(i_batch, len(i_batch))
                for i in i_batch:
                    data, freqseries, timeseries = _DA.read_from_csv(data_set[i][2])
                    data3D = np.expand_dims(data, axis=2)
                    x_batch.append(data3D)
                    y_batch.append(int(data_set[i][0]))
                x_batch = np.array(x_batch)
                y_batch = to_categorical(y_batch, num_classes = len(self.command_list))
                yield (x_batch, y_batch)
        
    def validate_generator(self, data_set):
        _DA = data_augmentation() 
        while True:
            index_array = range(0, len(data_set))
            # Create data batches
            for start in range(0, len(index_array), self._CNN_batches):
                x_batch = []
                y_batch = []
                
                end = min(start + self._CNN_batches, len(index_array))
                i_batch = index_array[start:end]
                i_batch = random.sample(i_batch, len(i_batch))
                for i in i_batch:
                    data, freqseries, timeseries = _DA.read_from_csv(data_set[i][2])
                    data3D = np.expand_dims(data, axis=2)
                    x_batch.append(data3D)
                    y_batch.append(int(data_set[i][0]))
                x_batch = np.array(x_batch)
                y_batch = to_categorical(y_batch, num_classes = len(self.command_list))
                yield (x_batch, y_batch)

    
    def test_generator(self, data_set):
        _DA = data_augmentation() 
        while True:
            index_array = range(0, len(data_set))
            # Create data batches
            for start in range(0, len(index_array), self._CNN_batches):
                x_batch = []
                
                end = min(start + self._CNN_batches, len(index_array))
                i_batch = index_array[start:end]
                for i in i_batch:
                    data, freqseries, timeseries = _DA.read_from_csv(data_set[i][2])
                    data3D = np.expand_dims(data, axis=2)
                    x_batch.append(data3D)
                x_batch = np.array(x_batch)
                yield x_batch


class speech_recognition_training(Thread):
    def __init__(self, 
                 _signal_message,
                 _CNN_feed_directory, 
                 _CNN_personalize_directory,
                 _personalized_percent,
                 _CNN_batches, 
                 _CNN_epochs):
        
        ''' Constructor. '''
        Thread.__init__(self)
        self._signal_message = _signal_message
        self._CNN_feed_directory = _CNN_feed_directory
        self._CNN_personalize_directory = _CNN_personalize_directory
        self._personalized_percent = _personalized_percent
        self._CNN_batches = _CNN_batches
        self._CNN_epochs = _CNN_epochs
        self.train_validate_test_split_over_bit = 0
        
        self.switch = 0
    
    def run(self):
        
        self.train_validate_test_split_over_bit = 1
        self.apply_train_validate_test_split(test_size=0.1, validate_size=0.4, random_state=42)
        while self.train_validate_test_split_over_bit:
            time.sleep(0.025)
        
        _DA = data_augmentation()
        data, freqseries, timeseries = _DA.read_from_csv(self.train_set_array[1][2])
        self.input_shape = (freqseries.shape[1], timeseries.shape[1], 1)
        self.run_model()
    
    def get_command_data_table(self, command):
        command_data_table_array_general = []
        file_names_array_general = []
        file_names_array_general = os.listdir(str(self._CNN_feed_directory + '/' + command))
        for file_item in file_names_array_general:
            command_data_table_array_general.append([str(self.command_list.index(command)), command , str(self._CNN_feed_directory + '/' + command + '/' + file_item)])
        
        command_data_table_array_personal = []
        file_names_array_personal = []
        file_names_array_personal = os.listdir(str(self._CNN_personalize_directory + '/' + command))
        for file_item in file_names_array_personal:
            command_data_table_array_personal.append([str(self.command_list.index(command)), command , str(self._CNN_personalize_directory + '/' + command + '/' + file_item)])
        
        command_data_table_array = command_data_table_array_general
        
        if self._personalized_percent > 0 and len(command_data_table_array_personal) > 1:
            step_size = int(100*max(100/self._personalized_percent,len(command_data_table_array_general)/len(command_data_table_array_personal)))+1
            for i in range(0,100*len(command_data_table_array_general),step_size):
                command_data_table_array[int(i/100)] = command_data_table_array_personal[int(i/step_size)]
        return(command_data_table_array)
    
    def apply_train_validate_test_split(self, test_size, validate_size, random_state):
        train_set_array = []
        validation_set_array = []
        test_set_array = []
        
        self.train_set_array = []
        self.validation_set_array = []
        self.test_set_array = []
        
        self.command_list = []
        item_names_array = []
        item_names_array = os.listdir(self._CNN_feed_directory)
        for item_name in item_names_array:
            if os.path.isdir(str(self._CNN_feed_directory + '/' + item_name)):
                self.command_list.append(item_name)
        
        train_set_array_len=[]
        validation_set_array_len=[]
        test_set_array_len=[]
        
        for command in self.command_list:
            train_array, test_array = train_test_split(self.get_command_data_table(command),test_size=test_size, random_state=random_state)
            train_array, validation_array = train_test_split(train_array, test_size=validate_size, random_state=random_state)
            
            train_set_array.append(train_array)
            validation_set_array.append(validation_array)
            test_set_array.append(test_array)
            
            train_set_array_len.append(len(train_array))
            validation_set_array_len.append(len(validation_array))
            test_set_array_len.append(len(test_array))
            
        train_set_array_len_min = min(min(train_set_array_len), 800)
        validation_set_array_len_min = min(min(validation_set_array_len), 640)
        test_set_array_len_min = min(min(test_set_array_len), 160)
        
        rand_train_set_array_index = random.sample(range(train_set_array_len_min), train_set_array_len_min)
        rand_validation_set_array_index = random.sample(range(validation_set_array_len_min), validation_set_array_len_min)
        rand_test_set_array_index = random.sample(range(test_set_array_len_min), test_set_array_len_min)
        
        for column in range(0, train_set_array_len_min):
            commands_slice = [i[rand_train_set_array_index[column]] for i in train_set_array]
            rand_commands_slice_index = random.sample(range(len(commands_slice)), len(commands_slice))
            self.train_set_array.extend([commands_slice[i] for i in rand_commands_slice_index])
        for column in range(0, validation_set_array_len_min):
            commands_slice = [i[rand_validation_set_array_index[column]] for i in validation_set_array]
            rand_commands_slice_index = random.sample(range(len(commands_slice)), len(commands_slice))
            self.validation_set_array.extend([commands_slice[i] for i in rand_commands_slice_index])
        for column in range(0, test_set_array_len_min):
            commands_slice = [i[rand_test_set_array_index[column]] for i in test_set_array]
            rand_commands_slice_index = random.sample(range(len(commands_slice)), len(commands_slice))
            self.test_set_array.extend([commands_slice[i] for i in rand_commands_slice_index])
        
        self.train_validate_test_split_over_bit = 0
        
    def run_model(self):
        name_classes = len(self.command_list)
        model = self.deep_cnn(self.input_shape, name_classes)
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
        callbacks = [EarlyStopping(monitor='val_acc', patience=25, verbose=1, mode='min')]
        
        _DG_1 = data_generator(self._CNN_batches, self.command_list)
        _DG_2 = data_generator(self._CNN_batches, self.command_list)
        _DG_3 = data_generator(self._CNN_batches, self.command_list)
    
        H = model.fit_generator(generator=_DG_1.train_generator(self.train_set_array),
                              steps_per_epoch=int(np.ceil(len(self.train_set_array)/self._CNN_batches)),
                              epochs=self._CNN_epochs,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=_DG_2.validate_generator(self.validation_set_array),
                              validation_steps=int(np.ceil(len(self.validation_set_array)/self._CNN_batches)))

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, len(H.history["loss"])), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("plot.png")


        y_true = [int(i[0]) for i in self.test_set_array]
        y_pred_proba = model.predict_generator(_DG_3.test_generator(self.test_set_array), 
                                      int(np.ceil(len(self.test_set_array)/self._CNN_batches)), 
                                      verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        acc_score = accuracy_score(y_true, y_pred)
        print(acc_score)
        
        self._signal_message.cnn_training_over_bit.emit(True)
        
        while self.switch == 0:
                time.sleep(0.025)
        if self.switch == 1:
            date_ = datetime.now().date()
            time_ = datetime.now().time()
            date_time = "CNN__" + str(date_.year) + "_" + str(date_.month) + "_" + str(date_.day) + "__" + str(time_.hour) + "_" + str(time_.minute) + "_" + str(time_.second)
            filename_hdf5 = str(QtCore.QDir.currentPath() + "/speech/models/" + date_time + ".hdf5")
            model.save(filename_hdf5)
            filename_csv = str(QtCore.QDir.currentPath() + "/speech/models/" + date_time + ".csv")
            self._signal_message.cnn_saved_name_bit.emit(filename_csv, self.command_list, str(int(acc_score*1000)/10))
        try:
            del model
        except:
            model = None
        
        if tf.compat.v1.keras.backend.tensorflow_backend._SESSION:
            tf.reset_default_graph() 
            tf.compat.v1.keras.backend.tensorflow_backend._SESSION.close()
            tf.compat.v1.keras.backend.tensorflow_backend._SESSION = None
 
    def deep_cnn(self, features_shape, num_classes, act='relu'):
 
        x = Input(name='inputs', shape=features_shape, dtype='float32')
        o = x
    
       # Block 1
        o = Conv2D(16, (7, 7), activation="relu", padding='same', strides=(1,1), name='block1_conv', input_shape=features_shape)(o)
        o = MaxPooling2D((3, 3), strides=(2,2), padding='valid', name='block1_pool')(o)
        o = BatchNormalization(name='block1_norm')(o)
    
        # Block 2
        o = Conv2D(32, (5, 5), activation="relu", padding='same', strides=(1,1), name='block2_conv')(o)
        o = MaxPooling2D((3, 3), strides=(2,2), padding='valid', name='block2_pool')(o)
        o = BatchNormalization(name='block2_norm')(o)
 
        # Block 3
        o = Conv2D(64, (3, 3), activation="relu", padding='same', strides=(1,1), name='block3_conv')(o)
        o = MaxPooling2D((3, 3), strides=(2,2), padding='valid', name='block3_pool')(o)
        o = BatchNormalization(name='block3_norm')(o)

        # Block 4
        o = Conv2D(128, (3, 3), activation="relu", padding='same', strides=(1,1), name='block4_conv')(o)
        o = MaxPooling2D((3, 3), strides=(2,2), padding='valid', name='block4_pool')(o)
        o = BatchNormalization(name='block4_norm')(o)

        # Block 5
        o = Conv2D(128, (3, 3), activation="relu", padding='same', strides=(1,1), name='block5_conv')(o)
        o = MaxPooling2D((3, 3), strides=(2,2), padding='valid', name='block5_pool')(o)
        o = BatchNormalization(name='block5_norm')(o)

        # Block 6
        o = Conv2D(256, (3, 3), activation="relu", padding='same', strides=(1,1), name='block6_conv')(o)
        o = MaxPooling2D((3, 3), strides=(2,2), padding='valid', name='block6_pool')(o)
        o = BatchNormalization(name='block6_norm')(o)

        # Flatten
        o = Flatten(name='flatten')(o)
    
        # Dense layer 1
        o = Dense(1024, activation="relu", name='dense1')(o)
        o = BatchNormalization(name='dense_norm1')(o)
        o = Dropout(0.5, name='dropout1')(o)

        # Predictions
        o = Dense(num_classes, activation='softmax', name='pred')(o)
 
        # Print network summary
        Model(inputs=x, outputs=o).summary()
    
        return Model(inputs=x, outputs=o)
    
    def change_switch(self, value):
        self.switch = value
        
class speech_recognition_listening(Thread):
    def __init__(self, 
                 _signal_message,
                 _filename_hdf5, 
                 _filename_csv):
        
        ''' Constructor. '''
        Thread.__init__(self)
        self._signal_message = _signal_message
        self._filename_hdf5 = _filename_hdf5
        self._filename_csv = _filename_csv

        self.switch = 0
    
    def run(self):
        try:
            readcsv=csv.reader(open(self._filename_csv))
            for row in readcsv:
                if row[0] == '_rate':
                    self._rate = int(row[1])
                elif row[0] == '_chunk':
                    self._chunk = int(row[1])
                elif row[0] == '_commandchunk':
                    self._commandchunk = int(row[1])
                elif row[0] == '_channels':
                    self._channels = int(row[1])
                elif row[0] == '_stftsegmentsize':
                    self._stftsegmentsize = int(row[1])
                elif row[0] == '_overlappercent':
                    self._overlappercent = int(row[1])
                elif row[0] == '_stftfreqres':
                    self._stftfreqres = int(row[1])
                elif row[0] == '_freqhighlim':
                    self._freqhighlim = int(row[1])
                elif row[0] == '_stft_lin_log_norm':
                    self._stft_lin_log_norm = int(row[1])
                elif row[0] == 'command_list':
                    self.command_list = row[1:]
            self._format = pyaudio.paInt16
            
            self.model = load_model(self._filename_hdf5)
            self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
            
            self._signal_message.cnn_loaded_bit.emit(True)
            self.idle()
        except:
            self._signal_message.cnn_unloaded_bit.emit(True)
    
    def idle(self):
        self._signal_message.cnn_idle_bit.emit(True)
        while self.switch == 0:
            time.sleep(0.025)
        if self.switch == 1:
            self.switch = 0
            self.listen()
        if self.switch == -1:
            self.switch = 0
            self.unload()
    
    def listen(self):
        self._signal_message.cnn_waiting_for_speech_bit.emit(True)
        while self.switch == 0:
            try:
                data = np.frombuffer(self.pa_stream.read(self._chunk), dtype=np.int16)
                amplitude = np.linalg.norm(data, np.inf)
                if amplitude > self._silencethreshold*32767/100:
                    voice_sample_array=np.array([])
                    voice_sample_array=np.concatenate((voice_sample_array, data))
                    data = np.frombuffer(self.pa_stream.read(self._commandchunk), dtype=np.int16)
                    
                    self.pa_stream.stop_stream()
                    
                    voice_sample_array=np.concatenate((voice_sample_array, data))
                    voice_sample_array=np.ravel(voice_sample_array)
                    freq_series, time_series, data_stft = stft(voice_sample_array, 
                                                               fs = self._rate, 
                                                               nperseg = self._stftsegmentsize, 
                                                               noverlap = int(self._stftsegmentsize*self._overlappercent/100), 
                                                               nfft = int(self._rate/self._stftfreqres), 
                                                               padded = False, 
                                                               boundary = None)
                    data_stft_np = np.array(data_stft, dtype = np.float16)
                    data_stft_abs = np.abs(data_stft_np)
                    if self._stft_lin_log_norm == 0:
                        data_stft_abs_norm = data_stft_abs / np.linalg.norm(data_stft_abs,np.inf)
                    else:
                        base = 10
                        data_stft_abs_norm = (np.log(data_stft_abs+1) / np.log(base)) / np.linalg.norm((np.log(data_stft_abs+1) / np.log(base)),np.inf)
                    freq_series = freq_series[:int(self._freqhighlim/self._stftfreqres+1)]
                    data_stft_abs_norm_limited = data_stft_abs_norm[:int(self._freqhighlim/self._stftfreqres+1),:]
                    zero_array = np.zeros((data_stft_abs_norm_limited.shape[0]-int(8000/self._stftfreqres+1),data_stft_abs_norm_limited.shape[1]), dtype = np.float16)
                    data_stft_abs_norm_limited[int(8000/self._stftfreqres+1):,:] = zero_array
                    filename = str(QtCore.QDir.currentPath() + "/speech/models/temp.csv")
                    with open(filename, 'w', newline='') as csvfile:
                        data_stft_csv = csv.writer(csvfile, dialect='excel')
                        data_stft_csv.writerow(freq_series)
                        data_stft_csv.writerow(time_series)
                        for i in range(0,data_stft_abs_norm_limited.shape[0]):
                            data_stft_csv.writerow(data_stft_abs_norm_limited[i,:])
                    self._signal_message.cnn_spectrogram_is_ready_bit.emit(filename)
                    x = []
                    data3D = np.expand_dims(data_stft_abs_norm_limited, axis=2)
                    x.append(data3D)
                    x = np.array(x)
                    y_pred_proba = self.model.predict(x, verbose=1)
                    y_pred = np.argmax(y_pred_proba, axis=1)
                    y_pred = int(y_pred[0])
                    print(y_pred_proba[0, y_pred])
                    if y_pred_proba[0, y_pred] > 0.9:
                        self._signal_message.cnn_recognized_bit.emit(self.command_list[y_pred])
                    else:
                        self._signal_message.cnn_recognized_bit.emit("UNRECOGNIZED")
                    
                    self.pa_stream.start_stream()
                    
                else:
                    self._signal_message.volume_update_bit.emit(amplitude*100/32767)
            except:
                self.switch = 0
                
        self._signal_message.volume_update_bit.emit(0)
        
        if self.switch == 1:
            self.switch = 0
            self.idle()
        if self.switch == -1:
            self.switch = 0
            self.unload()
    
    def update_mic_and_silence_threshold(self, _mic, _silencethreshold):
        self._mic = _mic
        self._silencethreshold = _silencethreshold
        # create audio stream
        try:
            self.pa = pyaudio.PyAudio()
            self.pa_stream = self.pa.open(format=self._format, 
                                channels=self._channels, 
                                rate=self._rate, input=True, 
                                frames_per_buffer=self._chunk, 
                                input_device_index=self._mic)
        except:
            self.pa = None
            self.pa_stream = None
            return None
        
    def unload(self):
        try:
            self.pa_stream.stop_stream()
            self.pa_stream.close()
            self.pa.terminate()
        except:
            self.pa = None
            self.pa_stream = None

        try:
            del self.model
        except:
            self.model = None
        
        if tf.compat.v1.keras.backend.tensorflow_backend._SESSION:
            tf.reset_default_graph() 
            tf.compat.v1.keras.backend.tensorflow_backend._SESSION.close()
            tf.compat.v1.keras.backend.tensorflow_backend._SESSION = None
            
        self._signal_message.cnn_unloaded_bit.emit(True)
        
    def change_switch(self, value):
        self.switch = value