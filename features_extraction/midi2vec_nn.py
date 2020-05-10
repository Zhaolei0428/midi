'''
# 6 features from midi as bellow
# average note num, average intensity, speed, average pitch, average time-value, average interval increase
'''

from struct import unpack
import time
import numpy as np
import os
from keras.utils import to_categorical
import random

def read_vlq(f):
    result = ''
    buffer = unpack('B', f.read(1))[0]
    length = 1
    while buffer > 127:
        # print(buffer)
        result += '{0:{fill}{n}b}'.format(buffer - 128, fill='0', n=7)
        buffer = unpack('B', f.read(1))[0]
        length += 1

    result += '{0:{fill}{n}b}'.format(buffer, fill='0', n=7)
    # print(result)
    return int(result, 2), length


def parse_event(evt, param):
    if 128 <= evt <= 143:
        print('Note Off event.',  'channel=', evt % 16, unpack('>BB', param))
    elif 144 <= evt <= 159:
        print('Note On event.', 'channel=', evt % 16, unpack('>BB', param))
    elif 176 <= evt <= 191:
        print('Control Change.', 'channel=', evt % 16, 'control number and value=', unpack('>BB', param))
    elif 192 <= evt <= 207:
        print('Program Change.', 'channel=', evt % 16, " instrument number=", unpack('B', param)[0])
    else:
        print('other channel event, type=', evt)


def parse_sys_event(f, event_type):
    if event_type == 240:
        count = 0
        while unpack('B', f.read(1))[0] != 247:
            count += 1
        return count+1
    elif event_type == 242:
        f.read(2)
        return 2
    elif event_type == 243:
        f.read(1)
        return 1
    else:
        return 0


def track2vec(f, note_list):
    track_head = f.read(4)
    if track_head != b'MTrk':
        if track_head != b'':
            print(f.read(20))
            raise Exception('not a midi file!')
        else:
            return None

    # length of track
    len_of_track = unpack('>L', f.read(4))[0]
    # print("==================================>")
    # print("MTrK len=", len_of_track)

    # {note_on:start_time}
    speed = -1
    note_on = {}
    # note_event_num = 0
    counter = 0
    t = 0
    instrument = [128] * 16  # ins type of 16 channels
    last_event = None
    # has = False
    while True:
        delta_t, len_ = read_vlq(f)
        counter += len_
        t += delta_t
        # print('T ', t, end=' ')
        event_code = f.read(1)
        event_type = unpack('B', event_code)[0]
        counter += 1
        # print(' event_type ', event_type, end='')
        if event_type == 255:
            meta_type = unpack('B', f.read(1))[0]
            counter += 1
            # print(' - meta_type ', meta_type, end='')
            data_len, len_ = read_vlq(f)
            counter += len_
            data = f.read(data_len)
            counter += data_len
            if meta_type == 0x51:  # microseconds per quarter note
                microminutes = int.from_bytes(data, 'big', signed=False)
                speed = int(60 * 1000000 / microminutes)
        elif 240 <= event_type < 255:
            counter += parse_sys_event(f, event_type)
        else:
            if event_type <= 127:
                event_type = last_event
                f.seek(-1, 1)
                counter -= 1
                # f.read(1)
                # counter += 1
                # has = True
            if 128 <= event_type <= 143:
                # print(' Note Off event.', end='')
                event_info = unpack('BB', f.read(2))
                counter += 2
                pitch = event_info[0]
                if note_on.get(pitch) is not None:
                    intensity = note_on.get(pitch)[1]
                    note_list.append([instrument[event_type % 16], note_on.get(pitch)[0], t, pitch, intensity])
                    note_on.pop(pitch)
                    # note_event_num += 1
                # parse_event(event_type, f.read(2))

            elif 144 <= event_type <= 159:
                # print(' Note On event.', end='')
                event_info = unpack('BB', f.read(2))
                counter += 2
                pitch = event_info[0]
                intensity = event_info[1]
                if note_on.get(pitch) is not None:
                    note_list.append([instrument[event_type % 16], note_on.get(pitch)[0], t, pitch, note_on.get(pitch)[1]])
                    note_on.pop(pitch)
                    # note_event_num += 1
                if intensity != 0:  # intensity==0 means note off
                    note_on[pitch] = [t, intensity]
            elif 160 <= event_type <= 175:
                # print(' after touch.', end='')
                f.read(2)
                counter += 2
            elif 176 <= event_type <= 191:
                # print(' Control Change.', end='')
                f.read(2)
                counter += 2
            elif 192 <= event_type <= 207:
                # print(' Program Change.', end='')
                ins = unpack('B', f.read(1))[0]
                instrument[event_type % 16] = ins
                counter += 1
                # print('*******', ins, event_type)
            elif 208 <= event_type <= 223:
                # print(' Channel pressure.', end='')
                f.read(1)
                counter += 1
            elif 224 <= event_type <= 239:
                # print(' pitch wheel.', end='')
                f.read(2)
                counter += 2
            last_event = event_type

        # print(counter)
        if counter == len_of_track:
            return speed


# 获取列表的第二个元素
def take_second(elem):
    return elem[1]


def midi2vec(path):
    print(path)
    with open(path, 'rb') as f:
        # HEADER
        if f.read(4) != b'MThd':
            raise Exception('not a midi file!')
        print('header_info_length=', unpack('>L', f.read(4))[0])
        header_info = unpack('>hhh', f.read(6))
        print('tracks number:', header_info[1])
        if header_info[2] < 0:
            raise Exception('not count based on ticks!')
        print('ticks of a quarter note=', header_info[2])
        note_64_ticks = header_info[2]/16
        total_note_list = []
        speed = 120

        result = track2vec(f, total_note_list)
        while result is not None:
            if result != -1:
                speed = result
            result = track2vec(f, total_note_list)

        if len(total_note_list) == 0:
            print(path, 'is not valid')
            return None

        for note in total_note_list:
            note[2] = int((note[2]-note[1])/note_64_ticks+1)  # time of note
            note[1] = int(note[1] / note_64_ticks)  # count by 64 division note

        total_note_list.sort(key=take_second)  # sort by start time

        feature_list = []
        # print(note_list)
        for sample_time in sample_steps:
            # average note num, average intensity, speed, average pitch, average time-value, average interval increase
            feature = np.zeros(feature_num, np.float)
            note_list = []
            last_pitch = 0
            for note in total_note_list:
                if sample_time <= note[1] <= sample_time + max_vec_num:
                    note_list.append(note)
                    feature[0] += 1
                    feature[1] += note[4]
                    feature[3] += note[3]
                    feature[4] += note[2]
                    if note[3] > last_pitch:
                        feature[5] += note[3] - last_pitch
                    last_pitch = note[3]
            if len(note_list) == 0:
                break
            feature[1] = feature[1]/feature[0]
            feature[2] = speed
            feature[3] = feature[3]/feature[0]
            feature[4] = feature[4]/feature[0]
            feature[5] = feature[5] / (max_vec_num / 16)
            feature[0] = feature[0] / (max_vec_num/16)
            # feature[3] = 0

            feature_list.append(feature)
        return feature_list


def prepare_dara():
        y_train = []
        features_train = []
        y_test = []
        features_test = []
        # dirs = os.listdir(midi_path)
        for emotion_name, emotion in emotion_dict.items():
            files = os.listdir(midi_path + emotion_name)
            print(files)
            for file in files:
                path = midi_path + emotion_name + '/' + file
                feature_list = midi2vec(path)
                if feature_list is not None:
                    for feature in feature_list:
                        select = random.randint(0, 9)
                        if select < 2:
                            y_test.append(emotion)
                            features_test.append(feature)
                        else:
                            y_train.append(emotion)
                            features_train.append(feature)

        y_train = np.array(y_train)
        features_train = np.array(features_train)
        y_test = np.array(y_test)
        features_test = np.array(features_test)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        print('x_train shape:', features_train.shape, 'y_train shape:', y_train.shape)
        print('x_test shape:', features_test.shape, 'y_test shape:', y_test.shape)
        # np.set_printoptions(threshold=np.inf)
        # print(features_test)
        np.savez(midi_path + 'vec.npz', x_train=features_train, y_train=y_train,
                 x_test=features_test, y_test=y_test)


emotion_dict = {'excited': 0, 'angry': 1, 'sad': 2, 'relaxed': 3}
feature_num = 6  # feature number like tempo, intensity
max_vec_num = 320  # time steps of sample(count by quarter notes)
sample_nums = 25  # max segment of one midi file
sample_steps = [max_vec_num * i for i in range(sample_nums)]

midi_path = '../datasets/'
prepare_dara()
# vec = midi2vec('/home/zhao/Desktop/datasets/train/excited/Hawaiian.mid')