from struct import unpack
import time
import numpy as np
import os
from keras.utils import to_categorical


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
    note_on = {}
    note_event_num = 0
    counter = 0
    t = 0
    instrument = [128] * 16  # ins type of 16 channels
    last_event = None
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
        elif event_type <= 127:
            f.read(1)
            counter += 1
        elif 240 <= event_type < 255:
            counter += parse_sys_event(f, event_type)
        else:
            if 128 <= event_type <= 143:
                # print(' Note Off event.', end='')
                event_info = unpack('BB', f.read(2))
                counter += 2
                pitch = event_info[0]
                intensity = event_info[1]
                if note_on.get(pitch) is not None:
                    note_list.append([instrument[event_type % 16], note_on.get(pitch), t, pitch])
                    note_on.pop(pitch)
                    note_event_num += 1
                # parse_event(event_type, f.read(2))

            elif 144 <= event_type <= 159:
                # print(' Note On event.', end='')
                event_info = unpack('BB', f.read(2))
                counter += 2
                pitch = event_info[0]
                intensity = event_info[1]
                if note_on.get(pitch) is not None:
                    note_list.append([instrument[event_type % 16], note_on.get(pitch), t, pitch])
                    note_event_num += 1
                note_on[pitch] = t
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
            return note_event_num


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
        note_list = []

        result = track2vec(f, note_list)
        while result is not None:
            result = track2vec(f, note_list)

        note_list.sort()
        ins_order = []
        ins = -1
        note_count = 1
        for note in note_list:
            note[2] = int((note[2]-note[1])/note_64_ticks+1)  # time of note
            note[1] = int(note[1] / note_64_ticks)  # count by 64 division note
            if note[0] == ins:
                note_count += 1
            else:
                ins_order.append((note_count, ins))
                ins = note[0]
                note_count = 1
        # find top3 instrument
        ins_order.append((note_count, ins))
        ins_order.sort(reverse=True)
        # print(len(note_list))
        if len(ins_order) > 3:
            ins_order = ins_order[:3]
        # print(ins_order)

        vec = np.zeros((max_vec_num, 96), dtype=np.uint8)
        ins_index = {}
        for i, r in enumerate(ins_order):
            ins_index[r[1]] = i
        # print(ins_index)

        note_list.sort(key=take_second)  # sort by start time
        for note in note_list:
            if ins_index.get(note[0], None) is None:
                continue
            index = ins_index.get(note[0])
            vec_num = note[1]/16
            if vec_num >= max_vec_num:
                continue
            note_time = note[2]
            pitch = note[3]
            location64 = note[1] % 16
            vec[int(vec_num), index*32+location64*2] = pitch
            vec[int(vec_num), index * 32 + location64 * 2 + 1] = note_time

        return vec


def prepare_dara():
    for data_type in ('train', 'test'):
        x = []
        y = []
        data_path = midi_path + data_type + '/'
        dirs = os.listdir(data_path)
        for emotion_dir in dirs:
            emotion = emotion_dict[emotion_dir]
            files = os.listdir(data_path + emotion_dir)
            print(files)
            for file in files:
                path = data_path + emotion_dir + '/' + file
                x.append(midi2vec(path))
                y.append(emotion)

        x = np.array(x)
        y = np.array(y)
        y = to_categorical(y)
        print(y.shape)
        print(x.shape)
        np.savez(midi_path + data_type + '_vec.npz', x=x, y=y)


emotion_dict = {'relaxed': 0, 'excited': 1, 'sad': 2, 'angry': 3}
max_vec_num = 120  # time steps of sample(count by quarter notes)

midi_path = '/home/zhao/Desktop/datasets/'
prepare_dara()
