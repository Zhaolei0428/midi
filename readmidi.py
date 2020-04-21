from struct import unpack
import time


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


def parse_sys_event(event_type):
    if event_type == 240:
        count = 0
        code = unpack('B', f.read(1))[0]
        while code != 247:
            print(code, end=',')
            count += 1
            code = unpack('B', f.read(1))[0]
        return count+1
    elif event_type == 242:
        f.read(2)
        return 2
    elif event_type == 243:
        f.read(1)
        return 1
    else:
        return 0

# /home/zhao/Desktop/datasets/excited/Darktown.mid
# /home/zhao/Downloads/watrmark.mid
with open('./datasets/excited/0020.MID', 'rb') as f:
    print(f.read(200))
    f.seek(0)
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

    ''' ================================== '''
    while True:
        track_head = f.read(4)
        if track_head != b'MTrk':
            if track_head != b'':
                print(f.read(20))
                raise Exception('not a midi file!')
            else:
                break

        # length of track
        len_of_track = unpack('>L', f.read(4))[0]
        # print('len_of_track ', len_of_track)

        counter = 0
        t = 0
        last_event = None
        print("==================================>")
        print("MTrK len=", len_of_track)
        while True:
            delta_t, len_ = read_vlq(f)
            counter += len_
            t += delta_t
            print('T ', t, end=' ')
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
                if meta_type == 0x51:
                    print('meta_event type=0x%x(设定速度，以微妙为单位，是四分音符的时值)' % meta_type, ' data=', data, '(', int.from_bytes(data, 'big', signed=False), ')')
                else:
                    print('meta_event type=0x%x' % meta_type, ' data=', data)
            elif event_type <= 127:
                print("********************************", end=' ')
                parse_event(last_event, event_code + f.read(1))
                counter += 1
            elif 240 <= event_type < 255:
                counter += parse_sys_event(event_type)
                print("******sys_event******", event_type)
            else:
                if 128 <= event_type <= 143:
                    # print(' Note Off event.', end='')
                    parse_event(event_type, f.read(2))
                    counter += 2
                elif 144 <= event_type <= 159:
                    # print(' Note On event.', end='')
                    parse_event(event_type, f.read(2))
                    counter += 2
                elif 160 <= event_type <= 175:
                    # print(' after touch.', end='')
                    parse_event(event_type, f.read(2))
                    counter += 2
                elif 176 <= event_type <= 191:
                    # print(' Control Change.', end='')
                    parse_event(event_type, f.read(2))
                    counter += 2
                elif 192 <= event_type <= 207:
                    # print(' Program Change.', end='')
                    parse_event(event_type, f.read(1))
                    counter += 1
                elif 208 <= event_type <= 223:
                    # print(' Channel pressure.', end='')
                    parse_event(event_type, f.read(1))
                    counter += 1
                elif 224 <= event_type <= 239:
                    # print(' pitch wheel.', end='')
                    parse_event(event_type, f.read(2))
                    counter += 2
                last_event = event_type

            # print(counter)
            if counter == len_of_track:
                break
