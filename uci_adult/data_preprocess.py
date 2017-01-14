#-*- coding: utf-8 -*-

from data_preprocess_with_pandas import get_feature_names

def get_all_str2num(fileName):
    all_str2num_dict = {}

    with open(fileName, 'r') as f:
        all = f.readlines();

    for line in all:
        line = line.strip()
        line = line.replace('.', '')
        colon = line.find(':')
        line = line[colon+1:]

        # print(line)

        line_arr = line.split(',')

        for i in range(len(line_arr)):
            str_now = line_arr[i].strip()
            all_str2num_dict[str_now] = i
        
        # print(all_str2num_dict)

    
    all_str2num_dict['?'] = 'NaN'
    all_str2num_dict['>50K'] = 0
    all_str2num_dict['<=50K'] = 1

    
    return all_str2num_dict

def getLine(fileName, lineNum):
    with open(fileName, 'r') as f:
        all = f.readlines()

    return all[lineNum].strip()

def data2nice(fileName, str2num_dict):
    all_lines = []
    all_lines_num = []
    now_line = []
    now_line_num = []
    with open(fileName, 'r') as f:
        all = f.readlines()

    for line in all:
        now_line = []
        now_line_num = []
        line = line.strip()
        line = line.replace('.', '')
        # print(line)

        line_arr = line.split(',')

        for i in range(len(line_arr)):
            str_now = line_arr[i].strip()
            now_line.append(str_now)
            if str_now in str2num_dict.keys():
                now_line_num.append(str2num_dict[str_now])
            else:
                now_line_num.append(str_now)

        all_lines.append(now_line)
        all_lines_num.append(now_line_num)

    return all_lines, all_lines_num        

def put_all_lines_num_to_file(to_save_file_name, all_lines_num):
    with open(to_save_file_name, 'w') as f:
        for line in all_lines_num:
            line_str = ','.join(str(e) for e in line)
            f.write(line_str + '\n')



if __name__ == '__main__':
    # baseDir = 'H:/practice/scikit_class/scikit_learning/uci_adult'
    baseDir = 'adult_data/'
    
    describe_file = baseDir+'describe.txt'

    feature_names = get_feature_names(describe_file)
    print(feature_names)

    to_change = ['adult.data','adult.test']

    for data_file in to_change:
        data_file = baseDir + data_file
        save_file = data_file+'.num'
        all_str2num_dict = get_all_str2num(describe_file)
    
        print(len(all_str2num_dict))
        # print(all_str2num_dict)

        all_lines, all_lines_num = data2nice(data_file, all_str2num_dict)
        print (len(all_lines))
        # print(all_lines)
        # print(all_lines_num)

        put_all_lines_num_to_file(save_file, all_lines_num)

        print('%s DONE!!' % data_file)

    print('ALL DONE!')        