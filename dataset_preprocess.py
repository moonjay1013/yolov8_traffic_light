from __future__ import print_function
import shutil
from tqdm import tqdm
from shutil import copy2
import os
import random
import glob
import xml.etree.ElementTree as ET


# VOC2YOLO
class VOC2YOLO:
    def __init__(self, input_voc_dir, output_yolo_dir):
        self.input_voc_dir = input_voc_dir
        self.output_yolo_dir = output_yolo_dir

    @staticmethod
    def xml_reader(file_name):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(file_name)
        size = tree.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            objects.append(obj_struct)
        return width, height, objects

    def voc2yolo(self, filename):
        classes_dict = {}
        with open(os.path.join(self.input_voc_dir, "class.txt")) as f:
            for idx, line in enumerate(f.readlines()):
                class_name = line.strip()
                classes_dict[class_name] = idx
        width, height, objects = VOC2YOLO.xml_reader(filename)
        '''
        1920 1080 [{'name': 'green', 'bbox': [616, 477, 633, 521]}, {'name': 'green', 'bbox': [941, 476, 956, 506]}]
        '''
        lines = []
        for obj in objects:
            x, y, x2, y2 = obj['bbox']
            class_name = obj['name']
            label = classes_dict[class_name]
            cx = (x2 + x) * 0.5 / width
            cy = (y2 + y) * 0.5 / height
            w = (x2 - x) * 1. / width
            h = (y2 - y) * 1. / height
            line = "%s %.6f %.6f %.6f %.6f\n" % (label, cx, cy, w, h)
            lines.append(line)
        txt_name = filename.split('\\')[-1].replace(".xml", ".txt")
        label_dir = self.output_yolo_dir
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        with open(label_dir+'/'+txt_name, "w") as f:
            f.writelines(lines)

    def get_image_list(self, image_path, image_dir, suffix=['jpg', 'jpeg', 'JPG', 'JPEG', 'png']):
        '''get all image path ends with suffix'''
        input_image_dir = os.path.join(image_path, image_dir)
        if not os.path.exists(input_image_dir):
            print("PATH:%s not exists" % image_dir)
            return []
        imglist = []
        for root, sdirs, files in os.walk(input_image_dir):
            # root: 'E:/DataSets/S2TLD/S2TLD（1080x1920）\\JPEGImages'
            if not files:
                continue
            for filename in files:
                filepath = os.path.join(root, filename) + "\n"
                if filename.split('.')[-1] in suffix:
                    imglist.append(filepath)
        return imglist

    def imglist2file(self, imglist):
        random.shuffle(imglist)
        train_list = imglist[:-100]
        valid_list = imglist[-100:]
        with open("train.txt", "w") as f:
            f.writelines(train_list)
        with open("valid.txt", "w") as f:
            f.writelines(valid_list)


def data_set_split(src_data_folder1, target_data_folder1, label_flag=False, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    # ----------------------------------------- Dataset_divide -----------------------------------------------#
    """
        读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
        :param src_data_folder1:    源文件夹
        :param target_data_folder1: 目标文件夹
        :param train_scale:         训练集比例
        :param val_scale:           验证集比例
        :param test_scale:          测试集比例
        :return:
    """
    print("开始数据集划分")
    #  按照比例划分数据集, 并进行数据图片的复制
    current_class_data_path = src_data_folder1
    current_all_data = os.listdir(current_class_data_path)
    current_data_length = len(current_all_data)
    current_data_index_list = list(range(current_data_length))
    random.seed(7)
    random.shuffle(current_data_index_list)  # shuffle the image list

    if label_flag:
        out_dir = os.path.join(target_data_folder1, "labels")
    else:
        out_dir = os.path.join(target_data_folder1, "images")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    split_names = ['train', 'val', 'test']  # 在目标目录下创建文件夹
    for split_name in split_names:
        split_path = os.path.join(out_dir, split_name)
        if os.path.exists(split_path):
            pass
        else:
            os.mkdir(split_path)

    train_folder = os.path.join(out_dir, 'train')
    val_folder = os.path.join(out_dir, 'val')
    test_folder = os.path.join(out_dir, 'test')
    train_stop_flag = current_data_length * train_scale
    val_stop_flag = current_data_length * (train_scale + val_scale)
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    print("current dir ------ ", current_class_data_path)
    for i in tqdm(range(current_data_length), desc="splitting..."):
        src_img_path = os.path.join(current_class_data_path, current_all_data[current_data_index_list[i]])
        if current_idx <= train_stop_flag:
            copy2(src_img_path, train_folder)  # print("{}复制到了{}".format(src_img_path, train_folder))
            train_num = train_num + 1
        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            copy2(src_img_path, val_folder)  # print("{}复制到了{}".format(src_img_path, val_folder))
            val_num = val_num + 1
        else:
            copy2(src_img_path, test_folder)  # print("{}复制到了{}".format(src_img_path, test_folder))
            test_num = test_num + 1
        current_idx = current_idx + 1
    print("**********************************************************************")
    print(
        "{}按照{}：{}：{}的比例划分完成，一共{}".format(label_flag, train_scale, val_scale, test_scale, current_data_length))
    print("训练集{}：{}张".format(train_folder, train_num))
    print("验证集{}：{}张".format(val_folder, val_num))
    print("测试集{}：{}张".format(test_folder, test_num))


def combine_datasets():
    # 源文件夹路径
    # source_folder = "swg_camera_traps/images100~500_targets"
    source_folder = "G:/SWG/over400_train_val_test_classes"
    dest_folder = "G:/SWG/over400_train_val_test"

    data_categories = ['train', 'test', 'val']

    # 遍历源文件夹内的所有文件
    for cate in data_categories:  # train、test、val
        # 构建源文件的完整路径
        root_path = os.path.join(source_folder, cate)
        dest_path = os.path.join(dest_folder, cate)
        for d in os.listdir(root_path):  # image dir 类别和id区分
            img_dir_path = os.path.join(root_path, d)
            img_dir_list = os.listdir(img_dir_path)
            for i in tqdm(range(len(img_dir_list)), desc="{} dir combining...".format(d)):
                source_file = os.path.join(img_dir_path, img_dir_list[i])
                if os.path.isfile(source_file):  # images
                    # 调用shutil模块的函数将文件复制到目标文件夹内
                    shutil.copy(source_file, dest_path)


if __name__ == '__main__':

    dirs = ['S2TLD（1080x1920）', 'S2TLD（720x1280）']
    dirs2 = ['normal_1', 'normal_2']
    dirs3 = ['JPEGImages', 'Annotations']

    # path_in = 'E:/DataSets/S2TLD/{}'.format(dirs[1])
    # path_in = 'E:/DataSets/S2TLD/{}/{}'.format(dirs[1], dirs2[0])
    path_in = 'E:/DataSets/S2TLD/{}/{}'.format(dirs[1], dirs2[1])
    # path_out = 'E:/DataSets/S2TLD/yolo'

    # for item in dirs3:
    #     label_flag = False
    #     src_data_folder = path_in + '/{}'.format(item)
    #     tar_data_folder = r"E:\DataSets\S2TLD\S2TLD_SMALL2"  # 输出路径
    #     if item == 'Annotations':
    #         label_flag = True
    #
    #     data_set_split(src_data_folder, tar_data_folder, label_flag)

    '''
    

    # split images func1
    imglist = obj.get_image_list(image_path=path_in, image_dir=dirs[2])
    obj.imglist2file(imglist)
    '''
    xml_root = 'E:/DataSets/S2TLD/S2TLD_YOLO'
    xml_dir =xml_root + '/labels'
    for i in ['train', 'val', 'test']:
        dir_path = xml_dir + '/' + i
        obj = VOC2YOLO(input_voc_dir=xml_root,
                       output_yolo_dir='E:/DataSets/S2TLD/S2TLD_YOLO/labels_txt/'+i)
        # process labels
        xml_path_list = glob.glob(xml_dir + "/{}/*.xml".format(i))
        for xml_path in xml_path_list:
            # print(xml_path)
            obj.voc2yolo(xml_path)


