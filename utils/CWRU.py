from utils.tools import concatenate_data, one_hot, convert_one_hot
import scipy.io
import numpy as np
from os.path import join 

def load_CWRU(opt):
    Normal_0_train, Normal_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/normal/Normal_0.mat')))
    Normal_0_label = one_hot(0, 64)
    Normal_1_train, Normal_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/normal/Normal_1.mat')))
    Normal_1_label = one_hot(1, 64)
    Normal_2_train, Normal_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/normal/Normal_2.mat')))
    Normal_2_label = one_hot(2, 64)
    Normal_3_train, Normal_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/normal/Normal_3.mat')))
    Normal_3_label = one_hot(3, 64)

    B007_0_train, B007_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B007_0.mat')))
    B007_0_label = one_hot(4, 64)
    B007_1_train, B007_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B007_1.mat')))
    B007_1_label = one_hot(5, 64)
    B007_2_train, B007_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B007_2.mat')))
    B007_2_label = one_hot(6, 64)
    B007_3_train, B007_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B007_3.mat')))
    B007_3_label = one_hot(7, 64)

    B014_0_train, B014_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B014_0.mat')))
    B014_0_label = one_hot(8, 64)
    B014_1_train, B014_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B014_1.mat')))
    B014_1_label = one_hot(9, 64)
    B014_2_train, B014_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B014_2.mat')))
    B014_2_label = one_hot(10, 64)
    B014_3_train, B014_3_test= concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B014_3.mat')))
    B014_3_label = one_hot(11, 64)

    B021_0_train, B021_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B021_0.mat')))
    B021_0_label = one_hot(12, 64)
    B021_1_train, B021_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B021_1.mat')))
    B021_1_label = one_hot(13, 64)
    B021_2_train, B021_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B021_2.mat')))
    B021_2_label = one_hot(14, 64)
    B021_3_train, B021_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B021_3.mat')))
    B021_3_label = one_hot(15, 64)

    B028_0_train, B028_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B028_0.mat')))
    B028_0_label = one_hot(16, 64)
    B028_1_train, B028_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B028_1.mat')))
    B028_1_label = one_hot(17, 64)
    B028_2_train, B028_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B028_2.mat')))
    B028_2_label = one_hot(18, 64)
    B028_3_train, B028_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/B028_3.mat')))
    B028_3_label = one_hot(19, 64)

    IR007_0_train, IR007_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR007_0.mat')))
    IR007_0_label = one_hot(20, 64)
    IR007_1_train, IR007_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR007_1.mat')))
    IR007_1_label = one_hot(21, 64)
    IR007_2_train, IR007_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR007_2.mat')))
    IR007_2_label = one_hot(22, 64)
    IR007_3_train, IR007_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR007_3.mat')))
    IR007_3_label = one_hot(23, 64)

    IR014_0_train, IR014_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR014_0.mat')))
    IR014_0_label = one_hot(24, 64)
    IR014_1_train, IR014_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR014_1.mat')))
    IR014_1_label = one_hot(25, 64)
    IR014_2_train, IR014_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR014_2.mat')))
    IR014_2_label = one_hot(26, 64)
    IR014_3_train, IR014_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR014_3.mat')))
    IR014_3_label = one_hot(27, 64)

    IR021_0_train, IR021_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR021_0.mat')))
    IR021_0_label = one_hot(28, 64)
    IR021_1_train, IR021_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR021_1.mat')))
    IR021_1_label = one_hot(29, 64)
    IR021_2_train, IR021_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR021_2.mat')))
    IR021_2_label = one_hot(30, 64)
    IR021_3_train, IR021_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR021_3.mat')))
    IR021_3_label = one_hot(31, 64)

    IR028_0_train, IR028_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR028_0.mat')))
    IR028_0_label = one_hot(32, 64)
    IR028_1_train, IR028_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR028_1.mat')))
    IR028_1_label = one_hot(33, 64)
    IR028_2_train, IR028_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR028_2.mat')))
    IR028_2_label = one_hot(34, 64)
    IR028_3_train, IR028_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/IR028_3.mat')))
    IR028_3_label = one_hot(35, 64)

    OR007_12_0_train, OR007_12_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@12_0.mat')))
    OR007_12_0_label = one_hot(36, 64)
    OR007_12_1_train, OR007_12_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@12_1.mat')))
    OR007_12_1_label = one_hot(37, 64)
    OR007_12_2_train, OR007_12_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@12_2.mat')))
    OR007_12_2_label = one_hot(38, 64)
    OR007_12_3_train, OR007_12_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@12_3.mat')))
    OR007_12_3_label = one_hot(39, 64)

    OR007_3_0_train, OR007_3_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@3_0.mat')))
    OR007_3_0_label = one_hot(40, 64)
    OR007_3_1_train, OR007_3_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@3_1.mat')))
    OR007_3_1_label = one_hot(41, 64)
    OR007_3_2_train, OR007_3_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@3_2.mat')))
    OR007_3_2_label = one_hot(42, 64)
    OR007_3_3_train, OR007_3_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@3_3.mat')))
    OR007_3_3_label = one_hot(43, 64)

    OR007_6_0_train, OR007_6_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@6_0.mat')))
    OR007_6_0_label = one_hot(44, 64)
    OR007_6_1_train, OR007_6_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@6_1.mat')))
    OR007_6_1_label = one_hot(45, 64)
    OR007_6_2_train, OR007_6_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@6_2.mat')))
    OR007_6_2_label = one_hot(46, 64)
    OR007_6_3_train, OR007_6_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR007@6_3.mat')))
    OR007_6_3_label = one_hot(47, 64)

    OR014_6_0_train, OR014_6_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR014@6_0.mat')))
    OR014_6_0_label = one_hot(48, 64)
    OR014_6_1_train, OR014_6_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR014@6_1.mat')))
    OR014_6_1_label = one_hot(49, 64)
    OR014_6_2_train, OR014_6_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR014@6_2.mat')))
    OR014_6_2_label = one_hot(50, 64)
    OR014_6_3_train, OR014_6_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR014@6_3.mat')))
    OR014_6_3_label = one_hot(51, 64)

    OR021_6_0_train, OR021_6_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@6_0.mat')))
    OR021_6_0_label = one_hot(52, 64)
    OR021_6_1_train, OR021_6_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@6_1.mat')))
    OR021_6_1_label = one_hot(53, 64)
    OR021_6_2_train, OR021_6_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@6_2.mat')))
    OR021_6_2_label = one_hot(54, 64)
    OR021_6_3_train, OR021_6_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@6_3.mat')))
    OR021_6_3_label = one_hot(55, 64)

    OR021_3_0_train, OR021_3_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@3_0.mat')))
    OR021_3_0_label = one_hot(56, 64)
    OR021_3_1_train, OR021_3_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@3_1.mat')))
    OR021_3_1_label = one_hot(57, 64)
    OR021_3_2_train, OR021_3_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@3_2.mat')))
    OR021_3_2_label = one_hot(58, 64)
    OR021_3_3_train, OR021_3_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@3_3.mat')))
    OR021_3_3_label = one_hot(59, 64)

    OR021_12_0_train, OR021_12_0_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@12_0.mat')))
    OR021_12_0_label = one_hot(60, 64)
    OR021_12_1_train, OR021_12_1_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@12_1.mat')))
    OR021_12_1_label = one_hot(61, 64)
    OR021_12_2_train, OR021_12_2_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@12_2.mat')))
    OR021_12_2_label = one_hot(62, 64)
    OR021_12_3_train, OR021_12_3_test = concatenate_data(scipy.io.loadmat(join(opt.data_dir, 'data/12k/OR021@12_3.mat')))
    OR021_12_3_label = one_hot(63, 64)

    all_data_0_train = np.concatenate((Normal_0_train, IR007_0_train, B007_0_train, OR007_6_0_train, OR007_3_0_train, OR007_12_0_train))
    all_data_0_test = np.concatenate((Normal_0_test, IR007_0_test, B007_0_test, OR007_6_0_test, OR007_3_0_test, OR007_12_0_test))
    
    Normal_0_label_all_train = convert_one_hot(Normal_0_label) * Normal_0_train.shape[0]
    IR007_0_label_all_train = convert_one_hot(IR007_0_label) * IR007_0_train.shape[0]
    B007_0_label_all_train = convert_one_hot(B007_0_label) * B007_0_train.shape[0]
    OR007_6_0_label_all_train = convert_one_hot(OR007_6_0_label) * OR007_6_0_train.shape[0]
    OR007_3_0_label_all_train = convert_one_hot(OR007_3_0_label) * OR007_3_0_train.shape[0]
    OR007_12_0_label_all_train = convert_one_hot(OR007_12_0_label) * OR007_12_0_train.shape[0]
    all_labels_0_train = np.concatenate((Normal_0_label_all_train, IR007_0_label_all_train, B007_0_label_all_train, OR007_6_0_label_all_train, OR007_3_0_label_all_train, OR007_12_0_label_all_train))
    
    Normal_0_label_all_test = convert_one_hot(Normal_0_label) * Normal_0_test.shape[0] 
    IR007_0_label_all_test = convert_one_hot(IR007_0_label) * IR007_0_test.shape[0]
    B007_0_label_all_test = convert_one_hot(B007_0_label) * B007_0_test.shape[0]
    OR007_6_0_label_all_test = convert_one_hot(OR007_6_0_label) * OR007_6_0_test.shape[0]
    OR007_3_0_label_all_test = convert_one_hot(OR007_3_0_label) * OR007_3_0_test.shape[0]
    OR007_12_0_label_all_test = convert_one_hot(OR007_12_0_label) * OR007_12_0_test.shape[0]
    all_labels_0_test = np.concatenate((Normal_0_label_all_test, IR007_0_label_all_test, B007_0_label_all_test, OR007_6_0_label_all_test, OR007_3_0_label_all_test, OR007_12_0_label_all_test))

    if opt.CWRU_case == "1":
        X_train, X_test, y_train, y_test = all_data_0_train, all_data_0_test, all_labels_0_train, all_labels_0_test
    
    all_data_1_train = np.concatenate((Normal_1_train, IR007_1_train, B007_1_train, OR007_6_1_train, OR007_3_1_train, OR007_12_1_train))
    Normal_1_label_all_train = convert_one_hot(Normal_1_label) * Normal_1_train.shape[0]
    IR007_1_label_all_train = convert_one_hot(IR007_1_label) * IR007_1_train.shape[0]
    B007_1_label_all_train = convert_one_hot(B007_1_label) * B007_1_train.shape[0]
    OR007_6_1_label_all_train = convert_one_hot(OR007_6_1_label) * OR007_6_1_train.shape[0]
    OR007_3_1_label_all_train = convert_one_hot(OR007_3_1_label) * OR007_3_1_train.shape[0]
    OR007_12_1_label_all_train = convert_one_hot(OR007_12_1_label) * OR007_12_1_train.shape[0]
    all_labels_1_train = np.concatenate((Normal_1_label_all_train, IR007_1_label_all_train, B007_1_label_all_train, OR007_6_1_label_all_train, OR007_3_1_label_all_train, OR007_12_1_label_all_train))
  
    all_data_1_test = np.concatenate((Normal_1_test, IR007_1_test, B007_1_test, OR007_6_1_test, OR007_3_1_test, OR007_12_1_test))
    Normal_1_label_all_test = convert_one_hot(Normal_1_label) * Normal_1_test.shape[0]
    IR007_1_label_all_test = convert_one_hot(IR007_1_label) * IR007_1_test.shape[0]
    B007_1_label_all_test = convert_one_hot(B007_1_label) * B007_1_test.shape[0]
    OR007_6_1_label_all_test = convert_one_hot(OR007_6_1_label) * OR007_6_1_test.shape[0]
    OR007_3_1_label_all_test = convert_one_hot(OR007_3_1_label) * OR007_3_1_test.shape[0]
    OR007_12_1_label_all_test = convert_one_hot(OR007_12_1_label) * OR007_12_1_test.shape[0]
    all_labels_1_test = np.concatenate((Normal_1_label_all_test, IR007_1_label_all_test, B007_1_label_all_test, OR007_6_1_label_all_test, OR007_3_1_label_all_test, OR007_12_1_label_all_test))

    all_data_1_train = np.concatenate((Normal_1_train, IR007_1_train, B007_1_train, OR007_6_1_train, OR007_3_1_train, OR007_12_1_train))
    Normal_1_label_all_train = convert_one_hot(Normal_1_label) * Normal_1_train.shape[0]
    IR007_1_label_all_train = convert_one_hot(IR007_1_label) * IR007_1_train.shape[0]
    B007_1_label_all_train = convert_one_hot(B007_1_label) * B007_1_train.shape[0]
    OR007_6_1_label_all_train = convert_one_hot(OR007_6_1_label) * OR007_6_1_train.shape[0]
    OR007_3_1_label_all_train = convert_one_hot(OR007_3_1_label) * OR007_3_1_train.shape[0]
    OR007_12_1_label_all_train = convert_one_hot(OR007_12_1_label) * OR007_12_1_train.shape[0]
    all_labels_1_train = np.concatenate((Normal_1_label_all_train, IR007_1_label_all_train, B007_1_label_all_train, OR007_6_1_label_all_train, OR007_3_1_label_all_train, OR007_12_1_label_all_train))

    all_data_1_test = np.concatenate((Normal_1_test, IR007_1_test, B007_1_test, OR007_6_1_test, OR007_3_1_test, OR007_12_1_test))
    Normal_1_label_all_test = convert_one_hot(Normal_1_label) * Normal_1_test.shape[0]
    IR007_1_label_all_test = convert_one_hot(IR007_1_label) * IR007_1_test.shape[0]
    B007_1_label_all_test = convert_one_hot(B007_1_label) * B007_1_test.shape[0]
    OR007_6_1_label_all_test = convert_one_hot(OR007_6_1_label) * OR007_6_1_test.shape[0]
    OR007_3_1_label_all_test = convert_one_hot(OR007_3_1_label) * OR007_3_1_test.shape[0]
    OR007_12_1_label_all_test = convert_one_hot(OR007_12_1_label) * OR007_12_1_test.shape[0]
    all_labels_1_test = np.concatenate((Normal_1_label_all_test, IR007_1_label_all_test, B007_1_label_all_test, OR007_6_1_label_all_test, OR007_3_1_label_all_test, OR007_12_1_label_all_test))

    all_data_2_train = np.concatenate((Normal_2_train, IR007_2_train, B007_2_train, OR007_6_2_train, OR007_3_2_train, OR007_12_2_train))
    Normal_2_label_all_train = convert_one_hot(Normal_2_label) * Normal_2_train.shape[0]
    IR007_2_label_all_train = convert_one_hot(IR007_2_label) * IR007_2_train.shape[0]
    B007_2_label_all_train = convert_one_hot(B007_2_label) * B007_2_train.shape[0]
    OR007_6_2_label_all_train = convert_one_hot(OR007_6_2_label) * OR007_6_2_train.shape[0]
    OR007_3_2_label_all_train = convert_one_hot(OR007_3_2_label) * OR007_3_2_train.shape[0]
    OR007_12_2_label_all_train = convert_one_hot(OR007_12_2_label) * OR007_12_2_train.shape[0]
    all_labels_2_train = np.concatenate((Normal_2_label_all_train, IR007_2_label_all_train, B007_2_label_all_train, OR007_6_2_label_all_train, OR007_3_2_label_all_train, OR007_12_2_label_all_train))
    
    all_data_2_test = np.concatenate((Normal_2_test, IR007_2_test, B007_2_test, OR007_6_2_test, OR007_3_2_test, OR007_12_2_test))
    Normal_2_label_all_test = convert_one_hot(Normal_2_label) * Normal_2_test.shape[0]
    IR007_2_label_all_test = convert_one_hot(IR007_2_label) * IR007_2_test.shape[0]
    B007_2_label_all_test = convert_one_hot(B007_2_label) * B007_2_test.shape[0]
    OR007_6_2_label_all_test = convert_one_hot(OR007_6_2_label) * OR007_6_2_test.shape[0]
    OR007_3_2_label_all_test = convert_one_hot(OR007_3_2_label) * OR007_3_2_test.shape[0]
    OR007_12_2_label_all_test = convert_one_hot(OR007_12_2_label) * OR007_12_2_test.shape[0]
    all_labels_2_test = np.concatenate((Normal_2_label_all_test, IR007_2_label_all_test, B007_2_label_all_test, OR007_6_2_label_all_test, OR007_3_2_label_all_test, OR007_12_2_label_all_test))

    all_data_3_train = np.concatenate((Normal_3_train, IR007_3_train, B007_3_train, OR007_6_3_train, OR007_3_3_train, OR007_12_3_train))
    Normal_3_label_all_train = convert_one_hot(Normal_3_label) * Normal_3_train.shape[0]
    IR007_3_label_all_train = convert_one_hot(IR007_3_label) * IR007_3_train.shape[0]
    B007_3_label_all_train = convert_one_hot(B007_3_label) * B007_3_train.shape[0]
    OR007_6_3_label_all_train = convert_one_hot(OR007_6_3_label) * OR007_6_3_train.shape[0]
    OR007_3_3_label_all_train = convert_one_hot(OR007_3_3_label) * OR007_3_3_train.shape[0]
    OR007_12_3_label_all_train = convert_one_hot(OR007_12_3_label) * OR007_12_3_train.shape[0]
    all_labels_3_train = np.concatenate((Normal_3_label_all_train, IR007_3_label_all_train, B007_3_label_all_train, OR007_6_3_label_all_train, OR007_3_3_label_all_train, OR007_12_3_label_all_train))
    
    all_data_3_test = np.concatenate((Normal_3_test, IR007_3_test, B007_3_test, OR007_6_3_test, OR007_3_3_test, OR007_12_3_test))
    Normal_3_label_all_test = convert_one_hot(Normal_3_label) * Normal_3_test.shape[0]
    IR007_3_label_all_test = convert_one_hot(IR007_3_label) * IR007_3_test.shape[0]
    B007_3_label_all_test = convert_one_hot(B007_3_label) * B007_3_test.shape[0]
    OR007_6_3_label_all_test = convert_one_hot(OR007_6_3_label) * OR007_6_3_test.shape[0]
    OR007_3_3_label_all_test = convert_one_hot(OR007_3_3_label) * OR007_3_3_test.shape[0]
    OR007_12_3_label_all_test = convert_one_hot(OR007_12_3_label) * OR007_12_3_test.shape[0]
    all_labels_3_test = np.concatenate((Normal_3_label_all_test, IR007_3_label_all_test, B007_3_label_all_test, OR007_6_3_label_all_test, OR007_3_3_label_all_test, OR007_12_3_label_all_test))

    # ------------------------------------ Gather data ------------------------------------
    all_data_4_train = np.concatenate((all_data_0_train, all_data_1_train, all_data_2_train, all_data_3_train))
    all_labels_4_train = np.concatenate((all_labels_0_train, all_labels_1_train, all_labels_2_train, all_labels_3_train))
    
    all_data_4_test = np.concatenate((all_data_0_test, all_data_1_test, all_data_2_test, all_data_3_test))
    all_labels_4_test = np.concatenate((all_labels_0_test, all_labels_1_test, all_labels_2_test, all_labels_3_test))

    if opt.CWRU_case == "2":
        X_train, X_test, y_train, y_test = all_data_4_train, all_data_4_test, all_labels_4_train, all_labels_4_test
    
    
    all_data_12_train = np.concatenate((all_data_4_train, IR021_0_train, B021_0_train, OR021_6_0_train, OR021_3_0_train, OR021_12_0_train,
                                IR021_1_train, B021_1_train, OR021_6_1_train, OR021_3_1_train, OR021_12_1_train,
                                IR021_2_train, B021_2_train, OR021_6_2_train, OR021_3_2_train, OR021_12_2_train,
                                IR021_3_train, B021_3_train, OR021_6_3_train, OR021_3_3_train, OR021_12_3_train))

    IR021_0_label_all_train = convert_one_hot(IR021_0_label) * IR021_0_train.shape[0]
    B021_0_label_all_train = convert_one_hot(B021_0_label) * B021_0_train.shape[0]
    OR021_6_0_label_all_train = convert_one_hot(OR021_6_0_label) * OR021_6_0_train.shape[0]
    OR021_3_0_label_all_train = convert_one_hot(OR021_3_0_label) * OR021_3_0_train.shape[0]
    OR021_12_0_label_all_train = convert_one_hot(OR021_12_0_label) * OR021_12_0_train.shape[0]
    
    IR021_1_label_all_train = convert_one_hot(IR021_1_label) * IR021_1_train.shape[0]
    B021_1_label_all_train = convert_one_hot(B021_1_label) * B021_1_train.shape[0]
    OR021_6_1_label_all_train = convert_one_hot(OR021_6_1_label) * OR021_6_1_train.shape[0]
    OR021_3_1_label_all_train = convert_one_hot(OR021_3_1_label) * OR021_3_1_train.shape[0]
    OR021_12_1_label_all_train = convert_one_hot(OR021_12_1_label) * OR021_12_1_train.shape[0]
    
    IR021_2_label_all_train = convert_one_hot(IR021_2_label) * IR021_2_train.shape[0]
    B021_2_label_all_train = convert_one_hot(B021_2_label) * B021_2_train.shape[0]
    OR021_6_2_label_all_train = convert_one_hot(OR021_6_2_label) * OR021_6_2_train.shape[0]
    OR021_3_2_label_all_train = convert_one_hot(OR021_3_2_label) * OR021_3_2_train.shape[0]
    OR021_12_2_label_all_train = convert_one_hot(OR021_12_2_label) * OR021_12_2_train.shape[0]
    
    IR021_3_label_all_train = convert_one_hot(IR021_3_label) * IR021_3_train.shape[0]
    B021_3_label_all_train = convert_one_hot(B021_3_label) * B021_3_train.shape[0]
    OR021_6_3_label_all_train = convert_one_hot(OR021_6_3_label) * OR021_6_3_train.shape[0]
    OR021_3_3_label_all_train = convert_one_hot(OR021_3_3_label) * OR021_3_3_train.shape[0]
    OR021_12_3_label_all_train = convert_one_hot(OR021_12_3_label) * OR021_12_3_train.shape[0]
    
    all_labels_12_train = np.concatenate((all_labels_4_train, IR021_0_label_all_train, B021_0_label_all_train, OR021_6_0_label_all_train, OR021_3_0_label_all_train, OR021_12_0_label_all_train,
                                    IR021_1_label_all_train, B021_1_label_all_train, OR021_6_1_label_all_train, OR021_3_1_label_all_train, OR021_12_1_label_all_train,
                                    IR021_2_label_all_train, B021_2_label_all_train, OR021_6_2_label_all_train, OR021_3_2_label_all_train, OR021_12_2_label_all_train,
                                    IR021_3_label_all_train, B021_3_label_all_train, OR021_6_3_label_all_train, OR021_3_3_label_all_train, OR021_12_3_label_all_train))
    
    
    all_data_12_test = np.concatenate((all_data_4_test, IR021_0_test, B021_0_test, OR021_6_0_test, OR021_3_0_test, OR021_12_0_test,
                                IR021_1_test, B021_1_test, OR021_6_1_test, OR021_3_1_test, OR021_12_1_test,
                                IR021_2_test, B021_2_test, OR021_6_2_test, OR021_3_2_test, OR021_12_2_test,
                                IR021_3_test, B021_3_test, OR021_6_3_test, OR021_3_3_test, OR021_12_3_test))
    
    IR021_0_label_all_test = convert_one_hot(IR021_0_label) * IR021_0_test.shape[0]
    B021_0_label_all_test = convert_one_hot(B021_0_label) * B021_0_test.shape[0]
    OR021_6_0_label_all_test = convert_one_hot(OR021_6_0_label) * OR021_6_0_test.shape[0]
    OR021_3_0_label_all_test = convert_one_hot(OR021_3_0_label) * OR021_3_0_test.shape[0]
    OR021_12_0_label_all_test = convert_one_hot(OR021_12_0_label) * OR021_12_0_test.shape[0]
    
    IR021_1_label_all_test = convert_one_hot(IR021_1_label) * IR021_1_test.shape[0]
    B021_1_label_all_test = convert_one_hot(B021_1_label) * B021_1_test.shape[0]
    OR021_6_1_label_all_test = convert_one_hot(OR021_6_1_label) * OR021_6_1_test.shape[0]
    OR021_3_1_label_all_test = convert_one_hot(OR021_3_1_label) * OR021_3_1_test.shape[0]
    OR021_12_1_label_all_test = convert_one_hot(OR021_12_1_label) * OR021_12_1_test.shape[0]
    
    IR021_2_label_all_test = convert_one_hot(IR021_2_label) * IR021_2_test.shape[0]
    B021_2_label_all_test = convert_one_hot(B021_2_label) * B021_2_test.shape[0]
    OR021_6_2_label_all_test = convert_one_hot(OR021_6_2_label) * OR021_6_2_test.shape[0]
    OR021_3_2_label_all_test = convert_one_hot(OR021_3_2_label) * OR021_3_2_test.shape[0]
    OR021_12_2_label_all_test = convert_one_hot(OR021_12_2_label) * OR021_12_2_test.shape[0]
    
    IR021_3_label_all_test = convert_one_hot(IR021_3_label) * IR021_3_test.shape[0]
    B021_3_label_all_test = convert_one_hot(B021_3_label) * B021_3_test.shape[0]
    OR021_6_3_label_all_test = convert_one_hot(OR021_6_3_label) * OR021_6_3_test.shape[0]
    OR021_3_3_label_all_test = convert_one_hot(OR021_3_3_label) * OR021_3_3_test.shape[0]
    OR021_12_3_label_all_test = convert_one_hot(OR021_12_3_label) * OR021_12_3_test.shape[0]
    
    all_labels_12_test = np.concatenate((all_labels_4_test, IR021_0_label_all_test, B021_0_label_all_test, OR021_6_0_label_all_test, OR021_3_0_label_all_test, OR021_12_0_label_all_test,
                                    IR021_1_label_all_test, B021_1_label_all_test, OR021_6_1_label_all_test, OR021_3_1_label_all_test, OR021_12_1_label_all_test,
                                    IR021_2_label_all_test, B021_2_label_all_test, OR021_6_2_label_all_test, OR021_3_2_label_all_test, OR021_12_2_label_all_test,
                                    IR021_3_label_all_test, B021_3_label_all_test, OR021_6_3_label_all_test, OR021_3_3_label_all_test, OR021_12_3_label_all_test))
    

    # ------------------------------------ Gather data ------------------------------------
    all_data_14_train = np.concatenate((all_data_12_train, IR014_0_train, IR014_1_train, IR014_2_train, IR014_3_train,
                                B014_0_train, B014_1_train, B014_2_train, B014_3_train,
                                OR014_6_0_train, OR014_6_1_train, OR014_6_2_train, OR014_6_3_train,
                                IR028_0_train, 	IR028_1_train, 	IR028_2_train, 	IR028_3_train,
                                B028_0_train, B028_1_train, B028_2_train, B028_3_train))
    
    IR014_0_label_all_train = convert_one_hot(IR014_0_label) * IR014_0_train.shape[0]
    IR014_1_label_all_train = convert_one_hot(IR014_1_label) * IR014_1_train.shape[0]
    IR014_2_label_all_train = convert_one_hot(IR014_2_label) * IR014_2_train.shape[0]
    IR014_3_label_all_train = convert_one_hot(IR014_3_label) * IR014_3_train.shape[0]

    B014_0_label_all_train = convert_one_hot(B014_0_label) * B014_0_train.shape[0]
    B014_1_label_all_train = convert_one_hot(B014_1_label) * B014_1_train.shape[0]
    B014_2_label_all_train = convert_one_hot(B014_2_label) * B014_2_train.shape[0]
    B014_3_label_all_train = convert_one_hot(B014_3_label) * B014_3_train.shape[0]

    OR014_6_0_label_all_train = convert_one_hot(OR014_6_0_label) * OR014_6_0_train.shape[0]
    OR014_6_1_label_all_train = convert_one_hot(OR014_6_1_label) * OR014_6_1_train.shape[0]
    OR014_6_2_label_all_train = convert_one_hot(OR014_6_2_label) * OR014_6_2_train.shape[0]
    OR014_6_3_label_all_train = convert_one_hot(OR014_6_3_label) * OR014_6_3_train.shape[0]

    IR028_0_label_all_train = convert_one_hot(IR028_0_label) * IR028_0_train.shape[0]
    IR028_1_label_all_train = convert_one_hot(IR028_1_label) * IR028_1_train.shape[0]
    IR028_2_label_all_train = convert_one_hot(IR028_2_label) * IR028_2_train.shape[0]
    IR028_3_label_all_train = convert_one_hot(IR028_3_label) * IR028_3_train.shape[0]

    B028_0_label_all_train = convert_one_hot(B028_0_label) * B028_0_train.shape[0]
    B028_1_label_all_train = convert_one_hot(B028_1_label) * B028_1_train.shape[0]
    B028_2_label_all_train = convert_one_hot(B028_2_label) * B028_2_train.shape[0]
    B028_3_label_all_train = convert_one_hot(B028_3_label) * B028_3_train.shape[0]
    
    all_labels_14_train = np.concatenate((all_labels_12_train, IR014_0_label_all_train, IR014_1_label_all_train,  IR014_2_label_all_train, IR014_3_label_all_train,
                                        B014_0_label_all_train,    B014_1_label_all_train,    B014_2_label_all_train,    B014_3_label_all_train,
                                        OR014_6_0_label_all_train, OR014_6_1_label_all_train, OR014_6_2_label_all_train, OR014_6_3_label_all_train,
                                        IR028_0_label_all_train,   IR028_1_label_all_train,   IR028_2_label_all_train,   IR028_3_label_all_train,
                                        B028_0_label_all_train,    B028_1_label_all_train,    B028_2_label_all_train,    B028_3_label_all_train))


    all_data_14_test = np.concatenate((all_data_12_test, IR014_0_test, IR014_1_test, IR014_2_test, IR014_3_test,
                                B014_0_test, B014_1_test, B014_2_test, B014_3_test,
                                OR014_6_0_test, OR014_6_1_test, OR014_6_2_test, OR014_6_3_test,
                                IR028_0_test, 	IR028_1_test, 	IR028_2_test, 	IR028_3_test,
                                B028_0_test, B028_1_test, B028_2_test, B028_3_test))
    
    IR014_0_label_all_test = convert_one_hot(IR014_0_label) * IR014_0_test.shape[0]
    IR014_1_label_all_test = convert_one_hot(IR014_1_label) * IR014_1_test.shape[0]
    IR014_2_label_all_test = convert_one_hot(IR014_2_label) * IR014_2_test.shape[0]
    IR014_3_label_all_test = convert_one_hot(IR014_3_label) * IR014_3_test.shape[0]

    B014_0_label_all_test = convert_one_hot(B014_0_label) * B014_0_test.shape[0]
    B014_1_label_all_test = convert_one_hot(B014_1_label) * B014_1_test.shape[0]
    B014_2_label_all_test = convert_one_hot(B014_2_label) * B014_2_test.shape[0]
    B014_3_label_all_test = convert_one_hot(B014_3_label) * B014_3_test.shape[0]

    OR014_6_0_label_all_test = convert_one_hot(OR014_6_0_label) * OR014_6_0_test.shape[0]
    OR014_6_1_label_all_test = convert_one_hot(OR014_6_1_label) * OR014_6_1_test.shape[0]
    OR014_6_2_label_all_test = convert_one_hot(OR014_6_2_label) * OR014_6_2_test.shape[0]
    OR014_6_3_label_all_test = convert_one_hot(OR014_6_3_label) * OR014_6_3_test.shape[0]

    IR028_0_label_all_test = convert_one_hot(IR028_0_label) * IR028_0_test.shape[0]
    IR028_1_label_all_test = convert_one_hot(IR028_1_label) * IR028_1_test.shape[0]
    IR028_2_label_all_test = convert_one_hot(IR028_2_label) * IR028_2_test.shape[0]
    IR028_3_label_all_test = convert_one_hot(IR028_3_label) * IR028_3_test.shape[0]

    B028_0_label_all_test = convert_one_hot(B028_0_label) * B028_0_test.shape[0]
    B028_1_label_all_test = convert_one_hot(B028_1_label) * B028_1_test.shape[0]
    B028_2_label_all_test = convert_one_hot(B028_2_label) * B028_2_test.shape[0]
    B028_3_label_all_test = convert_one_hot(B028_3_label) * B028_3_test.shape[0]
    
    all_labels_14_test = np.concatenate((all_labels_12_test,       IR014_0_label_all_test,   IR014_1_label_all_test,  IR014_2_label_all_test, IR014_3_label_all_test,
                                    B014_0_label_all_test,    B014_1_label_all_test,    B014_2_label_all_test,    B014_3_label_all_test,
                                    OR014_6_0_label_all_test, OR014_6_1_label_all_test, OR014_6_2_label_all_test, OR014_6_3_label_all_test,
                                    IR028_0_label_all_test,   IR028_1_label_all_test,   IR028_2_label_all_test,   IR028_3_label_all_test,
                                    B028_0_label_all_test,    B028_1_label_all_test,    B028_2_label_all_test,    B028_3_label_all_test))
    if opt.CWRU_case == "3":
        X_train, X_test, y_train, y_test = all_data_14_train, all_data_14_test, all_labels_14_train, all_labels_14_test

    Normal_5_train = np.concatenate((Normal_0_train, Normal_1_train, Normal_2_train, Normal_3_train))
    IR007_5_train = np.concatenate((IR007_0_train, IR007_1_train, IR007_2_train, IR007_3_train))
    B007_5_train = np.concatenate((B007_0_train, B007_1_train, B007_2_train, B007_3_train))
    OR007_6_5_train = np.concatenate((OR007_6_0_train, OR007_6_1_train, OR007_6_2_train, OR007_6_3_train))
    OR007_3_5_train = np.concatenate((OR007_3_0_train, OR007_3_1_train, OR007_3_2_train, OR007_3_3_train))
    OR007_12_5_train = np.concatenate((OR007_12_0_train, OR007_12_1_train, OR007_12_2_train, OR007_12_3_train))

    Normal_5_label_all_train = convert_one_hot(Normal_0_label) * Normal_5_train.shape[0]
    
    Normal_5_test = np.concatenate((Normal_0_test, Normal_1_test, Normal_2_test, Normal_3_test))
    IR007_5_test = np.concatenate((IR007_0_test, IR007_1_test, IR007_2_test, IR007_3_test))
    B007_5_test = np.concatenate((B007_0_test, B007_1_test, B007_2_test, B007_3_test))
    OR007_6_5_test = np.concatenate((OR007_6_0_test, OR007_6_1_test, OR007_6_2_test, OR007_6_3_test))
    OR007_3_5_test = np.concatenate((OR007_3_0_test, OR007_3_1_test, OR007_3_2_test, OR007_3_3_test))
    OR007_12_5_test = np.concatenate((OR007_12_0_test, OR007_12_1_test, OR007_12_2_test, OR007_12_3_test))

    Normal_5_label_all_test = convert_one_hot(Normal_0_label) * Normal_5_test.shape[0]

    IR021_13_train    = np.concatenate((IR007_5_train, IR021_0_train, IR021_1_train, IR021_2_train, IR021_3_train))
    B021_13_train     = np.concatenate((B007_5_train, B021_0_train, B007_1_train, B021_2_train, B021_3_train))
    OR021_6_13_train  = np.concatenate((OR007_6_5_train, OR021_6_0_train, OR021_6_1_train, OR021_6_2_train, OR021_6_3_train))
    OR021_3_13_train  = np.concatenate((OR007_3_5_train, OR021_3_0_train, OR021_3_1_train, OR021_3_2_train, OR021_3_3_train))
    OR021_12_13_train = np.concatenate((OR007_12_5_train, OR021_12_0_train, OR021_12_1_train, OR021_12_2_train, OR021_12_3_train))

    all_data_13_train = np.concatenate((Normal_5_train, IR021_13_train, B021_13_train, OR021_6_13_train, OR021_3_13_train, OR021_12_13_train))
    
    IR021_13_label_all_train = convert_one_hot(IR007_0_label) * IR021_13_train.shape[0]
    B021_13_label_all_train = convert_one_hot(B007_0_label) * B021_13_train.shape[0]
    OR021_6_13_label_all_train = convert_one_hot(OR007_6_0_label) * OR021_6_13_train.shape[0]
    OR021_3_13_label_all_train = convert_one_hot(OR007_3_0_label) * OR021_3_13_train.shape[0]
    OR021_12_13_label_all_train = convert_one_hot(OR007_12_0_label) * OR021_12_13_train.shape[0]
    all_labels_13_train = np.concatenate((Normal_5_label_all_train, IR021_13_label_all_train, B021_13_label_all_train, OR021_6_13_label_all_train, OR021_3_13_label_all_train, OR021_12_13_label_all_train))

    # ------------------------------------ Gather data ------------------------------------
    IR021_13_test    = np.concatenate((IR007_5_test, IR021_0_test, IR021_1_test, IR021_2_test, IR021_3_test))
    B021_13_test     = np.concatenate((B007_5_test, B021_0_test, B007_1_test, B021_2_test, B021_3_test))
    OR021_6_13_test  = np.concatenate((OR007_6_5_test, OR021_6_0_test, OR021_6_1_test, OR021_6_2_test, OR021_6_3_test))
    OR021_3_13_test  = np.concatenate((OR007_3_5_test, OR021_3_0_test, OR021_3_1_test, OR021_3_2_test, OR021_3_3_test))
    OR021_12_13_test = np.concatenate((OR007_12_5_test, OR021_12_0_test, OR021_12_1_test, OR021_12_2_test, OR021_12_3_test))

    all_data_13_test = np.concatenate((Normal_5_test, IR021_13_test, B021_13_test, OR021_6_13_test, OR021_3_13_test, OR021_12_13_test))
    
    IR021_13_label_all_test = convert_one_hot(IR007_0_label) * IR021_13_test.shape[0]
    B021_13_label_all_test = convert_one_hot(B007_0_label) * B021_13_test.shape[0]
    OR021_6_13_label_all_test = convert_one_hot(OR007_6_0_label) * OR021_6_13_test.shape[0]
    OR021_3_13_label_all_test = convert_one_hot(OR007_3_0_label) * OR021_3_13_test.shape[0]
    OR021_12_13_label_all_test = convert_one_hot(OR007_12_0_label) * OR021_12_13_test.shape[0]
    all_labels_13_test = np.concatenate((Normal_5_label_all_test, IR021_13_label_all_test, B021_13_label_all_test, OR021_6_13_label_all_test, OR021_3_13_label_all_test, OR021_12_13_label_all_test))

    if opt.CWRU_case == "4":
        X_train, X_test, y_train, y_test = all_data_13_train, all_data_13_test, all_labels_13_train, all_labels_13_test
    
    return X_train, X_test, y_train, y_test