import numpy as np
from os.path import join
from utils.tools import load_PU_data, load_table_10

def load_PU_table(opt):
    if opt.PU_table_8:
        print('\nLoad Training ---------------------------------------------\n')
        K002 = load_PU_data(join(opt.data_dir, 'K002'), opt)
        
        KA01 = load_PU_data(join(opt.data_dir, 'KA01'), opt)
        KA05 = load_PU_data(join(opt.data_dir, 'KA05'), opt)
        KA07 = load_PU_data(join(opt.data_dir, 'KA07'), opt)

        KI01 = load_PU_data(join(opt.data_dir, 'KI01'), opt)
        KI05 = load_PU_data(join(opt.data_dir, 'KI05'), opt)
        KI07 = load_PU_data(join(opt.data_dir, 'KI07'), opt)

        print('Load Testing ---------------------------------------------\n')
        K001 = load_PU_data(join(opt.data_dir, 'K001'), opt)
        
        KA22 = load_PU_data(join(opt.data_dir, 'KA22'), opt)
        KA04 = load_PU_data(join(opt.data_dir, 'KA04'), opt)
        KA15 = load_PU_data(join(opt.data_dir, 'KA15'), opt)
        KA30 = load_PU_data(join(opt.data_dir, 'KA30'), opt)
        KA16 = load_PU_data(join(opt.data_dir, 'KA16'), opt)

        KI14 = load_PU_data(join(opt.data_dir, 'KI14'), opt)
        KI21 = load_PU_data(join(opt.data_dir, 'KI21'), opt)
        KI17 = load_PU_data(join(opt.data_dir, 'KI17'), opt)
        KI18 = load_PU_data(join(opt.data_dir, 'KI18'), opt)
        KI16 = load_PU_data(join(opt.data_dir, 'KI16'), opt)
        
        print('Load all---------------------------------------------\n')
        min_ = np.min((K002.shape[1], KA01.shape[1], KA07.shape[1], KI01.shape[1], KI05.shape[1], KI07.shape[1],\
                        K001.shape[1], KA22.shape[1], KA04.shape[1], KA15.shape[1], KA30.shape[1], KA16.shape[1],\
                        KA05.shape[1], KI14.shape[1], KI21.shape[1], KI17.shape[1], KI18.shape[1], KI16.shape[1]))

        Healthy_train = K002[:, :min_]
        Healthy_train_label = np.array([0]*len(Healthy_train))

        OR_Damage_train = np.concatenate((KA01[:, :min_], KA05[:, :min_], KA07[:, :min_]))
        OR_Damage_train_label = np.array([1]*len(OR_Damage_train))

        IR_Damage_train = np.concatenate((KI01[:, :min_], KI05[:, :min_], KI07[:, :min_]))
        IR_Damage_train_label = np.array([2]*len(IR_Damage_train))

        Healthy_test = K001[:, :min_]
        Healthy_test_label = np.array([0]*len(Healthy_test))

        OR_Damage_test = np.concatenate((KA22[:, :min_], KA04[:, :min_], KA15[:, :min_], KA30[:, :min_], KA16[:, :min_]))
        OR_Damage_test_label = np.array([1]*len(OR_Damage_test))

        IR_Damage_test = np.concatenate((KI14[:, :min_], KI21[:, :min_], KI17[:, :min_], KI18[:, :min_], KI16[:, :min_]))
        IR_Damage_test_label = np.array([2]*len(IR_Damage_test))

        X_train = np.concatenate((Healthy_train, OR_Damage_train, IR_Damage_train))
        y_train = np.concatenate((Healthy_train_label, OR_Damage_train_label, IR_Damage_train_label))
        X_test = np.concatenate((Healthy_test, OR_Damage_test, IR_Damage_test))
        y_test = np.concatenate((Healthy_test_label, OR_Damage_test_label, IR_Damage_test_label))
        return X_train, y_train, X_test, y_test

def load_PU_data_10(opt):
    if opt.PU_table_10:
        print('Loading Healthy data...')
        K001 = load_PU_data(join(opt.data_dir, 'K001'), opt, 255990)
        K002 = load_PU_data(join(opt.data_dir, 'K002'), opt, 255990)
        K003 = load_PU_data(join(opt.data_dir, 'K003'), opt, 255990)
        K004 = load_PU_data(join(opt.data_dir, 'K004'), opt, 255990)
        K005 = load_PU_data(join(opt.data_dir, 'K005'), opt, 255990)
        
        print('Loading Outer ring damage...')
        KA04 = load_PU_data(join(opt.data_dir, 'KA04'), opt, 255990)
        KA15 = load_PU_data(join(opt.data_dir, 'KA15'), opt, 255990)
        KA16 = load_PU_data(join(opt.data_dir, 'KA16'), opt, 255990)
        KA22 = load_PU_data(join(opt.data_dir, 'KA22'), opt, 255990)
        KA30 = load_PU_data(join(opt.data_dir, 'KA30'), opt, 255990)
        
        print('Loading Inner ring damage...')
        KI04 = load_PU_data(join(opt.data_dir, 'KI04'), opt, 255990)
        KI14 = load_PU_data(join(opt.data_dir, 'KI14'), opt, 255990)
        KI16 = load_PU_data(join(opt.data_dir, 'KI16'), opt, 255990)
        KI18 = load_PU_data(join(opt.data_dir, 'KI18'), opt, 255990)
        KI21 = load_PU_data(join(opt.data_dir, 'KI21'), opt, 255990)
        
        print('Loading all data...')
        Healthy = np.concatenate((K001, K002, K003, K004, K005))
        Outer_ring_damage = np.concatenate((KA04, KA15, KA16, KA22, KA30))
        Inner_ring_damage = np.concatenate((KI04, KI14, KI16, KI18, KI21))
        return Healthy, Outer_ring_damage, Inner_ring_damage