import os
import random
import sys

import numpy as np
import pandas as pd
import talib
from talib import SUM
from tqdm import tqdm


def TD(CLOSE, HIGH, LOW, VOL, N, N1):
    C = CLOSE
    H = HIGH
    L = LOW
    V = VOL
    HTD = H * V
    LTD = L * V
    CTD = C * V
    VTD = SUM(V, N)
    AC = SUM(CTD, N)

    VTD1 = SUM(V, N1)
    AH1 = SUM(HTD, N1)
    AL1 = SUM(LTD, N1)
    AC1 = SUM(CTD, N1)

    HH = AH1 / VTD1
    LL = AL1 / VTD1
    CC = AC1 / VTD1
    CC_L = AC / VTD

    return HH, LL, CC, CC_L


def preprocess_data(data):
    # 选择需要的列作为特征
    selected_features = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg',
                         'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']

    # 提取选定的特征列
    data = data[selected_features].copy()

    # 删除包含NaN值的行
    data = data.dropna()

    # 对日期进行排序（如果数据未按日期顺序排列）
    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by='date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # 计算
    data['HH'], data['LL'], data['CC'], data['CC_L'] = TD(data['close'], data['high'], data['low'], data['volume'],
                                                          N=20, N1=60)

    # 计算MACD指标
    data['Macd'], data['Macd_signal'], data['Macd_hist'] = talib.MACD(data['close'], fastperiod=7, slowperiod=35,
                                                                      signalperiod=6)

    # 计算VOL指标
    data['Vol_120_sma'] = data['volume'].rolling(window=120).mean()
    data['Vol_65_sma'] = data['volume'].rolling(window=65).mean()
    data['Vol_20_sma'] = data['volume'].rolling(window=20).mean()

    # 计算CCI指标
    data['Cci'] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=14)

    # 计算BOLL指标
    data['Boll_upper'], data['Boll_middle'], data['Boll_lower'] = talib.BBANDS(data['close'], timeperiod=35, nbdevup=2,
                                                                               nbdevdn=2, matype=0)

    # 计算DMI指标
    data['Dmi_di1'], data['Dmi_di2'] = talib.PLUS_DI(data['high'], data['low'], data['close'],
                                                     timeperiod=14), talib.MINUS_DI(data['high'], data['low'],
                                                                                    data['close'], timeperiod=14)
    data['Dmi_adx'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
    data['Dmi_adxr'] = talib.ADXR(data['high'], data['low'], data['close'], timeperiod=6)

    # 计算close的MA指标
    ma_periods = [7, 12, 22, 35, 65, 135, 250]
    for period in ma_periods:
        col_name = f'close_MA_{period}'
        data[col_name] = data['close'].rolling(window=period, min_periods=period).mean()

    # 删除包含NaN值的行
    data = data.dropna()

    return data


def preprocess_train_data(folder_path):
    # 创建一个空的DataFrame用于存储所有处理过的数据
    all_data = pd.DataFrame()

    files = os.listdir(folder_path)
    # 随机打乱文件顺序
    random.shuffle(files)
    for file in tqdm(files):
        input_file = os.path.join(folder_path, file)

        # 读取数据文件，并将数值类型的列转换为单精度浮点数
        df = pd.read_csv(input_file)

        # 对数据进行添加指标及去NAN 预处理
        df_preprocessed = preprocess_data(df)

        # 将"code", "date"列以外的数据转换为float32.
        columns_to_convert = [col for col in df_preprocessed.columns if col not in ["code", "date"]]
        df_preprocessed[columns_to_convert] = df_preprocessed[columns_to_convert].astype('float32')

        # # 需要归一化的特性
        # columns = {
        #     'close',
        #     'high',
        #     'low',
        #     'open',
        #     'volume',
        #     'amount',
        #     'turn',
        #     'pctChg',
        #     'peTTM',
        #     'pbMRQ',
        #     'psTTM',
        #     'pcfNcfTTM',
        #     'Macd',
        #     'Macd_signal',
        #     'Macd_hist',
        #     'Vol_120_sma',
        #     'Vol_65_sma',
        #     'Vol_20_sma',
        #     'Cci',
        #     'Boll_upper',
        #     'Boll_middle',
        #     'Boll_lower',
        #     'Dmi_di1',
        #     'Dmi_di2',
        #     'Dmi_adx',
        #     'Dmi_adxr',
        #     'close_MA_7',
        #     'close_MA_12',
        #     'close_MA_22',
        #     'close_MA_35',
        #     'close_MA_65',
        #     'close_MA_135',
        #     'close_MA_250',
        #     'HH',
        #     'CC',
        #     'LL',
        #     'CC_L'
        # }
        #
        # # 进行归一化处理，以最大值和最小值的偏差来做归一化处理
        # for column in columns:
        #     if df_preprocessed[column].dtype == 'float32':
        #         if column == 'close':
        #             df_preprocessed['real_close'] = df_preprocessed[column]
        #         df_preprocessed[column] = (df_preprocessed[column] - df_preprocessed[column].min()) / (
        #                 df_preprocessed[column].max() - df_preprocessed[column].min() + 0.0001)  # + 0.0001防止除0

        # # 对数据长度进行裁切，以250的倍数只保留最近的数据
        # rows_to_keep = len(df_preprocessed) // 250 * 250
        # df_preprocessed = df_preprocessed.tail(rows_to_keep)

        # 随机抽取每支股票数据的连续250天数据
        if len(df_preprocessed) >= 750:
            start_index = np.random.randint(0, len(df_preprocessed) - 250 + 1)
            df_preprocessed = df_preprocessed.iloc[start_index: start_index + 250]
        else:
            df_preprocessed = None

        # 如果df_preprocessed不为None，将处理过的数据添加到all_data中
        if df_preprocessed is not None:
            all_data = pd.concat([all_data, df_preprocessed])

    # 检查数据中是否有NaN值
    if all_data.isnull().values.any():
        print("数据中有NaN值")
    else:
        print("数据中没有NaN值")

    # 打印每一列中NaN值的数量：
    print("每一列中NaN值的数量：")
    print(all_data.isnull().sum())

    # 将所有处理过的数据保存为PKL文件，重置索引
    all_data.reset_index(drop=True, inplace=True)
    all_data.to_pickle(f'{Output}/stock_data_non_nor.pkl')
    all_data.to_csv(f'{Output}/stock_data_non_nor.csv', index=False)
    print("训练数据已经预处理并保存")


def preprocess_test_data(test_data, start_date, end_date):
    files = os.listdir(test_data)
    for file in tqdm(files):
        input_file = os.path.join(test_data, file)

        # 读取数据文件，并将数值类型的列转换为单精度浮点数
        df = pd.read_csv(input_file)

        # 对数据进行添加指标及去NAN 预处理
        df_preprocessed = preprocess_data(df)

        # 将"code", "date"列以外的数据转换为float32.
        columns_to_convert = [col for col in df_preprocessed.columns if col not in ["code", "date"]]
        df_preprocessed[columns_to_convert] = df_preprocessed[columns_to_convert].astype('float32')

        # 需要归一化的特性
        columns = {
            'close',
            'high',
            'low',
            'open',
            'volume',
            'amount',
            'turn',
            'pctChg',
            'peTTM',
            'pbMRQ',
            'psTTM',
            'pcfNcfTTM',
            'Macd',
            'Macd_signal',
            'Macd_hist',
            'Vol_120_sma',
            'Vol_65_sma',
            'Vol_20_sma',
            'Cci',
            'Boll_upper',
            'Boll_middle',
            'Boll_lower',
            'Dmi_di1',
            'Dmi_di2',
            'Dmi_adx',
            'Dmi_adxr',
            'close_MA_7',
            'close_MA_12',
            'close_MA_22',
            'close_MA_35',
            'close_MA_65',
            'close_MA_135',
            'close_MA_250',
            'HH',
            'CC',
            'LL',
            'CC_L'
        }

        # 进行归一化处理，以最大值和最小值的偏差来做归一化处理
        for column in columns:
            if df_preprocessed[column].dtype == 'float32':
                if column == 'close':
                    df_preprocessed['real_close'] = df_preprocessed[column]
                df_preprocessed[column] = (df_preprocessed[column] - df_preprocessed[column].min()) / (
                        df_preprocessed[column].max() - df_preprocessed[column].min() + 0.0001)  # + 0.0001防止除0
        # 根据日期范围对数据进行筛选
        df_preprocessed = df_preprocessed[
            (df_preprocessed['date'] >= start_date) & (df_preprocessed['date'] <= end_date)]
        df_preprocessed.to_csv(f'{pre_test_data}/{file}')
        print("测试文件已经预处理并保存")


if __name__ == '__main__':
    stock_data = './stock_data/'
    Output = './temp/'
    test_data = './test_data'
    pre_test_data = './pre_test_data'

    choose = input('处理训练数据按1,处理测试数据按2,其他键退出:')
    if choose == '1':
        # 处理训练数据
        preprocess_train_data(stock_data)
    elif choose == '2':
        start_date = input('请输入开始日期：(yyyy-mm-dd) \n')
        end_date = input('输入结束日期：(yyyy-mm-dd) \n')
        # 处理测试数据
        preprocess_test_data(test_data, start_date, end_date)
    else:
        sys.exit()
