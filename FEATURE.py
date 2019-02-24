import pandas as pd
import numpy as np


def Feat(dataset_x, dataset_y):
    '平均车长'
    tmp = dataset_x[['class_id', 'car_length']]
    CAR_LEN_MEAN = tmp.groupby(by='class_id', as_index=False).mean()
    CAR_LEN_MEAN.columns = ['class_id', 'CAR_LEN_MEAN']
    dataset_y = pd.merge(dataset_y, CAR_LEN_MEAN, on='class_id', how='left')

    '车长中位数'
    tmp = dataset_x[['class_id', 'car_length']]
    CAR_LEN_MEDIAN = tmp.groupby(by='class_id', as_index=False).median()
    CAR_LEN_MEDIAN.columns = ['class_id', 'CAR_LEN_MEDIAN']
    dataset_y = pd.merge(dataset_y, CAR_LEN_MEDIAN, on='class_id', how='left')

    '平均车宽'
    tmp = dataset_x[['class_id', 'car_width']]
    CAR_WID_MEAN = tmp.groupby(by='class_id', as_index=False).mean()
    CAR_WID_MEAN.columns = ['class_id', 'CAR_WID_MEAN']
    dataset_y = pd.merge(dataset_y, CAR_WID_MEAN, on='class_id', how='left')

    '车宽中位数'
    tmp = dataset_x[['class_id', 'car_width']]
    CAR_WID_MEDIAN = tmp.groupby(by='class_id', as_index=False).median()
    CAR_WID_MEDIAN.columns = ['class_id', 'CAR_WID_MEDIAN']
    dataset_y = pd.merge(dataset_y, CAR_WID_MEDIAN, on='class_id', how='left')

    '平均车高'
    tmp = dataset_x[['class_id', 'car_height']]
    CAR_HIGH_MEAN = tmp.groupby(by='class_id', as_index=False).mean()
    CAR_HIGH_MEAN.columns = ['class_id', 'CAR_HIGH_MEAN']
    dataset_y = pd.merge(dataset_y, CAR_HIGH_MEAN, on='class_id', how='left')

    '车高中位数'
    tmp = dataset_x[['class_id', 'car_height']]
    CAR_HIGH_MEDIAN = tmp.groupby(by='class_id', as_index=False).median()
    CAR_HIGH_MEDIAN.columns = ['class_id', 'CAR_HIGH_MEDIAN']
    dataset_y = pd.merge(dataset_y, CAR_HIGH_MEDIAN, on='class_id', how='left')

    '每种class_id的销量和'
    tmp = dataset_x[['class_id', 'sale_quantity']]
    class_sum = tmp.groupby(by=['class_id'], as_index=False).sum()
    class_sum.columns = ['class_id', 'class_sum']
    dataset_y = pd.merge(dataset_y, class_sum, on='class_id', how='left')

    '每种class_id的销量平均值'
    tmp = dataset_x[['class_id', 'sale_quantity']]
    class_mean = tmp.groupby(by=['class_id'], as_index=False).mean()
    class_mean.columns = ['class_id', 'class_mean']
    dataset_y = pd.merge(dataset_y, class_mean, on='class_id', how='left')

    '每种class_id的销量最大值'
    tmp = dataset_x[['class_id', 'sale_quantity']]
    class_max = tmp.groupby(by=['class_id'], as_index=False).max()
    class_max.columns = ['class_id', 'class_max']
    dataset_y = pd.merge(dataset_y, class_max, on='class_id', how='left')

    '每种class_id的销量的中位数'
    tmp = dataset_x[['class_id', 'sale_quantity']]
    class_median = tmp.groupby(by=['class_id'], as_index=False).median()
    class_median.columns = ['class_id', 'class_median']
    dataset_y = pd.merge(dataset_y, class_median, on='class_id', how='left')

    '每种class_id的销量的最小值'
    tmp = dataset_x[['class_id', 'sale_quantity']]
    class_min = tmp.groupby(by=['class_id'], as_index=False).min()
    class_min.columns = ['class_id', 'class_min']
    dataset_y = pd.merge(dataset_y, class_min, on='class_id', how='left')

    '每种class_id的销量的方差'
    tmp = dataset_x[['class_id', 'sale_quantity']]
    class_min = tmp.groupby(by=['class_id'], as_index=False).var()
    class_min.columns = ['class_id', 'class_var']
    dataset_y = pd.merge(dataset_y, class_min, on='class_id', how='left')

    '每种brand的销量占总销量的比率'
    tmp = dataset_x[['class_id', 'brand_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'brand_id']).sum()
    grouped = pd.DataFrame(grouped).reset_index()
    grouped['brand_ratio'] = grouped['sale_quantity'] / grouped['sale_quantity'].sum()
    dataset_y = pd.merge(dataset_y, grouped[['class_id', 'brand_ratio']], on='class_id', how='left')

    '不同成交字段的销量占比'
    tmp = dataset_x[['class_id', 'price_level', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'price_level']).sum()
    grouped = grouped.rename(columns={'sale_quantity': 'price_level_count'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    grouped.columns = ['class_id', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    grouped['price_1'] = grouped['1'] / grouped['1'].sum()
    grouped['price_2'] = grouped['2'] / grouped['2'].sum()
    grouped['price_3'] = grouped['3'] / grouped['3'].sum()
    grouped['price_4'] = grouped['4'] / grouped['4'].sum()
    grouped['price_5'] = grouped['5'] / grouped['5'].sum()
    grouped['price_6'] = grouped['6'] / grouped['6'].sum()
    grouped['price_7'] = grouped['7'] / grouped['7'].sum()
    grouped['price_8'] = grouped['8'] / grouped['8'].sum()
    grouped['price_9'] = grouped['9'] / grouped['9'].sum()
    dataset_y = pd.merge(dataset_y, grouped.reset_index(), on='class_id', how='left')

    '不同成交字段的销量中位数'
    tmp = dataset_x[['class_id', 'price_level', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'price_level']).median()
    grouped = grouped.rename(columns={'sale_quantity': 'price_level_median'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    grouped.columns = ['class_id', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '不同成交字段的销量最大值'
    tmp = dataset_x[['class_id', 'price_level', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'price_level']).max()
    grouped = grouped.rename(columns={'sale_quantity': 'price_level_max'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    grouped.columns = ['class_id', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '不同成交字段的销量平均值'
    tmp = dataset_x[['class_id', 'price_level', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'price_level']).mean()
    grouped = grouped.rename(columns={'sale_quantity': 'price_level_mean'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    grouped.columns = ['class_id', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '不同成交字段的销量最小值'
    tmp = dataset_x[['class_id', 'price_level', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'price_level']).min()
    grouped = grouped.rename(columns={'sale_quantity': 'price_level_min'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    grouped.columns = ['class_id', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '每个class_id的品牌销售次数占比'
    temp = dataset_x[['class_id', 'brand_id', 'sale_quantity']]
    brand_num = temp.groupby(by=['class_id', 'brand_id'], as_index=False).count()
    brand_num = brand_num.rename(columns={'sale_quantity': 'sale_count'})
    brand_num['sale_ratio'] = brand_num['sale_count'] / brand_num['sale_count'].sum()
    dataset_y = pd.merge(dataset_y, brand_num[['class_id', 'sale_ratio']], on='class_id', how='left')

    'gearbox_type出现占比'
    # tmp = dataset_x[['class_id', 'gearbox_type', 'sale_quantity']]
    # grouped = tmp.groupby(by=['class_id', 'gearbox_type']).count()
    # grouped = grouped.rename(columns={'sale_quantity': 'gearbox_count'}).unstack()
    # grouped = grouped.fillna(0).reset_index()
    # grouped.columns=['class_id','AMT','AT','CVT','DCT','MT']
    # print(grouped)
    # grouped['AMT_r']=grouped['AMT']/grouped['AMT'].sum()
    # grouped['AT_r'] = grouped['AT'] / grouped['AT'].sum()
    # grouped['CVT_r'] = grouped['CVT'] / grouped['CVT'].sum()
    # grouped['DCT_r'] = grouped['DCT'] / grouped['DCT'].sum()
    # grouped['MT_r'] = grouped['MT'] / grouped['MT'].sum()
    # dataset_y = pd.merge(dataset_y, grouped[['class_id','AMT_r','AT_r','CVT_r','DCT_r','MT_r']].reset_index(), on='class_id', how='left')

    'MPV销量占比'
    tmp = dataset_x[['class_id', 'if_MPV_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_MPV_id']).sum()
    grouped = grouped.rename(columns={'sale_quantity': 'MPV_count'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    grouped.columns = ['class_id', '1', '2']
    grouped['MPV_1_ratio'] = grouped['1'] / grouped['1'].sum()
    grouped['MPV_2_ratio'] = grouped['2'] / grouped['2'].sum()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'MPV销量中位数'
    tmp = dataset_x[['class_id', 'if_MPV_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_MPV_id']).median()
    grouped = grouped.rename(columns={'sale_quantity': 'MPV_median'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    # grouped.columns = ['class_id', '1', '2']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'MPV销量最大值'
    tmp = dataset_x[['class_id', 'if_MPV_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_MPV_id']).max()
    grouped = grouped.rename(columns={'sale_quantity': 'MPV_max'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    # grouped.columns = ['class_id', '1', '2']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'MPV销量最小值'
    tmp = dataset_x[['class_id', 'if_MPV_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_MPV_id']).min()
    grouped = grouped.rename(columns={'sale_quantity': 'MPV_min'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    # grouped.columns = ['class_id', '1', '2']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'MPV销量平均值'
    tmp = dataset_x[['class_id', 'if_MPV_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_MPV_id']).mean()
    grouped = grouped.rename(columns={'sale_quantity': 'MPV_mean'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    # grouped.columns = ['class_id', '1', '2']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'luxurious销量占比'
    tmp = dataset_x[['class_id', 'if_luxurious_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_luxurious_id']).sum()
    grouped = grouped.rename(columns={'sale_quantity': 'luxurious_count'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    grouped.columns = ['class_id', '1', '2']
    grouped['lu1'] = grouped['1'] / grouped['1'].sum()
    grouped['lu2'] = grouped['2'] / grouped['2'].sum()
    grouped['lu1_ratio'] = grouped['1'] / (grouped['1'].sum() + grouped['2'].sum())
    grouped['lu2_ratio'] = grouped['2'] / (grouped['1'].sum() + grouped['2'].sum())
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'luxurious销量中位数'
    tmp = dataset_x[['class_id', 'if_luxurious_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_luxurious_id']).median()
    grouped = grouped.rename(columns={'sale_quantity': 'luxurious_median'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'luxurious销量最大值'
    tmp = dataset_x[['class_id', 'if_luxurious_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_luxurious_id']).max()
    grouped = grouped.rename(columns={'sale_quantity': 'luxurious_max'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'luxurious销量最小值'
    tmp = dataset_x[['class_id', 'if_luxurious_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_luxurious_id']).min()
    grouped = grouped.rename(columns={'sale_quantity': 'luxurious_min'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'luxurious销量平均值'
    tmp = dataset_x[['class_id', 'if_luxurious_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_luxurious_id']).mean()
    grouped = grouped.rename(columns={'sale_quantity': 'luxurious_mean'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'price字段'
    tmp = dataset_x[['class_id', 'price']]
    '单纯去掉price字段为\'-\'的class_id并不是一件非常好的事，应该使用5W填充'
    # extra=[]
    # for index,row in tmp.iterrows():
    #     row_price=row['price']
    #     if row_price=='-':
    #         extra.append(row['class_id'])
    # extra=list(set(extra))
    # for i in range(len(extra)):
    #     tmp=tmp[tmp['class_id']!=extra[i]]

    tmp['price'] = [str(i)[:10].replace('-', '0') for i in tmp['price']]
    tmp['price'] = [np.nan if i == 0 else i for i in tmp['price']]
    tmp['price'] = tmp['price'].astype('float64')
    tmp['price'] = tmp['price'].fillna(5)

    'price字段求和'
    grouped = tmp.groupby(by='class_id', as_index=False).sum()
    grouped = grouped.rename(columns={'price': 'price_sum'})
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'price中位数'
    grouped = tmp.groupby(by='class_id', as_index=False).median()
    grouped = grouped.rename(columns={'price': 'price_median'})
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'price最大值'
    grouped = tmp.groupby(by='class_id', as_index=False).max()
    grouped = grouped.rename(columns={'price': 'price_max'})
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'price平均值'
    grouped = tmp.groupby(by='class_id', as_index=False).mean()
    grouped = grouped.rename(columns={'price': 'price_mean'})
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '在一定厢数下的销售量占比'
    tmp = dataset_x[['class_id', 'compartment', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'compartment']).sum().unstack()
    grouped.columns = ['1', '2', '3']
    grouped = grouped.reset_index().fillna(0)
    grouped['compartment_1'] = grouped['1'] / grouped['1'].sum()
    grouped['compartment_2'] = grouped['2'] / grouped['2'].sum()
    grouped['compartment_3'] = grouped['3'] / grouped['3'].sum()
    grouped['c1'] = grouped['1'] / (grouped['1'].sum() + grouped['2'].sum() + grouped['3'].sum())
    grouped['c2'] = grouped['2'] / (grouped['1'].sum() + grouped['2'].sum() + grouped['3'].sum())
    grouped['c3'] = grouped['3'] / (grouped['1'].sum() + grouped['2'].sum() + grouped['3'].sum())
    dataset_y = pd.merge(dataset_y, grouped.drop(['1', '2', '3'], axis=1), on='class_id', how='left')

    '在一定厢数下的销售量中位数'
    tmp = dataset_x[['class_id', 'compartment', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'compartment']).median()
    grouped=grouped.rename(columns={'sale_quantity':'compartment_median'})
    grouped=grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '在一定厢数下的销售量最大值'
    tmp = dataset_x[['class_id', 'compartment', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'compartment']).max()
    grouped = grouped.rename(columns={'sale_quantity': 'compartment_max'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '在一定厢数下的销售量最小值'
    tmp = dataset_x[['class_id', 'compartment', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'compartment']).min()
    grouped = grouped.rename(columns={'sale_quantity': 'compartment_min'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '在一定厢数下的销售量平均值'
    tmp = dataset_x[['class_id', 'compartment', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'compartment']).mean()
    grouped = grouped.rename(columns={'sale_quantity': 'compartment_mean'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '在一定厢数下的销售量方差'
    tmp = dataset_x[['class_id', 'compartment', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'compartment']).var()
    grouped = grouped.rename(columns={'sale_quantity': 'compartment_var'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '排量displacement中位数'
    tmp = dataset_x[['class_id', 'displacement']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).median()
    grouped.columns = ['class_id', 'displacement_median']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '排量displacement最大值'
    tmp = dataset_x[['class_id', 'displacement']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).max()
    grouped.columns = ['class_id', 'displacement_max']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '排量displacement最小值'
    tmp = dataset_x[['class_id', 'displacement']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).min()
    grouped.columns = ['class_id', 'displacement_min']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '排量displacement平均值'
    tmp = dataset_x[['class_id', 'displacement']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).mean()
    grouped.columns = ['class_id', 'displacement_mean']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '是否增压if_charging销售量占比'
    tmp = dataset_x[['class_id', 'if_charging', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_charging']).sum().unstack().fillna(0)
    grouped = grouped.reset_index()
    grouped.columns = ['class_id', 'L', 'T']
    grouped['L_ratio1'] = grouped['L'] / grouped['L'].sum()
    grouped['T_ratio1'] = grouped['T'] / grouped['T'].sum()
    grouped['L_ratio2'] = grouped['L'] / (grouped['L'].sum() + grouped['T'].sum())
    grouped['T_ratio2'] = grouped['T'] / (grouped['L'].sum() + grouped['T'].sum())
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '是否增压if_charging销售量中位数'
    tmp = dataset_x[['class_id', 'if_charging', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_charging']).median()
    grouped=grouped.rename(columns={'sale_quantity':'if_charging_median'})
    grouped=grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '是否增压if_charging销售量最大值'
    tmp = dataset_x[['class_id', 'if_charging', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_charging']).max()
    grouped = grouped.rename(columns={'sale_quantity': 'if_charging_max'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '是否增压if_charging销售量最小值'
    tmp = dataset_x[['class_id', 'if_charging', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_charging']).min()
    grouped = grouped.rename(columns={'sale_quantity': 'if_charging_min'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '是否增压if_charging销售量平均值'
    tmp = dataset_x[['class_id', 'if_charging', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_charging']).mean()
    grouped = grouped.rename(columns={'sale_quantity': 'if_charging_mean'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '是否增压if_charging销售量方差'
    tmp = dataset_x[['class_id', 'if_charging', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'if_charging']).var()
    grouped = grouped.rename(columns={'sale_quantity': 'if_charging_var'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'power功率'
    tmp = dataset_x[['class_id', 'power']]
    extra = []
    for index, row in tmp.iterrows():
        if row['power'] == '81/70':
            extra.append(row['class_id'])
    extra = list(set(extra))
    for i in range(len(extra)):
        tmp = tmp[tmp['class_id'] != extra[i]]

    tmp['power'] = [float(i) for i in tmp['power']]

    'power中位数'
    grouped = tmp.groupby(by='class_id').median()
    grouped.columns = ['power_median']
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'power平均值'
    grouped = tmp.groupby(by='class_id').mean()
    grouped.columns = ['power_mean']
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'power最大值'
    grouped = tmp.groupby(by='class_id').max()
    grouped.columns = ['power_max']
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'power最小值'
    grouped = tmp.groupby(by='class_id').min()
    grouped.columns = ['power_min']
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'cylinder_number缸数'
    # tmp=dataset_x[['class_id','cylinder_number','sale_quantity']]
    # grouped=tmp.groupby(by=['class_id','cylinder_number']).sum().unstack().fillna(0)
    # grouped.columns=['3','4','6']
    # grouped['cy3']=grouped['3']/grouped['3'].sum()
    # grouped['cy4'] = grouped['4'] / grouped['4'].sum()
    # grouped['cy6'] = grouped['6'] / grouped['6'].sum()
    # grouped=grouped.reset_index()
    # dataset_y=pd.merge(dataset_y,grouped.drop(['3','4','6'],axis=1),on='class_id',how='left')

    'engine_torque发动机扭矩'
    tmp = dataset_x[['class_id', 'engine_torque']]
    tmp['engine_torque'] = [str(i)[:10].replace('-', '0') for i in tmp['engine_torque']]
    tmp['engine_torque'] = [float(i) for i in tmp['engine_torque']]

    '发动机扭矩平均值'
    ent_mean = tmp.groupby(by=['class_id'], as_index=False).mean()
    ent_mean.columns = ['class_id', 'ent_mean']
    dataset_y = pd.merge(dataset_y, ent_mean, on='class_id', how='left')

    '发动机扭矩最大值'
    ent_max = tmp.groupby(by=['class_id'], as_index=False).max()
    ent_max.columns = ['class_id', 'ent_max']
    dataset_y = pd.merge(dataset_y, ent_max, on='class_id', how='left')

    '发动机扭矩最小值'
    ent_min = tmp.groupby(by=['class_id'], as_index=False).min()
    ent_min.columns = ['class_id', 'ent_min']
    dataset_y = pd.merge(dataset_y, ent_min, on='class_id', how='left')

    '发动机扭矩中位数'
    ent_median = tmp.groupby(by=['class_id'], as_index=False).median()
    ent_median.columns = ['class_id', 'ent_median']
    dataset_y = pd.merge(dataset_y, ent_median, on='class_id', how='left')

    '总质量total_quality平均值'
    tmp = dataset_x[['class_id', 'total_quality']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).mean()
    grouped.columns = ['class_id', 'total_quality_mean']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '总质量total_quality中位数'
    tmp = dataset_x[['class_id', 'total_quality']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).median()
    grouped.columns = ['class_id', 'total_quality_median']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '总质量total_quality最大值'
    tmp = dataset_x[['class_id', 'total_quality']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).max()
    grouped.columns = ['class_id', 'total_quality_max']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '总质量total_quality最小值'
    tmp = dataset_x[['class_id', 'total_quality']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).min()
    grouped.columns = ['class_id', 'total_quality_min']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    '额定载客rated_passenger分段销售量占比'
    # tmp=dataset_x[['class_id','rated_passenger','sale_quantity']]
    # grouped=tmp.groupby(by=['class_id','rated_passenger']).sum().unstack().fillna(0)
    # grouped.columns=['p4-5','p5','p6-7','p6-8','p7','p7-8','p9']
    # grouped['p4-5r']=grouped['p4-5']/grouped['p4-5'].sum()
    # grouped['p5r'] = grouped['p5'] / grouped['p5'].sum()
    # grouped['p6-7r'] = grouped['p6-7'] / grouped['p6-7'].sum()
    # grouped['p6-8r'] = grouped['p6-8'] / grouped['p6-8'].sum()
    # grouped['p7r'] = grouped['p7'] / grouped['p7'].sum()
    # grouped['p7-8r'] = grouped['p7-8'] / grouped['p7-8'].sum()
    # grouped['p9r'] = grouped['p9'] / grouped['p9'].sum()
    # grouped=grouped.reset_index()
    # dataset_y=pd.merge(dataset_y,grouped.drop(['p4-5','p5','p6-7','p6-8','p7','p7-8','p9'],axis=1),on='class_id',how='left')

    'wheelbase轴距平均值'
    tmp = dataset_x[['class_id', 'wheelbase']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).mean()
    grouped.columns = ['class_id', 'wheelbase_mean']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'wheelbase中位数'
    tmp = dataset_x[['class_id', 'wheelbase']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).median()
    grouped.columns = ['class_id', 'wheelbase_median']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'wheelbase最大值'
    tmp = dataset_x[['class_id', 'wheelbase']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).max()
    grouped.columns = ['class_id', 'wheelbase_max']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'wheelbase最小值'
    tmp = dataset_x[['class_id', 'wheelbase']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).min()
    grouped.columns = ['class_id', 'wheelbase_min']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'front_track前轮距平均值'
    tmp = dataset_x[['class_id', 'front_track']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).mean()
    grouped.columns = ['class_id', 'front_track_mean']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'front_track前轮距最大值'
    tmp = dataset_x[['class_id', 'front_track']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).max()
    grouped.columns = ['class_id', 'front_track_max']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'front_track前轮距最小值'
    tmp = dataset_x[['class_id', 'front_track']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).min()
    grouped.columns = ['class_id', 'front_track_min']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'front_track前轮距中位数'
    tmp = dataset_x[['class_id', 'front_track']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).median()
    grouped.columns = ['class_id', 'front_track_median']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'rear_track后轮距平均值'
    tmp = dataset_x[['class_id', 'rear_track']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).mean()
    grouped.columns = ['class_id', 'rear_track_mean']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'rear_track后轮距最大值'
    tmp = dataset_x[['class_id', 'rear_track']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).max()
    grouped.columns = ['class_id', 'rear_track_max']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'rear_track后轮距最小值'
    tmp = dataset_x[['class_id', 'rear_track']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).min()
    grouped.columns = ['class_id', 'rear_track_min']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'rear_track后轮距中位数'
    tmp = dataset_x[['class_id', 'rear_track']]
    grouped = tmp.groupby(by=['class_id'], as_index=False).median()
    grouped.columns = ['class_id', 'rear_track_median']
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'type_id车型类别ID销售量占比'
    tmp = dataset_x[['class_id', 'type_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'type_id']).sum()
    grouped = grouped.rename(columns={'sale_quantity': 'type_id_count'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    grouped.columns = ['class_id', '1', '2', '3', '4']
    grouped['type_id_1_ratio'] = grouped['1'] / grouped['1'].sum()
    grouped['type_id_2_ratio'] = grouped['1'] / grouped['1'].sum()
    grouped['type_id_3_ratio'] = grouped['1'] / grouped['1'].sum()
    grouped['type_id_4_ratio'] = grouped['1'] / grouped['1'].sum()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'type_id车型类别ID销售量中位数'
    tmp = dataset_x[['class_id', 'type_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'type_id']).median()
    grouped = grouped.rename(columns={'sale_quantity': 'type_id_median'})
    grouped=grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'type_id车型类别ID销售量最大值'
    tmp = dataset_x[['class_id', 'type_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'type_id']).max()
    grouped = grouped.rename(columns={'sale_quantity': 'type_id_max'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'type_id车型类别ID销售量最小值'
    tmp = dataset_x[['class_id', 'type_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'type_id']).min()
    grouped = grouped.rename(columns={'sale_quantity': 'type_id_min'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'type_id车型类别ID销售量平均值'
    tmp = dataset_x[['class_id', 'type_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'type_id']).mean()
    grouped = grouped.rename(columns={'sale_quantity': 'type_id_mean'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'type_id车型类别ID销售量方差'
    tmp = dataset_x[['class_id', 'type_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'type_id']).var()
    grouped = grouped.rename(columns={'sale_quantity': 'type_id_var'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'level_id车型级别ID'
    # tmp=dataset_x[['class_id','level_id','sale_quantity']]
    # grouped=tmp.groupby(by=['class_id','level_id']).sum()
    # grouped=grouped.rename(columns={'sale_quantity':'level_id_count'}).unstack()
    # grouped=grouped.fillna(0).reset_index()
    # grouped.columns=['class_id','-','1','2','3','4']
    # grouped['level_id_-']=grouped['-']/grouped['-'].sum()
    # grouped['level_id_1'] = grouped['1'] / grouped['1'].sum()
    # grouped['level_id_2'] = grouped['2'] / grouped['2'].sum()
    # grouped['level_id_3'] = grouped['3'] / grouped['3'].sum()
    # grouped['level_id_4'] = grouped['4'] / grouped['4'].sum()
    # grouped=grouped[['class_id','level_id_1','level_id_2','level_id_3','level_id_4']]
    # dataset_y=pd.merge(dataset_y,grouped,on='class_id',how='left')

    'department_id车型系别ID销售量占比'
    tmp = dataset_x[['class_id', 'department_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'department_id']).sum()
    grouped = grouped.rename(columns={'sale_quantity': 'department_id_count'}).unstack()
    grouped = grouped.fillna(0).reset_index()
    grouped.columns = ['class_id', 'department_id_1_sum', 'department_id_2_sum', 'department_id_3_sum',\
                       'department_id_4_sum', 'department_id_5_sum', 'department_id_6_sum', 'department_id_7_sum']
    grouped['department_id_1_ratio'] = grouped['department_id_1_sum'] / grouped['department_id_1_sum'].sum()
    grouped['department_id_2_ratio'] = grouped['department_id_2_sum'] / grouped['department_id_2_sum'].sum()
    grouped['department_id_3_ratio'] = grouped['department_id_3_sum'] / grouped['department_id_3_sum'].sum()
    grouped['department_id_4_ratio'] = grouped['department_id_4_sum'] / grouped['department_id_4_sum'].sum()
    grouped['department_id_5_ratio'] = grouped['department_id_5_sum'] / grouped['department_id_5_sum'].sum()
    grouped['department_id_6_ratio'] = grouped['department_id_6_sum'] / grouped['department_id_6_sum'].sum()
    grouped['department_id_7_ratio'] = grouped['department_id_7_sum'] / grouped['department_id_7_sum'].sum()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'department_id车型系别ID销售量中位数'
    tmp = dataset_x[['class_id', 'department_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'department_id']).median()
    grouped = grouped.rename(columns={'sale_quantity': 'department_id_median'})
    grouped=grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'department_id车型系别ID销售量最大值'
    tmp = dataset_x[['class_id', 'department_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'department_id']).max()
    grouped = grouped.rename(columns={'sale_quantity': 'department_id_max'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'department_id车型系别ID销售量最小值'
    tmp = dataset_x[['class_id', 'department_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'department_id']).min()
    grouped = grouped.rename(columns={'sale_quantity': 'department_id_min'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'department_id车型系别ID销售量平均值'
    tmp = dataset_x[['class_id', 'department_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'department_id']).mean()
    grouped = grouped.rename(columns={'sale_quantity': 'department_id_mean'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    'department_id车型系别ID销售量方差'
    tmp = dataset_x[['class_id', 'department_id', 'sale_quantity']]
    grouped = tmp.groupby(by=['class_id', 'department_id']).var()
    grouped = grouped.rename(columns={'sale_quantity': 'department_id_var'})
    grouped = grouped.unstack().fillna(0)
    grouped = grouped.reset_index()
    dataset_y = pd.merge(dataset_y, grouped, on='class_id', how='left')

    return dataset_y
    pass



