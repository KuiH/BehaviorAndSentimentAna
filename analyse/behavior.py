import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyecharts import charts as pyc
from pyecharts import options as opts
from typing import *

# 解决matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 作图的字体默认设置
fontdict = {'fontsize': 15,
            'horizontalalignment': 'center'}


def ana_pv_uv(df: pd.DataFrame):
    pv_d = df.groupby("date").count()["user_id"]
    uv_d = df.groupby('date')["user_id"].apply(lambda x: x.drop_duplicates().count())
    pv_d = pv_d.iloc[25:35]  # iloc基于位置索引
    uv_d = uv_d.iloc[25:35]
    # 绘制折线图
    plt.figure(figsize=(10, 6))

    plt.plot(pv_d.index, pv_d.values, label='PV (Page Views)', color='blue', marker='o')
    plt.plot(uv_d.index, uv_d.values, label='UV (Unique Visitors)', color='green', marker='s')

    plt.title('PV and UV Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(r"D:\PythonProject\BehaviorAndSentimentAna\analyse\pics\pv_uv.png")


def ana_funnel(df: pd.DataFrame):
    count_users = df.userid.nunique()
    # 分别将fav和buy的df提取出来
    pv_df = df[df['type'] == 'pv']
    fav_df = df[df['type'] == 'fav']
    buy_df = df[df['type'] == 'buy']
    cart_df = df[df['type'] == 'cart']
    # 将fav_df和buy_df合并,得到收藏后购买的用户人数
    # process1 浏览->收藏->购买
    pv_fav_df = pd.merge(left=pv_df, right=fav_df, how='inner', on=[
        'userid', 'itemid'], suffixes=('_pv', '_fav'))
    fav_buy_df = pd.merge(left=fav_df, right=buy_df, how='inner', on=[
        'userid', 'itemid'], suffixes=('_fav', '_buy'))
    count_user_pv_fav = len(
        pv_fav_df[pv_fav_df.datetime_pv < pv_fav_df.datetime_fav].userid.value_counts())
    count_user_fav_buy = len(
        fav_buy_df[fav_buy_df.datetime_fav < fav_buy_df.datetime_buy].userid.value_counts())

    # process2 浏览->添加购物车->购买
    pv_cart_df = pd.merge(left=pv_df, right=cart_df, how='inner', on=[
        'userid', 'itemid', 'categoryid'], suffixes=('_pv', '_cart'))
    cart_buy_df = pd.merge(left=cart_df, right=buy_df, how='inner', on=[
        'userid', 'itemid', 'categoryid'], suffixes=('_cart', '_buy'))
    count_user_pv_cart = len(
        pv_cart_df[pv_cart_df.datetime_pv < pv_cart_df.datetime_cart].userid.value_counts())
    count_user_cart_buy = len(
        cart_buy_df[cart_buy_df.datetime_cart < cart_buy_df.datetime_buy].userid.value_counts())

    # 将数据合并成元组的列表是为了后面用pyecharts画图，如果不画图可以不这样合并
    process1 = [("浏览人数", count_users), ("收藏人数", count_user_pv_fav),
                ("购买人数", count_user_fav_buy)]
    process2 = [("浏览人数", count_users), ("添加购物车人数", count_user_pv_cart),
                ("购买人数", count_user_cart_buy)]

    conversion = pd.DataFrame(data=[[user[1] for user in process1], [user[1] for user in process2]], index=[
        'process1', 'process2'], columns=['pv', 'fav/cart', 'buy'])
    conversion['pv_fav/cart'] = conversion['fav/cart'].div(conversion['pv'])
    conversion['fav/cart_buy'] = conversion['buy'].div(conversion['fav/cart'])
    conversion['pv_buy'] = conversion['buy'].div(conversion['pv'])
    print(conversion)

    plt.figure(figsize=(10, 6))

    # 绘制 process1 的漏斗图
    plt.subplot(1, 2, 1)
    plt.title('Process 1: 浏览 -> 收藏 -> 购买', fontsize=14)
    categories = conversion.columns[:3][::-1]
    values = conversion.loc['process1'][:3][::-1]
    plt.barh(categories, values, color='skyblue')
    for i, v in enumerate(values):
        plt.text(v + 5, i, str(v), color='blue', va='center', fontweight='bold')

    # 绘制 process2 的漏斗图
    plt.subplot(1, 2, 2)
    plt.title('Process 2: 浏览 -> 添加购物车 -> 购买', fontsize=14)
    values = conversion.loc['process2'][:3][::-1]
    plt.barh(categories, values, color='lightgreen')
    for i, v in enumerate(values):
        plt.text(v + 5, i, str(v), color='green', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(r"D:\PythonProject\BehaviorAndSentimentAna\analyse\pics\funnel.png")


def RFM(df: pd.DataFrame):
    def R_score(x):
        if 0 <= x <= 1:
            return 4
        elif 2 <= x <= 3:
            return 3
        elif 4 <= x <= 6:
            return 2
        elif 7 <= x <= 9:
            return 1
        else:
            return 0

    def F_score(x):
        if x == 1:
            return 1
        elif x == 2:
            return 2
        elif x == 3:
            return 3
        elif x >= 4:
            return 4
        else:
            return 0

    def user_classfication(tup):
        R_score, F_score = tup
        if R_score >= 3 and F_score > 2:
            return "重要保持客户"
        elif R_score >= 3 and F_score <= 2:
            return "重要发展客户"
        elif R_score < 3 and F_score > 2:
            return "重要价值客户"
        elif R_score < 3 and F_score <= 2:
            return "重要挽留客户"
        else:
            return None

    # R:Recency最近一次交易日期；F:Frequency频率，交易次数
    buy_userid_group = df[df['type'] == 'buy'].groupby(by='userid')

    RF = pd.DataFrame(index=buy_userid_group.groups.keys(), columns=['R', 'F'])

    # 因为这边索引的顺序就是按照groupby之后的顺序，所以直接赋值即可
    RF['F'] = buy_userid_group.type.value_counts().values
    RF['last_buy_time'] = buy_userid_group.datetime.max().values
    RF['last_buy_time'] = pd.to_datetime(RF['last_buy_time']).dt.tz_localize(None)

    # 我们假设获取数据集的第二天进行分析，所以选择2017/12/04为对照日期，保留天数
    reference_date = pd.to_datetime('2017-12-04').tz_localize(None)  # 确保无时区

    RF['R'] = (reference_date - RF['last_buy_time']).dt.days

    # 根据R,F进行评分
    RF['R_score'] = RF.R.map(R_score)
    RF['F_score'] = RF.F.map(F_score)

    RF['user_classification'] = RF[['R_score', 'F_score']].apply(
        user_classfication, axis=1)
    classification_counts = RF['user_classification'].value_counts()

    plt.figure(figsize=(5, 6))
    bars = plt.bar(classification_counts.index, classification_counts.values, color='skyblue', width=0.4)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=12)

    plt.title('用户价值分类统计', fontsize=16)
    plt.xlabel('用户价值类别', fontsize=14)
    plt.ylabel('用户数量', fontsize=14)

    # 旋转 x 轴标签以便于阅读
    plt.xticks(rotation=45)

    # 显示网格
    plt.grid(axis='y')

    # 显示图形
    plt.tight_layout()
    # plt.show()
    plt.savefig(r"D:\PythonProject\BehaviorAndSentimentAna\analyse\pics\user_value_type.png")


def main():
    data_path = r"D:\PythonProject\BehaviorAndSentimentAna\dataset\UserBehavior_small_with_data.csv"
    df = pd.read_csv(data_path)
    # ana_funnel(df)
    RFM(df)


if __name__ == "__main__":
    main()
    pass
