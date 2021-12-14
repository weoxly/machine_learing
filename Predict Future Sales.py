import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings("ignore", module="lightgbm")
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgbm

pd.set_option('display.max_colwidth',None)

items = pd.read_csv('items.csv')
item_cat = pd.read_csv('item_categories.csv')
shops = pd.read_csv('shops.csv')
train = pd.read_csv('sales_train.csv')
test_dataset = pd.read_csv('test.csv')

train.head()
train.shape
#检测空值
train.isnull().sum()

train_dataset = train.copy()
train_dataset

#月销售
monthly_sales=train_dataset.groupby(["date_block_num","shop_id","item_id"])[
    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})
monthly_sales
monthly_sales.columns
sales_by_month = train_dataset.groupby(['date_block_num'])['item_cnt_day'].sum()
sales_by_month.plot()#销售额在几个月里不断下降，一些高峰出现在11月。
#检测相关性
corr = train_dataset.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(train_dataset.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(train_dataset.columns)
ax.set_yticklabels(train_dataset.columns)
plt.show()#相关性不强


#检查每一类别售出多少商品
items.head()
plt.rcParams['figure.figsize'] = (24, 9)
sns.barplot(items['item_category_id'], items['item_id'], palette = 'colorblind')
plt.title('The Sales of Category', fontsize = 30)
plt.xlabel('Category', fontsize = 15)
plt.ylabel('Sales', fontsize = 15)
plt.show()
#每月销售多少
plt.rcParams['figure.figsize'] = (24, 9)
sns.countplot(train_dataset['date_block_num'], palette = 'colorblind')
plt.title('The sales of month', fontsize = 30)
plt.xlabel('Month', fontsize = 15)
plt.ylabel('Sales', fontsize = 15)
plt.show()

# 唯一商店名称和类别名称的数量
print(item_cat['item_category_name'].nunique())
print(shops['shop_name'].nunique())

train_dataset['date'] = pd.to_datetime(train_dataset['date'], errors='coerce')

days = []
months = []
years = []

for day in train_dataset['date']:
    days.append(day.day)
for month in train_dataset['date']:
    months.append(month.month)
for year in train_dataset['date']:
    years.append(year.year)

# busy month
plt.rcParams['figure.figsize'] = (15, 7)
sns.countplot(months, palette= 'Blues')
plt.title('The busiest months for the shops', fontsize = 24)
plt.xlabel('Months', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)

plt.show()

# busy year
plt.rcParams['figure.figsize'] = (15, 7)
sns.countplot(years, palette= 'BuGn')
plt.title('The busiest years for the shops', fontsize = 24)
plt.xlabel('Years', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)

plt.show()

train_dataset['day'] = days
train_dataset['month'] = months
train_dataset['year'] = years

train_dataset

sns.countplot(train_dataset[(train_dataset.month == 2) & (train_dataset.year == 2013)]['shop_id'], palette='colorblind')

#去掉异常值
train_dataset.describe()
#商品价格箱型图
plt.figure(figsize=(10,4))
plt.xlim(train_dataset.item_price.min(), train_dataset.item_price.max()*1.1)
sns.boxplot(x=train_dataset.item_price)
plt.show()
#商品销售量箱型图
plt.figure(figsize=(10,4))
plt.xlim(train_dataset.item_cnt_day.min(), train_dataset.item_cnt_day.max()*1.1)
sns.boxplot(x=train_dataset.item_cnt_day)
plt.show()
#上面的箱型图判断，我们看到一个价格点比其他点更远。所以我们可以消去这个点。
train_dataset = train_dataset[(train_dataset["item_price"] > 0) & (train_dataset["item_price"] < 50000)]
train_dataset = train_dataset[(train_dataset["item_cnt_day"] > 0) & (train_dataset["item_cnt_day"] < 1000)]
#异常数据
train_dataset.shape
#从图表中我们可以看到，有一个价格点小于零。我们将用中值填充这个价格
train_dataset[train_dataset['item_price'] < 0]
median = train_dataset[(train_dataset.shop_id==32)&(train_dataset.item_id==2973)&(train_dataset.date_block_num==4)
                       &(train_dataset.item_price>0)].item_price.median()
median
#将价格小于0的替换成中位值
train_dataset["item_price"] = train_dataset["item_price"].map(lambda x: median if x<0 else x)
#替换后就无负的价格
train_dataset[train_dataset['item_price'] < 0]
#商品价格箱型图
plt.figure(figsize=(10,4))
plt.xlim(train_dataset.item_price.min(), train_dataset.item_price.max()*1.1)
sns.boxplot(x=train_dataset.item_price)
plt.show()

train_dataset[train_dataset['item_cnt_day'] < 0]
#将销售量小于0的替换成0
train_dataset["item_cnt_day"] = train_dataset["item_cnt_day"].map(lambda x: 0 if x<0 else x)
#替换后无负值
train_dataset[train_dataset['item_cnt_day'] < 0]
#商品销售量箱型图
plt.figure(figsize=(10,4))
plt.xlim(train_dataset.item_cnt_day.min(), train_dataset.item_cnt_day.max()*1.1)
sns.boxplot(x=train_dataset.item_cnt_day)
plt.show()

#Data Preprocessing
train_dataset.head(2)
#检查是否所有来自测试数据集的shop_id和item_id也出现在训练数据集中
print("唯一商品数: ", items['item_id'].nunique())
print("训练集中唯一商品数: ", train_dataset['item_id'].nunique())
print("测试集中唯一商品数: ", test_dataset['item_id'].nunique())

print("唯一商店数: ", shops['shop_id'].nunique())
print("训练集中唯一商店数: ", train_dataset['shop_id'].nunique())
print("测试集中唯一商店数: ", test_dataset['shop_id'].nunique())
#测试集和训练集中的项目数不相等，找出哪些item_id在test_set中，而不在train_set中
test_item_list = [x for x in (np.unique(test_dataset['item_id']))]
train_item_list = [x for x in (np.unique(train_dataset['item_id']))]
missing_item_ids_ = [element for element in test_item_list if element not in train_item_list]
len(missing_item_ids_)#在item_id在test_set中，而不在train_set中的数量

'''处理店铺数据'''
shops

# 去除！
shops['shop_name'] = shops['shop_name'].map(lambda x: x.split('!')[1] if x.startswith('!') else x)
shops['shop_name'] = shops["shop_name"].map(lambda x: 'СергиевПосад ТЦ "7Я"' if x == 'Сергиев Посад ТЦ "7Я"' else x)
#提取城市名称
shops['city'] = shops['shop_name'].map(lambda x: x.split(" ")[0])
# lets assign code to these city names too
shops['city_code'] = shops['city'].factorize()[0]
#显示操作后的表格
shops.head(2)
#增加特性
for shop_id in shops['shop_id'].unique():
    shops.loc[shop_id, 'num_products'] = train_dataset[train_dataset['shop_id'] == shop_id]['item_id'].nunique()
    shops.loc[shop_id, 'min_price'] = train_dataset[train_dataset['shop_id'] == shop_id]['item_price'].min()
    shops.loc[shop_id, 'max_price'] = train_dataset[train_dataset['shop_id'] == shop_id]['item_price'].max()
    shops.loc[shop_id, 'mean_price'] = train_dataset[train_dataset['shop_id'] == shop_id]['item_price'].mean()
#显示操作后的表格
shops.head(2)

'''处理类别数据'''
item_cat
#项目类别名称=类别的类型+子类型
cat_list = []
for name in item_cat['item_category_name']:
    cat_list.append(name.split('-'))

item_cat['split'] = (cat_list)
item_cat['cat_type'] = item_cat['split'].map(lambda x: x[0])
item_cat['cat_type_code'] = item_cat['cat_type'].factorize()[0]
item_cat['sub_cat_type'] = item_cat['split'].map(lambda x: x[1] if len(x)>1 else x[0])
item_cat['sub_cat_type_code'] = item_cat['sub_cat_type'].factorize()[0]
item_cat.head(2)
item_cat.drop('split', axis = 1, inplace=True)
#显示操作完后的表格
item_cat.head(2)


'''最后用于训练的DataFram'''
train_dataset = train_dataset[train_dataset["item_cnt_day"]>0]
train_dataset = train_dataset[["month", "date_block_num", "shop_id", "item_id", "item_price", "item_cnt_day"]].groupby(
    ["date_block_num", "shop_id", "item_id"]).agg(
    {"item_price": "mean","item_cnt_day": "sum", "month": "min"}).reset_index()
train_dataset.rename(columns={"item_cnt_day": "item_cnt_month"}, inplace=True)
train_dataset = pd.merge(train_dataset, items, on="item_id", how="inner")
train_dataset = pd.merge(train_dataset, shops, on="shop_id", how="inner")
train_dataset = pd.merge(train_dataset, item_cat, on="item_category_id", how="inner")

train_dataset.head(2)

#删除构件是不需要的列
train_dataset.drop(['item_name', 'shop_name', 'city', 'item_category_name', 'cat_type', 'sub_cat_type'], axis = 1, inplace=True)
train_dataset.head(1)

'''测试集数据'''
test_dataset.head()
test_dataset.shape
train_dataset.shape
#只保留train_dataset中test_dataset中的id
train_dataset = train_dataset[train_dataset['shop_id'].isin(test_dataset['shop_id'].unique())]
train_dataset = train_dataset[train_dataset['item_id'].isin(test_dataset['item_id'].unique())]
train_dataset.shape
train_dataset.head(2)

#将date_block_num = 34加入测试集以便能够做出销售预测
final_train_dataset = train_dataset.copy()
final_test_dataset = test_dataset.copy()

def data_preprocess(sales_train, test=None):
    indexlist = []
    for i in sales_train.date_block_num.unique():
        x = itertools.product(
            [i],
            sales_train.loc[sales_train.date_block_num == i].shop_id.unique(),
            sales_train.loc[sales_train.date_block_num == i].item_id.unique(),
        )
        indexlist.append(np.array(list(x)))
    df = pd.DataFrame(
        data=np.concatenate(indexlist, axis=0),
        columns=["date_block_num", "shop_id", "item_id"],
    )

    sales_train["item_revenue_day"] = sales_train["item_price"] * sales_train["item_cnt_month"]
    sales_train_grouped = sales_train.groupby(["date_block_num", "shop_id", "item_id"]).agg(
        item_cnt_month=pd.NamedAgg(column="item_cnt_month", aggfunc="sum"),
        item_revenue_month=pd.NamedAgg(column="item_revenue_day", aggfunc="sum"),
    )

    df = df.merge(
        sales_train_grouped, how="left", on=["date_block_num", "shop_id", "item_id"],
    )

    if test is not None:
        test["date_block_num"] = 34
        test["date_block_num"] = test["date_block_num"].astype(np.int8)
        test["shop_id"] = test.shop_id.astype(np.int8)
        test["item_id"] = test.item_id.astype(np.int16)
        test = test.drop(columns="ID")

        df = pd.concat([df, test[["date_block_num", "shop_id", "item_id"]]])

    # Fill empty item_cnt entries with 0
    df.item_cnt_month = df.item_cnt_month.fillna(0)
    df.item_revenue_month = df.item_revenue_month.fillna(0)

    return df

dataset_final = data_preprocess(final_train_dataset, final_test_dataset)

dataset_final = pd.merge(dataset_final, items, on="item_id", how="inner")
dataset_final = pd.merge(dataset_final, shops, on="shop_id", how="inner")
dataset_final = pd.merge(dataset_final, item_cat, on="item_category_id", how="inner")
dataset_final.head(3)
#删除不要的列
dataset_final.drop(['item_name', 'shop_name', 'city', 'item_category_name', 'cat_type', 'sub_cat_type'], axis = 1, inplace=True)
dataset_final.head(2)
dataset_final.shape

#为列名item_cnt_month和item_revenue_month添加特征
def lag_feature(matrix, lag_feature, lags):
    for lag in lags:
        newname = lag_feature + f"_lag_{lag}"
        print(f"Adding feature {newname}")
        targetseries = matrix.loc[:, ["date_block_num", "item_id", "shop_id"] + [lag_feature]]
        targetseries["date_block_num"] += lag
        targetseries = targetseries.rename(columns={lag_feature: newname})
        matrix = matrix.merge(
            targetseries, on=["date_block_num", "item_id", "shop_id"], how="left"
        )
#     print(matrix)
    return matrix

dataset_final = lag_feature(dataset_final, 'item_cnt_month', lags=[1,2,3])
dataset_final = lag_feature(dataset_final, 'item_revenue_month', lags=[1])
print("Lag features created..")
print(dataset_final.columns)

dataset_final.fillna(0, inplace= True)
dataset_final
dataset_final.head(2)

#除去2013年的销售记录
matrix = dataset_final[dataset_final.date_block_num>=12]
matrix.reset_index(drop=True, inplace=True)
matrix.head(2)
matrix.columns

'''创建模型'''
def fit_booster(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    params=None,
    test_run=False,
    categoricals=[],
    dropcols=[],
    early_stopping=True,
):
    if params is None:
        params = {"learning_rate": 0.1, "subsample_for_bin": 300000, "n_estimators": 50}

    early_stopping_rounds = None
    if early_stopping == True:
        early_stopping_rounds = 50

    if test_run:
        eval_set = [(X_train, y_train)]
    else:
        eval_set = [(X_train, y_train), (X_test, y_test)]

    booster = lgbm.LGBMRegressor(**params)

    categoricals = [c for c in categoricals if c in X_train.columns]

    booster.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        eval_metric=["rmse"],
        verbose=100,
        categorical_feature=categoricals,
        early_stopping_rounds=early_stopping_rounds,
    )

    return booster



keep_from_month = 2
test_month = 33

dropcols = [

    "item_id",
]

params = {
    "num_leaves": 966,
    "cat_smooth": 45.01680827234465,
    "min_child_samples": 27,
    "min_child_weight": 0.021144950289224463,
    "max_bin": 214,
    "learning_rate": 0.01,
    "subsample_for_bin": 300000,
    "min_data_in_bin": 7,
    "colsample_bytree": 0.8,
    "subsample": 0.6,
    "subsample_freq": 5,
    "n_estimators": 8000,
}
#lightgbm将以这些作为分类特征
categoricals = [
    "item_category_id",
    "month",
    "shop_id"
]



valid = matrix.drop(columns=dropcols).loc[matrix.date_block_num == test_month, :]
train__ = matrix.drop(columns=dropcols).loc[matrix.date_block_num < test_month, :]
train__ = train__[train__.date_block_num >= keep_from_month]
X_train = train__.drop(columns="item_cnt_month")
y_train = train__.item_cnt_month
X_valid = valid.drop(columns="item_cnt_month")
y_valid = valid.item_cnt_month

lgbooster = fit_booster(
    X_train,
    y_train,
    X_valid,
    y_valid,
    params=params,
    test_run=False,
    categoricals=categoricals,
)


'''预测'''
matrix['item_cnt_month'] = matrix['item_cnt_month'].clip(0,20)
keep_from_month = 2
test_month = 34
test__ = matrix.loc[matrix.date_block_num==test_month, :]
X_test = test__.drop(columns="item_cnt_month")
y_test = test__.item_cnt_month

X_test["item_cnt_month"] = lgbooster.predict(X_test.drop(columns=dropcols)).clip(0, 20)

'''提交结果'''
testing = test_dataset.merge(
    X_test[["shop_id", "item_id", "item_cnt_month"]],
    on=["shop_id", "item_id"],
    how="inner",
    copy=True,
)
assert test_dataset.equals(testing[["ID", "shop_id", "item_id"]])
testing[["ID", "item_cnt_month"]].to_csv("./submission.csv", index=False)
