import pandas as pd
import numpy as np
import keras
import tensorflow as tf

def read_csv_departments():
    depart = pd.read_csv("/Users/Romain/PycharmProjects/Instacard/Data/departments.csv")
    return depart[['department_id', 'department']]

def read_csv_order_products_prior():
    order_prior = pd.read_csv("/Users/Romain/PycharmProjects/Instacard/Data/order_products__prior.csv")
    return order_prior[['order_id', 'product_id']]

def read_csv_products():
    products = pd.read_csv("/Users/Romain/PycharmProjects/Instacard/Data/products.csv")
    return products[['product_id','department_id']]

def read_csv_orders():
    orders = pd.read_csv("/Users/Romain/PycharmProjects/Instacard/Data/orders.csv")
    return orders[['order_id', 'user_id']]


order_df = read_csv_orders()
orders_products_df = read_csv_order_products_prior()
product_df = read_csv_products()

orders_products_df = pd.merge(order_df,orders_products_df, on ='order_id')
orders_products_df = pd.merge(orders_products_df, product_df, on='product_id')

orders_products_df = orders_products_df[['user_id','order_id','department_id']]

orders_products_df.to_csv("/Users/Romain/PycharmProjects/Instacard/Data/merged_sample.csv")


def list_of_list(data):
    l = []
    sequences_client = []

    for user in data.groupby('user_id'):
        user_id = user[0]
        user_data = user[1]

        for order in user_data.groupby('order_id'):
            order_id = order[0]
            order_data = order[1]

            depart = list(order_data['department_id'])
            sequences_client.append(depart)
        l.append(sequences_client)
    return l
