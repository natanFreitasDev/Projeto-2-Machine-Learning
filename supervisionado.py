import os
import argparse
import warnings
import logging
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def haversine(lon1, lat1, lon2, lat2):
    """Retorna distância em km entre dois pontos (lon, lat)."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km


# Nota: remoção do manejo de ZIP porque os dados já estão na pasta do projeto.


def read_csvs(data_dir=None):

    project_root = os.path.dirname(os.path.abspath(__file__))
    if data_dir is None:
        data_dir = os.path.join(project_root, "olist")

    logging.info("Lendo CSVs de %s", data_dir)

    def p(fn):
        # Prioriza data_dir, depois tenta o diretório do projeto
        candidate = os.path.join(data_dir, fn)
        if os.path.exists(candidate):
            return candidate
        alt = os.path.join(project_root, fn)
        if os.path.exists(alt):
            logging.debug("Arquivo %s não encontrado em %s, usando %s", fn, data_dir, project_root)
            return alt
        # retorna o candidate para que o erro seja explícito se não existir
        return candidate

    orders = pd.read_csv(p("olist_orders_dataset.csv"),
                         parse_dates=['order_purchase_timestamp', 'order_approved_at',
                                      'order_delivered_carrier_date', 'order_delivered_customer_date',
                                      'order_estimated_delivery_date'])
    customers = pd.read_csv(p("olist_customers_dataset.csv"))
    order_items = pd.read_csv(p("olist_order_items_dataset.csv"))
    order_payments = pd.read_csv(p("olist_order_payments_dataset.csv"))
    order_reviews = pd.read_csv(p("olist_order_reviews_dataset.csv"))
    products = pd.read_csv(p("olist_products_dataset.csv"))
    sellers = pd.read_csv(p("olist_sellers_dataset.csv"))
    geolocation = pd.read_csv(p("olist_geolocation_dataset.csv"))
    cat_trans = pd.read_csv(p("product_category_name_translation.csv"))
    logging.info("Leitura completa.")
    return {
        "orders": orders, "customers": customers, "order_items": order_items,
        "order_payments": order_payments, "order_reviews": order_reviews,
        "products": products, "sellers": sellers, "geolocation": geolocation,
        "cat_trans": cat_trans
    }
    
    

def build_dataset(dfs):
    logging.info("Construindo dataset com target 'on_time' e features agregadas...")
    orders = dfs['orders'].copy()
    # Target: on_time
    orders = orders[~orders['order_delivered_customer_date'].isna()].copy()
    orders['delivery_delay_days'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days
    orders['on_time'] = (orders['delivery_delay_days'] <= 0).astype(int)

    # Payments agregados
    payments = dfs['order_payments']
    payments_agg = payments.groupby('order_id').agg(
        total_payment=('payment_value','sum'),
        n_payments=('payment_value','count'),
        payment_mean=('payment_value','mean')
    ).reset_index()
    # payment type mais frequente
    most_pt = payments.groupby(['order_id','payment_type']).size().reset_index(name='cnt')
    most_pt = most_pt.sort_values(['order_id','cnt'], ascending=[True,False]).drop_duplicates('order_id')[['order_id','payment_type']]
    payments_agg = payments_agg.merge(most_pt, on='order_id', how='left')


