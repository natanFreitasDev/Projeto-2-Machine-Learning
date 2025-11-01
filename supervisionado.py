import argparse
import os
import warnings
import logging
from math import radians, cos, sin, asin, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
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

    items = dfs['order_items']
    products = dfs['products']
    items_products = items.merge(products, on='product_id', how='left')
    items_products['volume_cm3'] = items_products['product_length_cm'] * items_products['product_height_cm'] * items_products['product_width_cm']

    items_agg = items_products.groupby('order_id').agg(
        n_items=('product_id', 'count'),
        n_sellers=('seller_id', 'nunique'),
        distinct_products=('product_id', 'nunique'),
        avg_weight_g=('product_weight_g', 'mean'),
       total_weight_g=('product_weight_g', 'sum'),
      avg_volume_cm3=('volume_cm3', 'mean'),
      total_volume_cm3=('volume_cm3', 'sum'),
      n_distinct_categories=('product_category_name', 'nunique')
    ).reset_index()  
    
    geo = dfs['geolocation']
    geo_agg = geo.groupby('geolocation_zip_code_prefix').agg(
       avg_lat=('geolocation_lat', 'mean'),
       avg_lng=('geolocation_lng', 'mean')).reset_index().rename(columns={'geolocation_zip_code_prefix': 'zip_code_prefix'})
    
    sellers = dfs['sellers'].merge(geo_agg, left_on='seller_zip_code_prefix', right_on='zip_code_prefix', how='left')
    sellers = sellers[['seller_id', 'avg_lat', 'avg_lng']].rename(columns={'avg_lat': 'seller_lat', 'avg_lng': 'seller_lng'})

    customers = dfs['customers'].merge(geo_agg, left_on='customer_zip_code_prefix', right_on='zip_code_prefix', how='left')
    customers = customers[['customer_id','avg_lat','avg_lng','customer_zip_code_prefix','customer_state']].rename(columns={'avg_lat':'customer_lat','avg_lng':'customer_lng'})
    
    oi = items[['order_id','seller_id']].merge(sellers, on='seller_id', how='left')
    oi = oi.merge(orders[['order_id', 'customer_id']], on='order_id', how='left')
    oi = oi.merge(customers, on='customer_id', how='left') 
    oi['dist_km'] = oi.apply(lambda r: haversine(r['seller_lng'], r['seller_lat'], r['customer_lng'], r['customer_lat']), axis=1)
    dist_agg = oi.groupby('order_id').agg(avg_seller_customer_km=('dist_km','mean')).reset_index()
    
    payments_order = dfs['order_payments'].groupby('order_id')['payment_value'].sum().reset_index().rename(columns={'payment_value':'order_payment_sum'})
    orders_pay = orders.merge(payments_order, on='order_id', how='left')
    cust_hist = orders_pay.groupby('customer_id').agg(
        customer_n_orders=('order_id', 'count'),
        customer_avg_ticket=('order_payment_sum','mean')
        ).reset_index()
    
    df = orders[['order_id','customer_id','order_purchase_timestamp','order_approved_at','order_estimated_delivery_date','order_delivered_customer_date','delivery_delay_days','on_time']].copy()
    df = df.merge(payments_agg, on='order_id', how='left')
    df = df.merge(items_agg, on='order_id', how='left')
    df = df.merge(dist_agg, on='order_id', how='left')
    df = df.merge(customers[['customer_id','customer_zip_code_prefix']],  on='customer_id', how='left')
    df = df.merge(cust_hist, on='customer_id', how='left')
    
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    df['time_to_estimated_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.days
    df['time_to_approved_hours'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds()/3600.0
    
    logging.info("Dataset montado. Shape: %s", df.shape)
    return df

def prepare_model_df(df):
    features = [
        'total_payment','n_payments','payment_type',
        'n_items','n_sellers','distinct_products','avg_weight_g','total_weight_g','avg_volume_cm3','total_volume_cm3','n_distinct_categories',
        'avg_seller_customer_km',
        'customer_n_orders','customer_avg_ticket',
        'purchase_hour','purchase_dayofweek','time_to_estimated_days','time_to_approved_hours',
        'customer_zip_code_prefix'
    ]
    
    target = 'on_time'
    df_model = df[features + [target]].copy()
    df_model['customer_zip_code_prefix'] = df_model['customer_zip_code_prefix'].astype(str)
    return df_model, features, target


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
     ('imputer', SimpleImputer(strategy='median')),
     ('scaler',StandardScaler())   
    ]
    )    
    
    categorical_transformer = Pipeline(
        steps=[
          ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
          ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop'
    )
    
    return preprocessor


def train_and_evaluate(clf, Xtr, Xte, ytr, yte, preprocessor, out_dir, show_roc=True):
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('clf', clf)])
    pipe.fit(Xtr, ytr)
    y_pred = pipe.predict(Xte)
    y_proba = pipe.predict_proba(Xte)[:,1] if hasattr(pipe.named_steps['clf'], "predict_proba") else None
    
    metrics = {
        "accuracy": accuracy_score(yte, y_pred),
        "precision": precision_score(yte, y_pred),
        "recall": recall_score(yte, y_pred),
        "f1": f1_score(yte, y_pred),
    }
    
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(yte, y_proba)
        except:
            pass

    logging.info("Modelo: %s - Metrics: %s", clf.__class__.__name__, metrics)

    cr = classification_report(yte, y_pred)
    cm = confusion_matrix(yte, y_pred)
    
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion matrix - {clf.__class__.__name__}')
    plt.ylabel('True')
    plt.xlabel('Pred')
    fig_path = os.path.join(out_dir, f'confusion_{clf.__class__.__name__}.png')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    logging.info("Confusion matrix salva em %s", fig_path)
    
    if show_roc and y_proba is not None:
        fpr, tpr, _ = roc_curve(yte, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{clf.__class__.__name__} (AUC {metrics.get("roc_auc", np.nan):.3f})')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
        roc_path = os.path.join(out_dir, f'roc_{clf.__class__.__name__}.png')
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
        logging.info("ROC curve salva em %s", roc_path)
    
    return pipe, metrics, cr
    
def plot_learning_curve(estimator, X, y, preprocessor, out_path, title='Learning Curve'):
    plt.figure(figsize=(8,5))
    train_sizes, train_scores, test_scores = learning_curve(
        Pipeline([('pre', preprocessor), ('clf', estimator)]),
        X, y, cv=5, scoring='f1', train_sizes=np.linspace(0.1,1.0,5), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Train score')
    plt.plot(train_sizes, test_scores_mean, 'o-', label='CV score')
    plt.xlabel('Training examples'); plt.ylabel('F1 score'); plt.title(title); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logging.info("Learning curve salva em %s", out_path)    


def main(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # zip handling removed: data is expected to be present in args.data_dir
        
    if not os.path.isdir(args.data_dir):
        logging.error("Diretório de dados não existe: %s", args.data_dir)
        return
    
    dfs = read_csvs(args.data_dir)
    df = build_dataset(dfs)
    df_model, features, target = prepare_model_df(df)
    
    df_model = df_model.dropna(subset=[target]).copy()
    X = df_model.drop(columns=[target])
    y = df_model[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)
    logging.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)

    numeric_features = [c for c in features if c not in ['payment_type','customer_zip_code_prefix']]
    categorical_features = ['payment_type','customer_zip_code_prefix']
    
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    preprocessor.fit(X_train)
    
    models = [
        LogisticRegression(class_weight='balanced', solver='liblinear', random_state=args.random_state, max_iter=1000),
        RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=args.random_state, n_jobs=-1),
        GradientBoostingClassifier(n_estimators=200, random_state=args.random_state)
    ]
    
    results = {}
    for clf in models:
        pipe, metrics, cr = train_and_evaluate(clf, X_train, X_test, y_train, y_test, preprocessor, out_dir, show_roc=True)
        results[clf.__class__.__name__] = metrics
        
        with open(os.path.join(out_dir, f'classif_report_{clf.__class__.__name__}.txt'), 'w') as f:
            f.write(cr)

    plot_learning_curve(RandomForestClassifier(n_estimators=200, random_state=args.random_state), X_train, y_train, preprocessor, os.path.join(out_dir, 'learning_curve_rf.png'))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state) 
    for clf in models:
        scores = cross_val_score(Pipeline([('pre', preprocessor), ('clf', clf)]), X_train, y_train, cv=skf, scoring='f1', n_jobs=-1)
        logging.info("%s CV F1 mean/std: %.4f +/- %.4f", clf.__class__.__name__, scores.mean(), scores.std())
        
    summary = pd.DataFrame(results).T
    summary.to_csv(os.path.join(out_dir, 'model_metrics_summary.csv'))
    logging.info("Resumo de métricas salvo em %s", os.path.join(out_dir, 'model_metrics_summary.csv'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Olist supervised pipeline")
    parser.add_argument('--zip', type=str, default=None, help='Caminho para o olist.zip.')
    parser.add_argument('--data-dir', type=str, default='olist', help='Diretório com CSVs extraídos.')
    parser.add_argument('--out-dir', type=str, default='out', help='Diretório de saída para fig/relatórios.')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()
    main(args)