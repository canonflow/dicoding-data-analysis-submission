from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
from streamlit import subheader

sns.set(style='dark')

def create_sum_order_items_df(df):
    sum_order_items_df = df.groupby(by="product_category_name_english").agg({
        "order_id": "nunique"
    }).sort_values(by="order_id", ascending=False).reset_index()

    return sum_order_items_df

def create_ratings(df):
    ratings = df['review_score'].value_counts().sort_values(ascending=False)
    return ratings

def create_monthly_orders_df(df):
    monthly_orders_df = df.resample(rule='M', on='order_approved_at').agg({
        "order_id": "size",
    })

    monthly_orders_df.index = monthly_orders_df.index.strftime('%Y-%m')
    monthly_orders_df.reset_index(inplace=True)
    monthly_orders_df.rename(columns={"order_id": "order_count"}, inplace=True)
    return monthly_orders_df

def create_monthly_revenue_df(df):
    monthly_revenue_df = df.resample(rule='M', on='order_approved_at').agg({
        "price": "sum",
    })

    monthly_revenue_df.index = monthly_revenue_df.index.strftime('%Y-%m')
    monthly_revenue_df.reset_index(inplace=True)
    monthly_revenue_df.rename(columns={"price": "revenue"}, inplace=True)
    return monthly_revenue_df

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": 'max',
        'order_id': 'size',
        'price': 'sum'
    })

    rfm_df.columns = ['customer_id', 'max_order_timestamp', 'frequency', 'monetary']

    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)

    # Create Score
    recency_score = rfm_df['recency'].rank(ascending=False)
    frequency_score = rfm_df['frequency'].rank(ascending=False)
    monetary_score = rfm_df['monetary'].rank(ascending=False)

    recency_score_norm = (recency_score - recency_score.min()) / (recency_score.max() - recency_score.min()) * 5
    frequency_score_norm = (frequency_score - frequency_score.min()) / (frequency_score.max() - frequency_score.min()) * 5
    monetary_score_norm = (monetary_score - monetary_score.min()) / (monetary_score.max() - monetary_score.min()) * 5

    rfm_df['recency_score'] = recency_score_norm
    rfm_df['frequency_score'] = frequency_score_norm
    rfm_df['monetary_score'] = monetary_score_norm

    rfm_score = (rfm_df['recency_score'] + rfm_df['frequency_score'] + rfm_df['monetary_score']) / 3
    rfm_df['rfm_score'] = rfm_score

    def customer_segmentation(df):
        if (df['rfm_score'] > 4.5):
            return "Champions"
        elif (df['rfm_score'] > 4):
            return "Potential Loyalist"
        elif (df['rfm_score'] > 3):
            return "Promising"
        elif (df['rfm_score'] > 2):
            return "About To Sleep"
        else:
            return "Lost"

    rfm_df['customer_segmentation'] = rfm_df.apply(customer_segmentation, axis=1)

    return rfm_df

all_df = pd.read_csv('./dashboard/all_data.csv')
datetime_columns = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date", "shipping_limit_date",
    "review_creation_date", "review_answer_timestamp"
]

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column], errors='coerce')

min_date = all_df['order_purchase_timestamp'].min()
max_date = all_df['order_purchase_timestamp'].max()

with st.sidebar:
    st.markdown("""
    # Interactive Dashboard
    Anda dapat memilih range tanggal di bawah ini
    
    """)
    input_date= st.date_input(
        label="Rentang Tanggal",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
    if len(input_date) == 2:
        start_date, end_date = input_date
        st.success(f"Range tanggal yang dipilih dari {start_date} sampai {end_date}")
    else:
        st.error("Mohon pilih rantang tanggal akhirnya!")
try:
    # ===== SETUP DATA =====
    main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & (all_df["order_purchase_timestamp"] <= str(end_date))]
    sum_order_items_df = create_sum_order_items_df(main_df)
    ratings = create_ratings(main_df)
    monthly_orders_df = create_monthly_orders_df(main_df)
    monthly_revenue_df = create_monthly_revenue_df(main_df)
    rfm_df = create_rfm_df(main_df)
    customer_segments = rfm_df.groupby(by="customer_segmentation", as_index=False).agg({
        "customer_id": "nunique",
    })

    st.header("🛒 Brazilian E-Commerce Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Order", value=f'{main_df["order_id"].nunique():,}')
    with col2:
        st.metric("Total Customer", value=f'{main_df["customer_id"].nunique():,}')

    st.metric("Total Revenue", value=format_currency(main_df["price"].sum(), 'BRL', locale='pt_BR'))

    # ===== Number 1 GOES HERE =====
    st.subheader("Best and Worst Performing Product by Number of Order")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 9))
    colors = ["#E46A76", "#FAD4D8", "#FAD4D8", "#FAD4D8", "#FAD4D8"]

    sns.barplot(
        x="order_id",
        y="product_category_name_english",
        data=sum_order_items_df.head(5),
        palette=colors,
        ax=ax[0]
    )

    ax[0].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[0].set_title("Best Performing Product", loc="center", fontsize=20)
    ax[0].tick_params(axis="y", labelsize=18)

    sns.barplot(
        x="order_id",
        y="product_category_name_english",
        data=sum_order_items_df.sort_values(by="order_id", ascending=True).head(5),
        palette=colors,
        ax=ax[1]
    )

    ax[1].set_ylabel(None)
    ax[1].set_xlabel(None)
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Worst Performing Product", loc="center", fontsize=20)
    ax[1].tick_params(axis="y", labelsize=18)
    plt.tight_layout()
    st.pyplot(fig)
    # ===== Number 1 END HERE =====

    # ===== Number 2 GOES HERE =====
    st.subheader("Customer Ratings")
    max_ratings = ratings.idxmax()
    colors = ["#FAD4D8", "#FAD4D8", "#E46A76", "#FAD4D8", "#FAD4D8"]
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=ratings.index,
        y=ratings.values,
        palette=['#E46A76' if score == max_ratings else "#FAD4D8" for score in ratings.index],
        order=ratings.index,
        ax=ax
    )

    ax.set_title("Rating Customers by E-Commerce Service", loc="center", fontsize=20)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_xlabel("Rating", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    # ===== Number 2 END HERE =====

    # ===== Number 3 and 4 GOES HERE =====
    st.subheader("Order and Revenue")
    fig, ax = plt.subplots(figsize=(12, 7), nrows=2, ncols=1)
    ax[0].plot(
        monthly_orders_df["order_approved_at"],
        monthly_orders_df["order_count"],
        marker='o',
        linewidth=2,
        color="#E46A76"
    )
    ax[0].set_title("Number of Orders per Month", loc="center", fontsize=20)
    ax[0].set_xlabel("", labelpad=10)
    ax[0].tick_params(axis="x", labelsize=12, labelrotation=45)
    ax[0].tick_params(axis='y', labelsize=10)
    ax[0].grid()

    ax[1].plot(
        monthly_revenue_df["order_approved_at"],
        monthly_revenue_df["revenue"],
        marker='o',
        linewidth=2,
        color="#E46A76"
    )
    ax[1].set_title("Number of Orders per Month", loc="center", fontsize=20)
    ax[1].set_xlabel("", labelpad=10)
    ax[1].tick_params(axis="x", labelsize=12, labelrotation=45)
    ax[1].tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax[1].grid()
    st.pyplot(fig)
    # ===== Number 3 and 4 END HERE =====

    # ===== RFM GOES HERE =====
    st.subheader("RFM Analysis")
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(30, 20))

    sns.histplot(
        rfm_df['recency'],
        bins=50,
        ax=ax[0],
        kde=True,
        color="#E46A76"
    ).set_title('Recency Distribution', fontsize=32)
    ax[0].tick_params(axis="x", labelsize=18)
    ax[0].tick_params(axis='y', labelsize=15)
    ax[0].grid()

    sns.histplot(
        rfm_df['frequency'],
        bins=50,
        ax=ax[1],
        kde=True,
        color="#E46A76"
    ).set_title('Frequency Distribution' , fontsize=32)

    ax[1].tick_params(axis="x", labelsize=18)
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].grid()

    sns.histplot(
        rfm_df['monetary'],
        bins=50,
        ax=ax[2],
        kde=True,
        color="#E46A76"
    ).set_title('Monetary Distribution', fontsize=32)

    ax[2].tick_params(axis="x", labelsize=18)
    ax[2].tick_params(axis='y', labelsize=15)
    ax[2].grid()

    plt.tight_layout()
    st.pyplot(fig)
    # ===== RFM END HERE =====

    # ===== Segmentation GOES HERE =====
    st.subheader("Customer Segmentations")
    categories = ["Champions", "Potential Loyalist", "Promising", "About To Sleep", "Lost"]

    customer_segments.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)

    customer_segments["customer_segmentation"] = pd.Categorical(customer_segments["customer_segmentation"], categories)
    colors = ["#FAD4D8", "#FAD4D8", "#E46A76", "#FAD4D8", "#FAD4D8"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x="customer_count",
        y="customer_segmentation",
        data=customer_segments.sort_values(by="customer_count", ascending=False),
        palette=colors,
        ax=ax
    )

    ax.set_title("Number of Customer by Segmentation", loc="center", fontsize=15)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    # ===== Segmentation END HERE =====
except Exception as ex:
    st.error(ex)
    st.error(f"Pilih tanggal mulai dan akhir!")