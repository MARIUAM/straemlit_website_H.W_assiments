# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="CSV Data Explorer", layout="wide")

# st.title("📊 CSV Data Explorer with Temperature Plot")

# # CSV upload
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     # Data preview
#     st.subheader("📋 Data Preview")
#     st.dataframe(df)

#     # Data summary
#     st.subheader("📈 Data Summary")
#     st.write(df.describe())

#     # Temperature Plot (if column exists)
#     st.subheader("🌡️ Temperature Plot")
#     temp_col = st.selectbox("Select temperature column", df.columns)

#     if pd.api.types.is_numeric_dtype(df[temp_col]):
#         plt.figure(figsize=(10, 4))
#         plt.plot(df[temp_col], marker='o', color='orange')
#         plt.title(f"Temperature Trend: {temp_col}")
#         plt.xlabel("Index")
#         plt.ylabel("Temperature")
#         st.pyplot(plt)
#     else:
#         st.warning("Selected column is not numeric.")


import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Theme Switcher
st.set_page_config(page_title="Advanced CSV Analyzer", layout="wide")

# Sidebar
st.sidebar.title("🔧 Options")
lang = st.sidebar.radio("🌐 Language / زبان", ["English", "Urdu"])
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"])

# Language Support
if lang == "Urdu":
    def t(msg_en, msg_ur): return msg_ur
else:
    def t(msg_en, msg_ur): return msg_en

st.title(t("📊 Advanced CSV Analyzer", "📊 ایڈوانس CSV تجزیہ کار"))

# Upload CSV
file = st.file_uploader(t("Upload your CSV file", "اپنی CSV فائل اپ لوڈ کریں"), type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success(t("File uploaded successfully!", "فائل کامیابی سے اپ لوڈ ہوگئی!"))

    # Data Preview
    st.subheader(t("🔍 Data Preview", "🔍 ڈیٹا کا جائزہ"))
    st.dataframe(df.head())

    # Data Summary
    st.subheader(t("📋 Data Summary", "📋 ڈیٹا کی سمری"))
    st.write(df.describe())

    # Missing Value Finder
    st.subheader(t("🚨 Missing Values", "🚨 غائب اقدار"))
    st.write(df.isnull().sum())
    if st.button(t("Remove Missing Values", "غائب اقدار کو ہٹائیں")):
        df.dropna(inplace=True)
        st.success(t("Missing values removed", "غائب اقدار ہٹا دی گئیں"))

    # Outlier Detection
    st.subheader(t("📌 Outlier Detection", "📌 آؤٹ لائر کی شناخت"))
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        col = st.selectbox(t("Select column for outlier check", "آؤٹ لائر کے لیے کالم منتخب کریں"), numeric_cols)
        fig, ax = plt.subplots(figsize=(6, 2.5))
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)

    # Column Selector
    st.subheader(t("📊 Select Columns to Plot", "📊 تجزیہ کے لیے کالم منتخب کریں"))
    selected_cols = st.multiselect(t("Choose columns", "کالم منتخب کریں"), df.columns.tolist())

    # Plotting Options
    st.subheader(t("📈 Plot Options", "📈 گراف کے اختیارات"))
    plot_type = st.selectbox(t("Select plot type", "گراف کی قسم منتخب کریں"), ["Line", "Bar", "Pie", "Histogram", "Correlation"])

    if plot_type == "Line" and selected_cols:
        st.plotly_chart(px.line(df[selected_cols]))
    elif plot_type == "Bar" and selected_cols:
        st.plotly_chart(px.bar(df[selected_cols]))
    elif plot_type == "Pie" and selected_cols:
        st.plotly_chart(px.pie(df, names=selected_cols[0]))
    elif plot_type == "Histogram" and selected_cols:
        st.plotly_chart(px.histogram(df[selected_cols]))
    elif plot_type == "Correlation":
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # Basic ML Prediction
    st.subheader(t("🧠 Basic ML Predictor", "🧠 بنیادی مشین لرننگ پیشگوئی"))
    target = st.selectbox(t("Select Target Column", "ٹارگٹ کالم منتخب کریں"), numeric_cols)
    features = st.multiselect(t("Select Features", "فیچرز منتخب کریں"), [col for col in numeric_cols if col != target])
    if features and target:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        st.write(t(f"Mean Squared Error: {mse}", f"ایم ایس ای: {mse}"))

    # Download CSV
    st.subheader(t("📤 Download Processed CSV", "📤 پراسیس شدہ CSV ڈاؤن لوڈ کریں"))
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)
    st.download_button(
        label=t("Download CSV", "CSV ڈاؤن لوڈ کریں"),
        data=csv,
        file_name='processed_data.csv',
        mime='text/csv',
    )

    # Feedback Form
    st.subheader(t("💬 Feedback", "💬 رائے"))
    feedback = st.text_area(t("What do you think about this tool?", "آپ اس ٹول کے بارے میں کیا سوچتے ہیں؟"))
    if feedback:
        st.success(t("Thanks for your feedback!", "آپ کی رائے کا شکریہ!"))

    feedback = st.text_area("Leave your feedback here:")
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thanks for your feedback!")
        else:
            st.warning("Feedback cannot be empty.")
            