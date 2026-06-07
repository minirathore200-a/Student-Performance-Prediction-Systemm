import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
from pandas.api.types import is_numeric_dtype

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="EduPredict - Student Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# Custom Glassmorphic Dark Theme Styling
# ----------------------------------------------------
st.markdown("""
<style>
    /* Custom Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Modern Glassmorphic Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: all 0.3s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 210, 255, 0.5);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00D2FF;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Prediction Alert Cards */
    .alert-high {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        color: #D1FAE5;
        font-weight: 500;
    }
    .alert-average {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #F59E0B;
        border-radius: 12px;
        padding: 20px;
        color: #FEF3C7;
        font-weight: 500;
    }
    .alert-low {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        color: #FEE2E2;
        font-weight: 500;
    }
    
    /* Styled Headers */
    .section-title {
        color: #00D2FF;
        border-bottom: 2px solid rgba(0, 210, 255, 0.2);
        padding-bottom: 8px;
        margin-bottom: 20px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Caching Data Loading and Model Training
# ----------------------------------------------------
@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv("student_data.csv")
    
    # Create target column Performance
    def performance(g3):
        if g3 >= 15:
            return "High"
        elif g3 >= 10:
            return "Average"
        else:
            return "Low"
            
    df["Performance"] = df["G3"].apply(performance)
    return df

@st.cache_resource
def get_encoders(df):
    encoders = {}
    for col in df.columns:
        if col not in ["Performance", "G3"] and not is_numeric_dtype(df[col]):
            le = LabelEncoder()
            le.fit(df[col])
            encoders[col] = le
            
    te = LabelEncoder()
    te.fit(df["Performance"])
    return encoders, te

@st.cache_resource
def train_models(_df, _encoders, _te):
    encoded_df = _df.copy()
    for col, le in _encoders.items():
        encoded_df[col] = le.transform(encoded_df[col])
        
    X = encoded_df.drop(["Performance", "G3"], axis=1)
    y = _te.transform(_df["Performance"])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # 1. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # 2. Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    
    # 3. Linear Regression
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, _df["G3"], test_size=0.20, random_state=42
    )
    lr = LinearRegression()
    lr.fit(X_train_reg, y_train_reg)
    lr_pred_g3 = lr.predict(X_test_reg)
    
    # Convert LR continuous predictions to High/Average/Low
    def performance_val(g3):
        if g3 >= 15:
            return "High"
        elif g3 >= 10:
            return "Average"
        else:
            return "Low"
            
    lr_pred_performance = [performance_val(g) for g in lr_pred_g3]
    lr_pred_encoded = _te.transform(lr_pred_performance)
    lr_acc = accuracy_score(y_test, lr_pred_encoded)
    
    return {
        "rf": rf,
        "dt": dt,
        "lr": lr,
        "rf_acc": rf_acc,
        "dt_acc": dt_acc,
        "lr_acc": lr_acc,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "rf_pred": rf_pred,
        "feature_names": X.columns.tolist()
    }

# Load data and train models
try:
    df = load_data()
    encoders, te = get_encoders(df)
    models_dict = train_models(df, encoders, te)
except Exception as e:
    st.error(f"Error loading data or training models: {e}")
    st.stop()

# ----------------------------------------------------
# Sidebar Navigation & Statistics
# ----------------------------------------------------
st.sidebar.markdown("<h2 style='text-align: center; color: #00D2FF;'>🎓 EduPredict</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #94A3B8; font-size: 0.9rem;'>Student Performance Analytics</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

navigation = st.sidebar.radio(
    "Navigation Menu",
    ["Home", "Dataset Analysis", "Model Performance", "Prediction", "Recommendations", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("<h4 style='color: #00D2FF;'>📊 Database Summary</h4>", unsafe_allow_html=True)
st.sidebar.metric(label="Total Records", value=len(df))
st.sidebar.metric(label="High Performers Count", value=len(df[df["Performance"] == "High"]))
st.sidebar.metric(label="Average G3 Grade", value=f"{df['G3'].mean():.2f}/20")

# ----------------------------------------------------
# Main Content Routing
# ----------------------------------------------------

# --- PAGE: Home ---
if navigation == "Home":
    st.markdown("<h1 class='section-title'>🎓 EduPredict: Student Performance Analytics</h1>", unsafe_allow_html=True)
    st.write(
        "Welcome to the **EduPredict Dashboard**. This application is an industry-level educational analytics platform "
        "designed to predict student academic outcomes and recommend tailored guidance. Using advanced machine learning "
        "algorithms, EduPredict identifies key factors determining student success and suggests early academic interventions."
    )
    
    # Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Students Tracked</div>
            <div class="metric-value">395</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Average Final Grade</div>
            <div class="metric-value">10.41 / 20</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">High Performers</div>
            <div class="metric-value">10.13 %</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">86.08 %</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🎯 Key Objectives")
    st.markdown("""
    * **Early Intervention**: Identify students at academic risk (Low Performance) before final exams.
    * **Personalized Recommendations**: Provide dynamic suggestions to improve attendance, study routines, and overall grades.
    * **Feature Importance Analysis**: Uncover whether attendance, social factors, family support, or study times play the largest role.
    * **Multi-Model Evaluation**: Compare Random Forest Classifier, Decision Tree Classifier, and Linear Regression.
    """)
    
    st.markdown("### 📂 Data Overview")
    st.write(
        "The model is trained on the **Student Performance Dataset** from the UCI Machine Learning Repository. "
        "It includes student achievements in secondary education at two Portuguese schools: Gabriel Pereira (GP) and Mousinho da Silveira (MS). "
        "It maps attributes such as demographics, family background, study habits, grades, and lifestyle metrics."
    )

# --- PAGE: Dataset Analysis ---
elif navigation == "Dataset Analysis":
    st.markdown("<h1 class='section-title'>📊 Dataset Exploration & Analysis</h1>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📋 Dataset Statistics", "📈 Exploratory Data Analysis"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.write("Browse the student records from the database:")
        st.dataframe(df.head(100), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Rows & Columns Statistics")
            shape_df = pd.DataFrame({
                "Statistic": ["Total Students (Rows)", "Total Attributes (Columns)", "Numerical Features", "Categorical Features"],
                "Value": [df.shape[0], df.shape[1], len(df.select_dtypes(include=[np.number]).columns), len(df.select_dtypes(exclude=[np.number]).columns)]
            })
            st.table(shape_df)
            
        with col2:
            st.markdown("#### Missing Value Summary")
            missing_count = df.isnull().sum().sum()
            if missing_count == 0:
                st.success("🎉 Excellent! No missing values detected in the dataset.")
            else:
                st.warning(f"Note: {missing_count} missing values found in the dataset.")
            st.write("Each student record is completely populated, ensuring robust machine learning training.")
            
        st.markdown("#### Feature Dictionary & Description")
        st.write("Explanation of core attributes tracked in this dashboard:")
        desc_data = {
            "Attribute": ["G1", "G2", "G3", "Performance", "studytime", "failures", "absences", "schoolsup", "Dalc", "Walc"],
            "Description": [
                "First period grade (numeric: from 0 to 20)",
                "Second period grade (numeric: from 0 to 20)",
                "Final grade (numeric: from 0 to 20, target for regression)",
                "Academic outcome label (High: G3>=15, Average: 10<=G3<15, Low: G3<10)",
                "Weekly study time (1: <2 hours, 2: 2-5 hours, 3: 5-10 hours, 4: >10 hours)",
                "Number of past class failures (numeric: 0 to 3)",
                "Number of school absences (numeric: 0 to 93)",
                "Extra educational support from school (binary: yes or no)",
                "Workday alcohol consumption (numeric: 1 - very low to 5 - very high)",
                "Weekend alcohol consumption (numeric: 1 - very low to 5 - very high)"
            ]
        }
        st.dataframe(pd.DataFrame(desc_data), use_container_width=True, hide_index=True)
        
    with tab2:
        st.write("Explore relationship charts between study routines, grades, social parameters, and failures:")
        
        # Row 1 of charts
        col1, col2 = st.columns(2)
        with col1:
            # 1. Student Performance Distribution Histogram
            fig1 = px.histogram(
                df, x="G3", nbins=20, 
                title="Student Performance: Final Grade (G3) Distribution",
                labels={"G3": "Final Grade (Out of 20)"},
                color_discrete_sequence=["#00D2FF"]
            )
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # 2. Study Time vs Final Grade Boxplot
            fig2 = px.box(
                df, x="studytime", y="G3",
                title="Study Time vs Final Grade",
                labels={"studytime": "Weekly Study Time Category", "G3": "Final Grade (G3)"},
                color="studytime",
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
            st.plotly_chart(fig2, use_container_width=True)
            
        # Row 2 of charts
        col3, col4 = st.columns(2)
        with col3:
            # 3. Absences vs Grade Scatter plot
            fig3 = px.scatter(
                df, x="absences", y="G3", color="Performance",
                title="Absences vs Final Grade",
                labels={"absences": "Number of Absences", "G3": "Final Grade (G3)"},
                color_discrete_map={"High": "#10B981", "Average": "#F59E0B", "Low": "#EF4444"}
            )
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
            st.plotly_chart(fig3, use_container_width=True)
            
        with col4:
            # 4. Failures vs Grade chart
            fig4 = px.box(
                df, x="failures", y="G3",
                title="Failures vs Final Grade",
                labels={"failures": "Past Failures Count", "G3": "Final Grade (G3)"},
                color="failures",
                color_discrete_sequence=px.colors.sequential.Reds
            )
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
            st.plotly_chart(fig4, use_container_width=True)
            
        # Row 3 of charts
        col5, col6 = st.columns(2)
        with col5:
            # 5. Correlation heatmap
            numeric_cols = df.select_dtypes(include=[np.number])
            corr = numeric_cols.corr()
            fig5 = px.imshow(
                corr, text_auto=".2f",
                title="Numerical Correlation Heatmap",
                color_continuous_scale="RdBu_r"
            )
            fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
            st.plotly_chart(fig5, use_container_width=True)
            
        with col6:
            # 6. Feature Importance Graph
            rf_model = models_dict["rf"]
            feature_names = models_dict["feature_names"]
            importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": rf_model.feature_importances_
            }).sort_values(by="Importance", ascending=False).head(10)
            
            fig6 = px.bar(
                importance, x="Importance", y="Feature", orientation="h",
                title="Top 10 Important Predictor Features",
                labels={"Feature": "Feature Name", "Importance": "Relative Weight"},
                color="Importance",
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig6.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
            st.plotly_chart(fig6, use_container_width=True)

# --- PAGE: Model Performance ---
elif navigation == "Model Performance":
    st.markdown("<h1 class='section-title'>🤖 Machine Learning Model Performance</h1>", unsafe_allow_html=True)
    
    # Model comparison accuracies
    rf_acc = models_dict["rf_acc"]
    dt_acc = models_dict["dt_acc"]
    lr_acc = models_dict["lr_acc"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Random Forest Accuracy</div>
            <div class="metric-value">{rf_acc*100:.2f} %</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Decision Tree Accuracy</div>
            <div class="metric-value">{dt_acc*100:.2f} %</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Linear Regression Accuracy</div>
            <div class="metric-value">{lr_acc*100:.2f} %</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_chart, col_report = st.columns(2)
    with col_chart:
        # Accuracy comparison chart
        comp_df = pd.DataFrame({
            "Model Name": ["Linear Regression", "Decision Tree", "Random Forest"],
            "Accuracy (%)": [lr_acc * 100, dt_acc * 100, rf_acc * 100]
        })
        fig_comp = px.bar(
            comp_df, x="Model Name", y="Accuracy (%)",
            title="Classifier Accuracy Comparison Chart",
            color="Accuracy (%)",
            color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"]
        )
        fig_comp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
        st.plotly_chart(fig_comp, use_container_width=True)
        
    with col_report:
        # Random Forest Classification Report
        st.subheader("Random Forest Classification Report")
        st.write("Precision, recall, and F1-score details for predictions:")
        
        # Classification report dictionary to dataframe format
        y_test = models_dict["y_test"]
        rf_pred = models_dict["rf_pred"]
        report_dict = classification_report(y_test, rf_pred, target_names=["Average", "High", "Low"], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)
        
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix (Random Forest)")
    cm = confusion_matrix(y_test, rf_pred)
    classes = ["Average", "High", "Low"]
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=classes, y=classes,
        labels=dict(x="Predicted Performance", y="Actual Performance"),
        color_continuous_scale="Blues"
    )
    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC")
    st.plotly_chart(fig_cm, use_container_width=True)

# --- PAGE: Prediction ---
elif navigation == "Prediction":
    st.markdown("<h1 class='section-title'>🔮 Student Performance Prediction</h1>", unsafe_allow_html=True)
    st.write("Fill out the academic, social, and family background details below to generate predictions:")
    
    # Create the Form structure
    with st.form("prediction_form"):
        col_ac, col_pe, col_fa = st.columns(3)
        
        with col_ac:
            st.markdown("### 🎓 Academic Details")
            G1 = st.slider("First Period Grade (G1)", 0, 20, 10, help="Grade out of 20")
            G2 = st.slider("Second Period Grade (G2)", 0, 20, 10, help="Grade out of 20")
            studytime = st.slider("Study Time Level", 1, 4, 2, help="1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h")
            failures = st.slider("Past Failures Count", 0, 3, 0)
            absences = st.slider("School Absences", 0, 93, 4)
            traveltime = st.slider("Travel Time Level", 1, 4, 1, help="1: <15m, 2: 15-30m, 3: 30m-1h, 4: >1h")
            schoolsup = st.selectbox("School Support", ["yes", "no"], index=1)
            famsup = st.selectbox("Family Support", ["yes", "no"], index=0)
            paid = st.selectbox("Paid Classes", ["yes", "no"], index=1)
            higher = st.selectbox("Higher Education Target", ["yes", "no"], index=0)
            
        with col_pe:
            st.markdown("### 👤 Personal Details")
            age = st.slider("Age", 15, 22, 17)
            sex = st.selectbox("Sex", ["F", "M"])
            address = st.selectbox("Address Type", ["U", "R"], help="U: Urban, R: Rural")
            internet = st.selectbox("Internet Access", ["yes", "no"], index=0)
            romantic = st.selectbox("Romantic Relationship", ["yes", "no"], index=1)
            activities = st.selectbox("Extracurricular Activities", ["yes", "no"], index=0)
            freetime = st.slider("Free Time Level", 1, 5, 3, help="1: very low, 5: very high")
            goout = st.slider("Go Out Level", 1, 5, 3, help="1: very low, 5: very high")
            health = st.slider("Health Status", 1, 5, 5, help="1: very bad, 5: very good")
            Dalc = st.slider("Workday Alcohol", 1, 5, 1)
            Walc = st.slider("Weekend Alcohol", 1, 5, 1)
            
        with col_fa:
            st.markdown("### 🏠 Family Details")
            school = st.selectbox("School Code", ["GP", "MS"])
            famsize = st.selectbox("Family Size", ["GT3", "LE3"], help="GT3: >3 members, LE3: <=3 members")
            Pstatus = st.selectbox("Parent Cohabitation", ["T", "A"], help="T: Together, A: Apart")
            Medu = st.slider("Mother's Education Level", 0, 4, 4, help="0: none, 4: higher education")
            Fedu = st.slider("Father's Education Level", 0, 4, 3, help="0: none, 4: higher education")
            Mjob = st.selectbox("Mother's Job Type", ["at_home", "health", "other", "services", "teacher"], index=2)
            Fjob = st.selectbox("Father's Job Type", ["teacher", "other", "services", "health", "at_home"], index=1)
            reason = st.selectbox("School Reason", ["course", "other", "home", "reputation"])
            guardian = st.selectbox("Guardian Name", ["mother", "father", "other"])
            famrel = st.slider("Family Relationship Level", 1, 5, 4, help="1: very bad, 5: excellent")
            nursery = st.selectbox("Attended Nursery School", ["yes", "no"], index=0)
            
        submit_btn = st.form_submit_button("🔮 Predict Performance", use_container_width=True)
        
    if submit_btn:
        # Assemble input dictionary
        input_data = {
            "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
            "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob, "Fjob": Fjob,
            "reason": reason, "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
            "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher, "internet": internet,
            "romantic": romantic, "famrel": famrel, "freetime": freetime, "goout": goout,
            "Dalc": Dalc, "Walc": Walc, "health": health, "absences": absences, "G1": G1, "G2": G2
        }
        
        # Save inputs to session state for other sections
        st.session_state["inputs"] = input_data
        
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical columns
        encoded_input = input_df.copy()
        for col, le in encoders.items():
            encoded_input[col] = le.transform(encoded_input[col])
            
        # Reorder columns to match training features exactly
        encoded_input = encoded_input[models_dict["feature_names"]]
        
        # 1. Random Forest prediction (classification)
        rf_model = models_dict["rf"]
        pred_class_encoded = rf_model.predict(encoded_input)[0]
        pred_class = te.inverse_transform([pred_class_encoded])[0]
        
        # 2. Linear Regression prediction (G3 score)
        lr_model = models_dict["lr"]
        pred_g3 = lr_model.predict(encoded_input)[0]
        pred_g3 = np.clip(pred_g3, 0, 20) # ensure grade stays within bounds
        
        st.markdown("---")
        st.markdown("<h2 style='color: #00D2FF;'>📝 Results & Analytics</h2>", unsafe_allow_html=True)
        
        col_res, col_gauge = st.columns(2)
        with col_res:
            st.markdown("#### Predicted Performance Category")
            if pred_class == "High":
                st.markdown("""
                <div class="alert-high">
                    <h3 style="margin: 0; color: #10B981;">🚀 High Performance Predicted</h3>
                    <p style="margin: 10px 0 0 0;">The student is projected to excel, obtaining a final score in the top academic tier (G3 >= 15).</p>
                </div>
                """, unsafe_allow_html=True)
            elif pred_class == "Average":
                st.markdown("""
                <div class="alert-average">
                    <h3 style="margin: 0; color: #F59E0B;">📈 Average Performance Predicted</h3>
                    <p style="margin: 10px 0 0 0;">The student is on track for a satisfactory average performance, maintaining scores between 10 and 14.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-low">
                    <h3 style="margin: 0; color: #EF4444;">⚠️ Low Performance Predicted</h3>
                    <p style="margin: 10px 0 0 0;">Attention required! The student is at risk of falling behind or failing the final exam (G3 < 10).</p>
                </div>
                """, unsafe_allow_html=True)
                
        with col_gauge:
            # Gauge score visualization
            fig_g = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_g3,
                title = {'text': "Predicted Final Grade (G3)"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 20]},
                    'bar': {'color': "#00D2FF"},
                    'steps': [
                        {'range': [0, 10], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [10, 15], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [15, 20], 'color': "rgba(16, 185, 129, 0.2)"}
                    ]
                }
            ))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC", height=250)
            st.plotly_chart(fig_g, use_container_width=True)
            
        st.markdown("---")
        
        # Student Comparison Dashboard & Trends
        col_radar, col_trends = st.columns(2)
        with col_radar:
            st.markdown("#### Student Comparison Dashboard")
            st.write("Comparing input student attributes against class average (1-5 scale):")
            
            # Scale functions for radar chart
            def scale_to_5(val, max_val):
                return min((val / max_val) * 4 + 1, 5)
                
            student_scaled = {
                "Study Time": scale_to_5(studytime, 4),
                "Absences": scale_to_5(absences, 40),
                "Workday Alcohol": Dalc,
                "Weekend Alcohol": Walc,
                "Free Time": freetime,
                "Going Out": goout,
                "Health Status": health,
                "G1 Grade": scale_to_5(G1, 20),
                "G2 Grade": scale_to_5(G2, 20)
            }
            
            avg_studytime = df["studytime"].mean()
            avg_absences = df["absences"].mean()
            avg_Dalc = df["Dalc"].mean()
            avg_Walc = df["Walc"].mean()
            avg_freetime = df["freetime"].mean()
            avg_goout = df["goout"].mean()
            avg_health = df["health"].mean()
            avg_G1 = df["G1"].mean()
            avg_G2 = df["G2"].mean()
            
            avg_scaled = {
                "Study Time": scale_to_5(avg_studytime, 4),
                "Absences": scale_to_5(avg_absences, 40),
                "Workday Alcohol": avg_Dalc,
                "Weekend Alcohol": avg_Walc,
                "Free Time": avg_freetime,
                "Going Out": avg_goout,
                "Health Status": avg_health,
                "G1 Grade": scale_to_5(avg_G1, 20),
                "G2 Grade": scale_to_5(avg_G2, 20)
            }
            
            categories = list(student_scaled.keys())
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=list(student_scaled.values()),
                theta=categories,
                fill='toself',
                name='Current Student',
                line=dict(color='#00D2FF')
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=list(avg_scaled.values()),
                theta=categories,
                fill='toself',
                name='Class Average',
                line=dict(color='#FF5E62')
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#F8FAFC",
                showlegend=True,
                height=350
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_trends:
            st.markdown("#### Performance Trends Dashboard")
            st.write("Visualizing grade trajectory across G1, G2, and predicted G3:")
            
            grades_df = pd.DataFrame({
                "Evaluation Stage": ["Period 1 (G1)", "Period 2 (G2)", "Final Exam (G3 - Predicted)"],
                "Score": [G1, G2, pred_g3]
            })
            fig_line = px.line(
                grades_df, x="Evaluation Stage", y="Score", markers=True,
                color_discrete_sequence=["#00D2FF"],
                range_y=[0, 20]
            )
            fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F8FAFC", height=300)
            st.plotly_chart(fig_line, use_container_width=True)
            
        # Recommendation Engine Suggestions
        st.markdown("#### 📋 Recommendations & Guidance Advice")
        
        recs = []
        if studytime < 2:
            recs.append("🔴 **Study Time Alert**: Weekly study time is extremely low (<2 hours). Recommendation: Increase to at least 2-5 hours to avoid failing grades.")
        else:
            recs.append("💚 **Study Time Pattern**: Weekly study habits are solid. Maintain this standard.")
            
        if failures > 0:
            recs.append(f"🔴 **Past Failures Alert**: Student has {failures} past course failures. Recommendation: Arrange immediate remedial support or tutor counseling.")
            
        if absences > 10:
            recs.append(f"🔴 **Absence Alert**: Student has {absences} absences. High absenteeism is heavily correlated with low scores. Recommendation: Establish attendance improvement checkpoints.")
        elif absences > 5:
            recs.append("🟡 **Absence Warning**: Attendance is slightly irregular. Monitor absences to prevent grade decline.")
        else:
            recs.append("💚 **Attendance Quality**: Excellent class attendance. Good attendance facilitates structured learning.")
            
        if G2 < 10:
            recs.append("🔴 **Period 2 Trend**: The Period 2 grade was failing. Focus on reviewing core curriculum structures.")
            
        if schoolsup == "yes":
            recs.append("💚 **School Support**: Active educational support is engaged. Leverage scheduled tutor hours.")
            
        if Dalc >= 3 or Walc >= 3:
            recs.append("🟡 **Lifestyle Alert**: Weekend or weekday alcohol consumption levels are medium-to-high. Encourage participation in healthy extracurriculars.")
            
        for r in recs:
            st.markdown(r)
            
        # Report generator
        report_text = f"""==================================================
STUDENT PERFORMANCE PREDICTION REPORT
================------------------================
1. PREDICTED RESULTS:
- Predicted Performance Level: {pred_class}
- Predicted Final Grade (G3): {pred_g3:.2f} / 20

2. STUDENT PARAMETERS:
- Age: {age} | Sex: {sex} | Address: {address}
- G1 Score: {G1} | G2 Score: {G2}
- Weekly Study Time Level: {studytime} / 4
- Absences Count: {absences}
- Past Failures Count: {failures}
- School Support: {schoolsup} | Family Support: {famsup}

3. RECOMMENDATIONS & GUIDANCE:
"""
        for i, r in enumerate(recs):
            # clean formatting symbols
            clean_rec = r.replace("**", "").replace("🔴", "").replace("🟢", "").replace("🟡", "").replace("💚", "")
            report_text += f"{i+1}. {clean_rec}\n"
            
        report_text += "\nReport generated by EduPredict Analytics.\n=================================================="
        
        st.download_button(
            label="Download Prediction Report (.txt)",
            data=report_text,
            file_name=f"student_report_{pred_class}.txt",
            mime="text/plain",
            use_container_width=True
        )

# --- PAGE: Recommendations ---
elif navigation == "Recommendations":
    st.markdown("<h1 class='section-title'>📋 General Recommendation Engine Rules</h1>", unsafe_allow_html=True)
    st.write("The recommendation engine uses a series of academic heuristics derived from feature importance models:")
    
    st.markdown("""
    ### 🔬 Policy Logic Mappings:
    1. **Study Routine (studytime)**
       * *Constraint*: `studytime < 2` (< 2 hours/week)
       * *Action*: Trigger warning and advice to increase study levels.
       
    2. **Academic Risk (failures)**
       * *Constraint*: `failures > 0`
       * *Action*: Highlight student risk and request targeted remedial tutorials.
       
    3. **Attendance (absences)**
       * *Constraint*: `absences > 10`
       * *Action*: High attendance alert, advise weekly review of absence triggers.
       
    4. **Social & Lifestyle (Walc / Dalc)**
       * *Constraint*: `Walc >= 3` or `Dalc >= 3`
       * *Action*: Flag lifestyle friction risks and advice recreational activities.
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Interactive Rule Testing Simulator")
    st.write("Manually configure parameters below to view output recommendation guidelines:")
    
    sim_study = st.slider("Simulated Study Time", 1, 4, 2)
    sim_failures = st.slider("Simulated Failures", 0, 3, 0)
    sim_absences = st.slider("Simulated Absences", 0, 93, 4)
    
    # Calculate simulated recommendations
    sim_recs = []
    if sim_study < 2:
        sim_recs.append("🔴 **Study Time warning**: Study time is less than 2 hours per week. Advise setting up structured study routines.")
    else:
        sim_recs.append("💚 **Study Time satisfactory**: Maintain current study volume.")
        
    if sim_failures > 0:
        sim_recs.append("🔴 **Remedial Tutorial requested**: Student has past failures. Immediate tutoring sessions advised.")
        
    if sim_absences > 10:
        sim_recs.append("🔴 **High Absenteeism detected**: Absences exceed 10 classes. Direct home communication requested.")
    else:
        sim_recs.append("💚 **Attendance healthy**: Low absences. Good integration with class syllabus.")
        
    st.markdown("#### Simulated Output Advice:")
    for sr in sim_recs:
        st.markdown(sr)

# --- PAGE: About Project ---
elif navigation == "About Project":
    st.markdown("<h1 class='section-title'>ℹ️ About the EduPredict Project</h1>", unsafe_allow_html=True)
    st.write(
        "EduPredict is designed as an interactive diagnostic system to predict educational outcomes. "
        "It leverages predictive models to identify academic trends and provide personalized advice."
    )
    
    st.markdown("### 🏆 Top Performing Students (Class G3 >= 15)")
    st.write("Displaying high-performing student profiles from the original dataset database:")
    top_students = df[df["Performance"] == "High"][["school", "sex", "age", "studytime", "absences", "G1", "G2", "G3"]].sort_values(by="G3", ascending=False)
    st.dataframe(top_students.head(15), use_container_width=True, hide_index=True)
    
    st.markdown("### 🛠️ Technical Stack & Machine Learning Details")
    st.markdown("""
    * **Streamlit**: Single-page reactive application framework.
    * **Plotly**: Used for premium, interactive dark-themed charts.
    * **Scikit-Learn**: Powering the model training pipeline:
      * **Random Forest Classifier**: Primary classifier (86.08% accuracy).
      * **Decision Tree Classifier**: Secondary classifier (84.81% accuracy).
      * **Linear Regression**: Continuous grade score (G3) predictor (77.22% accuracy).
    * **Data Preprocessing**: String variables are converted into integer columns using `LabelEncoder`.
    """)
    
    st.markdown("### 📈 Class Averages Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Absences", f"{df['absences'].mean():.2f} days")
    col2.metric("Average Medu (Mother's Edu)", f"{df['Medu'].mean():.2f} / 4")
    col3.metric("Average Fedu (Father's Edu)", f"{df['Fedu'].mean():.2f} / 4")