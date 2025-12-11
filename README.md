# E-Commerce Customer Lifetime Value (CLV) Prediction and Segmentation

A comprehensive data science project demonstrating advanced customer lifetime value prediction using probabilistic models, RFM analysis, and machine learning clustering techniques.

## 📋 Project Overview

This project implements a complete end-to-end pipeline for predicting customer lifetime value (CLV) in e-commerce settings. It combines traditional RFM (Recency, Frequency, Monetary) analysis with advanced probabilistic models (BG/NBD and Pareto/NBD) and K-Means clustering to provide actionable business insights.

### Key Features

- **Data Processing**: Complete data cleaning, validation, and preparation pipeline
- **RFM Analysis**: Customer segmentation using Recency, Frequency, and Monetary metrics
- **K-Means Clustering**: Optimal cluster determination with Silhouette Score analysis
- **Advanced CLV Modeling**: 
  - BG/NBD (Beta-Geometric/Negative Binomial Distribution)
  - Pareto/NBD models for probabilistic CLV prediction
- **Business Insights**: Actionable recommendations for each customer segment
- **Comprehensive Visualization**: RFM distributions, cluster analysis, and CLV predictions

## 🎯 Why This Project Stands Out

1. **Rare Technical Knowledge**: BG/NBD and Pareto/NBD models are advanced probabilistic techniques that 90% of junior data scientists don't know
2. **Business Value**: Not just model building—translates predictions into marketing strategies
3. **End-to-End Pipeline**: Complete workflow from data ingestion to business recommendations
4. **Production-Ready Code**: Clean, modular, well-documented Python code
5. **Scalability**: Designed to handle large datasets efficiently

## 📊 Dataset

- **Source**: E-commerce transaction data (100,000 transactions)
- **Customers**: 10,000 unique customers
- **Time Period**: 2018-2025 (7 years)
- **Features**: Transaction ID, Customer ID, Date, Amount, Product Category, Payment Method, Demographics

## 🏗️ Project Structure

```
ecommerce-clv-prediction/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                           # Project setup
├── .gitignore                         # Git ignore rules
│
├── data/
│   ├── raw/                           # Raw data
│   │   └── ecommerce_transactions.csv
│   ├── processed/                     # Processed data
│   │   ├── rfm_with_clv.parquet
│   │   ├── segmented_customers.parquet
│   │   └── rfm_analysis.parquet
│   └── external/                      # External data sources
│
├── notebooks/
│   ├── 01_complete_clv_analysis.ipynb # Complete analysis notebook
│   └── 01_complete_clv_analysis_executed.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py             # Data cleaning and preparation
│   ├── rfm_analysis.py                # RFM metrics and analysis
│   ├── segmentation.py                # K-Means clustering
│   ├── clv_modeling.py                # BG/NBD and Pareto/NBD models
│   ├── visualization.py               # Visualization functions
│   └── utils.py                       # Utility functions
│
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_rfm_analysis.py
│   ├── test_segmentation.py
│   └── test_clv_modeling.py
│
├── reports/
│   ├── figures/                       # Generated visualizations
│   ├── executive_summary.md           # Executive summary
│   └── detailed_report.html           # Detailed report
│
├── config/
│   └── config.yaml                    # Project configuration
│
├── generate_sample_data.py            # Sample data generation
└── run_complete_analysis.py           # Complete analysis script
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ecommerce-clv-prediction.git
cd ecommerce-clv-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Generate sample data** (or use your own)
```bash
python generate_sample_data.py
```

2. **Run complete analysis**
```bash
python run_complete_analysis.py
```

3. **Run Jupyter Notebook**
```bash
jupyter notebook notebooks/01_complete_clv_analysis.ipynb
```

## 📈 Analysis Pipeline

### Step 1: Data Loading and Exploration
- Load transaction data
- Explore data characteristics
- Check data quality

### Step 2: Data Cleaning and Preparation
- Handle missing values
- Remove duplicates
- Detect and treat outliers
- Convert data types

### Step 3: RFM Analysis
- Calculate Recency, Frequency, Monetary metrics
- Generate RFM scores (1-5 scale)
- Segment customers into 8 RFM segments:
  - Champions
  - Loyal Customers
  - Potential Loyalists
  - At Risk
  - Need Attention
  - New Customers
  - Lost
  - Others

### Step 4: Customer Segmentation (K-Means)
- Standardize RFM features
- Determine optimal number of clusters using Silhouette Score
- Apply K-Means clustering
- Analyze cluster profiles

### Step 5: Advanced CLV Modeling

#### BG/NBD Model
- Beta-Geometric/Negative Binomial Distribution
- Predicts:
  - Probability of customer being active
  - Expected number of transactions
  - Customer lifetime value

#### Pareto/NBD Model
- Pareto/Negative Binomial Distribution
- Alternative probabilistic approach
- Better for certain customer patterns

### Step 6: Model Evaluation
- Compare model performance
- Calculate prediction accuracy
- Generate business metrics

### Step 7: Business Insights
- Identify high-value customers
- Analyze segment characteristics
- Generate marketing recommendations

## 📊 Key Metrics

### RFM Metrics
- **Recency**: Days since last purchase
- **Frequency**: Number of purchases
- **Monetary**: Total spending

### CLV Metrics
- **Average CLV**: Mean customer lifetime value
- **Median CLV**: Median customer lifetime value
- **Total CLV**: Sum of all customer values
- **CLV by Segment**: Value distribution across segments

### Model Metrics
- **Mean Absolute Error (MAE)**: Average prediction error
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors
- **Mean Absolute Percentage Error (MAPE)**: Percentage error
- **R² Score**: Proportion of variance explained

## 💡 Business Recommendations

### Champions (Highest Value Customers)
- **Strategy**: VIP treatment, exclusive offers, loyalty programs
- **Action**: Personalized communication, early access to new products
- **Investment**: High—these customers drive significant revenue

### Loyal Customers
- **Strategy**: Retention programs, personalized recommendations
- **Action**: Regular engagement, special discounts, membership benefits
- **Investment**: Medium-High—maintain and grow relationships

### At Risk (High Value but Low Activity)
- **Strategy**: Win-back campaigns, special discounts, re-engagement
- **Action**: Personalized offers, feedback surveys, exclusive deals
- **Investment**: Medium—recover lost revenue

### New Customers
- **Strategy**: Onboarding programs, welcome offers, education
- **Action**: Product education, first-purchase incentives, nurture campaigns
- **Investment**: Medium—build long-term relationships

## 🔧 Technologies Used

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Polars**: High-performance data processing (optional)

### Machine Learning
- **Scikit-learn**: K-Means clustering, preprocessing
- **Lifetimes**: BG/NBD and Pareto/NBD models
- **SciPy**: Statistical functions

### Visualization
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive visualizations

### Development
- **Jupyter**: Interactive notebooks
- **pytest**: Unit testing
- **Black**: Code formatting
- **Flake8**: Code linting

## 📚 Key Concepts

### RFM Analysis
RFM is a quantitative method to rank and segment customers based on three key metrics:
- **Recency**: How recently a customer made a purchase
- **Frequency**: How often a customer makes purchases
- **Monetary**: How much money a customer spends

### BG/NBD Model
The Beta-Geometric/Negative Binomial Distribution model:
- Assumes customers can become inactive
- Models purchase frequency with negative binomial distribution
- Estimates customer lifetime value probabilistically
- Particularly useful for non-contractual business settings

### Pareto/NBD Model
The Pareto/Negative Binomial Distribution model:
- Alternative probabilistic approach
- Assumes exponential lifetime distribution
- Better for certain customer behavior patterns
- Provides complementary insights to BG/NBD

### K-Means Clustering
Unsupervised learning algorithm that:
- Partitions customers into k clusters
- Minimizes within-cluster variance
- Optimal k determined using Silhouette Score
- Useful for customer segmentation

## 📈 Expected Results

### Sample Analysis Results
- **Total Customers**: 10,000
- **Total Transactions**: 100,000
- **Total Revenue**: $10,015,143.57
- **Average CLV**: $1,001.51
- **Optimal Clusters**: 3

### Segment Distribution
- Champions: 15.3% of customers
- Loyal Customers: 35.1% of customers
- At Risk: 4.4% of customers
- New Customers: 5.4% of customers
- Others: 40.2% of customers

## 🎓 Learning Outcomes

By studying this project, you'll learn:

1. **Advanced Statistical Modeling**: Probabilistic models for customer behavior
2. **Machine Learning**: Clustering and segmentation techniques
3. **Data Engineering**: Complete data pipeline from raw to insights
4. **Business Analytics**: Translating models into actionable strategies
5. **Software Engineering**: Professional code structure and documentation
6. **Data Visualization**: Effective communication of insights

## 🔬 Advanced Topics Covered

- **Probabilistic Modeling**: BG/NBD and Pareto/NBD
- **Maximum Likelihood Estimation**: Parameter optimization
- **Bayesian Inference**: Posterior probability calculations
- **Feature Scaling**: StandardScaler for clustering
- **Cross-Validation**: Model evaluation techniques
- **Hyperparameter Optimization**: Optimal cluster determination

## 📝 Code Quality

- **Modular Design**: Separate modules for different functionalities
- **Documentation**: Comprehensive docstrings and comments
- **Type Hints**: Python type annotations for clarity
- **Error Handling**: Robust error management
- **Logging**: Detailed logging for debugging
- **Testing**: Unit tests for core functions

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💼 About the Author

This project demonstrates advanced data science skills including:
- Statistical modeling
- Machine learning
- Business analytics
- Software engineering
- Data visualization
- Professional communication

Perfect for portfolio showcasing to Fortune 500 companies and fintech firms.

## 🔗 References

1. **BG/NBD Model**: Fader, P. S., & Hardie, B. G. (2005). "A Note on Deriving the Pareto/NBD and Related Models"
2. **RFM Analysis**: Pfeifer, P. E., & Carraway, R. L. (2000). "Modeling customer relationships as Markov chains"
3. **Lifetimes Library**: https://github.com/CamDavidsonPilon/lifetimes
4. **Customer Segmentation**: Wedel, M., & Kamakura, W. A. (2012). "Market Segmentation: Conceptual and Methodological Foundations"

## 📞 Contact

For questions or collaboration opportunities, please reach out through GitHub or email.

---

**Last Updated**: December 2025  
**Status**: Production Ready  
**Version**: 1.0.0
