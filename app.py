import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.title("Loan Approval System")
st.subheader("Know your loan approval probability")

Gender = st.selectbox(
    'Gender',
    ('Male', 'Female'))
Married = st.selectbox(
    'Married',
    ('Yes', 'No'))
Dependents = st.selectbox(
    'Dependents',
    ('Yes', 'No'))
Education = st.selectbox(
    'Education',
    ('Graduate', 'Not Graduate'))
Self_Employed = st.selectbox(
    'Are you self_employed?',
    ('Yes', 'No'))
ApplicantIncome = st.number_input('Applicant Income (in thousands)', value=0, step=1)
Co_ApplicantIncome = st.number_input('Co. Applicant Income(in thousands)', value=0, step=1)
LoanAmount = st.number_input('Required Loan Amount (in thousands)', value=0, step=1)
Loan_Amount_Term = st.selectbox(
    'Loan Amount Duration',
    ('180 months', '360 months'))
Credit_History = st.selectbox(
    'Do you have a good Credit History?',
    ('Good', 'Bad'))
Property_Area = st.selectbox(
    'Property Area',
    ('Rural', 'Urban', 'Semi-Urban'))

cr = pd.read_csv("CreditRisk.csv")
cr.Gender.fillna("Male", inplace=True)
cr.Married.fillna("No", inplace=True)
cr.Dependents.fillna(0, inplace=True)
cr.Self_Employed.fillna("No", inplace=True)
cr.LoanAmount.fillna(cr.LoanAmount.median(), inplace=True)
cr.Loan_Amount_Term.fillna(cr.Loan_Amount_Term.median(), inplace=True)
cr.Credit_History.fillna(1, inplace=True)
cr = cr.drop(["Loan_ID"], axis=1)
cr.Gender.replace({'Male': 0, 'Female': 1}, inplace=True)
cr.Married.replace({'No': 0, 'Yes': 1}, inplace=True)
cr.Education.replace({'Not Graduate': 0, 'Graduate': 1}, inplace=True)
cr.Self_Employed.replace({'No': 0, 'Yes': 1}, inplace=True)
cr.Property_Area.replace({'Rural': 0, 'Semi-urban': 1, 'Urban': 2}, inplace=True)
cr.Loan_Status = cr.Loan_Status.replace({"Y": 1, "N": 0})

cr_train, cr_test = train_test_split(cr, test_size=.2)
cr_train_x = cr_train.iloc[:, 0:-1]
cr_train_y = cr_train.iloc[:, -1]
cr_test_x = cr_test.iloc[:, 0:-1]
cr_test_y = cr_test.iloc[:, -1]

dt = DecisionTreeClassifier(class_weight="balanced", max_depth=8,
                            min_samples_split=50)
dt.fit(cr_train_x, cr_train_y)


def Loan_approval_calculator():
    df = pd.DataFrame()
    print("Please input the details as required")
    df['Gender'] = [Gender.lower()]
    df['Married'] = [Married.lower()]
    df['Dependents'] = [Dependents]
    df['Education'] = [Education.lower()]
    df['Self_Employed'] = [Self_Employed.lower()]
    df['ApplicantIncome'] = ['ApplicantIncome']
    df['Co_ApplicantIncome'] = [Co_ApplicantIncome]
    df['LoanAmount'] = [LoanAmount]
    df['Loan_Amount_Term'] = [Loan_Amount_Term]
    df['Credit_History'] = [Credit_History]
    df['Property_Area'] = [Property_Area]
    df.Gender.replace({'male': 0, 'female': 1}, inplace=True)
    df.Married.replace({'no': 0, 'yes': 1}, inplace=True)
    df.Dependents.replace({'no': 0, 'yes': 1}, inplace=True)
    df.Loan_Amount_Term.replace({'180 months': 180, '360 months': 360}, inplace=True)
    df.Education.replace({'not graduate': 0, 'graduate': 1}, inplace=True)
    df.Self_Employed.replace({'no': 0, 'yes': 1}, inplace=True)
    df.Credit_History.replace({'bad': 0, 'good': 1}, inplace=True)
    df.Property_Area.replace({'rural': 0, 'semi-urban': 1, 'urban': 2}, inplace=True)
    a = dt.predict_proba(df)[0][1]
    return a * 100
