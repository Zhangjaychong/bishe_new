import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('./XGBoost.pkl')

# 定义特征选项
cp_options = {
    1: '典型心绞痛 (1)',
    2: '非典型心绞痛 (2)',
    3: '非心绞痛疼痛 (3)',
    4: '无症状 (4)'
}
restecg_options = {
    0: '正常 (0)',
    1: 'ST-T波异常 (1)',
    2: '左心室肥厚 (2)'
}
slope_options = {
    1: '上坡型 (1)',
    2: '平坦型 (2)',
    3: '下坡型 (3)'
}
thal_options = {
    1: '正常 (1)',
    2: '固定缺陷 (2)',
    3: '可逆缺陷 (3)'
}

# 定义特征名称
feature_names = [
    "年龄", "性别", "胸痛类型", "静息血压", "血清胆固醇",
    "空腹血糖 > 120 mg/dl", "静息心电图结果", "最大心率", "运动诱发的心绞痛",
    "相对休息时的ST段下降", "峰值运动ST段斜率", "由荧光透视着色的主要血管数量", "地中海贫血"
]

# Streamlit 用户界面
st.title("心脏病预测")

# 用户输入
age = st.number_input("年龄:", min_value=1, max_value=120, value=50)
sex = st.selectbox("性别 (0=女性, 1=男性):", options=[0, 1], format_func=lambda x: '女性 (0)' if x == 0 else '男性 (1)')
cp = st.selectbox("胸痛类型:", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])
trestbps = st.number_input("静息血压 (mm Hg):", min_value=50, max_value=200, value=120)
chol = st.number_input("血清胆固醇 (mg/dl):", min_value=100, max_value=600, value=200)
fbs = st.selectbox("空腹血糖 > 120 mg/dl:", options=[0, 1], format_func=lambda x: '否 (0)' if x == 0 else '是 (1)')
restecg = st.selectbox("静息心电图结果:", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])
thalach = st.number_input("最大心率:", min_value=50, max_value=250, value=150)
exang = st.selectbox("运动诱发的心绞痛:", options=[0, 1], format_func=lambda x: '否 (0)' if x == 0 else '是 (1)')
oldpeak = st.number_input("相对休息时的ST段下降:", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("峰值运动ST段斜率:", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])
ca = st.number_input("由荧光透视着色的主要血管数量:", min_value=0, max_value=4, value=0)
thal = st.selectbox("地中海贫血:", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

# 处理输入并进行预测
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
features = np.array([feature_values])

if st.button("预测"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**预测类别:** {predicted_class}")
    st.write(f"**预测概率:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"根据我们的模型，您有较高的心脏病风险。 "
            f"模型预测您患心脏病的概率为 {probability:.1f}%。 "
            "虽然这只是估计值，但它表明您可能面临显著的风险。 "
            "我们建议您尽快咨询心脏病专家以进行进一步评估，并确保获得准确的诊断和必要的治疗。"
        )
    else:
        advice = (
            f"根据我们的模型，您有较低的心脏病风险。 "
            f"模型预测您不患心脏病的概率为 {probability:.1f}%。 "
            "然而，保持健康的生活方式仍然非常重要。 "
            "我们建议定期检查心脏健康状况，并在出现任何症状时及时寻求医疗建议。"
        )
    st.write(advice)

    # 计算SHAP值并显示force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")