import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import itertools
import numpy as np


st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa - Adinda Salsabila", layout="centered")

st.title("Prediksi Kelulusan Mata Kuliah Mahasiswa dengan Bayesian Network")

st.markdown("""
          
#### Nama: Adinda Salsabila
#### NPM : 2215061035    
#### Kelas: Kecerdasan Buatan C
            
<br>
<br>

Faktor diinput sebagai **Low** atau **High**. Probabilitas lulus akan dihitung dengan mempertimbangkan bobot tiap faktor:

- **Presensi & Etika:** 10%
- **Keaktifan & Kuis:** 10%
- **Tugas:** 30%
- **UTS:** 25%
- **UAS:** 25%
""", unsafe_allow_html=True)

edges = [
    ("P", "G"),  # Presensi & Etika → Kelulusan
    ("K", "G"),  # Keaktifan & Kuis → Kelulusan
    ("T", "G"),  # Tugas → Kelulusan
    ("U", "G"),  # UTS → Kelulusan
    ("A", "G"),  # UAS → Kelulusan
]
model = DiscreteBayesianNetwork(edges)

cpd_p = TabularCPD('P', 2, [[0.3], [0.7]])  
cpd_k = TabularCPD('K', 2, [[0.3], [0.7]])
cpd_t = TabularCPD('T', 2, [[0.3], [0.7]])
cpd_u = TabularCPD('U', 2, [[0.3], [0.7]])
cpd_a = TabularCPD('A', 2, [[0.3], [0.7]])

weights = {
    'P': 10,
    'K': 10,
    'T': 30,
    'U': 25,
    'A': 25,
}

def get_prob_lulus(parents):
    score = (
        parents[0]*weights['P'] +
        parents[1]*weights['K'] +
        parents[2]*weights['T'] +
        parents[3]*weights['U'] +
        parents[4]*weights['A']
    )

    min_score, max_score = 0, 100
    raw_prob = 0.1 + 0.85 * ((score - min_score) / (max_score - min_score))
    raw_prob = max(0.01, min(0.99, raw_prob))
    return [1-raw_prob, raw_prob] 

parent_states = list(itertools.product([0,1], repeat=5))
cpd_g_values = [get_prob_lulus(list(state)) for state in parent_states]
cpd_g = TabularCPD(
    variable='G', variable_card=2,
    values=np.array(cpd_g_values).T,
    evidence=['P', 'K', 'T', 'U', 'A'],
    evidence_card=[2,2,2,2,2]
)

model.add_cpds(cpd_p, cpd_k, cpd_t, cpd_u, cpd_a, cpd_g)
infer = VariableElimination(model)

st.header("Input Faktor (Low/High)")

col1, col2, col3 = st.columns(3)
with col1:
    presensi = st.selectbox("Presensi & Etika", ["-", "Low", "High"])
    keaktifan = st.selectbox("Keaktifan & Kuis", ["-", "Low", "High"])
with col2:
    tugas = st.selectbox("Tugas", ["-", "Low", "High"])
with col3:
    uts = st.selectbox("UTS", ["-", "Low", "High"])
    uas = st.selectbox("UAS", ["-", "Low", "High"])

evidence = {}
if presensi != "-": evidence["P"] = 0 if presensi == "Low" else 1
if keaktifan != "-": evidence["K"] = 0 if keaktifan == "Low" else 1
if tugas != "-": evidence["T"] = 0 if tugas == "Low" else 1
if uts != "-": evidence["U"] = 0 if uts == "Low" else 1
if uas != "-": evidence["A"] = 0 if uas == "Low" else 1

st.subheader("Hasil Prediksi Kelulusan")
try:
    result = infer.query(variables=["G"], evidence=evidence if evidence else None)
    prob_lulus = result.values[1]*100
    prob_tidak = result.values[0]*100
    st.success(f"**Probabilitas Lulus:** {prob_lulus:.2f}%")
    st.error(f"**Probabilitas Tidak Lulus:** {prob_tidak:.2f}%")
    st.write(result)
except Exception as e:
    st.error(f"Error saat inferensi: {e}")

st.subheader("Visualisasi Struktur Bayesian Network")
G_vis = nx.DiGraph(edges)
fig, ax = plt.subplots(figsize=(8, 5))
pos = nx.spring_layout(G_vis, seed=42)
nx.draw(G_vis, pos, with_labels=True, node_color="lightpink", node_size=2000, font_size=14, font_weight="bold", ax=ax)
ax.set_title("Struktur BN Kelulusan Mahasiswa", fontsize=16)
st.pyplot(fig)

with st.expander("Keterangan Node"):
    st.markdown("""
    - **P**: Presensi & Etika (`Low`/`High`)
    - **K**: Keaktifan & Kuis (`Low`/`High`)
    - **T**: Tugas (`Low`/`High`)
    - **U**: UTS (`Low`/`High`)
    - **A**: UAS (`Low`/`High`)
    - **G**: Kelulusan (`Tidak Lulus`/`Lulus`)
    """)
