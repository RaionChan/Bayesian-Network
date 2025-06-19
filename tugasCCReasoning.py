from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

student_model = DiscreteBayesianNetwork([('I', 'G'), ('D', 'G'), ('G', 'L'), ('I', 'S')])

cpd_d = TabularCPD(
    variable='D',
    variable_card=2,
    values=[[0.6], [0.4]],
    state_names={'D': ['Easy', 'Hard']}
)

cpd_i = TabularCPD(
    variable='I',
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={'I': ['Low', 'High']}
)

cpd_g = TabularCPD(
    variable='G',
    variable_card=3,
    values=[[0.3, 0.05, 0.9, 0.5],
            [0.4, 0.25, 0.08, 0.3],
            [0.3, 0.7, 0.02, 0.2]],
    evidence=['I', 'D'],
    evidence_card=[2, 2],
    state_names={
        'G': ['A', 'B', 'C'],
        'I': ['Low', 'High'],
        'D': ['Easy', 'Hard']
    }
)

cpd_l = TabularCPD(
    variable='L',
    variable_card=2,
    values=[[0.1, 0.4, 0.99],
            [0.9, 0.6, 0.01]],
    evidence=['G'],
    evidence_card=[3],
    state_names={
        'L': ['Weak', 'Strong'],
        'G': ['A', 'B', 'C']
    }
)

cpd_s = TabularCPD(
    variable='S',
    variable_card=2,
    values=[[0.95, 0.2],
            [0.05, 0.8]],
    evidence=['I'],
    evidence_card=[2],
    state_names={
        'S': ['Low', 'High'],
        'I': ['Low', 'High']
    }
)

student_model.add_cpds(cpd_d, cpd_i, cpd_g, cpd_l, cpd_s)

student_model.check_model()

inference = VariableElimination(student_model)

result = inference.query(
    variables=['G'],
    evidence={'I': 'High', 'D': 'Hard'}
)

print(result)