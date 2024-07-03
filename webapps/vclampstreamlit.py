import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Initializing parameters
Cm = 1            # Membrane capacitance, in uF/cm^2
gmax_Na = 120     # Sodium (Na) maximum conductances, in mS/cm^2
gmax_K = 36       # Potassium (K) maximum conductances, in mS/cm^2
gmax_L = 0.3      # Leak maximum conductances, in mS/cm^2

VNa = 115.0       # Sodium (Na) Nernst reversal potentials, in mV
VK = -12.0        # Potassium (K) Nernst reversal potentials, in mV
VL = 10.613       # Leak Nernst reversal potentials, in mV

# Gating variable functions
alpha_n = lambda Vm: (0.01 * (10.0 - Vm)) / (np.exp((10 - Vm) / 10) - 1.0)
beta_n = lambda Vm: 0.125 * np.exp(-Vm / 80.0)

alpha_m = lambda Vm: (0.1 * (25.0 - Vm)) / (np.exp((25 - Vm) / 10) - 1.0)
beta_m = lambda Vm: 4.0 * np.exp(-Vm / 18.0)

alpha_h = lambda Vm: 0.07 * np.exp(-Vm / 20)
beta_h = lambda Vm: 1.0 / (np.exp((30 - Vm) / 10) + 1.0)

# Conductance density (in mS/cm^2) of Sodium and Potassium
gNa = lambda m, h: gmax_Na * m**3 * h
gK = lambda n: gmax_K * n**4

# Currents
JNa = lambda Vm, m, h: gNa(m, h) * (Vm - VNa)
JK = lambda Vm, n: gK(n) * (Vm - VK)
JL = lambda Vm: gmax_L * (Vm - VL)

# Steady-state functions
def n_inf(Vm=0.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))

def m_inf(Vm=0.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))

def h_inf(Vm=0.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))

# ODE system
def dALLdt(X, t, I1_amp, I1_dur, I1_delay, I2_amp, I2_dur, I2_delay):
    V, m, h, n = X
    I_inj = I1_amp * (t > I1_delay) * (t <= I1_delay + I1_dur) + I2_amp * (t > (I1_delay + I1_dur + I2_delay)) * (t <= (I1_delay + I1_dur + I2_delay + I2_dur))
    dVdt = (I_inj - JNa(V, m, h) - JK(V, n) - JL(V)) / Cm
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    return dVdt, dmdt, dhdt, dndt

# Simulation function
def simulate_iclamp(I1_amp, I1_dur, I1_delay, I2_amp, I2_dur, I2_delay, t_n=150, delta_t=0.01):
    t = np.arange(0, t_n, delta_t)
    init_values = [0.0, n_inf(), m_inf(), h_inf()]
    X = odeint(dALLdt, init_values, t, args=(I1_amp, I1_dur, I1_delay, I2_amp, I2_dur, I2_delay))
    V = X[:, 0]
    m = X[:, 1]
    h = X[:, 2]
    n = X[:, 3]
    ina = JNa(V, m, h)
    ik = JK(V, n)
    il = JL(V)
    
    return t, V, m, h, n, ina, ik, il


st.set_page_config(page_title="Voltage Clamp Simulation", layout="wide")
st.title("Interactive Voltage Clamp Simulation with Dual Current Injection")


st.sidebar.header("Simulation Parameters")

# Current Injection 1
st.sidebar.subheader("Current Amplitudes")
I1_amp = st.sidebar.slider("Injection Current 1 Amplitude (µA/cm²)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
I2_amp = st.sidebar.slider("Injection Current 2 Amplitude (µA/cm²)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)

I1_delay = 10.0  # Always start after a delay of 10 ms

# Current Injection 2
st.sidebar.subheader("Current Durations")
I1_dur = st.sidebar.slider("Injection Current 1 Duration (ms)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
I2_dur = st.sidebar.slider("Injection Current 2 Duration (ms)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

st.sidebar.subheader("Delay Inbetween")
I2_delay = st.sidebar.slider("(ms, relative to end of Injection 1)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)


# Simulation parameters
st.sidebar.subheader("Simulation Time (ms)")
t_n = st.sidebar.slider("",min_value=50, max_value=200, value=100, step=1)
delta_t = 0.01
st.sidebar.markdown("""
    Made with ❤️ by Sandun Induwara
    """)
# Run simulation
t, V, m, h, n, ina, ik, il = simulate_iclamp(I1_amp, I1_dur, I1_delay, I2_amp, I2_dur, I2_delay, t_n, delta_t)

# Create smaller subplots
fig1, ax1 = plt.subplots(figsize=(6, 1))
fig2, ax2 = plt.subplots(figsize=(6, 1))
fig3, ax3 = plt.subplots(figsize=(6, 1))
fig4, ax4 = plt.subplots(figsize=(6, 1))

# Injected current
i_inj_values = I1_amp * (t > I1_delay) * (t <= I1_delay + I1_dur) + I2_amp * (t > (I1_delay + I1_dur + I2_delay)) * (t <= (I1_delay + I1_dur + I2_delay + I2_dur))
ax1.plot(t, i_inj_values, 'k')
ax1.set_ylabel('$I_{inj}$ $(uA/cm^2)$')
ax1.set_xlabel('Time (ms)')

# Membrane potential
ax2.plot(t, V, 'k')
ax2.set_title('Voltage Clamp Simulation')
ax2.set_ylabel('Membrane Potential (mV)')

# Gating variables
ax3.plot(t, m, 'r', label='m')
ax3.plot(t, h, 'g', label='h')
ax3.plot(t, n, 'b', label='n')
ax3.set_ylabel('Gating Variables')
ax3.legend()

# Currents
ax4.plot(t, ina, 'c', label='I_Na')
ax4.plot(t, ik, 'y', label='I_K')
ax4.plot(t, il, 'm', label='I_L')
ax4.set_ylabel('Currents $(uA/cm^2)$')
ax4.legend()

# Layout for Streamlit
st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)
st.pyplot(fig4)
