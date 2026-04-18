# Physics Derivations and Mathematical Formulation
**UTDTB v5.0 BEAST — Technical Reference**

This document provides the complete mathematical foundations for the Universal Turbofan Digital Twin Benchmark.

---

## 1. Atmospheric and Flow Modeling (EQ-C1)
The simulation uses a standard ISA atmosphere model to define the ambient environment based on altitude ($alt$).

### 1.1 Troposphere ($alt \le 11,000$ m)
$$T_0 = (288.15 + \Delta T_{ISA}) - 0.0065 \times alt$$
$$P_0 = 101325 \times \left(\frac{T_0}{T_{sl}}\right)^{5.2561}$$

### 1.2 Stratosphere ($alt > 11,000$ m)
$$T_0 = T_{11} \quad (\text{isothermal})$$
$$P_0 = P_{11} \times \exp\left(-\frac{g \cdot \Delta alt}{R \cdot T_{11}}\right)$$

### 1.3 Corrected Quantities
To normalize performance across flight envelopes:
$$\theta = \frac{T_0}{T_{ref}}, \quad \delta = \frac{P_0}{P_{ref}}$$
$$Nc_{corrected} = \frac{Nc}{\sqrt{\theta}}, \quad \dot{m}_{corrected} = \frac{\dot{m} \cdot \delta}{\sqrt{\theta}}$$

---

## 2. Thermodynamic Brayton Cycle (EQ-C2 to C5)

### 2.1 Compressor (EQ-C2)
$$\tau_c = OPR^{\frac{\gamma_c - 1}{\gamma_c}}$$
$$T_3 = T_2 \times \left(1 + \frac{\tau_c - 1}{\eta_c}\right)$$
$$\eta_c = \eta_{c0} \times (1 - \alpha_1 D_{fat} - \alpha_2 D_{cor} - \alpha_3 \cdot fouling)$$

### 2.2 Combustor (EQ-C4)
The turbine entry temperature ($T_4$) is driven by fuel flow and heating value:
$$T_4 = \frac{c_{p,c} \cdot T_3 + FAR \cdot LHV \cdot \eta_{comb} \cdot (1 - f_{contam})}{c_{p,t} \cdot (1 + FAR)}$$

### 2.3 Turbine Expansion (EQ-C5)
$$T_5 = T_{45} - \eta_t \cdot 0.97 \cdot T_{45} \cdot \left(1 - \left(\frac{P_5}{P_{45}}\right)^{\frac{\gamma_t-1}{\gamma_t}}\right)$$
**Thermal Mass ODE:**
$$\frac{dT_{core}}{dt} = \frac{Q_{in} - W_{out} - Q_{loss}}{m_{th} \cdot c_p}$$

---

## 3. Structural Degradation & Fracture

### 3.1 Fatigue: Paris-Erdogan with Walker Correction (EQ-F1, F2)
$$\Delta K_{eff} = \frac{\Delta K}{(1 - R)^{1 - c_w}}$$
$$\frac{dD_{fat}}{dN} = C_{fat} \times \Delta K_{eff}^{m_{fat}} \times (1 + \lambda_{cor} \cdot D_{cor}) \times phase_{mult} \times \epsilon$$

### 3.2 Creep: Norton-Bailey (EQ-CR1)
$$\dot{\varepsilon}_{crp} = A \times \sigma^n \times \exp\left(-\frac{Q}{R \cdot T_4}\right)$$

### 3.3 Corrosion: Arrhenius Model (EQ-RX1)
$$\frac{dD_{cor}}{dt} = k_{cor} \times RH \times \exp\left(-\frac{Q_{cor}}{R \cdot T_{hot}}\right) \times (1 + (k_{salt} - 1) \cdot salt) \times (1 + 0.5 \cdot sand)$$

### 3.4 Fracture & Burst (EQ-BU1)
Disk burst probability follows a Weibull distribution:
$$P_{burst} = 1 - \exp\left(-\left(\frac{\sigma}{\sigma_0}\right)^k\right)$$
Failure occurs when crack length $a \ge a_{crit}$ (12mm).

---

## 4. Health Index (HI) and RUL

### 4.1 Health Index Weighted Sum
The engine health is a normalized weighted sum of 9 components ($c_1 \dots c_9$):
$$HI(c) = 1 - \frac{\sum w_i \cdot c_i}{\sum w_i}$$
*Weights: Fatigue (0.22), Creep (0.16), Corrosion (0.10), Thermal Fatigue (0.08), etc.*

### 4.2 Failure Probability
$$P_{fail}(c) = \frac{1}{1 + \exp(10 \cdot (HI(c) - 0.30))}$$

### 4.3 Uncertainty Derivation
The model decomposes uncertainty into Aleatoric ($\sigma_a$) and Epistemic ($\sigma_e$):
$$\sigma_{total}(c) = \sqrt{\sigma_{alea}^2 + \sigma_{epi}^2}$$
$$RUL_{lower/upper} = RUL \pm 1.96 \cdot \sigma_{total}$$

---

## 5. Sensor Noise Model
A three-layer noise process is applied to the ground truth ($gt$):
1. **Layer 1:** Gaussian Base Noise $\epsilon_G \sim \mathcal{N}(0, \sigma_i^2)$.
2. **Layer 2:** Heavy-tail Student-t spikes ($p=0.05$).
3. **Layer 3:** Per-engine calibration bias $b_i \sim \mathcal{N}(0, 0.004)$.

$$Noisy = gt \times (1 + b_i + drift_{frac}) + \epsilon_G + \epsilon_t + drift_{fault}$$
