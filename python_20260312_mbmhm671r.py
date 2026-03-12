#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV Tonal Noise Predictor - Interactive Dashboard
"""

import numpy as np
import plotly.graph_objects as go
from scipy.special import jv
from scipy.constants import pi
import streamlit as st

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

class UAVAcousticsConfig:
    C_SOUND = 343.0
    P_REF = 2e-5
    DEFAULT_BLADES = 4
    DEFAULT_RPM = 15000
    DEFAULT_RADIUS = 0.127
    DEFAULT_THRUST = 5.0
    DEFAULT_TORQUE = 0.2
    DEFAULT_OBSERVER_DIST = 10.0
    NUM_HARMONICS = 10

# =============================================================================
# CORE ACOUSTICS MODULE
# =============================================================================

class UAVAcousticsModel:
    def __init__(self, config=UAVAcousticsConfig()):
        self.config = config
        self.c = config.C_SOUND
        self.p_ref = config.P_REF
        
    def calculate_angular_velocity(self, rpm):
        return rpm * (2 * pi / 60)
    
    def calculate_bpf(self, blades, rpm):
        return (blades * rpm) / 60.0
    
    def calculate_tip_mach(self, omega, radius_eff):
        return (omega * radius_eff) / self.c
    
    def a_weighting(self, frequency):
        f = frequency
        if f <= 0: return 0.0
        f1, f2, f3, f4 = 20.6, 107.7, 737.9, 12194.0
        r_a = (f4**2 * f**4) / ((f**2 + f1**2) * 
                                 np.sqrt((f**2 + f2**2) * (f**2 + f3**2)) * 
                                 (f**2 + f4**2))
        return 20 * np.log10(r_a) + 2.0
    
    def calculate_harmonic_pressure(self, m, blades, omega, thrust, torque, 
                                   radius_eff, observer_dist, theta):
        M_e = self.calculate_tip_mach(omega, radius_eff)
        bessel_arg = m * blades * M_e * np.sin(theta)
        bessel_val = np.abs(jv(m * blades, bessel_arg))
        thrust_term = -thrust * np.cos(theta)
        torque_term = (self.c * torque) / (omega * radius_eff**2)
        loading_term = np.abs(thrust_term + torque_term)
        p_m = (m * blades * omega / (2 * pi * self.c * observer_dist)) * \
              loading_term * bessel_val
        return p_m
    
    def calculate_spl(self, p_amplitude):
        p_rms = np.maximum(p_amplitude / np.sqrt(2), 1e-12)
        spl = 20 * np.log10(p_rms / self.p_ref)
        return np.maximum(spl, 0)
    
    def get_directivity_pattern(self, blades, rpm, thrust, torque, 
                               radius, observer_dist, harmonic=1):
        omega = self.calculate_angular_velocity(rpm)
        radius_eff = 0.8 * radius
        theta_rad = np.linspace(0, 2 * pi, 360)
        p_m = self.calculate_harmonic_pressure(
            m=harmonic, blades=blades, omega=omega,
            thrust=thrust, torque=torque,
            radius_eff=radius_eff,
            observer_dist=observer_dist,
            theta=theta_rad
        )
        spl = self.calculate_spl(p_m)
        theta_deg = np.degrees(theta_rad)
        return theta_deg, spl
    
    def get_spectrum(self, blades, rpm, thrust, torque, radius, 
                    observer_dist, theta_deg=45, num_harmonics=10):
        omega = self.calculate_angular_velocity(rpm)
        radius_eff = 0.8 * radius
        theta_rad = np.radians(theta_deg)
        bpf = self.calculate_bpf(blades, rpm)
        frequencies, spl_values, spl_dba = [], [], []
        for m in range(1, num_harmonics + 1):
            freq = m * bpf
            p_m = self.calculate_harmonic_pressure(
                m=m, blades=blades, omega=omega,
                thrust=thrust, torque=torque,
                radius_eff=radius_eff,
                observer_dist=observer_dist,
                theta=theta_rad
            )
            spl = self.calculate_spl(p_m)
            a_weight = self.a_weighting(freq)
            frequencies.append(freq)
            spl_values.append(spl)
            spl_dba.append(spl + a_weight)
        return np.array(frequencies), np.array(spl_values), np.array(spl_dba)
    
    def calculate_total_spl(self, spl_values):
        return 10 * np.log10(np.sum(10**(spl_values / 10)))

# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

class UAVAcousticsVisualizer:
    @staticmethod
    def plot_directivity(theta_deg, spl, title="Directivity Pattern"):
        fig = go.Figure(go.Scatterpolar(
            r=spl, theta=theta_deg, mode='lines',
            line=dict(color='blue', width=2),
            fill='toself', name='SPL (dB)'
        ))
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(spl) + 10]),
                angularaxis=dict(rotation=90, direction="clockwise")
            ),
            showlegend=True, height=600
        )
        return fig
    
    @staticmethod
    def plot_spectrum(frequencies, spl, spl_dba, title="Frequency Spectrum"):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=frequencies, y=spl, name='SPL (dB)', marker_color='blue', opacity=0.7))
        fig.add_trace(go.Bar(x=frequencies, y=spl_dba, name='SPL (dBA)', marker_color='green', opacity=0.7))
        fig.update_layout(title=title, xaxis_title="Frequency (Hz)", yaxis_title="Sound Pressure Level (dB)", barmode='group', height=500, showlegend=True)
        return fig
    
    @staticmethod
    def plot_parametric_sweep(x_values, y_values, x_label, y_label, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', line=dict(color='red', width=2), name=y_label))
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, height=500, showlegend=True)
        return fig

# =============================================================================
# STREAMLIT DASHBOARD
# =============================================================================

def main():
    st.set_page_config(page_title="UAV Tonal Noise Predictor", page_icon="🚁", layout="wide")
    st.title("🚁 UAV Tonal Noise Predictor")
    st.markdown("**Analytical characterization of tonal noise generation in multi-rotor UAVs**")
    
    model = UAVAcousticsModel()
    visualizer = UAVAcousticsVisualizer()
    
    st.sidebar.header("🔧 Rotor Configuration")
    blades = st.sidebar.slider("Number of Blades (B)", 2, 8, 4)
    rpm = st.sidebar.slider("Rotational Speed (RPM)", 5000, 25000, 15000)
    radius = st.sidebar.number_input("Rotor Radius (m)", value=0.127, min_value=0.05, max_value=0.5)
    
    st.sidebar.header("⚙️ Operating Conditions")
    thrust = st.sidebar.number_input("Thrust per Rotor (N)", value=5.0, min_value=0.1, max_value=50.0)
    torque = st.sidebar.number_input("Torque per Rotor (N·m)", value=0.2, min_value=0.01, max_value=5.0)
    observer_dist = st.sidebar.number_input("Observer Distance (m)", value=10.0, min_value=1.0, max_value=100.0)
    observer_angle = st.sidebar.slider("Observer Angle (degrees)", 0, 180, 45)
    
    bpf = model.calculate_bpf(blades, rpm)
    omega = model.calculate_angular_velocity(rpm)
    radius_eff = 0.8 * radius
    mach_tip = model.calculate_tip_mach(omega, radius_eff)
    
    st.subheader("📊 Key Acoustic Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Blade Passage Frequency", f"{bpf:.1f} Hz")
    with col2: st.metric("Tip Mach Number", f"{mach_tip:.3f}")
    with col3: st.metric("Angular Velocity", f"{omega:.1f} rad/s")
    with col4: st.metric("Effective Radius", f"{radius_eff:.3f} m")
    
    st.subheader("📈 Acoustic Analysis")
    tab1, tab2, tab3 = st.tabs(["🎯 Directivity", "📊 Spectrum", "📉 Parametric Study"])
    
    with tab1:
        st.markdown("### Directivity Pattern (1st Harmonic)")
        theta_deg, spl = model.get_directivity_pattern(blades, rpm, thrust, torque, radius, observer_dist, harmonic=1)
        fig_directivity = visualizer.plot_directivity(theta_deg, spl)
        st.plotly_chart(fig_directivity, use_container_width=True)
        st.info("**Interpretation:** Thrust noise dominates on-axis (0° and 180°), while torque noise dominates in the rotor plane (90° and 270°).")
    
    with tab2:
        st.markdown("### Frequency Spectrum")
        frequencies, spl, spl_dba = model.get_spectrum(blades, rpm, thrust, torque, radius, observer_dist, theta_deg=observer_angle, num_harmonics=10)
        fig_spectrum = visualizer.plot_spectrum(frequencies, spl, spl_dba, title=f"Spectrum at {observer_angle}° Observer Angle")
        st.plotly_chart(fig_spectrum, use_container_width=True)
        st.markdown("### Harmonic Breakdown")
        harmonic_data = {
            'Harmonic': range(1, 11),
            'Frequency (Hz)': frequencies,
            'SPL (dB)': np.round(spl, 1),
            'A-Weight (dB)': np.round(spl_dba - spl, 1),
            'SPL (dBA)': np.round(spl_dba, 1)
        }
        st.dataframe(harmonic_data, use_container_width=True)
        total_spl = model.calculate_total_spl(spl)
        total_spl_dba = model.calculate_total_spl(spl_dba)
        st.markdown(f"**Overall Levels:** - Total SPL: **{total_spl:.1f} dB** - Total SPL (A-weighted): **{total_spl_dba:.1f} dBA**")
    
    with tab3:
        st.markdown("### Parametric Studies")
        study_type = st.selectbox("Select Parameter to Sweep", ["RPM", "Blade Count", "Thrust", "Observer Distance"])
        
        if study_type == "RPM":
            rpm_values = np.linspace(8000, 22000, 20)
            spl_values = [model.calculate_total_spl(model.get_spectrum(blades, r, thrust, torque, radius, observer_dist, theta_deg=observer_angle)[1]) for r in rpm_values]
            fig_sweep = visualizer.plot_parametric_sweep(rpm_values, spl_values, "RPM", "Total SPL (dB)", "Effect of RPM on Overall Noise Level")
            st.plotly_chart(fig_sweep, use_container_width=True)
        elif study_type == "Blade Count":
            blade_values = np.arange(2, 9)
            spl_values = [model.calculate_total_spl(model.get_spectrum(b, rpm, thrust, torque, radius, observer_dist, theta_deg=observer_angle)[1]) for b in blade_values]
            fig_sweep = visualizer.plot_parametric_sweep(blade_values, spl_values, "Number of Blades", "Total SPL (dB)", "Effect of Blade Count on Overall Noise Level")
            st.plotly_chart(fig_sweep, use_container_width=True)
        elif study_type == "Thrust":
            thrust_values = np.linspace(2, 10, 20)
            spl_values = [model.calculate_total_spl(model.get_spectrum(blades, rpm, t, torque, radius, observer_dist, theta_deg=observer_angle)[1]) for t in thrust_values]
            fig_sweep = visualizer.plot_parametric_sweep(thrust_values, spl_values, "Thrust (N)", "Total SPL (dB)", "Effect of Thrust Loading on Overall Noise Level")
            st.plotly_chart(fig_sweep, use_container_width=True)
        elif study_type == "Observer Distance":
            dist_values = np.linspace(5, 50, 20)
            spl_values = [model.calculate_total_spl(model.get_spectrum(blades, rpm, thrust, torque, radius, d, theta_deg=observer_angle)[1]) for d in dist_values]
            fig_sweep = visualizer.plot_parametric_sweep(dist_values, spl_values, "Distance (m)", "Total SPL (dB)", "Effect of Observer Distance on Overall Noise Level")
            st.plotly_chart(fig_sweep, use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Export Data")
    if st.sidebar.button("Download Current Results"):
        frequencies, spl, spl_dba = model.get_spectrum(blades, rpm, thrust, torque, radius, observer_dist, theta_deg=observer_angle, num_harmonics=10)
        csv_data = "Harmonic,Frequency_Hz,SPL_dB,A_Weight_dB,SPL_dBA\n"
        for i in range(len(frequencies)):
            csv_data += f"{i+1},{frequencies[i]:.2f},{spl[i]:.2f},{spl_dba[i]-spl[i]:.2f},{spl_dba[i]:.2f}\n"
        st.sidebar.download_button(label="📥 Download CSV", data=csv_data, file_name="uav_noise_spectrum.csv", mime="text/csv")
    
    st.markdown("---")
    st.markdown("**Developed at:** Concordia University, Aerospace Engineering")

if __name__ == "__main__":
    main()