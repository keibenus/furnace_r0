import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import pandas as pd

# ==============================================================================
# 1. ユーザー設定 (CONFIG)
# ==============================================================================
CONFIG = {
    # --- 計算制御 ---
    'n_cycles': 20,
    'cycle_time': 90.0,
    'log_interval': 5.0,
    
    # --- 安定化設定 ---
    'ramp_time': 300.0,
    'solver_atol': 1.0,
    
    # --- グリッド ---
    'N_mono': 30,
    'N_pack': 10,

    # --- 形状 ---
    'L_mono': 1.2,
    'L_pack': 0.15,
    'S_sect': 3.3,            
    'cc_vol': 4.0,

    # --- 運転条件 ---
    'flow_rate_Nm3h': 20000.0,
    'T_inlet': 443.0,
    'Y_voc_inlet': 6.0e-4,    
    
    # --- 燃焼室制御 ---
    'T_set_final': 1130.0, #1073.0,
    'LHV_voc': 40.6e6,        
    'LHV_ch4': 50.0e6,        
    
    # --- 固体物性 ---
    'solid': {
        #'rho': 2300.0, 'c': 1000.0, 'k': 1.5,
        'rho': 2300.0, 'c': 1000.0, 'k': 1.5,
        'mono': {'eps': 0.70, 'd': 0.003, 'av': 2000.0, 'phi': 1.0}, 
        'pack': {'eps': 0.45, 'd': 0.025, 'av': 800.0,  'phi': 0.3}
    }
}

# ==============================================================================
# 2. 物理計算クラス
# ==============================================================================
class RTOPhysics:
    def __init__(self):
        self.R_gas = 287.05
        self.P_atm = 101325.0

    def get_air_properties(self, T_arr):
        T = np.clip(T_arr, 250.0, 3000.0)
        rho = self.P_atm / (self.R_gas * T)
        mu = 1.716e-5 * ((T / 273.15)**1.5) * (383.55 / (T + 110.4))
        k = 0.0241 * ((T / 273.15)**0.85)
        t = T / 1000.0
        cp = 1005.0 + (t - 0.3) * 150.0
        return rho, cp, k, mu

# ==============================================================================
# 3. 蓄熱塔モデル
# ==============================================================================
class RTOCanister:
    def __init__(self):
        self.p = CONFIG
        N_m, N_p = self.p['N_mono'], self.p['N_pack']
        L_m, L_p = self.p['L_mono'], self.p['L_pack']
        
        self.N = N_m + N_p
        dx_m = L_m / N_m
        dx_p = L_p / N_p
        self.dx = np.concatenate([np.full(N_m, dx_m), np.full(N_p, dx_p)])
        
        x_m = np.linspace(0, L_m, N_m, endpoint=False) + dx_m/2
        x_p = np.linspace(L_m, L_m+L_p, N_p, endpoint=False) + dx_p/2
        self.x = np.concatenate([x_m, x_p])
        
        s = self.p['solid']
        self.eps = np.concatenate([np.full(N_m, s['mono']['eps']), np.full(N_p, s['pack']['eps'])])
        self.d   = np.concatenate([np.full(N_m, s['mono']['d']),   np.full(N_p, s['pack']['d'])])
        self.av  = np.concatenate([np.full(N_m, s['mono']['av']),  np.full(N_p, s['pack']['av'])])
        self.is_mono = np.concatenate([np.ones(N_m, dtype=bool), np.zeros(N_p, dtype=bool)])
        
    def get_ht_coeff(self, rho, mu, k, cp, u):
        u_abs = np.abs(u) + 1e-6
        Re = np.clip((rho * u_abs * self.d) / mu, 0.1, 1.0e5)
        Pr = (cp * mu) / k
        Sc = 0.7 
        Nu = np.zeros_like(Re)
        
        m = self.is_mono
        if np.any(m):
            d_over_L = self.d[m] / self.p['L_mono']
            term = d_over_L * Re[m] * Sc
            Sh = 0.766 * (term ** 0.4833)
            Nu[m] = Sh * ((Pr[m]/Sc)**(1.0/3.0))
            Nu[m] = np.maximum(Nu[m], 4.0)

        p = ~self.is_mono
        if np.any(p):
            Nu[p] = 1.0 + 1.8 * (Re[p]**0.5) * (Pr[p]**(1.0/3.0))
            
        h = Nu * k / self.d
        return h

# ==============================================================================
# 4. システムソルバー
# ==============================================================================
class RTOSystem:
    def __init__(self):
        self.phys = RTOPhysics()
        self.p = CONFIG
        self.bed1 = RTOCanister()
        self.bed2 = RTOCanister()
        
        N = self.bed1.N
        self.y1 = np.zeros(2*N)
        self.y2 = np.zeros(2*N)
        for y in [self.y1, self.y2]:
            y[0:N]   = self.p['T_inlet']
            y[N:2*N] = self.p['T_inlet']
            
        self.flow_dir = 1
        self.cycle_start_time = 0.0 
        self.history_data = []
        self.profile_data_t0 = None
        self.profile_data_t90 = None

    def get_fuel_power(self, T_in, cp, current_time):
        m_dot = (self.p['flow_rate_Nm3h']/3600)*1.293
        T_start = self.p['T_inlet']
        T_final = self.p['T_set_final']
        
        if current_time < self.p['ramp_time']:
            ratio = current_time / self.p['ramp_time']
            T_target = T_start + (T_final - T_start) * ratio
        else:
            T_target = T_final
        T_target = max(T_target, T_in)
        
        Q_voc = m_dot * self.p['Y_voc_inlet'] * self.p['LHV_voc']
        divisor = m_dot * cp if m_dot*cp > 1e-3 else 1e-3
        Q_req = divisor * (T_target - T_in)
        Q_fuel = max(0.0, Q_req - Q_voc)
        return Q_fuel, T_target

    def solve_cc_thermal_soft(self, T_in, cp, current_time):
        m_dot = (self.p['flow_rate_Nm3h']/3600)*1.293
        Q_fuel, T_target = self.get_fuel_power(T_in, cp, current_time)
        Q_voc = m_dot * self.p['Y_voc_inlet'] * self.p['LHV_voc']
        divisor = m_dot * cp if m_dot*cp > 1e-3 else 1e-3
        if Q_fuel > 0:
            T_out = T_target
        else:
            T_out = T_in + Q_voc / divisor
        return T_out

    def derivs(self, t, y_all):
        real_time = self.cycle_start_time + t
        N = self.bed1.N
        y1 = y_all[:2*N]; y2 = y_all[2*N:]
        y1 = np.clip(y1, 200.0, 3000.0); y2 = np.clip(y2, 200.0, 3000.0)
        rho1, cp1, k1, mu1 = self.phys.get_air_properties(y1[:N])
        rho2, cp2, k2, mu2 = self.phys.get_air_properties(y2[:N])
        m_dot = (self.p['flow_rate_Nm3h']/3600)*1.293
        u1 = m_dot / (np.maximum(rho1, 0.1) * self.p['S_sect'])
        u2 = m_dot / (np.maximum(rho2, 0.1) * self.p['S_sect'])
        
        if self.flow_dir == 1:
            dy1, T_out1 = self.calc_bed(y1, self.bed1, u1, rho1, cp1, k1, mu1, self.p['T_inlet'], dir=1)
            T_cc_out = self.solve_cc_thermal_soft(T_out1, cp1[-1], real_time)
            dy2, _      = self.calc_bed(y2, self.bed2, u2, rho2, cp2, k2, mu2, T_cc_out, dir=-1)
        else:
            dy2, T_out2 = self.calc_bed(y2, self.bed2, u2, rho2, cp2, k2, mu2, self.p['T_inlet'], dir=1)
            T_cc_out = self.solve_cc_thermal_soft(T_out2, cp2[-1], real_time)
            dy1, _      = self.calc_bed(y1, self.bed1, u1, rho1, cp1, k1, mu1, T_cc_out, dir=-1)
        return np.concatenate([dy1, dy2])

    def calc_bed(self, y, bed, u, rho, cp, k, mu, T_in, dir):
        N = bed.N
        Tg, Ts = y[0:N], y[N:2*N]
        h = bed.get_ht_coeff(rho, mu, k, cp, u)
        dx = bed.dx
        dT = np.zeros(N)
        if dir == 1: 
            dT[0] = (Tg[0]-T_in)/dx[0]; dT[1:] = (Tg[1:]-Tg[:-1])/dx[1:]; T_exit = Tg[-1]
        else: 
            dT[-1] = (Tg[-1]-T_in)/dx[-1]; dT[:-1] = (Tg[:-1]-Tg[1:])/dx[:-1]; T_exit = Tg[0]
        d2Ts = np.zeros(N)
        d2Ts[1:-1] = (Ts[2:] - 2*Ts[1:-1] + Ts[:-2]) / (dx[1:-1]**2)
        d2Ts[0] = (Ts[1]-Ts[0])/(dx[0]**2); d2Ts[-1] = (Ts[-2]-Ts[-1])/(dx[-1]**2)
        dTg_dt = (-rho*cp*np.abs(u)*dT + h*bed.av*(Ts-Tg)) / (bed.eps*rho*cp)
        s = self.p['solid']
        dTs_dt = ((1-bed.eps)*s['k']*d2Ts + h*bed.av*(Tg-Ts)) / ((1-bed.eps)*s['rho']*s['c'])
        return np.concatenate([dTg_dt, dTs_dt]), T_exit

    def calculate_profiles(self, y_vec, flow_dir):
        N = self.bed1.N
        g1, s1 = y_vec[0:N], y_vec[N:2*N]
        g2, s2 = y_vec[2*N:3*N], y_vec[3*N:4*N]
        rho1, cp1, k1, mu1 = self.phys.get_air_properties(g1)
        rho2, cp2, k2, mu2 = self.phys.get_air_properties(g2)
        m_dot = (self.p['flow_rate_Nm3h']/3600)*1.293
        u1 = m_dot / (np.maximum(rho1, 0.1) * self.p['S_sect'])
        u2 = m_dot / (np.maximum(rho2, 0.1) * self.p['S_sect'])
        h1 = self.bed1.get_ht_coeff(rho1, mu1, k1, cp1, u1)
        h2 = self.bed2.get_ht_coeff(rho2, mu2, k2, cp2, u2)
        return {'g1': g1, 's1': s1, 'h1': h1, 'u1': u1, 'g2': g2, 's2': s2, 'h2': h2, 'u2': u2}

    def run(self):
        print(f"--- Simulation Start (Plotting h & u added) ---")
        y_curr = np.concatenate([self.y1, self.y2])
        start_real = time.time()
        self.cycle_start_time = 0.0
        
        for i in range(self.p['n_cycles']):
            if i == self.p['n_cycles'] - 1:
                self.profile_data_t0 = self.calculate_profiles(y_curr, self.flow_dir)

            t_cycle = 0.0
            t_end = self.p['cycle_time']
            dt_log = self.p['log_interval']
            
            while t_cycle < t_end:
                t_next = min(t_cycle + dt_log, t_end)
                sol = solve_ivp(self.derivs, (t_cycle, t_next), y_curr, 
                                method='Radau', rtol=1e-2, atol=self.p['solver_atol'])
                if not sol.success: print(f"Error: {sol.message}"); return
                y_curr = sol.y[:, -1]
                t_cycle = t_next
                
                current_time = self.cycle_start_time + t_cycle
                N = self.bed1.N
                # Data for history CSV
                g1, s1 = y_curr[0:N], y_curr[N:2*N]
                g2, s2 = y_curr[2*N:3*N], y_curr[3*N:4*N]
                if self.flow_dir == 1: 
                    T_cc_in = g1[-1]
                    cp_in = self.phys.get_air_properties(np.array([T_cc_in]))[1][0]
                else:
                    T_cc_in = g2[-1]
                    cp_in = self.phys.get_air_properties(np.array([T_cc_in]))[1][0]
                fuel_W, _ = self.get_fuel_power(T_cc_in, cp_in, current_time)
                
                # 簡易熱収支計算
                profs = self.calculate_profiles(y_curr, self.flow_dir)
                q_b1 = np.sum(profs['h1'] * self.bed1.av * (profs['s1'] - profs['g1']) * self.bed1.dx * self.p['S_sect'])
                q_b2 = np.sum(profs['h2'] * self.bed2.av * (profs['s2'] - profs['g2']) * self.bed2.dx * self.p['S_sect'])
                
                self.history_data.append({
                    'Time_s': current_time, 'Cycle': i+1, 'Fuel_kW': fuel_W/1000.0,
                    'Heat_Ex_B1_kW': q_b1/1000.0, 'Heat_Ex_B2_kW': q_b2/1000.0,
                    'Tg_B1_Bot': g1[0], 'Tg_B1_Top': g1[-1], 'Ts_B1_Bot': s1[0], 'Ts_B1_Top': s1[-1],
                    'Tg_B2_Bot': g2[0], 'Tg_B2_Top': g2[-1], 'Ts_B2_Bot': s2[0], 'Ts_B2_Top': s2[-1]
                })
                
                T_max = np.max(y_curr[:self.bed1.N])
                T_tgt = self.solve_cc_thermal_soft(0, 1005.0, current_time)
                print(f"\r[Cycle {i+1}/{self.p['n_cycles']}] t={t_cycle:4.0f}s | T_tgt:{T_tgt:.0f}K | MaxT:{T_max:.0f}K", end="")
            
            self.flow_dir *= -1
            self.cycle_start_time += t_end
            
        self.profile_data_t90 = self.calculate_profiles(y_curr, self.flow_dir)
        print(f"\nCompleted in {time.time()-start_real:.2f} sec.")
        self.y1 = y_curr[:2*self.bed1.N]; self.y2 = y_curr[2*self.bed1.N:]

    def save_csv(self):
        df_hist = pd.DataFrame(self.history_data)
        df_hist.to_csv('rto_time_history.csv', index=False)
        print("Saved 'rto_time_history.csv'")
        
        def make_prof_df(p_data, label):
            df = pd.DataFrame({
                'Position_m': self.bed1.x,
                f'Tg_B1_{label}': p_data['g1'], f'Ts_B1_{label}': p_data['s1'],
                f'h_B1_{label}': p_data['h1'],  f'u_B1_{label}': p_data['u1'],
                f'Tg_B2_{label}': p_data['g2'], f'Ts_B2_{label}': p_data['s2'],
                f'h_B2_{label}': p_data['h2'],  f'u_B2_{label}': p_data['u2']
            })
            return df
        df_t0 = make_prof_df(self.profile_data_t0, 't0')
        df_t90 = make_prof_df(self.profile_data_t90, 't90')
        df_prof = pd.merge(df_t0, df_t90, on='Position_m')
        df_prof.to_csv('rto_profiles.csv', index=False)
        print("Saved 'rto_profiles.csv'")

if __name__ == "__main__":
    sim = RTOSystem()
    sim.run()
    sim.save_csv()
    
    # ==========================================
    # グラフ描画 (温度、熱伝達率、流速)
    # ==========================================
    N = sim.bed1.N
    x1 = sim.bed1.x
    x2 = sim.bed2.x
    
    # データ取得
    p0 = sim.profile_data_t0
    p90 = sim.profile_data_t90
    
    # 共通レイアウト関数
    def setup_ax(axL, axR, title, ylabel):
        axL.set_xlabel("Height [m]"); axL.set_ylabel(ylabel); axL.grid(True)
        axL.set_xlim(0, 1.35); axL.set_title("Canister 1 (Left)")
        axR.set_xlabel("Height [m]"); axR.grid(True)
        axR.set_xlim(1.35, 0); axR.set_title("Canister 2 (Right)") # 左右反転
        axR.set_yticklabels([])
    
    # --- 1. 温度分布 (Comparison t=0 vs t=90) ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace': 0.05})
    setup_ax(ax1, ax2, "", "Temperature [K]")
    
    # Bed 1
    ax1.plot(x1, p0['g1'], 'b-', label='Gas t=0')
    ax1.plot(x1, p90['g1'], 'r--', label='Gas t=90')
    ax1.plot(x1, p90['s1'], 'k:', label='Solid t=90') # Solidはt=90のみ代表表示
    ax1.legend(loc='upper left')
    ax1.set_ylim(300, 1200)
    
    # Bed 2
    ax2.plot(x2, p0['g2'], 'b-', label='Gas t=0')
    ax2.plot(x2, p90['g2'], 'r--', label='Gas t=90')
    ax2.plot(x2, p90['s2'], 'k:', label='Solid t=90')
    ax2.set_ylim(300, 1200)
    
    fig1.suptitle('Gas Temperature Distribution', fontsize=14)
    plt.savefig('rto_temp_dist.png')
    print("Saved 'rto_temp_dist.png'")

    # --- 2. 熱伝達率 (h) 分布 ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace': 0.05})
    setup_ax(ax3, ax4, "", "Heat Transfer Coeff. [W/m2K]")
    
    # Bed 1
    ax3.plot(x1, p0['h1'], 'b-', label='t=0')
    ax3.plot(x1, p90['h1'], 'r--', label='t=90')
    ax3.legend()
    
    # Bed 2
    ax4.plot(x2, p0['h2'], 'b-')
    ax4.plot(x2, p90['h2'], 'r--')
    
    fig2.suptitle('Heat Transfer Coefficient Distribution', fontsize=14)
    plt.savefig('rto_ht_coeff.png')
    print("Saved 'rto_ht_coeff.png'")

    # --- 3. ガス流速 (u) 分布 ---
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'wspace': 0.05})
    setup_ax(ax5, ax6, "", "Gas Velocity [m/s]")
    
    # Bed 1
    ax5.plot(x1, p0['u1'], 'b-', label='t=0')
    ax5.plot(x1, p90['u1'], 'r--', label='t=90')
    ax5.legend()
    
    # Bed 2
    ax6.plot(x2, p0['u2'], 'b-')
    ax6.plot(x2, p90['u2'], 'r--')
    
    fig3.suptitle('Gas Velocity Distribution', fontsize=14)
    plt.savefig('rto_velocity.png')
    print("Saved 'rto_velocity.png'")