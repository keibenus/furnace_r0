import cantera as ct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ==========================================
# 1. 設定セクション (Configuration)
# ==========================================
class SimulationConfig:
    # --- ファイル入出力設定 ---
    input_file = 'cfd_pathline_data.txt'
    dummy_file = 'dummy_cfd_data.csv'
    output_csv = 'result_data.csv'
    output_png = 'result_plot.png'
    
    # 【追加】ダミーデータ生成時にCSV保存するかどうかのフラグ
    export_dummy_csv = False  # True: 出力する, False: 出力しない
    
    # --- 化学反応設定 ---
    mech_file = 'gri30.yaml'
    dt_sim = 1.0e-5
    
    # --- 成分設定 ---
    major_species = ['O2', 'H2O', 'CO2'] 
    balance_species = 'N2'
    
    # 燃料の初期濃度設定（ppm入力用）
    # ※内部計算でモル分率に変換されます
    initial_voc_ppm = {
        'C3H8': 100.0,  # トルエン
        'CH4':  50.0    # メタン
    }
    
    # ※GRI-Mech 3.0用 (C7H8がない場合用)
    # initial_voc_ppm = {'C3H8': 100.0}

# ==========================================
# 2. データ読み込み & 補間クラス
# ==========================================
class CFDDataLoader:
    def __init__(self, config):
        self.cfg = config
        self.columns = [
            'TIME', 'TEMPERATURE', 'X', 'Y', 'Z', 'U', 'V', 'W', 
            'DENSITY', 'PRESSURE', 
            'X_O2', 'X_N2', 'X_CO2', 'X_H2O'
        ]

    def load_data(self):
        if os.path.exists(self.cfg.input_file):
            print(f"Loading CFD data from: {self.cfg.input_file}")
            return self._read_file(self.cfg.input_file)
        else:
            print(f"File not found: {self.cfg.input_file}")
            print(">> Generating DUMMY data (Mole Fraction Mode)...")
            df_dummy = self._generate_dummy_data()
            
            # 【変更】設定フラグに基づいてCSV出力を制御
            if self.cfg.export_dummy_csv:
                print(f">> Saving dummy data to: {self.cfg.dummy_file}")
                df_dummy.to_csv(self.cfg.dummy_file, index=False, sep=' ')
            else:
                print(">> Skipping dummy data export (config.export_dummy_csv = False).")
            
            return df_dummy

    def _read_file(self, filepath):
        try:
            df = pd.read_csv(
                filepath, skiprows=6, header=None, sep=r'\s+', engine='python'
            )
            if df.shape[1] < 14:
                pass
            df = df.iloc[:, :14]
            df.columns = self.columns
            return df
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)

    def _generate_dummy_data(self):
        steps = 100
        t_max = 0.02
        t = np.linspace(0, t_max, steps)
        
        T = 300 + 1200 * np.exp(-((t - 0.005)**2) / (2 * 0.002**2))
        P = 101325 * np.ones_like(t)
        
        X_O2 = 0.21 * np.ones_like(t)
        X_CO2 = 0.05 * (1 - np.exp(-t/0.005))
        X_H2O = 0.10 * (1 - np.exp(-t/0.005))
        X_N2 = 1.0 - (X_O2 + X_CO2 + X_H2O)
        
        df = pd.DataFrame(index=range(steps), columns=self.columns)
        df['TIME'] = t
        df['TEMPERATURE'] = T
        df['PRESSURE'] = P
        df['X_O2'] = X_O2
        df['X_N2'] = X_N2
        df['X_CO2'] = X_CO2
        df['X_H2O'] = X_H2O
        df.fillna(0.0, inplace=True)
        return df

    def interpolate_data(self, df_raw):
        t_raw = df_raw['TIME'].values
        _, unique_idx = np.unique(t_raw, return_index=True)
        t_raw = t_raw[np.sort(unique_idx)]
        
        t_new = np.arange(t_raw[0], t_raw[-1], self.cfg.dt_sim)
        df_interp = pd.DataFrame({'time': t_new})
        
        target_cols = ['TEMPERATURE', 'PRESSURE', 'X_O2', 'X_CO2', 'X_H2O', 'X_N2']
        for col in target_cols:
            if col == 'TEMPERATURE': new_col = 'T'
            elif col == 'PRESSURE': new_col = 'P'
            else: new_col = col 
            df_interp[new_col] = np.interp(t_new, df_raw['TIME'], df_raw[col])
            
        return df_interp

# ==========================================
# 3. メインシミュレーションクラス
# ==========================================
class LagrangianReactor:
    def __init__(self, config):
        self.cfg = config
        try:
            self.gas = ct.Solution(self.cfg.mech_file)
        except Exception as e:
            print(f"Error loading mechanism: {e}")
            sys.exit(1)

        self.r = ct.IdealGasReactor(self.gas)
        self.sim = ct.ReactorNet([self.r])
        self.results = []
        
        self.voc_indices = {}
        for sp in self.cfg.initial_voc_ppm.keys():
            idx = self.gas.species_index(sp)
            if idx >= 0:
                self.voc_indices[sp] = idx
            else:
                print(f"Warning: Species '{sp}' not found in mechanism.")

    def set_initial_condition(self, T_init, P_init, major_specs_mole):
        # 1. 燃料 (ppm -> モル分率)
        voc_str_parts = []
        for sp, ppm_val in self.cfg.initial_voc_ppm.items():
            mole_frac = ppm_val * 1.0e-6  
            voc_str_parts.append(f"{sp}:{mole_frac}")
        voc_str = ", ".join(voc_str_parts)
        
        # 2. 主要成分
        major_str_parts = []
        for sp, val in major_specs_mole.items():
            major_str_parts.append(f"{sp}:{val}")
        major_str = ", ".join(major_str_parts)
        
        # 3. 初期セット (TPX)
        comp_str = f"{voc_str}, {major_str}, {self.cfg.balance_species}:1.0"
        self.gas.TPX = T_init, P_init, comp_str
        
        self.r.syncState()
        self.sim.reinitialize()

    def run(self, df_interp):
        print("Starting Simulation Loop...")
        total_steps = len(df_interp)
        log_interval = max(1, total_steps // 10)
        
        for i in range(total_steps):
            if i % log_interval == 0:
                print(f"Progress: {i}/{total_steps} ({i/total_steps*100:.0f}%)")

            row = df_interp.iloc[i]
            t_next = df_interp.iloc[i+1]['time'] if i < total_steps - 1 else row['time']
            if t_next == row['time']: break

            # --- Step 1: Mixing ---
            T_cfd = row['T']
            P_cfd = row['P']
            X_curr = self.gas.X
            X_new = X_curr.copy()
            sum_others = 0.0
            
            for sp in self.cfg.major_species:
                col_name = f'X_{sp}'
                idx = self.gas.species_index(sp)
                if idx >= 0:
                    X_new[idx] = row[col_name]
            
            idx_n2 = self.gas.species_index(self.cfg.balance_species)
            for k in range(self.gas.n_species):
                if k != idx_n2:
                    sum_others += X_new[k]
            X_new[idx_n2] = max(0.0, 1.0 - sum_others)
            
            self.gas.TPX = T_cfd, P_cfd, X_new
            self.r.syncState()
            self.sim.reinitialize()
            
            # --- Step 2: Reaction ---
            self.sim.advance(t_next)
            self.store_results(t_next)

        print("Done.")
        return pd.DataFrame(self.results)

    def store_results(self, t):
        res = {'time': t, 'T': self.gas.T, 'P': self.gas.P}
        
        # 【変更】ppm変換をやめ、モル分率(X)をそのまま保存
        # キー名からも _ppm を削除
        for sp, idx in self.voc_indices.items():
            res[f'X_{sp}'] = self.gas.X[idx]
            
        for nox in ['NO', 'NO2', 'N2O']:
            idx = self.gas.species_index(nox)
            if idx >= 0:
                res[f'X_{nox}'] = self.gas.X[idx]
            else:
                res[f'X_{nox}'] = 0.0
                
        self.results.append(res)

# ==========================================
# 4. 実行 & 可視化
# ==========================================
if __name__ == "__main__":
    cfg = SimulationConfig()
    
    loader = CFDDataLoader(cfg)
    df_raw = loader.load_data()
    df_interp = loader.interpolate_data(df_raw)
    
    solver = LagrangianReactor(cfg)
    
    row0 = df_interp.iloc[0]
    initial_majors = {
        'O2': row0['X_O2'], 'CO2': row0['X_CO2'], 'H2O': row0['X_H2O']
    }
    solver.set_initial_condition(row0['T'], row0['P'], initial_majors)
    
    df_res = solver.run(df_interp)
    df_res.to_csv(cfg.output_csv, index=False)
    print(f"Results saved to {cfg.output_csv}")
    
    # --- プロット (単位変更に対応) ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # 1. 温度
    ax1.plot(df_res['time'], df_res['T'], 'r')
    ax1.set_ylabel('Temp (K)')
    ax1.grid(True, alpha=0.3)
    
    # 2. VOC (Mole Fraction, Log Scale)
    for sp in cfg.initial_voc_ppm.keys():
        col = f'X_{sp}'
        if col in df_res.columns:
            ax2.plot(df_res['time'], df_res[col], label=sp)
    
    ax2.set_ylabel('VOC (Mole Fraction)')
    ax2.set_yscale('log') # モル分率でも小さい値なので対数が推奨されます
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. NOx (Mole Fraction)
    # 値が小さい(10^-5程度)ので、必要に応じて指数表記を確認してください
    ax3.plot(df_res['time'], df_res['X_NO'], label='NO')
    ax3.plot(df_res['time'], df_res['X_NO2'], label='NO2')
    if 'X_N2O' in df_res.columns and df_res['X_N2O'].max() > 1e-9:
        ax3.plot(df_res['time'], df_res['X_N2O'], label='N2O')

    ax3.set_ylabel('NOx (Mole Fraction)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(cfg.output_png)
    print("Plot Saved.")