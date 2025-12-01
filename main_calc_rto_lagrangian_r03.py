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
    
    export_dummy_csv = False
    
    # --- 化学反応設定 ---
    #mech_file = 'mech_toluene_LLNL_mod.yaml'    
    mech_file = 'GalwayMech1.0_June_2025.yaml'
    dt_sim = 1.0e-3
    
    # --- 成分設定 ---
    major_species = ['O2', 'H2O', 'CO2'] 
    balance_species = 'N2'
    
    # 燃料の設定
    initial_voc_ppm = {
        'C6H5CH3': 100.0,
        'CH4':  50.0
    }

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
            print(">> Generating DUMMY data...")
            df_dummy = self._generate_dummy_data()
            if self.cfg.export_dummy_csv:
                print(f">> Saving dummy data to: {self.cfg.dummy_file}")
                df_dummy.to_csv(self.cfg.dummy_file, index=False, sep=' ')
            return df_dummy

    def _read_file(self, filepath):
        try:
            df = pd.read_csv(filepath, skiprows=6, header=None, sep=r'\s+', engine='python')
            df = df.iloc[:, :14]
            df.columns = self.columns
            return df
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)

    def _generate_dummy_data(self):
        #steps = 100
        #t_max = 0.02
        steps = 100
        t_max = 2.0
        t = np.linspace(0, t_max, steps)
        #T = 300 + 1200 * np.exp(-((t - 0.005)**2) / (2 * 0.002**2))
        #P = 101325 * np.ones_like(t)
        #X_O2 = 0.21 * np.ones_like(t)
        #X_CO2 = 0.05 * (1 - np.exp(-t/0.005))
        #X_H2O = 0.10 * (1 - np.exp(-t/0.005))
        #X_N2 = 1.0 - (X_O2 + X_CO2 + X_H2O)
        T = 300 + 800 * np.exp(-((t - 0.5)**2) / (2 * 0.2**2))
        P = 101325 * np.ones_like(t)
        X_O2 = 0.21 * np.ones_like(t)
        X_CO2 = 0.05 * (1 - np.exp(-t/0.5))
        X_H2O = 0.10 * (1 - np.exp(-t/0.5))
        X_N2 = 1.0 - (X_O2 + X_CO2 + X_H2O)
        
        df = pd.DataFrame(index=range(steps), columns=self.columns)
        df['TIME'] = t; df['TEMPERATURE'] = T; df['PRESSURE'] = P
        df['X_O2'] = X_O2; df['X_N2'] = X_N2; df['X_CO2'] = X_CO2; df['X_H2O'] = X_H2O
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
            new_col = 'T' if col == 'TEMPERATURE' else 'P' if col == 'PRESSURE' else col
            df_interp[new_col] = np.interp(t_new, df_raw['TIME'], df_raw[col])
        return df_interp

# ==========================================
# 3. メインシミュレーションクラス (修正版)
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
        
        # --- 安全な化学種インデックス取得処理 (try-exceptを追加) ---
        
        # 1. 燃料 (VOC)
        self.voc_indices = {}
        for sp in self.cfg.initial_voc_ppm.keys():
            try:
                idx = self.gas.species_index(sp)
                self.voc_indices[sp] = idx
            except ValueError:
                # 存在しない場合はスキップ
                print(f"Warning: Fuel species '{sp}' not found. Skipping.")

        # 2. NOx (存在チェック)
        target_nox = ['NO', 'NO2', 'N2O']
        self.nox_indices = {}
        
        print("Checking for NOx species in mechanism...")
        for sp in target_nox:
            try:
                idx = self.gas.species_index(sp)
                self.nox_indices[sp] = idx
                print(f"  - {sp}: Found (Index {idx})")
            except ValueError:
                print(f"  - {sp}: NOT Found (Skipping analysis for this species)")
        
        # 3. バランスガス (N2) のインデックス確保
        try:
            self.idx_n2 = self.gas.species_index(self.cfg.balance_species)
        except ValueError:
            print(f"Critical Warning: Balance species '{self.cfg.balance_species}' not found!")
            self.idx_n2 = -1

    def set_initial_condition(self, T_init, P_init, major_specs_mole):
        # 燃料初期設定 (存在する燃料のみ)
        voc_str_parts = []
        for sp, ppm_val in self.cfg.initial_voc_ppm.items():
            if sp in self.voc_indices: # 存在する燃料だけ追加
                mole_frac = ppm_val * 1.0e-6
                voc_str_parts.append(f"{sp}:{mole_frac}")
        voc_str = ", ".join(voc_str_parts)
        
        # 主要成分
        major_str_parts = []
        for sp, val in major_specs_mole.items():
            major_str_parts.append(f"{sp}:{val}")
        major_str = ", ".join(major_str_parts)
        
        # 結合
        if voc_str:
            comp_str = f"{voc_str}, {major_str}"
        else:
            comp_str = f"{major_str}"
            
        # バランス種が存在する場合のみ追加
        if self.idx_n2 >= 0:
            comp_str += f", {self.cfg.balance_species}:1.0"

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
            
            # 主要成分の上書き
            for sp in self.cfg.major_species:
                try:
                    idx = self.gas.species_index(sp)
                    col_name = f'X_{sp}'
                    X_new[idx] = row[col_name]
                except ValueError:
                    # 主要成分すらメカニズムにない場合は無視（通常ありえないが安全のため）
                    pass
            
            # バランス計算
            if self.idx_n2 >= 0:
                for k in range(self.gas.n_species):
                    if k != self.idx_n2:
                        sum_others += X_new[k]
                
                X_new[self.idx_n2] = max(0.0, 1.0 - sum_others)
            
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
        
        # 存在するVOCのみ保存
        for sp, idx in self.voc_indices.items():
            res[f'X_{sp}'] = self.gas.X[idx]
            
        # 存在するNOxのみ保存
        for sp, idx in self.nox_indices.items():
            res[f'X_{sp}'] = self.gas.X[idx]
                
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
    
    # --- 安全なプロット処理 ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # 1. 温度
    ax1.plot(df_res['time'], df_res['T'], 'r')
    ax1.set_ylabel('Temp (K)')
    ax1.grid(True, alpha=0.3)
    
    # 2. VOC (存在する列のみプロット)
    voc_plotted = False
    for sp in cfg.initial_voc_ppm.keys():
        col = f'X_{sp}'
        if col in df_res.columns:
            ax2.plot(df_res['time'], df_res[col], label=sp)
            voc_plotted = True
    
    ax2.set_ylabel('VOC (Mole Fraction)')
    if voc_plotted:
        ax2.set_yscale('log')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No VOC species found", ha='center', transform=ax2.transAxes)
    ax2.grid(True, alpha=0.3)
    
    # 3. NOx (存在する列のみプロット)
    nox_plotted = False
    potential_nox = ['NO', 'NO2', 'N2O']
    colors = {'NO': 'g', 'NO2': 'b', 'N2O': 'y'}
    
    for sp in potential_nox:
        col = f'X_{sp}'
        if col in df_res.columns:
            ax3.plot(df_res['time'], df_res[col], label=sp, color=colors.get(sp, 'k'))
            nox_plotted = True

    ax3.set_ylabel('NOx (Mole Fraction)')
    ax3.set_xlabel('Time (s)')
    if nox_plotted:
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No NOx species found", ha='center', transform=ax3.transAxes)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(cfg.output_png)
    print("Plot Saved.")