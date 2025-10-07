import math
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Numeric Parameters
A = (5e-6) ** 2  # junction area, m² (example: 1 µm²)
d_min = 0.1e-10  # minimum barrier thickness (e.g. 0.1 Å)

# I/O Parameters
file = "Brinkmanfit-analysis.xlsx"
sheet = "Simmons-Temp"
ind_var = "Voltage"  # independent column header
dep_var = "Current"  # dependent column header

# Mathamatical Constants
e = 1.602176634e-19  # elementary charge, C
h = 6.62607015e-34  # Planck's constant, J*s
m_e = 9.10938356e-31  # electron mass, kg
π = math.pi

# Model
def simmons(V, α, d, Φ):
	"""
	Simmons tunneling model for symmetric barriers
	
	Args:
		V (np.ndarry|float): Applied voltage (V)
		α (float): Effective mass factor (dimensionless)
		d (float): Barrier thickness (m)
		Φ (float): Barrier height (J)
	"""
	# Support float as well as numpy array
	V = np.asarray(V, dtype=float)
	
	# to avoid recalculations
	_2πd = 2*π*d
	eV2 = e*V/2
	Φ_minus = Φ - eV2
	Φ_plus = Φ + eV2
	_2αm_e = 2*α*m_e
	h_inv = 1/h

	# for reabability
	term1 = Φ_minus * np.exp(-2 * _2πd * h_inv * np.sqrt(_2αm_e * Φ_minus))
	term2 = Φ_plus  * np.exp(-2 * _2πd * h_inv * np.sqrt(_2αm_e * Φ_plus ))
	
	return e/(_2πd * h * d) * (term1 - term2)

# Model Bounds
α_bounds = (0, 1)
d_bounds = (d_min, np.sqrt(A))
def Φ_bounds(V_bounds):
	V_min, V_max = V_bounds
	Φ_min = np.maximum(-e * V_min / 2, e * V_max / 2)
	return (Φ_min, np.inf)

# Main
if __name__ == "__main__":
	# Load and Extract Data
	df = pd.read_excel(pd.ExcelFile(file), sheet_name=sheet)[[ind_var, dep_var]]
	p_data = df[df[ind_var] >= 0]  # non-negative ("positive") data
	n_data = df[df[ind_var] <= 0]  # non-positive ("negative") data

	# Curve Fitting (for both "Positive" and "Negative" sides)
	def fit(data):
		# convert pandas series to numpy arrays explicitally to make debugging easier
		V = data[ind_var].to_numpy(dtype=float)
		I = data[dep_var].to_numpy(dtype=float)

		# parameters need bounds
		V_bounds = (V.min(), V.max())
		bounds = list(zip(α_bounds, d_bounds, Φ_bounds(V_bounds)))

		return curve_fit(simmons, V, I, bounds=bounds)  # tuple(params, covariance)
	
	# Output Results as Plot (for both "Positive" and "Negative" sides)
	def plot(data, fit, label_suffix):
		V = data[ind_var].to_numpy(dtype=float)
		I = data[dep_var].to_numpy(dtype=float)
		_params, covariance = fit
		α, d, Φ = _params
		
		plt.scatter(V, I, color="blue", label=f"Experimental Data ({label_suffix})", s=10)
		
		I_fit = simmons(V, α, d, Φ)
		plt.plot(V, I_fit, color="red", label="Simmons Fit", linewidth=2)
		
		plt.xlabel("Voltage (V)")
		plt.ylabel("Current Density (A/m^2)")
		plt.title(f"Experimental vs Simmons Fit ({label_suffix})")
		plt.legend()
		plt.tight_layout()
		plt.show()

		print(f"α: {α:.4f}, d: {d*1e10:.4f} Å, Φ: {Φ:.4f} eV")

	plot(p_data, fit(p_data), '"Positive"')
	plot(n_data, fit(n_data), '"Negative"')
