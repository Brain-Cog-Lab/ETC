import math


class short_time():
	def __init__(self,SizeHistOutput):
		super().__init__()
		self.SizeHistOutput = SizeHistOutput
	# Computes variables for short-term synaptic plasticity
	# See Tsodyks and Markram 1997 for details
	# Input: Plastic synapse struct (Syn), inter-spike interval (ISI), # presynaptic spikes (Nsp)
	# Output: synaptic modification factor (R*u)
	def syndepr(self,Syn=None, ISI=None, Nsp=None):
		SizeHistOutput=self.SizeHistOutput
		qu = Syn.uprev[Nsp] * math.exp(-ISI / Syn.tc_fac)
		qR = math.exp(-ISI / Syn.tc_rec)
		u = qu + Syn.use * (1.0 - qu)
		R = Syn.Rprev[Nsp] * (1.0 - Syn.uprev[Nsp]) * qR + 1.0 - qR
		Syn.uprev[(Nsp + 1) % SizeHistOutput] = u
		Syn.Rprev[(Nsp + 1) % SizeHistOutput] = R
		return R * u



	# double set_Isyn (struct Neuron np, double dt, double v)
	# Compute synaptic current Isyn from the three synapse types
	# implementing a double-exponential decay with time (for spikes, see below)
	def set_gsyn(self,np=None, dt=None, v=None, NoiseSyn=None):
		Isyn = 0
		gsyn_AN = 0
		gsyn_G = 0

		for j in range(np.NumSynType):
			syn = np.STList[j]
			sgate = 1.0
			if (syn.Mg_gate > 0.0):
				sgate = syn.Mg_gate / (1.0 + syn.Mg_fac * math.exp(syn.Mg_slope * (syn.Mg_half - v[0])))
			Isyn += sgate * (
						np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on)) * (
							syn.Erev - v[0])
			if (syn.Erev == 0.0):
				gsyn_AN = gsyn_AN + sgate * (
						np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on))
			else:
				gsyn_G = gsyn_G + sgate * (
						np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on))

		# Compute noise input
		for j in range(NoiseSyn.NumSyn):
			syn = NoiseSyn.Syn[j].STPtr
			sgate = 1.0
			if (syn.Mg_gate > 0.0):  # use Mg gate if flag is on ( for NMDA only)
				sgate = syn.Mg_gate / (1.0 + syn.Mg_fac * math.exp(syn.Mg_slope * (syn.Mg_half - v)))
			Isyn += sgate * (
					np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on)) * (
							syn.Erev - v)
			if (syn.Erev == 0.0):
				gsyn_AN = gsyn_AN + sgate * (
						np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on))
			else:
				gsyn_G = gsyn_G + sgate * (
						np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on))

		I_tot = Isyn + np.Iinj
		return gsyn_AN, I_tot, gsyn_G

	# ODEs that define the model
	# Computes left-hand side of ODEs for a single neuron
	# Input: neuron (np), current variables (v), time step (dt)
	# Output: derivative of variables (dv)
	def IDderiv(self, np=None, v=None, dt=None, dv=None, NoiseSyn=None, flag_dv=None):
		Isyn = 0
		gsyn_G = 0
		gsyn_AN = 0
		for j in range(np.NumSynType):
			syn = np.STList[j]
			sgate = 1.0
			if (syn.Mg_gate > 0.0):  # use Mg gate if flag is on ( for NMDA only)
				sgate = syn.Mg_gate / (1.0 + syn.Mg_fac * math.exp(syn.Mg_slope * (syn.Mg_half - v[0])))
			Isyn += sgate * (
					np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on)) * (
							syn.Erev - v[0])
			if (syn.Erev == 0.0):
				gsyn_AN = gsyn_AN + sgate * (
						np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on))
			else:
				gsyn_G = gsyn_G + sgate * (
						np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on))

		# Compute noise input
		for j in range(NoiseSyn.NumSyn):
			syn = NoiseSyn.Syn[j].STPtr
			sgate = 1.0
			if (syn.Mg_gate > 0.0):  # use Mg gate if flag is on ( for NMDA only)
				sgate = syn.Mg_gate / (1.0 + syn.Mg_fac * math.exp(syn.Mg_slope * (syn.Mg_half - v[0])))
			Isyn += sgate * (
					np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on)) * (
							syn.Erev - v[0])
			if (syn.Erev == 0.0):
				gsyn_AN = gsyn_AN + sgate * (
						np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on))
			else:
				gsyn_G = gsyn_G + sgate * (
						np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on))

		# Exponential term
		I_ex = np.gL * np.sf * math.exp((v[0] - np.Vth) / np.sf)
		# V-nullcline at v[0]
		wV = np.Iinj + Isyn - np.gL * (v[0] - np.EL) + I_ex
		# Calculation of D_0
		D0 = (np.Cm / np.gL) * wV

		# Compute membrane potential derivative from all currents
		if ((
				np.Iinj + Isyn) >= np.I_ref and flag_dv == 0):  # use flag_dv for restriction of depolarization block to time x after spike
			dv[0] = -(np.gL / np.Cm) * (v[0] - np.v_dep)
			flag_regime_osc = 0
		else:
			dv[0] = (np.Iinj - np.gL * (v[0] - np.EL) - v[1] + I_ex + Isyn) / np.Cm
			flag_regime_osc = 1

		# derivative of D_0 with respect to V
		dD0 = np.Cm * (math.exp((v[0] - np.Vth) / np.sf) - 1)

		# second differential equation
		if ((v[1] > wV - D0 / np.tcw) and (v[1] < wV + D0 / np.tcw) and v[0] <= np.Vth and (
				np.Iinj + Isyn) < np.I_ref):
			dv[1] = -(np.gL * (1 - math.exp((v[0] - np.Vth) / np.sf)) + dD0 / np.tcw) * dv[0]
		else:
			dv[1] = 0
		I_tot = Isyn + np.Iinj

		return wV, D0, gsyn_AN, gsyn_G, I_tot, dv




	# Integrates ODEs by an explicit Runge-Kutta method of 2nd order
	# Input: neuron (np), time step (dt)
	# Output: updated neuron (np)
	def update(self,np=None, dt=None, NoiseSyn=None, flag_dv=None):
		nvar = 2
		v = [0] * 2
		dv1 = [0] * 2
		dv2 = [0] * 2
		for i in range(nvar):
			v[i] = np.v[i]
		wV, D0, gsyn_AN, gsyn_G, I_tot, dv1 =short_time(self.SizeHistOutput).IDderiv(np, v, 0.0, dv1, NoiseSyn, flag_dv)
		for i in range(nvar):
			v[i] += dt * dv1[i]
		wV, D0, gsyn_AN, gsyn_G, I_tot, dv2 = short_time(self.SizeHistOutput).IDderiv(np, v, 0.0, dv2, NoiseSyn, flag_dv)
		for i in range(nvar):
			np.v[i] += dt / 2.0 * (dv1[i] + dv2[i])
			np.dv[i] = dt / 2.0 * (dv1[i] + dv2[i])

		# Make a jump in w when it approaches the nullcline wV from the right to avoid singularities
		if ((np.v[1] > wV - D0 / np.tcw) and (np.v[1] < wV + D0 / np.tcw) and np.v[0] <= np.Vth):
			np.v[1] = wV - (D0 / np.tcw)

		return np, gsyn_AN, gsyn_G, I_tot