from scipy.optimize import minimize
# import torch
# from torch import tensor
from scipy.optimize import NonlinearConstraint, LinearConstraint
from collections import defaultdict
import display
from potential_initialization import *
import ndsplines

def init_analytic_properties(interp):
    jac = [interp.derivative(i) for i in range(3)]
    hess = [[interp.derivative(i).derivative(j) for i in range(3)] for j in range(3)]
    return lambda x: np.array([func(x) for func in jac]).reshape(3),\
           lambda x: np.array([[func(x) for func in col] for col in hess]).reshape((3, 3)),\
           lambda x: np.array([func(x) for func in jac]).reshape(3) * 1e3,\
           lambda x: np.array([[func(x) for func in col] for col in hess]).reshape((3, 3)) * 1e6,\


def get_pseudo_potential_interpolation(model, pscoef, srange, stepsize, ppi=None):
    text = "range_" + str(srange) + "stepsize_" + str(stepsize)
    # if ppi is not None and text in ppi:
    #     return ppi[model]
    field_points = get_field_points(srange, stepsize)
    pb, pp = get_potential_basis(model, field_points, text=text)
    pp *= pscoef
    # print(pp)
    interp = ndsplines.make_interp_spline(field_points, pp)
    return interp, *init_analytic_properties(interp)
# field_points.shape, pp.shape

def get_potential_basis_interpolation(model, srange, stepsize, pbi=None, regenerate=False):
    
    text = "range_" + str(srange) + "stepsize_" + str(stepsize)
    def list_function(function_list):
        
        def result_function(input):
            return np.array([function(input) for function in function_list])

        return result_function
    if pbi is not None and text in pbi:
        return pbi[text]
    
    field_points = get_field_points(srange, stepsize)
    pb, pp = get_potential_basis(model, field_points=field_points, regenerate=regenerate, text=text)

    interp, jac, hess, jac_SI, hess_SI = [], [], [], [], []
    for i in tqdm(range(pb.shape[-1])):
        itp = ndsplines.make_interp_spline(field_points, pb[...,i])
        interp.append(itp)
        j, h, jsi, hsi = init_analytic_properties(itp)
        jac.append(j)
        hess.append(h)
        jac_SI.append(jsi)
        hess_SI.append(hsi)
    pbi[text] = (list_function(interp), list_function(jac), list_function(hess)\
                , list_function(jac_SI), list_function(hess_SI))
    return pbi[text]

def find_radial_minimum(interp, grad, srange, stepsize, plot=False):
    xtick, ytick, ztick = get_field_point_ticks(srange, stepsize=stepsize)
    locations = []
    for x in tqdm(xtick):
        func = lambda y: interp([x, y[0], y[1]])
        jac = lambda y: grad([x, y[0], y[1]])[1:]
        # hess = lambda y: hessian([x, y[0], y[1]])[1:, 1:] 
        res = minimize(func, x0=[0, 0.07], jac=jac, tol=1e-15, bounds=[(-0.1, 0.1), (0.05, 0.1)])
        locations.append(res.x)
    locations = np.array(locations)
    if plot:
        figure = plt.figure()
        axes = figure.add_subplot(projection='3d')
        axes.scatter(xtick, locations[:, 0], locations[:, 1])
        plt.show()
    return xtick, locations


def get_rf_null(pseudo_interp, pseudo_jac, srange, stepsize):
    xtick, locations = find_radial_minimum(pseudo_interp, pseudo_jac, srange, stepsize)
    return ndsplines.make_interp_spline(xtick, locations)



class trap_model:
    def __init__(self, name, V_rf, omega_rf, shuttle_range, stepsize, pbi=None, ppi=None, regenerate=False) -> None:
        print("这tqdm有大病")
        data = np.load("models\%s.npz" % name, allow_pickle=True)
        self.name = name
        self.pairs = data["pairs"].item()
        self.npairs = len(self.pairs)
        self.dual = dict()
        self.idx2name = dict()
        self.shuttle_range = shuttle_range
        self.stepsize = stepsize
        for k in self.pairs:
            p1, p2 = self.pairs[k]
            self.dual[p1] = p2
            self.dual[p2] = p1
            self.idx2name[p1] = self.idx2name[p2] = k
        self.idx2sector = data["idx2sector"]
        self.sector2idx = defaultdict(list)
        for idx, sector in enumerate(self.idx2sector):
            self.sector2idx[sector].append(idx)
        self.interp, self.jac, self.hess, self.jac_SI, self.hess_SI = get_potential_basis_interpolation(name, srange=shuttle_range, stepsize=stepsize, pbi=pbi, regenerate=regenerate)  
        self.reset_rf(V_rf, omega_rf)
        self.rf_null = get_rf_null(self.pseudo_interp, self.pseudo_jac, shuttle_range, stepsize)
        self.dc_sectors = data["dc_sectors"]
        
        self.weight = [100, 10000, 1000, 1000, 300, 1, 1, 100, 200, 200]
        self.mesh = display.get_mesh(name)
        self.nparts = len(trimesh.graph.connected_components(self.mesh.edges))

    def reset_rf(self, V_rf, omega_rf):
        self.pscoef = (q / (4 * m * omega_rf ** 2)) * V_rf ** 2
        self.pseudo_interp, self.pseudo_jac, self.pseudo_hess, self.pseudo_jac_SI, self.pseudo_hess_SI  \
            = get_pseudo_potential_interpolation(self.name, self.pscoef, srange=self.shuttle_range, stepsize=self.stepsize)

    def depth_of_trap(self, x, ploting=False):
        p = self.rf_null_point(x)
        zrange = np.linspace(*self.shuttle_range[2], 100)
        potential_list = [self.potential(np.zeros(self.nparts), [p[0], p[1], z]) for z in zrange]
        #np.zeros(94)长度94的0向量；p[0]和p[1]为x=0.55截面赝势零点的xy坐标，p[2]赝势零点z坐标；
        opt = minimize(lambda z: -self.potential(np.zeros(self.nparts), [p[0], p[1], z]), x0=p[2] + 0.005, bounds=[(p[2], 0.2)])
        if ploting is True:
            plt.plot(zrange, potential_list)
            plt.xlabel("z(mm)")
            plt.ylabel("potential")
            plt.vlines(x=p[2], ymin=-0.1, ymax=0.5, label="rf null", color="orange", linestyles='dashed') #p[2]赝势零点z坐标，虚线位置不对应呀？；
            plt.vlines(x=opt.x, ymin=-0.1, ymax=0.5, label="escape point", color="red", linestyles='dashed')
            plt.title("x=%fmm" % x)
            plt.legend()
        return opt.x[0] - p[2]


    def plot(self, part_id=None, electrodode=None, sector=None):
        if part_id is not None:
            if isinstance(part_id, int):
                display.part(self.mesh, part_id)
            else:
                display.part(self.mesh, [e for e in part_id])
        if electrodode is not None:
            if isinstance(electrodode, str):
                display.part(self.mesh, self.pairs[electrodode])
            else:
                display.part(self.mesh, [self.pairs[e] for e in electrodode])
        if sector is not None:
            if isinstance(sector, int): 
                display.part(self.mesh, self.sector2idx[sector])
            else:
                display.part(self.mesh, [self.sector2idx[s] for s in sector])

    def plot_slice(self, voltage, x=None, z=None, ax=None):
        if x is not None:
            display.potential_slicex(self.shuttle_range, self.stepsize, x,  interp=lambda xx: self.potential(voltage, xx), ax=ax)
        if z is not None:
            display.potential_slicez(self.shuttle_range, self.stepsize, z,  interp=lambda xx: self.potential(voltage, xx), ax=ax)

    def rf_null_point(self, x):
        if isinstance(x, (list, np.ndarray)):
            y, z = self.rf_null(x).reshape(len(x), 2).T
            return np.array([x, y, z]).T
        else:
            y, z = self.rf_null(x)
            return np.array([x, y, z])
        
    
    def field_properties(self, point):
        return self.interp(point), self.jac(point), self.hess(point),\
                self.pseudo_interp(point), self.pseudo_jac(point), self.pseudo_hess(point)
    
    def get_dual(self, indices):
        return [self.dual[idx] for idx in indices]
    
    def sector_subset(self, property_list, indices, paired=True, compensate_z=True):

        def sector_subset_one(prop):
            if paired:
                res =  prop[indices] + prop[self.get_dual(indices)]
            else:
                res =  prop[indices]
            if compensate_z:
                res = list(res)
                res.append(prop[self.get_all_from_sectors(self.dc_sectors)].sum(axis=0))
            return res
        
        if indices is None:
            return property_list
        if type(property_list) is not list:
            return  sector_subset_one(property_list)
        return [sector_subset_one(prop) for prop in property_list]

        
    
    def loss_function(self, point, util_indices=None, all_indices=None, method="profile target loss", 
                      omega=[0.5, 3, 3], w=None, confine_voltage=5, paired=True, compensate_z=True, rotation=np.eye(3)):

        def oscillator_energy(voltage):
            E = (jac * voltage[:, None]).sum(axis=0) + pseudo_jac
            hessian = (hess * voltage[:, None, None]).sum(axis=0) + pseudo_hess
            k, eigv = np.linalg.eig(hessian)
            E = E @ eigv
            return (E ** 2 / k).sum()
        
        def profile(voltage, display=False, details=False):
            E = (jac * voltage[:, None]).sum(axis=0) + pseudo_jac
            V = (interp * voltage[:, None]).sum(axis=0) + pseudo_interp
            hessian = (hess * voltage[:, None, None]).sum(axis=0) + pseudo_hess
            # print(hessian.shape)
            hessian = rotation.T @ hessian @ rotation
            # print(hessian.shape)
            P = np.array([V[0], E[0], E[1], E[2], hessian[0, 0], hessian[1, 1], hessian[2, 2], hessian[0, 1], hessian[0, 2], hessian[1, 2]])
            # print(P.shape)
            omega = (np.array([hessian[0, 0], hessian[1, 1], hessian[2, 2]]) / 1e6 * q / m) ** 0.5 / (2 * np.pi)
            if display:
                print(P - target)
                print("E: " + str(E))
                print("Omega: " + str(omega))
            loss = ((P - target) ** 2 * w).sum()
            if details == True:
                return loss, E, omega, pseudo_jac
            return loss
        
        interp, jac, hess, pseudo_interp, pseudo_jac, pseudo_hess = self.field_properties(point)

        null_voltage = np.zeros(self.nparts)
        null_voltage[all_indices] = confine_voltage
        null_voltage[util_indices] = 0
        null_voltage[self.get_dual(util_indices)] = 0

        
        pseudo_interp += (interp * null_voltage[:, None]).sum(axis=0)
        pseudo_jac += (jac * null_voltage[:, None]).sum(axis=0)
        pseudo_hess += (hess * null_voltage[:, None, None]).sum(axis=0)
        interp, jac, hess = self.sector_subset([interp, jac, hess], util_indices, paired, compensate_z)

        if method == "oscillator energy":
            return oscillator_energy
        if method == "profile target loss":
            if w is None:
                w = self.weight
            omega = np.array(omega)
            k = (m / q * (omega * 2 * np.pi) ** 2) * 1e6
            target = np.array([confine_voltage, 0, 0, 0, k[0], k[1], k[2], 0, 0, 0])
            return profile
    
    # def loss_gradient(self, point, indices=None, method="profile target loss", omega=[0.5, 3, 3], w=None, analytical_grad=False, confine_voltage=5, paired=True):

    #     def oscillator_energy(voltage):

    #         voltage = torch.tensor(voltage, requires_grad=True)
    #         E = (jac * voltage[:, None]).sum(axis=0) + pseudo_jac
    #         hessian = (hess * voltage[:, None, None]).sum(axis=0) + pseudo_hess
    #         k, eigv = torch.linalg.eigh(hessian)
    #         Energy = ((E @ eigv) ** 2 / k).sum()
    #         Energy.backward()
    #         return np.array(voltage.grad)
            

    #     def profile(voltage):
    #         voltage = torch.tensor(voltage, requires_grad=True)
    #         V = (interp * voltage[:, None]).sum(axis=0) + pseudo_interp
    #         E = (jac * voltage[:, None]).sum(axis=0) + pseudo_jac
    #         hessian = (hess * voltage[:, None, None]).sum(axis=0) + pseudo_hess
    #         P = torch.stack([V[0], E[0], E[1], E[2], hessian[0, 0], hessian[1, 1], hessian[2, 2], hessian[0, 1], hessian[0, 2], hessian[1, 2]])
    #         # print(P - target)
    #         loss = ((P - target) ** 2 * w).sum()
    #         loss.backward()
    #         return np.array(voltage.grad)

    #     if analytical_grad is False:
    #         return None
    #     interp, jac, hess, pseudo_interp, pseudo_jac, pseudo_hess = self.field_properties(point)

    #     null_voltage = np.ones(self.nparts) * confine_voltage
    #     null_voltage[indices] = 0
    #     null_voltage[self.get_dual(indices)] = 0

    #     pseudo_interp += (interp * null_voltage[:, None]).sum(axis=0)
    #     pseudo_jac += (jac * null_voltage[:, None]).sum(axis=0)
    #     pseudo_hess += (hess * null_voltage[:, None, None]).sum(axis=0)

    #     interp, jac, hess, pseudo_interp, pseudo_jac, pseudo_hess =\
    #           tensor(interp.astype(np.float64)), tensor(jac), tensor(hess), tensor(pseudo_interp.astype(np.float64)), tensor(pseudo_jac), tensor(pseudo_jac)
    #     interp, jac, hess = self.sector_subset([interp, jac, hess], indices, paired)


    #     if method == "oscillator energy":
    #         return oscillator_energy
    #     if method == "profile target loss":
    #         if w is None:
    #             w = self.weight
    #         w = tensor(w)
    #         omega = tensor(omega)
    #         k = tensor((m / q * (omega * 2 * np.pi) ** 2) * 1e6)
    #         target = tensor([confine_voltage, 0, 0, 0, k[0], k[1], k[2], 0, 0, 0])
    #         return profile
    
    def OE_constraints(self, point, indices=None, lb=0.1):
        
        def cons(voltage):
            hessian = (hess * voltage[:, None, None]).sum(axis=0) + pseudo_hess
            k, eigv = np.linalg.eig(hessian)
            return k
        
        # print(point.shape)

        interp, jac, hess, pseudo_interp, pseudo_jac, pseudo_hess = self.field_properties(point)
        jac, hess = self.sector_subset([jac, hess], indices)
        return NonlinearConstraint(cons, lb=np.ones(3) * lb, ub=[np.inf, np.inf, np.inf])
    
    def get_indices(self, use_sectors=None, point=None, top_nearest=None, paired=True, plot=False):
        if top_nearest is not None:
            if not paired:
                indicator = -abs((self.jac(point) ** 2).sum(axis=1))
                indicator[self.sector2idx[0]] = 0
                indices = np.argsort(indicator)[:top_nearest]
            else:
                jac = self.jac(point)
                indicator = -np.array([(jac[self.pairs[pname], :] ** 2).sum() for pname in self.pairs])
                arg = np.argsort(indicator)
                pind = np.array(list(self.pairs))[arg][:top_nearest]
                indices = np.array([self.pairs[pname][0] for pname in pind]).ravel()
        elif use_sectors is not None:
            indices = [idx for sector in use_sectors for idx in self.sector2idx[sector]]
        else:
            indices = [i for i in range(self.nparts)]
        if plot:
             display.part(self.mesh, indices)
            #  plt.scatter(point[0], point[1], point[2])
        return sorted(indices)
    

    def coord_voltage(self, x, omega=[0.5, 3, 3], rotation=np.eye(3), w=None,
                        confine_voltage=0, max_voltage=10, top_nearest=3, paired=True, use_sectors=None, message=False):
        if use_sectors is None:
            use_sectors = self.dc_sectors
        if w is None:
            w = self.weight
        all_indices = self.get_all_from_sectors(use_sectors)
        point = self.rf_null_point(x)
        indices = self.get_indices(use_sectors, point, top_nearest, paired)
        initial_guess = np.ones(len(indices)) * (confine_voltage - max_voltage) / 2
        bounds = [(-max_voltage, confine_voltage) for guess in initial_guess]
        opt =  minimize(self.loss_function(point, indices, all_indices, "profile target loss", \
                                           omega, w, confine_voltage, paired, False, rotation), 
                        x0=initial_guess,
                        method="SLSQP",
                        bounds=bounds,
                        )
        # print(opt)
        voltage = np.zeros(self.nparts)
        voltage[all_indices] = confine_voltage
        voltage[indices] = opt.x
        if paired:
            voltage[self.get_dual(indices)] = opt.x
        if message:
            print("Ion location at " + str(point))
            if paired:
                for idx, v in zip(indices, opt.x):
                    if abs(v) > 1e-5:
                        print(self.idx2name[idx] + ": " + str(v) + " V")
            else:
                for idx, v in zip(indices, opt.x):
                    if abs(v) > 1e-5:
                        print("electrode " + str(idx) + ": " + str(v) + " V")
        return point, voltage
        # return opt
    

    def optimized_voltage(self, point, util_indices, all_indices, initial_guess, bounds, method="profile target loss"\
                          , omega=[0.5, 3, 3], w=None, confine_voltage=5,\
                            paired=True, compensate_z=False):

        # if method == "oscillator energy":
        #     cons = self.OE_constraints(point, indices)
        if compensate_z:
            initial_guess = list(initial_guess)
            initial_guess.append(0)
            bounds.append((-2, 2))
        return minimize(self.loss_function(point, util_indices, all_indices, method, omega, w, confine_voltage, paired, compensate_z), 
                        # jac=self.loss_gradient(point, indices, method, omega, w, analytical_grad, confine_voltage, paired),
                        x0=initial_guess,
                        method="SLSQP",
                        bounds=bounds,
                        # constraints=cons,
                        )
    
    def get_all_from_sectors(self, sectors):
        if sectors is None:
            sectors = self.dc_sectors
        return [p for sector in sectors for p in self.sector2idx[sector]]

    def optimized_voltage_profile(self, x, use_sectors=None, top_nearest=None, method="profile target loss", 
                                  omega=[0.5, 3, 3], w=None, initial_guess=None,
                                  plot_v=True, alpha=50,
                                  max_voltage=10, confine_voltage=0,
                                  paired=True, epack=None, apack=None,
                                  fix_voltage_dict=None, dynamic_selection=True, 
                                  compensate_z=False, calc_position=False):

        res, voltages, grads, frequencies, psEs = [], [], [], [], []
        if calc_position:
            positions = []
        indices = None
        all_indices = self.get_all_from_sectors(use_sectors)
        # alpha是电压随离子坐标的变化率上限，单位V/mm
        # alpha * 输运速度 = 电压对时间的变化率上限
        deltav=alpha * (x[1] - x[0])
        for i in tqdm(x):
            point = self.rf_null_point(i)
            new_idx = self.get_indices(use_sectors, point, top_nearest, paired)
            
            bounds = None
            # print(new_idx is None)
            if indices is None:
                indices = new_idx
                initial_guess = np.ones(len(indices)) * (confine_voltage - max_voltage) / 2
                bounds = [(-max_voltage, confine_voltage) for guess in initial_guess]
            elif (indices != new_idx):
                # print("changed")
                idx2guess = defaultdict(lambda: confine_voltage)
                for idx, guess in zip(indices, initial_guess):
                    idx2guess[idx] = guess
                indices = new_idx
                initial_guess = [idx2guess[idx] for idx in indices]
            
            if bounds is None:
                bounds = [(max(-max_voltage, guess - deltav), min(guess + deltav, confine_voltage)) for guess in initial_guess]
            # print(bounds)
            # print(len(initial_guess))
            opt_res = self.optimized_voltage(point=point, 
                                             initial_guess=initial_guess, bounds = bounds,
                                             util_indices=indices, all_indices=all_indices,
                                             method=method, omega=omega, w=w,
                                             confine_voltage=confine_voltage,
                                             paired=paired,
                                             compensate_z=compensate_z)
            
            _, E, omega, psE = self.loss_function(point, indices, all_indices, method, omega, w, confine_voltage, compensate_z=compensate_z)(opt_res.x, details=True)
            grads.append(E)
            frequencies.append(omega)
            psEs.append(psE),

            res.append(opt_res)
            sol = opt_res.x[:-1] if compensate_z else opt_res.x
            initial_guess = sol
            
            voltage = np.ones(self.nparts) * confine_voltage
            voltage[indices] = sol
            if paired:
                voltage[self.get_dual(indices)] = sol
            if compensate_z:
                voltage[self.get_all_from_sectors(self.dc_sectors)] += opt_res.x[-1]
            
            voltages.append(voltage)
            if calc_position:
                position = self.position(voltage, i)
                positions.append(position)
            # frequencies.append(self.freq(point, voltage))
        voltages = np.array(voltages)
        grads = np.array(grads)
        frequencies = np.array(frequencies)
        psEs = np.array(psEs)
        if plot_v:
            plt.figure(figsize=(13, 8), dpi=80)
            for eidx in range(self.nparts):
                if (voltages[:, eidx] != confine_voltage).any():
                    plt.plot(x, voltages[:, eidx], label="electordode %s" % self.idx2name[eidx])
            plt.legend()
            plt.show()
            def plot_profile_params(param, text):
                plt.plot(x, param[:, 0], label="x")
                plt.plot(x, param[:, 1], label="y")
                plt.plot(x, param[:, 2], label="z")
                plt.legend()
                plt.title(text)
                plt.show()
            plot_profile_params(grads, text="electronic field at given points")
            plot_profile_params(psEs, "pseudo electronic field at given points")
            if calc_position:
                freq = np.array([self.freq(position, voltage) for position, voltage in zip(positions, voltages)])
                positions = np.array(positions)
                plt.plot(x, positions[:, 0])
                plt.plot(x, x, alpha=0.5)
                plt.xlabel("set x")
                plt.ylabel('actual x')
                plt.show()
                plt.plot(positions[:, 0], positions[:, 2], label='actual z')
                plt.plot(positions[:, 0], self.rf_null_point(positions[:, 0])[:, 2], label='rf null z')
                plt.legend()
                plt.xlabel('actual x')
                plt.ylabel('z')
                plt.show()
                plot_profile_params(freq, "trap frequencies at actual points")
                plot_profile_params(frequencies, "trap frequencies at given points")
            
        if calc_position:
            return voltages, grads, frequencies, psEs, positions
        else:
            return voltages, grads, frequencies, psEs
    

    def potential(self, voltage, points):
        return (self.interp(points) * voltage[:, None]).sum(axis=0) + self.pseudo_interp(points)
        
    def potential_curvature_x(self, voltage, ax=None, points=None):
        x, y, z = get_field_point_ticks(self.shuttle_range)
        points = self.rf_null_point(x)
        if ax is None:
            plt.plot(x, self.potential(voltage, points))
        else:
            ax.plot(x, self.potential(voltage, points))

    def position(self, voltage, guess_x):
        func = lambda point: self.potential(voltage, point)
        guess = [guess_x, 0, 0.07]
        return minimize(func, x0=guess, tol=1e-9, method='Nelder-Mead').x
    
    def freq(self, point, voltages, rotate=False):
        hessian = self.pseudo_hess(point) + (self.hess(point) * voltages[:, None, None]).sum(axis=0)
        if rotate is False:
            eig = np.array([hessian[0, 0], hessian[1, 1], hessian[2, 2]])
            # print(eig)
            freq = (q * eig * 1e-6 / m) ** 0.5 / (2 * np.pi )
            return freq
        else:
            eig, eigv = np.linalg.eig(hessian)
            freq = (q * eig * 1e-6 / m) ** 0.5 / (2 * np.pi )
            return freq, eigv
    
