from scipy.optimize import minimize
from copy import deepcopy
# import torch
# from torch import tensor
from scipy.optimize import NonlinearConstraint, LinearConstraint
from collections import defaultdict
import display
from potential_initialization import *
import ndsplines
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from scipy import constants
from scipy.misc import derivative

def mesh3d(mesh,**kwargs):
    vertices = mesh.vertices
    triangles = mesh.faces
    x, y, z = vertices.T
    I, J, K = triangles.T              
    Xe = []
    Ye = []
    Ze = []
    for T in vertices[triangles] :
        Xe.extend([T[k%3][0] for k in range(4)]+[None])
        Ye.extend([T[k%3][1] for k in range(4)]+[None])
        Ze.extend([T[k%3][2] for k in range(4)]+[None])
    return (go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,flatshading=True,showscale=False,**kwargs),)
            #go.Scatter3d(x=Xe,y=Ye,z=Ze,mode='lines',line=dict(color='rgb(70,70,70)',width=1),name='')) 
            
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

class group_constrain_info:
    # stores grouping and constrain information for electrododes
    def __init__(self, fixed=None, gi=None, cons=None) -> None:
        self.gi = gi
        self.fixed = fixed
        self.cons = cons

    def group2idx(self):
        res = defaultdict(lambda:[])
        for idx in self.gi:
            res[self.gi[idx]].append(idx)
        return res 
    
    def guess(self):
        return np.array(list(self.cons.values())).mean(1)


class trap_model:
    def __init__(self, name, V_rf, omega_rf, shuttle_range, stepsize, pbi=None, ppi=None, regenerate=False) -> None:
        #print("这tqdm有大病")
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
        
        self.weight = [0, 100, 0, 1000, 1, 0.001, 0.001, 0, 0.0005, 0]
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
        electrodes = trimesh.graph.split(self.mesh,False)
        if electrodode is not None:
            if isinstance(electrodode, str):
                electrodode = [electrodode]
            part_id = []
            for e in electrodode:
                part_id += self.pairs[e]
        elif sector is not None:
            if isinstance(sector, int):
                sector = [sector]
            part_id = []
            for s in sector:
                part_id += self.sector2idx[s]
        if isinstance(part_id, int):
            part_id = [part_id]
        part_id = set(part_id)
        scene = dict(aspectratio=dict(x=3,y=2,z=1),zaxis=dict(range=[0,0.1]),camera=dict(eye=dict(x=0,y=-1.5,z=3)))
        layout = go.Layout(scene=scene,margin=dict(r=0,l=0,b=0,t=0))#,height=400)
        data = []
        for i in range(len(electrodes)):
            data += mesh3d(electrodes[i],color='goldenrod' if i in part_id else 'grey',hoverinfo='name',name=str(i),opacity=1 if i in part_id else 0.5)
        return go.Figure(data=data,layout=layout)

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
        
    def pack(self, props, gc):
        def _pack(prop):
            return np.array([prop[g2i[g]].sum(0) for g in gc.cons])
        
        g2i = gc.group2idx()
        if type(props) is not list:
            return  _pack(props)
        return [_pack(prop) for prop in props]
    
    def top_nearest(self, point, ntop, base_gc):
        
        ind = self.pack((self.jac(point)**2), base_gc).sum(axis=1)
        sl = sorted([(i, idx) for idx, i in zip(list(base_gc.cons), ind)], reverse=True)
        # print(sl)
        gc = deepcopy(base_gc)
        for i, idx in sl[ntop:]:
            gc.fixed[idx] = 0
            gc.cons.pop(idx)
        return gc
    
    def base_gc(self, paired=True, ground_bias=True, max_v=10):
        gi, cons, fixed = dict(), dict(), dict()

        for p in self.pairs:
            l, r = self.pairs[p]
            if paired:
                gi[l] = p
                gi[r] = p
                cons[p] = (-max_v, 0)
            else:
                gi[l] = l
                gi[r] = r
                cons[l] = cons[r] = (-max_v, 0)
        if ground_bias:
            gi[1] = 'dc ground'
            cons['dc ground'] = (-max_v/2, max_v/2)  
        return group_constrain_info(fixed, gi, cons)
        
    
    def coder(self, gc, point):
        interp, jac, hess, pseudo_interp, pseudo_jac, pseudo_hess = self.field_properties(point)
        null_voltage = np.zeros(self.nparts)
        g2i = gc.group2idx()
        for f in gc.fixed:
            null_voltage[g2i[f]] = gc.fixed[f] 
        pseudo_interp += (interp * null_voltage[:, None]).sum(axis=0)
        pseudo_jac += (jac * null_voltage[:, None]).sum(axis=0)
        pseudo_hess += (hess * null_voltage[:, None, None]).sum(axis=0)
        interp, jac, hess = self.pack([interp, jac, hess], gc)
        return interp, jac, hess, pseudo_interp, pseudo_jac, pseudo_hess


    def loss_fn(self, gc, point, omega, w, rotation):
        def func(v):
            E = (jac * v[:, None]).sum(axis=0) + pseudo_jac
            V = (interp * v[:, None]).sum(axis=0) + pseudo_interp
            hessian = (hess * v[:, None, None]).sum(axis=0) + pseudo_hess
            hessian = rotation.T @ hessian @ rotation
            P = np.array([V[0], E[0], E[1], E[2], hessian[0, 0], hessian[1, 1], hessian[2, 2], hessian[0, 1], hessian[0, 2], hessian[1, 2]])
            loss = ((P - target) ** 2 * w).sum()
            # print("loss", loss)
            return loss

        interp, jac, hess, pseudo_interp, pseudo_jac, pseudo_hess = self.coder(gc, point)
        omega = np.array(omega)
        k = (m / q * (omega * 2 * np.pi) ** 2) * 1e6
        target = np.array([0, 0, 0, 0, k[0], k[1], k[2], 0, 0, 0])
        return func
    
    
    def decoder(self, sol_v, gc):
        voltages = np.zeros(self.nparts)
        g2i = gc.group2idx()
        for g in gc.fixed:
            voltages[g2i[g]] = gc.fixed[g]
        for i, g in enumerate(gc.cons):
            voltages[g2i[g]] = sol_v[i]
        return voltages
    
    def optimize_voltage(self, x, gc, omega=[0.5, 3, 3], w=None, rotation=np.eye(3)):
        if w is None:
            w = self.weight
        loss_fn = self.loss_fn(gc, self.rf_null_point(x), omega, w, rotation)
        sol_v = minimize(loss_fn, x0=gc.guess(), bounds=list(gc.cons.values()), method="SLSQP").x
        # print(gc.cons, '\n', sol_v)
        return self.decoder(sol_v, gc)
    
    def smooth_constrain(self, gc, base_gc, v, alphadx):
        g2i = gc.group2idx()
        for active in gc.cons:
            vlast = v[g2i[active][0]]
            lb, ub = base_gc.cons[active]
            gc.cons[active] = (max(vlast-alphadx, lb), min(vlast+alphadx, ub))
    
    def optimize_voltage_profile(self, profile, alpha=5000, paired=True, top_nearest=3, ground_bias=True, max_v=10, omega=[0.5, 3, 3], w=None, plot=False):
        # alpha是电压随离子坐标的变化率上限，单位V/mm
        # alpha * 输运速度 = 电压对时间的变化率上限
        w = self.weight if w is None else w
        base_gc = self.base_gc(paired, ground_bias, max_v)
        volts, points, actuals, freqs = [], [], [], []
        dx = profile[1] - profile[0]
        top_nearest = top_nearest + 1 if ground_bias else top_nearest
        for x in tqdm(profile):
            point = self.rf_null_point(x)
            gc = self.top_nearest(point, top_nearest, base_gc)
            if len(volts) != 0:
                self.smooth_constrain(gc, base_gc, volts[-1], alpha * dx)
            v = self.optimize_voltage(x, gc, omega=omega, w=w)
            volts.append(v)
            # print("point ideal", point)
            # print("frequency",self.freq(point, v))
            # print("elec field", self.electric_field(v, point))
            point_actual = self.position(v, x, 1)
            points.append(point)
            actuals.append(point_actual.flatten())
            freqs.append(self.freq(point_actual, v).flatten())
            # print("point actual", point_actual)

        # junction.plot(part_id=elec)
        volts, points, actuals, freqs = np.array(volts), np.array(points), np.array(actuals), np.array(freqs)
        if plot:
            g2i = base_gc.group2idx()
            for g in base_gc.cons:
                v = volts[:, g2i[g][0]]
                if np.linalg.norm(v) > 1e-3:
                    plt.plot(profile, v, label=g)
            plt.legend()
            plt.show()
            for i, d in enumerate(['x', 'y', 'z']):
                plt.plot(profile, points[:, i], label='rf_null')
                plt.plot(profile, actuals[:, i], label='potential minimum')
                plt.xlabel('x/mm')
                plt.ylabel(d+'/mm')
                plt.legend()
                plt.show()
            plt.plot(profile, freqs)
        return volts

    def field_properties(self, point):
        return self.interp(point), self.jac(point), self.hess(point),\
                self.pseudo_interp(point), self.pseudo_jac(point), self.pseudo_hess(point)
    
    def get_dual(self, indices):
        return [self.dual[idx] for idx in indices]
    
    def get_all_from_sectors(self, sectors):
        if sectors is None:
            sectors = self.dc_sectors
        return [p for sector in sectors for p in self.sector2idx[sector]]
        
    def transport_motion(self, T, x_i, x_f, voltages, N_ions, protocol, Th=1e-5):
        n_seg = len(voltages)
        profile = np.linspace(x_i, x_f, n_seg)
        func = ndsplines.make_interp_spline(profile, voltages)
        q0_of_t = lambda t: protocol(t, tspan=(0, T), qspan=(x_i, x_f))
        v_of_t = lambda t: func(q0_of_t(t))
        tspan, y = self.motion(T + Th, v_of_t, self.position(voltages[0], x_i, N_ions).flatten())
        return q0_of_t, tspan, y

    def heating_one_ion_during_transportation(self, q0_of_t, tspan, y, freq=5e5*2*np.pi, plot=True):
        # computing the heating rate during single ion transportation without axis pivoting.
        heating, momentums, Vs = [], [], []
        quanta = constants.hbar * freq
        for t, vec in zip(tspan, y):
            x = vec[0]
            dqdt = vec[3] - derivative(q0_of_t, t, dx=1e-9)
            q0 = q0_of_t(t)
            q = x - q0
            # print(q, dqdt)
            momentum = 0.5 * m * (dqdt*1e-3)**2
            V = 0.5 * m * freq**2 * (q*1e-3)**2
            E =  V + momentum
            heating.append(E/quanta)
            momentums.append(momentum/quanta)
            Vs.append(V/quanta)
        if plot:
            # plt.plot(tspan, heating)
            # plt.xlabel('t/s')
            # plt.ylabel('heating rate')
            # plt.show()
            plt.plot(tspan, momentums, label='momentum')
            # plt.plot(tspan, Vs, label='potential')
            plt.legend()
        return heating
        
    def heating_one_ion(self, q0_of_t, tspan, T, y, freq=5e5*2*np.pi, plot=True):
        yvalues = y[0, tspan>T]
        maxy = np.max(yvalues)
        miny = np.min(yvalues)
        A = (maxy-miny)/2
        return (A*1e-3)**2*m*np.abs(freq)/(2*constants.hbar)

    def motion(self, T, voltage_of_t, x0):
        def func(t, v):
            v = v.reshape((N_ions*2, 3))
            x = v[:N_ions]
            dxdt = v[N_ions:].flatten()
            eom = []
            voltage = voltage_of_t(t)
            for i in range(N_ions):
                d2xdt2 = (q / m * self.electric_field(voltage, x[i])) * 1e3
                for j in range(N_ions):
                    if i == j:
                        continue
                    r = (x[i] - x[j]) * 1e-3 
                    d2xdt2 += (k0 * q ** 2 / m * r / np.linalg.norm(r) ** 3) * 1e3
                eom = np.concatenate([eom, d2xdt2])
            # print("balance point: ", self.position(voltage, x[0, 0]))
            # print(t, x, dxdt, eom)
            print(' %.4f %%' % (t / T * 100), end='\r')
            return np.concatenate([dxdt, eom])
        
        N_ions = int(len(x0) / 3)
        dxdt0 = np.zeros((N_ions, 3))
        sol = solve_ivp(func, (0, T), np.concatenate([x0, dxdt0.flatten()]))
        return sol.t, sol.y
        

    def potential(self, voltage, points):
        return (self.interp(points) * voltage[:, None]).sum(axis=0) + self.pseudo_interp(points)
    
    def electric_field(self, voltage, points):
        return -((self.jac_SI(points) * voltage[:, None]).sum(axis=0) + self.pseudo_jac_SI(points))
    
    def hess_at_point(self, voltage, point):
        return self.pseudo_hess(point) + (self.hess(point) * voltage[:, None, None]).sum(axis=0)
       
    def potential_curvature_x(self, voltage, ax=None, points=None):
        x, y, z = get_field_point_ticks(self.shuttle_range, stepsize=self.stepsize)
        points = self.rf_null_point(x)
        if ax is None:
            plt.plot(x, self.potential(voltage, points))
        else:
            ax.plot(x, self.potential(voltage, points))

    def position(self, voltage, guess_x, N_ions=1):

        def totol_energy(vector):
            # print(vector)
            points = vector.reshape((N_ions, 3))
            E = np.array([self.potential(voltage, point) for point in points]).sum()
            if N_ions > 1:
                d = np.array([np.linalg.norm(points[i] - points[j]) for i in range(N_ions) for j in range(i+1, N_ions)])
                E += (k0 * q / (d * 1e-3)).sum()
            return E
        p = self.rf_null_point(guess_x)
        guess = np.array([[guess_x + (i - N_ions / 2 + 0.5) * 0.005, p[1], p[2]] for i in range(N_ions)]).flatten() + 0.001*np.random.uniform()
        # bounds = [(guess_x-0.05, guess_x+0.05), (-0.02, 0.02), (0.5, 1)] * N_ions
        return minimize(totol_energy, x0=guess).x.reshape((N_ions, 3))
    
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
    
    
    def freq(self, point, voltages, rotate=False):
    # 返回的单位是 2*pi MHz

        hessian = self.pseudo_hess(point) + (self.hess(point) * voltages[:, None, None]).sum(axis=0)
        if rotate is False:
            eig = np.array([hessian[0, 0], hessian[1, 1], hessian[2, 2]])
            freq = (q * eig * 1e-6 / m) ** 0.5 / (2 * np.pi )
            return freq
        else:
            eig, eigv = np.linalg.eig(hessian)
            freq = (q * eig * 1e-6 / m) ** 0.5 / (2 * np.pi )
            return freq, eigv
    
