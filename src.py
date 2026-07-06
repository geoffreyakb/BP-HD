import os
import sys
sys.path.append(os.getenv("IDEFIX_DIR"))
import numpy as np
import matplotlib.pyplot as plt
from pytools.vtk_io import readVTK
import inifix
import cv2
import matplotlib.ticker as tkr
import matplotlib.colors as mcolors

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'text.latex.preamble':r'\usepackage{amsmath}',
    'text.latex.preamble':r'\usepackage{fourier}'
})
w, l, l_log = 1.25, 10, 6
pad, lpad, shrink = 0.1, 5, 0.8

def READ_BOX_AVERAGE():
    fid = open("output/analysis/globalAverage.dat", "r")
    varnames = fid.readline().split()
    fid.close()
    data = np.loadtxt("output/analysis/globalAverage.dat",skiprows=1)
    V = {}
    i = 0
    for name in varnames:
        V[name] = data[:,i]           
        i += 1

    return V["t"], V["Mtot"]

def READ_RADIAL_AVERAGE(n_average, n_r, beta_0):
    Sigma = np.zeros((n_average, n_r))
    rho = np.zeros((n_average, n_r))
    L = np.zeros((n_average, n_r, 3))
    beta = np.zeros((n_average, n_r))
    gamma = np.zeros((n_average, n_r))
    LBH = np.zeros((n_average, n_r, 3))
    betaBH = np.zeros((n_average, n_r))
    gammaBH = np.zeros((n_average, n_r))
    rho_Vr = np.zeros((n_average, n_r))
    rho_Vperp = np.zeros((n_average, n_r))

    current_number = 0
    for i in range(n_average):
        if i < 10:
            current_number = f"000{i}"
        elif i < 100:
            current_number = f"00{i}"
        elif i < 1000:
            current_number = f"0{i}"
        else:
            current_number = i

        fid=open(f"output/analysis/radialAverage_{current_number}.dat","r")
        varnames=fid.readline().split()
        fid.close()
        data=np.loadtxt(f"output/analysis/radialAverage_{current_number}.dat",skiprows=1)
        V={}
        j=0
        for name in varnames:
            V[name]=data[:,j]
            j=j+1

        Sigma[i,:] = V['Sigma']
        rho[i,:] = Sigma[i,:] / (2*V["r"])
        L[i,:,0] = V['Lx']
        L[i,:,1] = V['Ly']
        L[i,:,2] = V['Lz']
        LBH[i,:,0] = np.cos(beta_0)*L[i,:,0] + np.sin(beta_0)*L[i,:,2]
        LBH[i,:,1] = L[i,:,1]
        LBH[i,:,2] = -np.sin(beta_0)*L[i,:,0] + np.cos(beta_0)*L[i,:,2]

        # norm = np.sqrt(L[i,:,0]**2 + L[i,:,1]**2 + L[i,:,2]**2)
        # beta[i,:] = np.arccos(L[i,:,2] / norm) * 180/np.pi
        # gamma[i,:] = np.arctan2(L[i,:,1], L[i,:,0]) * 180/np.pi
        norm = np.sqrt(LBH[i,:,0]**2 + LBH[i,:,1]**2 + LBH[i,:,2]**2)
        betaBH[i,:] = np.arccos(LBH[i,:,2] / norm) * 180/np.pi
        gammaBH[i,:] = np.arctan2(LBH[i,:,1], LBH[i,:,0]) * 180/np.pi

        rho_Vr[i,:] = V['rho_Vr']
        rho_Vperp[i,:] = V['rho_Vperp']
    
    # wh = (beta < 1e-1)
    # gamma[wh] *= 0
    wh = (betaBH < 1e-1)
    gammaBH[wh] *= 0
    
    # return V["r"], Sigma, rho, L, beta, gamma, rho_Vr, rho_Vperp
    return V["r"], Sigma, rho, LBH, betaBH, gammaBH, rho_Vr, rho_Vperp

def READ_VTK(n_vtk):
    NVAR = 5
    RHO = 0
    VX1 = 1
    VX2 = 2
    VX3 = 3
    INVDT = 4

    if n_vtk >= 1000:
        current_number = str(n_vtk)
    elif n_vtk >= 100:
        current_number = '0' + str(n_vtk)
    elif n_vtk >= 10:
        current_number = '00' + str(n_vtk)
    else:
        current_number = '000' + str(n_vtk)
    current_VTK = readVTK('output/vtk/data.' + current_number + '.vtk', geometry='spherical')    
    
    r_vtk = current_VTK.r
    theta_vtk = current_VTK.theta
    phi_vtk = current_VTK.phi

    vtk = np.zeros((NVAR, phi_vtk.size, theta_vtk.size, r_vtk.size), dtype=float)
    vtk[RHO,:,:,:] = np.moveaxis(current_VTK.data['RHO'], [0, 2], [2, 0])
    vtk[VX1,:,:,:] = np.moveaxis(current_VTK.data['VX1'], [0, 2], [2, 0])
    vtk[VX2,:,:,:] = np.moveaxis(current_VTK.data['VX2'], [0, 2], [2, 0])
    vtk[VX3,:,:,:] = np.moveaxis(current_VTK.data['VX3'], [0, 2], [2, 0])
    vtk[INVDT,:,:,:] = np.moveaxis(current_VTK.data['InvDt'], [0, 2], [2, 0])

    rho = vtk[RHO,:,:,:]
    v_r = vtk[VX1,:,:,:]
    v_theta = vtk[VX2,:,:,:]
    v_phi = vtk[VX3,:,:,:]
    InvDT = vtk[INVDT,:,:,:]

    return r_vtk, theta_vtk, phi_vtk, rho, v_r, v_theta, v_phi, InvDT

def MOVIE(plots, name):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    images = plots
    height, width, _ = cv2.imread(images[0]).shape
    video_summary = cv2.VideoWriter(filename=f"./plots/{name}.mp4", fourcc=fourcc, fps=fps, frameSize=(width, height))
    for image in images:
        video_summary.write(cv2.imread(image))
    video_summary.release()

def MASS_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, zoom, list_plots, plots_name, time):
    densityFloor = quantities["densityFloor"]
    rho = quantities["rho"]
    beta_0 = quantities["beta_0"]
    r_norm = quantities["r_norm"]

    R, TH = np.meshgrid(r_vtk, theta_vtk)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6.5))
    
    phi_cut_plus = np.where(phi_vtk >= 0)[0][0]
    phi_cut_minus = np.where(phi_vtk >= np.pi)[0][0]
    X_cut_plus, X_cut_minus, Z = R*np.sin(TH)*np.cos(phi_vtk[phi_cut_plus]), R*np.sin(TH)*np.cos(phi_vtk[phi_cut_minus]), R*np.cos(TH)
    x_label = r"$x$ [$r_g$]"
    y_label = r"$z$ [$r_g$]"
    ax = axs[0]
    ax.set_title(r'$\log(\rho)$ $[$Code Units$]$')
    ticks = np.linspace(np.log10(densityFloor), 0, 5)
    pc0 = ax.pcolormesh(X_cut_plus, Z, np.log10(rho[phi_cut_plus,:,:]), cmap="jet", vmin=np.log10(densityFloor), vmax=0)
    pc0 = ax.pcolormesh(X_cut_minus, Z, np.log10(rho[phi_cut_minus,:,:]), cmap="jet", vmin=np.log10(densityFloor), vmax=0)
    formats = tkr.FormatStrFormatter('%.0f')
    cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    z = np.linspace(-r_max, r_max, int(1e3))
    ax.plot(np.sin(-beta_0)*z, np.cos(-beta_0)*z, color="black")
    ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
    ax.set_facecolor("black")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad=3)
    if zoom == True:
        ax.set_xlim((-3*r_min,3*r_min))
        xplot = np.linspace(-3*r_min, 3*r_min, 5)
        xl = [f"{i:.1f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-3*r_min, 3*r_min))
        ax.set_yticks(xplot)
    elif zoom == False:
        ax.set_xlim((-r_max,r_max))
        xplot = np.linspace(-r_max, r_max, 5)
        xl = [f"{i:.0f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-r_max, r_max))
        ax.set_yticks(xplot)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    phi_cut_plus = np.where(phi_vtk >= np.pi/2)[0][0]
    phi_cut_minus = np.where(phi_vtk >= 3*np.pi/2)[0][0]
    X_cut_plus, X_cut_minus, Z = R*np.sin(TH)*np.sin(phi_vtk[phi_cut_plus]), R*np.sin(TH)*np.sin(phi_vtk[phi_cut_minus]), R*np.cos(TH)
    x_label = r"$y$ [$r_g$]"
    y_label = r"$z$ [$r_g$]"
    ax = axs[1]
    ax.set_title(r'$\log(\rho)$ $[$Code Units$]$')
    ticks = np.linspace(np.log10(densityFloor), 0, 5)
    pc0 = ax.pcolormesh(X_cut_plus, Z, np.log10(rho[phi_cut_plus,:,:]), cmap="jet", vmin=np.log10(densityFloor), vmax=0)
    pc0 = ax.pcolormesh(X_cut_minus, Z, np.log10(rho[phi_cut_minus,:,:]), cmap="jet", vmin=np.log10(densityFloor), vmax=0)
    formats = tkr.FormatStrFormatter('%.0f')
    cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    z = np.linspace(-r_max, r_max, int(1e3))
    ax.plot(0*z, np.cos(-beta_0)*z, color="black")
    ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
    ax.set_facecolor("black")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad=3)
    if zoom == True:
        ax.set_xlim((-3*r_min,3*r_min))
        xplot = np.linspace(-3*r_min, 3*r_min, 5)
        xl = [f"{i:.1f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-3*r_min, 3*r_min))
        ax.set_yticks(xplot)
    elif zoom == False:
        ax.set_xlim((-r_max,r_max))
        xplot = np.linspace(-r_max, r_max, 5)
        xl = [f"{i:.0f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-r_max, r_max))
        ax.set_yticks(xplot)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    fig.suptitle(rf"$t\Omega_0(r={r_norm}r_g) =$ " + f"{time:.0f}")
    fig.tight_layout()
    plt.savefig(f"./output/plots/{plots_name}.png", bbox_inches='tight', dpi=200)
    plt.close()
    list_plots.append(f"./output/plots/{plots_name}.png")

def VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, cut, zoom, list_plots, plots_name, time):
    q_r = quantities["q_r"]
    q_th = quantities["q_th"]
    q_phi = quantities["q_phi"]
    buff_r = quantities["buff_r"]
    buff_th = quantities["buff_th"]
    buff_phi = quantities["buff_phi"]
    title_r = quantities["title_r"]
    title_th = quantities["title_th"]
    title_phi = quantities["title_phi"]
    beta_0 = quantities["beta_0"]
    r_norm = quantities["r_norm"]
    rho = quantities["rho"]
    densityFloor = quantities["densityFloor"]

    # coordinates -----------------------------------------------------------------------------------------------------------------------
    R, TH = np.meshgrid(r_vtk, theta_vtk)
    if cut == "xz":
        phi_cut_plus = np.where(phi_vtk >= 0)[0][0]
        phi_cut_minus = np.where(phi_vtk >= np.pi)[0][0]
        X_cut_plus, X_cut_minus, Z = R*np.sin(TH)*np.cos(phi_vtk[phi_cut_plus]), R*np.sin(TH)*np.cos(phi_vtk[phi_cut_minus]), R*np.cos(TH)
        x_label = r"$x$ [$r_g$]"
    elif cut == "yz":
        phi_cut_plus = np.where(phi_vtk >= np.pi/2)[0][0]
        phi_cut_minus = np.where(phi_vtk >= 3*np.pi/2)[0][0]
        X_cut_plus, X_cut_minus, Z = R*np.sin(TH)*np.sin(phi_vtk[phi_cut_plus]), R*np.sin(TH)*np.sin(phi_vtk[phi_cut_minus]), R*np.cos(TH)
        x_label = r"$y$ [$r_g$]"
    y_label = r"$z$ [$r_g$]"
    # ------------------------------------------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, gridspec_kw={'wspace': 0.01}, figsize=(15, 6.5))

    ax = axs[0]
    ax.set_title(title_r)
    # buff = np.max(np.abs(q_r))
    buff = buff_r
    ticks = np.linspace(-buff, buff, 5)
    pc0 = ax.pcolormesh(X_cut_plus, Z, q_r[phi_cut_plus,:,:], cmap="RdYlBu_r", vmin=-buff, vmax=buff)
    pc0 = ax.pcolormesh(X_cut_minus, Z, q_r[phi_cut_minus,:,:], cmap="RdYlBu_r", vmin=-buff, vmax=buff)
    ax.contour(X_cut_plus, Z, rho[phi_cut_plus,:,:], levels=[densityFloor*2], alpha=0.75)
    ax.contour(X_cut_minus, Z, rho[phi_cut_minus,:,:], levels=[densityFloor*2], alpha=0.75)
    formats = tkr.FormatStrFormatter('%.1e')
    cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    if cut == "xz":
        z = np.linspace(-r_max, r_max, int(1e3))
        ax.plot(np.sin(-beta_0)*z, np.cos(-beta_0)*z, color="black")
    elif cut == "yz":
        z = np.linspace(-r_max, r_max, int(1e3))
        ax.plot(0*z, np.cos(-beta_0)*z, color="black")
    ax.tick_params(axis='both', direction='in', color='black', width=w, length=l, pad=lpad)
    ax.set_facecolor("black")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad=3)
    if zoom == True:
        ax.set_xlim((-3*r_min,3*r_min))
        xplot = np.linspace(-3*r_min, 3*r_min, 5)
        xl = [f"{i:.1f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-3*r_min, 3*r_min))
        ax.set_yticks(xplot)
    elif zoom == False:
        ax.set_xlim((-r_max,r_max))
        xplot = np.linspace(-r_max, r_max, 5)
        xl = [f"{i:.0f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-r_max, r_max))
        ax.set_yticks(xplot)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    ax = axs[1]
    ax.set_title(title_th)
    # buff = np.max(np.abs(q_th))
    buff = buff_th
    ticks = np.linspace(-buff, buff, 5)
    pc0 = ax.pcolormesh(X_cut_plus, Z, q_th[phi_cut_plus,:,:], cmap="RdYlBu_r", vmin=-buff, vmax=buff)
    pc0 = ax.pcolormesh(X_cut_minus, Z, q_th[phi_cut_minus,:,:], cmap="RdYlBu_r", vmin=-buff, vmax=buff)
    ax.contour(X_cut_plus, Z, rho[phi_cut_plus,:,:], levels=[densityFloor*2], alpha=0.75)
    ax.contour(X_cut_minus, Z, rho[phi_cut_minus,:,:], levels=[densityFloor*2], alpha=0.75)
    formats = tkr.FormatStrFormatter('%.1e')
    cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    if cut == "xz":
        z = np.linspace(-r_max, r_max, int(1e3))
        ax.plot(np.sin(-beta_0)*z, np.cos(-beta_0)*z, color="black")
    elif cut == "yz":
        z = np.linspace(-r_max, r_max, int(1e3))
        ax.plot(0*z, np.cos(-beta_0)*z, color="black")
    ax.tick_params(axis='both', direction='in', color='black', width=w, length=l, pad=lpad)
    ax.set_facecolor("black")
    ax.set_xlabel(x_label)
    if zoom == True:
        ax.set_xlim((-3*r_min,3*r_min))
        xplot = np.linspace(-3*r_min, 3*r_min, 5)
        xl = [f"{i:.1f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-3*r_min, 3*r_min))
        yl = ["" for i in xplot]
        ax.set_yticks(xplot, yl)
    elif zoom == False:
        ax.set_xlim((-r_max,r_max))
        xplot = np.linspace(-r_max, r_max, 5)
        xl = [f"{i:.0f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-r_max, r_max))
        yl = ["" for i in xplot]
        ax.set_yticks(xplot, yl)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    ax = axs[2]
    ax.set_title(title_phi)
    # buff = np.max(np.abs(q_phi))
    buff = buff_phi
    ticks = np.linspace(-buff, buff, 5)
    pc0 = ax.pcolormesh(X_cut_plus, Z, q_phi[phi_cut_plus,:,:], cmap="RdYlBu_r", vmin=-buff, vmax=buff)
    pc0 = ax.pcolormesh(X_cut_minus, Z, q_phi[phi_cut_minus,:,:], cmap="RdYlBu_r", vmin=-buff, vmax=buff)
    ax.contour(X_cut_plus, Z, rho[phi_cut_plus,:,:], levels=[densityFloor*2], alpha=0.75)
    ax.contour(X_cut_minus, Z, rho[phi_cut_minus,:,:], levels=[densityFloor*2], alpha=0.75)
    formats = tkr.FormatStrFormatter('%.1e')
    cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    if cut == "xz":
        z = np.linspace(-r_max, r_max, int(1e3))
        ax.plot(np.sin(-beta_0)*z, np.cos(-beta_0)*z, color="black")
    elif cut == "yz":
        z = np.linspace(-r_max, r_max, int(1e3))
        ax.plot(0*z, np.cos(-beta_0)*z, color="black")
    ax.tick_params(axis='both', direction='in', color='black', width=w, length=l, pad=lpad)
    ax.set_facecolor("black")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad=3)
    if zoom == True:
        ax.set_xlim((-3*r_min,3*r_min))
        xplot = np.linspace(-3*r_min, 3*r_min, 5)
        xl = [f"{i:.1f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-3*r_min, 3*r_min))
        ax.set_yticks(xplot)
    elif zoom == False:
        ax.set_xlim((-r_max,r_max))
        xplot = np.linspace(-r_max, r_max, 5)
        xl = [f"{i:.0f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-r_max, r_max))
        ax.set_yticks(xplot)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    fig.suptitle(rf"$t\Omega_0(r={r_norm}r_g) =$ " + f"{time:.0f}")
    fig.tight_layout()
    plt.savefig(f"./output/plots/{plots_name}.png", bbox_inches='tight', dpi=200)
    plt.close()
    list_plots.append(f"./output/plots/{plots_name}.png")

def INVDT_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, zoom, list_plots, plots_name, time):
    InvDT = quantities["InvDT"]
    beta_0 = quantities["beta_0"]
    r_norm = quantities["r_norm"]

    R, TH = np.meshgrid(r_vtk, theta_vtk)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6.5))
    
    phi_cut_plus = np.where(phi_vtk >= 0)[0][0]
    phi_cut_minus = np.where(phi_vtk >= np.pi)[0][0]
    X_cut_plus, X_cut_minus, Z = R*np.sin(TH)*np.cos(phi_vtk[phi_cut_plus]), R*np.sin(TH)*np.cos(phi_vtk[phi_cut_minus]), R*np.cos(TH)
    x_label = r"$x$ [$r_g$]"
    y_label = r"$z$ [$r_g$]"
    ax = axs[0]
    ax.set_title(r'$\log(1/dt)$ $[$Code Units$]$')
    buff = np.max(np.abs(np.log10(InvDT)))
    ticks = np.linspace(-buff, buff, 5)
    pc0 = ax.pcolormesh(X_cut_plus, Z, np.log10(InvDT[phi_cut_plus,:,:]), cmap="jet", vmin=-buff, vmax=buff)
    pc0 = ax.pcolormesh(X_cut_minus, Z, np.log10(InvDT[phi_cut_minus,:,:]), cmap="jet", vmin=-buff, vmax=buff)
    formats = tkr.FormatStrFormatter('%.1f')
    cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    z = np.linspace(-r_max, r_max, int(1e3))
    ax.plot(np.sin(-beta_0)*z, np.cos(-beta_0)*z, color="black")
    ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
    ax.set_facecolor("black")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad=3)
    if zoom == True:
        ax.set_xlim((-3*r_min,3*r_min))
        xplot = np.linspace(-3*r_min, 3*r_min, 5)
        xl = [f"{i:.1f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-3*r_min, 3*r_min))
        ax.set_yticks(xplot)
    elif zoom == False:
        ax.set_xlim((-r_max,r_max))
        xplot = np.linspace(-r_max, r_max, 5)
        xl = [f"{i:.0f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-r_max, r_max))
        ax.set_yticks(xplot)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    phi_cut_plus = np.where(phi_vtk >= np.pi/2)[0][0]
    phi_cut_minus = np.where(phi_vtk >= 3*np.pi/2)[0][0]
    X_cut_plus, X_cut_minus, Z = R*np.sin(TH)*np.sin(phi_vtk[phi_cut_plus]), R*np.sin(TH)*np.sin(phi_vtk[phi_cut_minus]), R*np.cos(TH)
    x_label = r"$y$ [$r_g$]"
    y_label = r"$z$ [$r_g$]"
    ax = axs[1]
    ax.set_title(r'$\log(1/dt)$ $[$Code Units$]$')
    buff = np.max(np.abs(np.log10(InvDT)))
    ticks = np.linspace(-buff, buff, 5)
    pc0 = ax.pcolormesh(X_cut_plus, Z, np.log10(InvDT[phi_cut_plus,:,:]), cmap="jet", vmin=-buff, vmax=buff)
    pc0 = ax.pcolormesh(X_cut_minus, Z, np.log10(InvDT[phi_cut_minus,:,:]), cmap="jet", vmin=-buff, vmax=buff)
    formats = tkr.FormatStrFormatter('%.1f')
    cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    z = np.linspace(-r_max, r_max, int(1e3))
    ax.plot(0*z, np.cos(-beta_0)*z, color="black")
    ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
    ax.set_facecolor("black")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad=3)
    if zoom == True:
        ax.set_xlim((-3*r_min,3*r_min))
        xplot = np.linspace(-3*r_min, 3*r_min, 5)
        xl = [f"{i:.1f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-3*r_min, 3*r_min))
        ax.set_yticks(xplot)
    elif zoom == False:
        ax.set_xlim((-r_max,r_max))
        xplot = np.linspace(-r_max, r_max, 5)
        xl = [f"{i:.0f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-r_max, r_max))
        ax.set_yticks(xplot)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    fig.suptitle(rf"$t\Omega_0(r={r_norm}r_g) =$ " + f"{time:.0f}")
    fig.tight_layout()
    plt.savefig(f"./output/plots/{plots_name}.png", bbox_inches='tight', dpi=200)
    plt.close()
    list_plots.append(f"./output/plots/{plots_name}.png")


def KEPLER(r_vtk):
    return np.sqrt(1/r_vtk**3)