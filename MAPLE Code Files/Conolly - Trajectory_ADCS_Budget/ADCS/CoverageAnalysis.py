import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Pool
from numba import njit

# --- 1) MARS & ORBIT CONSTANTS ---
marsRadius   = 3389.5
muMars       = 4.282837e4
obliquityRad = np.deg2rad(25.19)
spinAxis     = np.array([0.0, np.sin(obliquityRad), np.cos(obliquityRad)])
spinAxis    /= np.linalg.norm(spinAxis)

marsRotPeriod = 24.6167 * 3600
omegaMars     = 2 * np.pi / marsRotPeriod

altitude    = 380.0
a           = marsRadius + altitude
J2_Mars     = 1.96045e-3
T_MarsYear  = 686.98 * 86400
omega_req   = 2 * np.pi / T_MarsYear
n           = np.sqrt(muMars / a**3)
orbitPeriod = 2 * np.pi / n

cos_i = -omega_req / ((1.5) * J2_Mars * (marsRadius / a)**2 * n)
inc    = np.arccos(cos_i)
incDeg = np.rad2deg(inc)

# --- 2) HIGH-RES GRID & AREA SETUP (0.05°) ---
dt       = 0.1
lat0     = -90.0
lon0     = -180.0
dLat     = 0.05
dLon     = 0.05
latBins  = int(180.0 / dLat)
lonBins  = int(360.0 / dLon)

latEdges = np.linspace(lat0, 90.0, latBins+1)
lonEdges = np.linspace(lon0, 180.0, lonBins+1)

dLonRad    = np.deg2rad(dLon)
latCenters = (latEdges[:-1] + latEdges[1:]) / 2
cellAreas  = marsRadius**2 * dLonRad * (
    np.sin(np.deg2rad(latCenters + dLat/2)) -
    np.sin(np.deg2rad(latCenters - dLat/2))
)
areaMatrix       = np.repeat(cellAreas[:, None], lonBins, axis=1)
area_flat        = areaMatrix.ravel()
totalSurfaceArea = 4 * np.pi * marsRadius**2

# --- 3) PRECOMPUTE ORBIT FRAME VECTORS ---
X_eq = np.array([1.0, 0.0, 0.0], dtype=np.float64)
X_eq -= np.dot(X_eq, spinAxis) * spinAxis
X_eq /= np.linalg.norm(X_eq)
Y_eq = np.cross(spinAxis, X_eq)

h_inertial = -np.sin(inc) * Y_eq + np.cos(inc) * spinAxis
P = X_eq - np.dot(X_eq, h_inertial) * h_inertial
P /= np.linalg.norm(P)
Q = np.cross(h_inertial, P)

RAD2DEG = 180.0 / np.pi

@njit
def orbit_hits(orbitIdx):
    t0 = (orbitIdx - 1) * orbitPeriod
    t1 = orbitIdx * orbitPeriod
    N  = int((t1 - t0) / dt) + 1
    hits = np.empty(N, np.int64)
    cnt  = 0
    for i in range(N):
        t = t0 + i * dt
        θ = n * (t - t0)
        xi = a*(P[0]*np.cos(θ)+Q[0]*np.sin(θ))
        yi = a*(P[1]*np.cos(θ)+Q[1]*np.sin(θ))
        zi = a*(P[2]*np.cos(θ)+Q[2]*np.sin(θ))

        α = omegaMars * t
        cx, sx = np.cos(α), np.sin(α)
        x, y, z = spinAxis

        # build rotation rows
        r11 = x*x*(1-cx)+cx; r12 = x*y*(1-cx)-z*sx; r13 = x*z*(1-cx)+y*sx
        r21 = x*y*(1-cx)+z*sx; r22 = y*y*(1-cx)+cx;   r23 = y*z*(1-cx)-x*sx
        r31 = x*z*(1-cx)-y*sx; r32 = y*z*(1-cx)+x*sx; r33 = z*z*(1-cx)+cx

        xb = r11*xi + r12*yi + r13*zi
        yb = r21*xi + r22*yi + r23*zi
        zb = r31*xi + r32*yi + r33*zi

        rnorm = np.sqrt(xb*xb + yb*yb + zb*zb)
        xs, ys, zs = marsRadius*xb/rnorm, marsRadius*yb/rnorm, marsRadius*zb/rnorm

        lon = np.arctan2(ys, xs)*RAD2DEG
        lat = np.arctan2(zs, np.hypot(xs, ys))*RAD2DEG

        i_lat = int((lat - lat0)/dLat)
        i_lon = int((lon - lon0)/dLon)
        if 0 <= i_lat < latBins and 0 <= i_lon < lonBins:
            hits[cnt] = i_lat*lonBins + i_lon
            cnt += 1

    return hits[:cnt]

if __name__ == '__main__':
    print(f'Sun‑synch orbit @ {altitude:.1f} km, inc={incDeg:.2f}°')

    monthsTotal      = 10 * 12
    covered_flat     = np.zeros(latBins*lonBins, dtype=bool)
    coverageOverTime = np.zeros(monthsTotal)

    monthly_masks = []
    prevOrbits    = 0

    with Pool() as pool:
        for m in range(1, monthsTotal+1):
            numOrbits = int((m*30*86400)//orbitPeriod)
            newOrbits = range(prevOrbits+1, numOrbits+1)
            prevOrbits = numOrbits

            for hits in pool.imap_unordered(orbit_hits, newOrbits):
                uniq = np.unique(hits)
                covered_flat[uniq] = True

            # store boolean grid for animation
            monthly_masks.append(covered_flat.copy().reshape(latBins, lonBins))
            coverageOverTime[m-1] = covered_flat.dot(area_flat) / totalSurfaceArea
            print(f'Month {m:3d}: {100*coverageOverTime[m-1]:.2f}%')

    # compute new coverage line
    newCoverage = np.empty_like(coverageOverTime)
    newCoverage[0]  = coverageOverTime[0]
    newCoverage[1:] = coverageOverTime[1:] - coverageOverTime[:-1]
    months = np.arange(1, monthsTotal+1)

    # Downsample for animation (1°)
    ds = int(1.0/dLat)
    latAnim = latEdges[::ds]
    lonAnim = lonEdges[::ds]
    anim_masks = [mask[::ds,::ds] for mask in monthly_masks]

    # Animation
    fig, ax = plt.subplots(figsize=(6,4))
    mesh = ax.pcolormesh(lonAnim, latAnim, anim_masks[0], cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    #ax.set_title('Coverage Month 1')
    def update(frame):
        mesh.set_array(anim_masks[frame].ravel())
        #ax.set_title(f'Coverage Month {frame+1}')
        return mesh,
    ani = animation.FuncAnimation(fig, update, frames=monthsTotal, blit=True, interval=50)
    ani.save('coverage_progression.gif', writer='pillow', fps=20)

    # Combined plot
    plt.figure()
    plt.plot(months, 100*coverageOverTime,    '-o', markersize=1, label='Cumulative')
    plt.plot(months, 100*newCoverage,         '-o', markersize=1,label='New')
    plt.xlabel('Month'); plt.ylabel('% Surface')
    plt.title('Coverage & New Coverage'); plt.legend(); plt.grid()

    # Separate plots
    plt.figure()
    plt.plot(months, 100*coverageOverTime, '-o', markersize=1)
    plt.xlabel('Month'); plt.ylabel('Cumulative Coverage (%)')
    plt.title('Cumulative Coverage'); plt.grid()

    plt.figure()
    plt.plot(months, 100*newCoverage, '-o', markersize=1)
    plt.xlabel('Month'); plt.ylabel('Monthly New Coverage (%)')
    plt.title('Monthly New Coverage'); plt.grid()

    plt.show()

