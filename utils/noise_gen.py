import torch
import numpy as np, math
from pyperlin import FractalPerlin2D

import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.path import Path
import pickle

def perlin(shape:tuple, wavelengths:tuple, num_channels=3, black_prop:float=0.3,device='cpu', seed=None):
    """
        Makes fractal noise with dominant wavelength equal to max_wavelength.

        Args:
        shape : (B,H,W) tuple or array, size of tensor
        wavelength : int, wavelength of the noise in pixels
        black_prop : percentage of black (0) regions. Defaut is .5=50% of black.
        device : device for returned tensor

        Returns :
        (B,3,H,W) torch tensor of perlin noise.
    """    
    B,H,W = tuple(shape)
    lams = tuple(int(wave) for wave in wavelengths)
    # Extend image so that its integer wavelengths of noise
    W_new=int(W+(lams[0]-W%lams[0]))
    H_new=int(H+(lams[1]-H%lams[1]))
    frequency = [H_new//lams[0],W_new//lams[1]]
    if not seed:
        seed = np.random.randint(2**32)
    gen = torch.Generator().manual_seed(seed)
    # Strange 1/0.7053 factor to get images noise in range (-1,1), quirk of implementation I think...
    fp = FractalPerlin2D((B*num_channels,H_new,W_new), [frequency], [1/0.7053], generator=gen)()[:,:H,:W].reshape(B,num_channels,H,W) # (B,C,H,W) noise)

    return seed, torch.clamp((fp+(0.5-black_prop)*2)/(2*(1.-black_prop)),0,1).to(device)

def perlin_fractal(shape:tuple, max_wavelength:int, persistence=0.5,num_channels=3,black_prop:float=0.3,device='cpu'):
    """
        Makes fractal noise with dominant wavelength equal to max_wavelength.
    """
    max_num = min(6,int(math.log2(max_wavelength)))
    normalization = float(sum([persistence**(i+1) for i in range(max_num)]))
    return 1./normalization*sum([persistence**(i+1)*perlin(shape,[int(2**(-i)*max_wavelength)]*2,black_prop=black_prop,num_channels=num_channels,device=device)[1] for i in range(max_num)])


def load_pattern(pattern, array_sizes):
    array_mids = [size // 2 for size in array_sizes]
    array = np.zeros([1] + array_sizes)
    pattern = np.asarray(pattern)
    _, w, h = pattern.shape
    x1 = array_mids[0] - w//2;  x2 = x1 + w
    y1 = array_mids[1] - h//2;  y2 = y1 + h
    array[:, x1:x2, y1:y2] = pattern
    return array

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def crop_mask(a):
    try:
        coords = np.argwhere(a)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        cropped = a[x_min:x_max + 1, y_min:y_max + 1]
    except:
        return np.zeros(a.shape)
    return cropped

def generate_random_polygons(array_size, rand_sizes, samples):
    try:
        with open('unif_random_voronoi/polygons.pickle', 'rb') as handle:
            AREA = pickle.load(handle)
    except:
        AREA = dict([(n, []) for n in rand_sizes])


    for rand_size in rand_sizes:
        L = 0
        while L < samples:
            if rand_size < array_size/2:
                k = 1
                rand_points_range = range(4, 50)
            else:
                k = 2
                rand_points_range = range(3, 8)

            x, y = np.meshgrid(np.arange(k * array_size), np.arange(k * array_size))  # make a canvas with coordinates
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            for rand_points in rand_points_range:
                for _ in range(1):
                    rpoints = np.random.uniform(0, k*array_size, (rand_points, 2))
                    vor = Voronoi(rpoints)
                    regions, vertices = voronoi_finite_polygons_2d(vor)
                    rand_vertices = [[vertices[i] for i in region] for region in regions]
                    ps = [Path(rv) for rv in rand_vertices]  # make a polygon
                    grids = [p.contains_points(points) for p in ps]
                    masks = [np.asarray(grid.reshape(k*array_size, k*array_size)) for grid in grids]
                    masks = [crop_mask(m) for m in masks]

                    masks = [m for m in masks if max(m.shape)<array_size+1]
                    areas = [int(np.round(np.sqrt(np.sum(mask)))) for mask in masks]
                    # print(rand_points, areas)
                    for index in range(len(areas)):
                        try:
                            if len(AREA[areas[index]])<samples:
                                AREA[areas[index]].append(masks[index])
                        except:
                            continue
            L = len(AREA[rand_size])
            Ls = np.asarray([len(a) for a in AREA.values()])
            print(k, Ls)
    with open(f'polygons{array_size}.pickle', 'wb') as handle:
        pickle.dump(AREA, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return AREA

def plot_voronoi(array_size, rand_points):
    points = np.random.uniform(0, array_size, (rand_points, 2))
    vor = Voronoi(points)
    fig = voronoi_plot_2d(vor)

    # regions = [reg for reg in vor.regions if reg and -1 not in reg]
    regions, vertices = voronoi_finite_polygons_2d(vor)

    index = np.random.randint(0, len(regions))
    rand_region = regions[index]
    rand_vertices = [vertices[i] for i in rand_region]

    x, y = np.meshgrid(np.arange(array_size), np.arange(array_size))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    #
    p = Path(rand_vertices)  # make a polygon
    grid = p.contains_points(points)
    mask = np.transpose(np.asarray(grid.reshape(array_size, array_size), dtype=int))
    area = np.sum(mask)
    print("area: ", area, "sqrt area:",  int(np.round(np.sqrt(area))))

    plotx_points = []
    ploty_points = []

    nplotx_points = []
    nploty_points = []
    for x in range(array_size):
        for y in range(array_size):
            if mask[x, y]:
                plotx_points.append(x)
                ploty_points.append(y)
            else:
                nplotx_points.append(x)
                nploty_points.append(y)

    plt.scatter(plotx_points, ploty_points, color="darkred", s = 1)
    plt.scatter(nplotx_points, nploty_points, color="darkgray", s = 1)
    plt.savefig("voronoi.png")

def plot_voronoi_from_file(array_size, rand_size, sample):
    with open(f'polygons{array_size}.pickle', 'rb') as handle:
        data = pickle.load(handle)
    mask = np.asarray(data[rand_size][sample], dtype=int)
    init_config = load_pattern(mask.reshape(1, *mask.shape), [array_size, array_size]).reshape(array_size, array_size)
    plt.matshow(init_config)
    plt.savefig(f'polygons_area{rand_size}_sample{sample}.png')

def classify(phases):
    if phases == {"order"}:
        return "order"
    elif phases == {"chaos"} or phases == {"chaos", "max"}:
        return "chaos"
    elif phases == {"order", "chaos", "max"}:
        return "max"
    elif phases == {"order", "chaos"}:
        return "trans"
    else:
        return "no phase"


""" array_size = 30
rand_sizes = [5,6,7,8,9,10,11,12,13,14,15]
samples = 32

generate_random_polygons(array_size, rand_sizes, samples) """