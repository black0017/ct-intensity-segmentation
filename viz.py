import glob

from sklearn.cluster import KMeans

from utils import *

basepath = './Images/slice*.nii.gz'
vessels = './Vessels/'
overlay_path = './Vessel_overlayed/'
paths = sorted(glob.glob(basepath))
myFile = open('vessel_volumes.csv', 'w')
lung_areas_csv = []
ratios = []


def split_array_coords(array, indx=0, indy=1):
    x = [array[i][indx] for i in range(len(array))]
    y = [array[i][indy] for i in range(len(array))]
    return x, y


def create_vessel_mask(lung_mask, ct_numpy, denoise=False):
    vessels = lung_mask * ct_numpy  # isolate lung area
    vessels[vessels == 0] = -1000
    vessels[vessels >= -500] = 1
    vessels[vessels < -500] = 0
    show_slice(vessels)
    if denoise:
        return denoise_vessels(lungs_contour, vessels)
    show_slice(vessels)

    return vessels


for c, exam_path in enumerate(paths):
    img_name = exam_path.split("/")[-1].split('.nii')[0]
    vessel_name = vessels + img_name + "_vessel_only_mask"
    overlay_name = overlay_path + img_name + "_vessels"

    ct_img = nib.load(exam_path)
    pixdim = find_pix_dim(ct_img)
    ct_numpy = ct_img.get_fdata()

    contours = intensity_seg(ct_numpy, -1000, -300)

    lungs_contour = find_lungs(contours)
    lung_mask = create_mask_from_polygon(ct_numpy, lungs_contour)

    lung_area = compute_area(lung_mask, find_pix_dim(ct_img))

    vessels_only = create_vessel_mask(lung_mask, ct_numpy, denoise=True)

    overlay_plot(ct_numpy, vessels_only)
    plt.savefig(overlay_name)
    plt.close()

    save_nifty(vessels_only, vessel_name, affine=ct_img.affine)

    vessel_area = compute_area(vessels_only, find_pix_dim(ct_img))
    ratio = (vessel_area / lung_area) * 100
    print(img_name, 'Vessel %:', ratio)
    lung_areas_csv.append([img_name, lung_area, vessel_area, ratio])
    ratios.append(ratio)

# Data viz here
x, y = split_array_coords(lung_areas_csv, indx=1, indy=2)
data2d = np.stack([np.asarray(x), np.asarray(y)], axis=1)

kmeans = KMeans(init="random", n_clusters=2)
kmeans.fit(data2d)
print('Centers 2D:', kmeans.cluster_centers_)
print('Slice labels:', kmeans.labels_)

class1 = [data2d[i, :] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 0]
class2 = [data2d[i, :] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 1]

plt.figure()
x, y = split_array_coords(class1)
plt.scatter(x, y, c='r')
x, y = split_array_coords(class2)
plt.scatter(x, y, c='b')
plt.title('K-means clustered data')
plt.savefig('kmeans-clustered')

plt.figure()
x, _ = split_array_coords(lung_areas_csv, indx=3, indy=2)
plt.scatter(x, np.arange(len(x)))
plt.title('Ratios of different slices')
plt.savefig('./Ratios')
plt.close()

# assign categories
categories = np.zeros(len(x), dtype=int)

for c, i in enumerate(x):
    if i > 6:
        categories[c] = int(1)

colormap = np.array(['r', 'b'])

plt.figure()
plt.scatter(np.arange(len(x)) + 1, x, c=colormap[categories])
plt.title('clustered Ratios')
plt.savefig('./clustered_ratios')
plt.close()
