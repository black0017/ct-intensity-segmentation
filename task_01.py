import glob
import csv
from utils import *

basepath = './Images/slice*.nii.gz'
outpath = './LUNGS/'
contour_path = './Contours/'
paths = sorted(glob.glob(basepath))
myFile = open('lung_volumes.csv', 'w')
lung_areas = []

for c, exam_path in enumerate(paths):
    img_name = exam_path.split("/")[-1].split('.nii')[0]
    out_mask_name = outpath + img_name + "_mask"
    contour_name = contour_path + img_name + "_contour"

    ct_img = nib.load(exam_path)
    pixdim = find_pix_dim(ct_img)
    ct_numpy = ct_img.get_fdata()

    contours = intensity_seg(ct_numpy, min=-1000, max=-300)

    lungs = find_lungs(contours)
    show_contour(ct_numpy, lungs, contour_name,save=True)
    lung_mask = create_mask_from_polygon(ct_numpy, lungs)
    save_nifty(lung_mask, out_mask_name, ct_img.affine)

    lung_area = compute_area(lung_mask, find_pix_dim(ct_img))
    lung_areas.append([img_name,lung_area]) # int is ok since the units are already mm^2


with myFile:
    writer = csv.writer(myFile)
    writer.writerows(lung_areas)
