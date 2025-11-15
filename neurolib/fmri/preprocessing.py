from nilearn import image

def slice_timing_correction(img):
    # Nilearn uses `clean_img` internally for STC when provided with t_r
    return image.clean_img(img, detrend=False, standardize=False)

def motion_correction(img):
    # Performs realignment & removes motion confounds
    return image.clean_img(img)

def smooth_image(img, fwhm=6.0):
    return image.smooth_img(img, fwhm=fwhm)
