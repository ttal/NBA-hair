from PIL import Image
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def get_images(active=True, outdir='player_images', outlist='player_names.csv'):
    import bs4, urlgrabber, httplib

    if active:
        list = 'http://stats.nba.com/frags/stats-site-page-players-directory-active.html'
    else:
        list = 'http://stats.nba.com/players.html'

    # prepare player list
    flist = open(outlist, 'w')
    flist.write('# name\n')

    # fetch and parse the NBA player list
    player_page = urlgrabber.urlread(list)
    soup = bs4.BeautifulSoup(player_page)

    # loop through the player list
    for p in soup('a', 'playerlink'):
        phref = str(p['href'])

        ## exclude "historical" players
        #if (len(phref.split('HISTADD')) == 1):

        # verify that player pages exist
        pname = phref.split('/')[-1]
        conn = httplib.HTTPConnection('i.cdn.turner.com')
        conn.request('HEAD', '/nba/nba/.element/img/2.0/sect/statscube/players/large/'+pname+'.png')
        if (conn.getresponse().status != 404):

            # download and save player images
            img_link = 'http://i.cdn.turner.com/nba/nba/.element/img/2.0/sect/statscube/players/large/'+pname+'.png'
            urlgrabber.urlgrab(img_link, filename=outdir+'/'+pname+'.png')

            # write player names to list
            flist.write(pname+'\n')

    # close name list
    flist.close()

    return

def convert_to_greyscale(name_list, indir='player_images', outdir='greyscale'):

    # loop through the list and write greyscale images
    for player in name_list.name:
        Image.open(indir+'/'+player+'.png').convert('LA').save(outdir+'/'+player+'.png')

    return

def combine_lum_alpha(infile):

    # load pixel values of greyscale image
    img = Image.open(infile)
    pix = img.load()
    global imsize
    imsize = img.size
    
    # save luminance and alpha values to numpy arrays
    img_lum, img_alpha = np.zeros([imsize[1], imsize[0]]), np.zeros([imsize[1], imsize[0]])
    for i in range(imsize[0]):
        for j in range(imsize[1]):
            img_lum[j,i], img_alpha[j,i] = pix[i,j][0], pix[i,j][1]
            
    # normalize pixel values
    img_lum /= np.max(img_lum)
    img_alpha /= np.max(img_alpha)

    # combine luminance with alpha and set background to "white"
    combined = img_lum * img_alpha
    combined[img_alpha < 0.5] = 1

    return combined

def flatten_array(inarray):
    # flatten image into a single array
    return np.hstack(inarray)

def median_filter(inimg, thresh=2.5, sm_fac=5.0, display_image=False):

    med_val = np.median(inimg[inimg < 1.0])
    smoothed = ndimage.uniform_filter(inimg, sm_fac)

    w_gt_thresh = smoothed > med_val / thresh
    w_lt_thresh = smoothed <= med_val / thresh

    smoothed[w_gt_thresh] = 0.0
    smoothed[w_lt_thresh] = 1.0
        
    if display_image:
        plt.imshow(smoothed, cmap='Greys'); raw_input(); plt.clf()

    return smoothed

def identify_hair(inarray, minpix=10, ythresh=30):

    # identify continuous regions
    labels, n_labels = ndimage.label(inarray)
    label_indices = [(labels == i).nonzero() for i in xrange(1, n_labels+1)]

    # find topmost region which is larger than minpix pixels
    ymin, ntop = 1e3, 1e3
    for n in range(n_labels):
        ymin_n = np.min(label_indices[n][0])
        if ((len(label_indices[n][0]) >= minpix) & (ymin_n < ymin) & (ymin_n < ythresh)):
            ymin, ntop = ymin_n, n
    
    if ntop == 1e3: ntop = np.NaN

    return label_indices, ntop

def extend_selection(inarray, ntop, sigma=3.0, thresh=0.1):

    g_smoothed = np.zeros(imsize)
    g_smoothed[inarray[ntop][1], inarray[ntop][0]] = 1
    g_smoothed = ndimage.filters.gaussian_filter(g_smoothed, 3.0)
    g_smoothed[g_smoothed <= thresh] = 0
    g_smoothed[g_smoothed > thresh] = 1

    return g_smoothed

def make_dict(inlist, target_names, greyscale_dir='greyscale'):

    flat_img, target_coverage = np.array([]), np.array([])
    for p, player in enumerate(inlist.name):
        reduced_arr = combine_lum_alpha(greyscale_dir+'/'+player+'.png')
        flat_img = np.append(flat_img, flatten_array(reduced_arr))
        target_coverage = np.append(target_coverage, inlist.coverage[p])

    flat_img = flat_img.reshape(len(inlist.name), len(flat_img)/len(inlist.name))
    img_dict = {'data':flat_img, 'target':target_coverage, 'target_names':target_names}

    return img_dict

def train_nnw(training_dict):
    from sklearn import svm#, cross_validation

    clf = svm.SVC(C=1, kernel='linear')
    clf.fit(training_dict['data'], training_dict['target'])

    return clf

def fit_images(clf, name_list, display_images=False):

    for player in name_list['name']:
        reduced_arr = combine_lum_alpha('greyscale/'+player+'.png')
        print player, clf.predict(flatten_array(reduced_arr))[0]

        if display_images:
            img = Image.open('player_images/'+player+'.png')
            plt.imshow(img); raw_input(); plt.clf()


if __name__ == '__main__':
    #get_images()

    name_list = read_csv('player_names.csv')

    #convert_to_greyscale(name_list)

    #training_list = read_csv('training_binary2.csv')
    #training_dict = make_dict(training_list, target_names=['no', 'yes'])

    #clf = train_nnw(training_dict)
    
    #fit_images(clf, name_list, display_images=True)

    for player in name_list['name']:
        print player
        colimg = Image.open('player_images/'+player+'.png')
        img = combine_lum_alpha('greyscale/'+player+'.png')
        smoothed = median_filter(img, display_image=False)
        features, ntop = identify_hair(smoothed)
        if (ntop == ntop): 
            #g_smoothed = extend_selection(features, ntop)
            #plt.contour(zip(*g_smoothed), color='r'); raw_input(); plt.clf()
            
            hair_arr = np.zeros(imsize)
            hair_arr[features[ntop][1], features[ntop][0]] = 1
            plt.imshow(colimg); plt.contour(zip(*hair_arr), color='r'); raw_input(); plt.clf()

        else:
            plt.imshow(colimg); raw_input(); plt.clf()
