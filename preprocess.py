from resampler import cleanResampling, batchResampling
from xml2npz import createTargetsInCollection, cleanTargetsInCollection
import sys

#targets_to_clean = ['resampled', 'resampled10', 'resampled20']
#resampling_prefixes = ['resampled10', 'resampled20']
targets_to_clean = ['resampled', 'resampled100','resampled1024']
#resampling_prefixes = ['resampled100']
#resampling_bases = [.95,] #this should be same size and correspond to resampling_bases

resampling_prefixes = ['resampled1024']
resampling_bases = [1024,] #this should be same size and correspond to resampling_bases



def clean(r):
    print("Cleaning target folder ({})".format(r))
    cleanResampling(r, extension_prefixes=targets_to_clean)
    cleanTargetsInCollection(r)
    print("All xml->image targets and resampled targets have been deleted.")

if __name__=='__main__':
    if len(sys.argv) == 3:
        repertoire = sys.argv[2]
        if sys.argv[1] == 'info':
            raise NotImplementedError
        if sys.argv[1] == 'clean':
            clean(repertoire)
        elif sys.argv[1] == 'clean_resampled':
            print("Cleaning resampled images on target folder ({})".format(repertoire))
            cleanResampling(repertoire, extension_prefixes=targets_to_clean)
        elif sys.argv[1] == 'xml2images':
            clean(repertoire)
            print("Will search target folder ('{}') for xmls and create corresponding image files".format(repertoire))
            createTargetsInCollection(repertoire)
        elif sys.argv[1] == 'resample':
            print("Cleaning target folder ({}) and resampling targets to 10% and 20% of original size".format(repertoire))
            cleanResampling(repertoire, extension_prefixes=resampling_prefixes)
            batchResampling(repertoire, extension_prefixes=resampling_prefixes, resample_bases=resampling_bases)
        else:
            raise NotImplementedError
    elif len(sys.argv) == 1:
        print("""
        preprocess.py
            Preprocess layout/text-line document collection data. 
            Modified to store xml targets as npz files.
            G.Sfikas Oct 2020
        
        Examples of use:
        
        python preprocess.py clean fixtures/bessarion-mini                 #Clean all extra data
        python preprocess.py clean_resampled fixtures/bessarion-mini       #Clean only resampled images        
        python preprocess.py xml2images fixtures/bessarion-mini            #Create all text-nontext image targets given page XMLs
        python preprocess.py resample fixtures/bessarion-mini              #Create all resampled images for original documents and text-nontext image targets
        """)
    else:
        raise NotImplementedError
